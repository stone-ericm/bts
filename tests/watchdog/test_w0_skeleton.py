"""W0: the watchdog skeleton (C1 rank-2 build plan W0; registration R1, R5, §2, §4.2 gate 2; W0 review r1 B1-B7).

Covers:
- the descriptor-anchored root and its race controls;
- the kernel-enforced write gate and its red controls;
- notification episodes (fault, recovery, recurrence and escalation);
- fenced UTC leases;
- state validation;
- check isolation and output-failure reporting;
- queue-only mode.
"""
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from bts.watchdog import notify as N
from bts.watchdog import root as R
from bts.watchdog.clock import FixedClock
from bts.watchdog.result import CheckResult, Status
from bts.watchdog.root import JobBusy, OwnedRoot, RootError
from bts.watchdog.runner import run_job
from tests.watchdog.conftest import needs_sandbox, run_confined

ET = ZoneInfo("America/New_York")
T0 = datetime(2026, 10, 6, 12, 0, tzinfo=ET)


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (tmp_path / "outside").mkdir()
    return d


@pytest.fixture
def root(data):
    return OwnedRoot.under(data)


# ---- B1: the descriptor-anchored root ----------------------------------------------------------------------------

def test_names_are_plain_components(root):
    assert root.child("notify", "state.json") == root.path / "notify" / "state.json"
    for bad in ("..", ".", "", "a/b", "/etc", "x y"):
        with pytest.raises(RootError, match="refused path component"):
            root.child(bad)


def test_a_symlinked_root_is_refused(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "data" / "watchdog").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(RootError, match="is a symlink: refused"):
        OwnedRoot.under(tmp_path / "data")


def test_a_symlinked_component_or_leaf_is_refused_before_any_write(root, tmp_path):
    outside = tmp_path / "outside"
    (root.path / "notify").symlink_to(outside)
    with pytest.raises(RootError, match="symlink"):
        root.write_atomic(("notify", "state.json"), b"x")
    (root.path / "real").mkdir()
    (root.path / "alias").symlink_to(root.path / "real")                 # a symlink pointing inside the root
    with pytest.raises(RootError, match="symlink"):
        root.write_atomic(("alias", "x.json"), b"x")
    (root.path / "locks").mkdir()
    (root.path / "locks" / "job-t.lock").symlink_to(outside / "planted.lock")
    with pytest.raises(RootError, match="symlink"):
        with root.job_lock("t"):
            pass
    assert list(outside.iterdir()) == []


def _swap_before(monkeypatch, fn_name, when, swap):
    """Run `swap()` once, just before the root module's `os.<fn_name>` is called with a first argument matching
    `when`. This interleaving is the round-1 race."""
    real = getattr(os, fn_name)
    done = {"x": False}

    def wrapped(*a, **k):
        if not done["x"] and a and when(a[0]):
            done["x"] = True
            swap()
        return real(*a, **k)
    monkeypatch.setattr(R.os, fn_name, wrapped)


def test_a_directory_swapped_to_a_symlink_after_admission_cannot_redirect_a_write(root, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    root.ensure_dir("results")

    def swap():
        os.rename(root.path / "results", root.path / "results.moved")
        (root.path / "results").symlink_to(outside)
    _swap_before(monkeypatch, "open", lambda name: isinstance(name, str) and name.endswith(".tmp"), swap)
    root.write_atomic(("results", "r.json"), b"escaped?")
    assert list(outside.iterdir()) == []
    assert (root.path / "results.moved" / "r.json").read_bytes() == b"escaped?"   # the admitted directory's inode


def test_ensure_dir_swap_before_mkdir_cannot_create_outside(root, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    root.ensure_dir("branch")

    def swap():
        os.rename(root.path / "branch", root.path / "branch.moved")
        (root.path / "branch").symlink_to(outside)
    _swap_before(monkeypatch, "mkdir", lambda name: name == "created", swap)
    root.ensure_dir("branch", "created")
    assert list(outside.iterdir()) == [] and (root.path / "branch.moved" / "created").is_dir()


def test_a_root_replaced_after_construction_cannot_redirect(root, tmp_path):
    outside = tmp_path / "outside"
    os.rename(root.path, root.path.with_name("watchdog.moved"))
    root.path.symlink_to(outside)
    root.ensure_dir("new", "nested")
    assert list(outside.iterdir()) == []
    assert (root.path.with_name("watchdog.moved") / "new" / "nested").is_dir()


def test_concurrent_atomic_writes_never_touch_each_others_files(root, monkeypatch):
    """Unique, exclusive temps: a paused writer cannot rewrite another writer's published file, and both
    replacements succeed whole."""
    real_write = os.write
    gate, paused = threading.Event(), threading.Event()
    errors, first = [], {"x": True}

    def slow_write(fd, data):
        if first["x"] and bytes(data) == b"AAAA":
            first["x"] = False
            paused.set()
            assert gate.wait(10)
        return real_write(fd, data)
    monkeypatch.setattr(R.os, "write", slow_write)

    def a():
        try:
            root.write_atomic(("x", "t.json"), b"AAAA")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    th = threading.Thread(target=a)
    th.start()
    try:
        assert paused.wait(10)
        root.write_atomic(("x", "t.json"), b"BBBB")
        assert (root.path / "x" / "t.json").read_bytes() == b"BBBB"
    finally:
        gate.set()
        th.join(10)
    assert not th.is_alive() and errors == []
    assert (root.path / "x" / "t.json").read_bytes() == b"AAAA"                  # a later, whole replacement
    assert [p for p in os.listdir(root.path / "x") if p.endswith(".tmp")] == []


# ---- results and the runner (B6) ---------------------------------------------------------------------------------

def ok_check(ctx):
    return [CheckResult("W-ok", ctx.et_date, Status.VERIFIED, "fine")]


def faulty_check(ctx):
    return [CheckResult("W-bad", ctx.et_date, Status.FAULT, "missing delivery", incident="I-test",
                        selection="802415@822934")]


def recovered_check(ctx):
    return [CheckResult("W-bad", ctx.et_date, Status.VERIFIED, "delivered", selection="802415@822934")]


def raising_check(ctx):
    raise RuntimeError("boom")


def test_every_check_runs_and_an_exception_is_a_checker_failure(root):
    out = run_job("t", [raising_check, ok_check, faulty_check], root=root, clock=FixedClock(T0), notifier=None)
    by = {r.check: r for r in out.results}
    assert by["raising_check"].status is Status.CHECKER_FAILURE and "RuntimeError" in by["raising_check"].detail
    assert by["W-ok"].status is Status.VERIFIED and by["W-bad"].status is Status.FAULT
    saved = json.loads(out.path.read_text())
    assert saved["job"] == "t" and saved["et_date"] == "2026-10-06" and len(saved["results"]) == 3 and out.ok


class Unnamed:
    def __call__(self, ctx):
        raise ValueError("x")

    def __repr__(self):
        raise RuntimeError("repr")


class BadStr(Exception):
    def __str__(self):
        raise RuntimeError("str")


def bad_str_check(ctx):
    raise BadStr()


def partial_generator(ctx):
    yield CheckResult("W-part", ctx.et_date, Status.FAULT, "half", selection="s")
    raise RuntimeError("midway")


def not_a_result(ctx):
    return ["nope"]


def unserializable(ctx):
    return [CheckResult("W-u", ctx.et_date, Status.VERIFIED, "x", evidence={"o": object()})]


def bad_date(ctx):
    return [CheckResult("W-d", "not-a-date", Status.VERIFIED, "x")]


def test_isolation_survives_hostile_checks_and_outputs(root):
    """Round-1 B6: a raising __repr__, a raising __str__, invalid output and a partial generator each become one
    checker_failure, and every following check still runs."""
    checks = [Unnamed(), bad_str_check, partial_generator, not_a_result, unserializable, bad_date, ok_check]
    out = run_job("t", checks, root=root, clock=FixedClock(T0), notifier=None)
    statuses = [r.status for r in out.results]
    assert statuses == [Status.CHECKER_FAILURE] * 6 + [Status.VERIFIED] and out.ok
    assert out.results[0].check == "check#0" and "<unprintable>" in out.results[1].detail
    assert "1 partial result(s) discarded" in out.results[2].detail                  # transactional, stated
    assert json.loads(out.path.read_text())["results"][-1]["check"] == "W-ok"


def test_a_check_returning_nothing_is_a_checker_failure_not_silence(root):
    out = run_job("t", [lambda ctx: []], root=root, clock=FixedClock(T0), notifier=None)
    assert [r.status for r in out.results] == [Status.CHECKER_FAILURE]


def test_a_job_is_a_singleton(root):
    with root.job_lock("t"):
        with pytest.raises(JobBusy):
            run_job("t", [ok_check], root=root, clock=FixedClock(T0), notifier=None)
    assert run_job("u", [ok_check], root=root, clock=FixedClock(T0), notifier=None).results


# ---- notifications ------------------------------------------------------------------------------------------------

class Transport:
    def __init__(self, fail=0, uncertain=False):
        self.sent, self.fail, self.uncertain = [], fail, uncertain

    def __call__(self, recipient, text):
        if self.fail:
            self.fail -= 1
            raise RuntimeError("send failed")
        self.sent.append((recipient, text))
        return "" if self.uncertain else f"msg-{len(self.sent)}"


def notifier(root, transport, clock=None, recipient="watchdog-test.invalid"):
    return N.Notifier(root, recipient=recipient, send=transport, clock=clock or FixedClock(T0))


def notices(root):
    return list(N.load_state(root)["notices"].values())


def test_a_results_write_failure_is_reported_and_notification_still_runs(root, monkeypatch):
    t = Transport()
    real = root.write_atomic

    def failing(parts, data):
        if parts[0] == "results":
            raise OSError("disk full")
        return real(parts, data)
    monkeypatch.setattr(root, "write_atomic", failing)
    out = run_job("t", [faulty_check], root=root, clock=FixedClock(T0), notifier=notifier(root, t))
    assert out.persist_error and "disk full" in out.persist_error and not out.ok
    assert len(t.sent) == 1 and out.results[0].status is Status.FAULT


def test_a_corrupt_notification_state_is_reported_with_results_kept(root):
    root.write_atomic(N.STATE, b'{"bad": {}}')
    out = run_job("t", [faulty_check], root=root, clock=FixedClock(T0), notifier=notifier(root, Transport()))
    assert out.notify_error and "NotifyStateError" in out.notify_error and out.path.exists() and not out.ok


def test_a_fault_is_sent_once_per_episode_across_restarts(root):
    t = Transport()
    for _ in range(3):                                          # repeated polling, each a new Notifier (restart)
        run_job("t", [faulty_check], root=root, clock=FixedClock(T0), notifier=notifier(root, t))
    assert len(t.sent) == 1 and "episode 1" in t.sent[0][1]
    (n,) = notices(root)
    assert n["status"] == "sent" and n["message_id"] == "msg-1" and n["recipient"] == "watchdog-test.invalid"


def test_fault_recovery_and_recurrence_through_the_real_path(root):
    """Round-1 B3: fault, then verified (recovery), then the same fault again is a new episode, through
    enqueue/flush with restarts in between."""
    t = Transport()
    for check in (faulty_check, faulty_check, recovered_check, recovered_check, faulty_check, faulty_check):
        run_job("t", [check], root=root, clock=FixedClock(T0), notifier=notifier(root, t))
    texts = [m for _, m in t.sent]
    assert len(texts) == 3
    assert "episode 1" in texts[0] and "RECOVERED" in texts[1] and "episode 2" in texts[2]


def test_an_escalation_within_an_episode_is_notified_once(root):
    t = Transport()
    n = notifier(root, t)
    base = dict(check="W-x", et_date="2026-10-06", detail="d", incident="I-1", selection="s")
    for status in (Status.FAULT, Status.FAULT, Status.CHECKER_FAILURE, Status.CHECKER_FAILURE):
        n.enqueue([CheckResult(status=status, **base)])
        n.flush()
    assert len(t.sent) == 2 and "escalated to checker_failure" in t.sent[1][1]


def test_distinct_incidents_dates_and_selections_are_separate_targets(root):
    t = Transport()
    n = notifier(root, t)
    n.enqueue([CheckResult("W-x", d, Status.FAULT, "x", incident=i, selection=s)
               for i, d, s in (("I-1", "2026-10-06", "s"), ("I-2", "2026-10-06", "s"),
                               ("I-1", "2026-10-07", "s"), ("I-1", "2026-10-06", "other"))])
    n.flush()
    assert len(t.sent) == 4


def test_pending_and_unverifiable_change_no_episode(root):
    t = Transport()
    n = notifier(root, t)
    n.enqueue([CheckResult("a", "2026-10-06", s, "x") for s in (Status.VERIFIED, Status.PENDING, Status.UNVERIFIABLE)])
    n.flush()
    assert t.sent == [] and N.load_state(root)["notices"] == {}


def test_failed_and_uncertain_sends_stay_pending_and_retry(root):
    run_job("t", [faulty_check], root=root, clock=FixedClock(T0), notifier=notifier(root, Transport(fail=1)))
    (n,) = notices(root)
    assert n["status"] == "pending" and n["attempts"] == 1 and n["last_error"] == "RuntimeError"
    run_job("t", [faulty_check], root=root, clock=FixedClock(T0), notifier=notifier(root, Transport(uncertain=True)))
    (n,) = notices(root)
    assert n["status"] == "pending" and n["attempts"] == 2 and n["last_error"] == "no message id"
    ok = Transport()
    run_job("t", [faulty_check], root=root, clock=FixedClock(T0), notifier=notifier(root, ok))
    (n,) = notices(root)
    assert n["status"] == "sent" and len(ok.sent) == 1


def test_queue_only_mode_keeps_the_alert_for_a_later_send(root):
    """Round-1 B7: --no-send queues; a later configured flush sends that same notice exactly once."""
    run_job("t", [faulty_check], root=root, clock=FixedClock(T0), notifier=notifier(root, None, recipient=None))
    (n,) = notices(root)
    assert n["status"] == "pending"
    later = Transport()
    notifier(root, later).flush()
    notifier(root, later).flush()
    assert len(later.sent) == 1


def _blocking_sender(started, release, out):
    def send(recipient, text):
        started.set()
        if not release.wait(10):
            raise AssertionError("never released")
        out.append(text)
        return "msg-slow"
    return send


def test_the_state_lock_is_free_while_a_send_is_in_flight(root):
    """Round-1 test fix: the first sender blocks until an explicit release. A contending flush must complete while
    it is still blocked (the lock is not held during network work), and must not send the claimed notice."""
    started, release, slow_out, errors = threading.Event(), threading.Event(), [], []
    n1 = notifier(root, _blocking_sender(started, release, slow_out))
    n1.enqueue([CheckResult("W-bad", "2026-10-06", Status.FAULT, "x", incident="I-1", selection="s")])

    def run1():
        try:
            n1.flush()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    th = threading.Thread(target=run1)
    th.start()
    try:
        assert started.wait(10)
        other, done = Transport(), threading.Event()
        th2 = threading.Thread(target=lambda: (notifier(root, other).flush(), done.set()))
        th2.start()
        th2.join(5)
        assert done.is_set() and not th2.is_alive(), "a contending flush blocked behind an in-flight send"
        assert other.sent == [] and not slow_out
    finally:
        release.set()
        th.join(10)
    assert not th.is_alive() and errors == [] and slow_out
    (n,) = notices(root)
    assert n["status"] == "sent" and n["message_id"] == "msg-slow"


def _claim_state(root, *, claimed_at, lease_until, token="t0"):
    with N.state_lock(root):
        st = N.load_state(root)
        (k,) = st["notices"]
        st["notices"][k].update(status="sending", claim_token=token, claimed_at=claimed_at, lease_until=lease_until)
        N.save_state(root, st)


def test_a_stale_sender_never_overwrites_a_newer_confirmation(root):
    """Round-1 B4: sender 1 claims and stalls; after its lease expires sender 2 sends and confirms; sender 1's late
    failure must not reset the notice to pending."""
    started, release = threading.Event(), threading.Event()
    clock = FixedClock(T0)

    def failing_slow(recipient, text):
        started.set()
        release.wait(10)
        raise RuntimeError("late failure")
    n1 = N.Notifier(root, recipient="r", send=failing_slow, clock=clock)
    n1.enqueue([CheckResult("W-bad", "2026-10-06", Status.FAULT, "x", incident="I-1", selection="s")])
    th = threading.Thread(target=n1.flush)
    th.start()
    try:
        assert started.wait(10)
        clock.at = T0 + timedelta(minutes=11)
        second = Transport()
        N.Notifier(root, recipient="r", send=second, clock=clock).flush()
    finally:
        release.set()
        th.join(10)
    (n,) = notices(root)
    assert len(second.sent) == 1 and n["status"] == "sent" and n["message_id"] == "msg-1"
    third = Transport()
    N.Notifier(root, recipient="r", send=third, clock=clock).flush()
    assert third.sent == []


@pytest.mark.parametrize("now, due", [
    ("2026-11-01T01:55:00-04:00", False),        # 05:55Z: before the 06:05Z expiry, though its wall label is "later"
    ("2026-11-01T01:04:59-05:00", False),        # 06:04:59Z
    ("2026-11-01T01:05:00-05:00", True),         # 06:05Z: expired
    ("2026-11-01T01:00:00-04:00", True),         # 05:00Z: before the claim, so the clock rolled back (retry)
])
def test_leases_use_utc_elapsed_time_across_the_fall_back(root, now, due):
    n = notifier(root, Transport())
    n.enqueue([CheckResult("W-bad", "2026-11-01", Status.FAULT, "x", incident="I-1", selection="s")])
    _claim_state(root, claimed_at="2026-11-01T05:55:00+00:00", lease_until="2026-11-01T06:05:00+00:00")
    later = Transport()
    notifier(root, later, clock=FixedClock(datetime.fromisoformat(now))).flush()
    assert (len(later.sent) == 1) is due


def test_a_claim_lease_is_computed_in_utc(root):
    clock = FixedClock(datetime.fromisoformat("2026-11-01T01:55:00-04:00"))
    started, release = threading.Event(), threading.Event()
    n = N.Notifier(root, recipient="r", send=_blocking_sender(started, release, []), clock=clock)
    n.enqueue([CheckResult("W-bad", "2026-11-01", Status.FAULT, "x", incident="I-1", selection="s")])
    th = threading.Thread(target=n.flush)
    th.start()
    try:
        assert started.wait(10)
        (st,) = notices(root)
        assert datetime.fromisoformat(st["lease_until"]) - datetime.fromisoformat(st["claimed_at"]) == timedelta(minutes=10)
        assert datetime.fromisoformat(st["lease_until"]).utcoffset() == timedelta(0)
    finally:
        release.set()
        th.join(10)


def test_a_send_accepted_before_a_crash_is_retried_after_the_lease(root):
    """Pre-send crash and accepted-but-unrecorded send both leave a claim; after the lease it is sent again
    (a duplicate is the documented limit, never a lost alert)."""
    n = notifier(root, Transport())
    n.enqueue([CheckResult("W-bad", "2026-10-06", Status.FAULT, "x", incident="I-1", selection="s")])
    _claim_state(root, claimed_at=(T0 - timedelta(minutes=20)).astimezone(timezone.utc).isoformat(),
                 lease_until=(T0 - timedelta(minutes=10)).astimezone(timezone.utc).isoformat())
    later = Transport()
    notifier(root, later).flush()
    (st,) = notices(root)
    assert len(later.sent) == 1 and st["status"] == "sent"


@pytest.mark.parametrize("raw", [b"{not json", b"[]", b'{"bad": {}}',
                                 b'{"version": 1, "targets": {}, "notices": {"k": {"status": "weird"}}}',
                                 b'{"version": 1, "targets": {}, "notices": {"k": {"status": "pending", "text": "x", '
                                 b'"target": "missing", "attempts": 0}}}'])
def test_invalid_notification_state_raises_and_is_preserved(root, raw):
    root.write_atomic(N.STATE, raw)
    with pytest.raises(N.NotifyStateError):
        notifier(root, Transport()).enqueue([CheckResult("W", "2026-10-06", Status.FAULT, "x")])
    assert root.read_bytes(N.STATE) == raw


def test_the_flush_is_bounded(root):
    clock = FixedClock(T0)
    sent = []

    def slow(recipient, text):
        sent.append(text)
        clock.at = clock.at + timedelta(seconds=60)
        return f"m{len(sent)}"
    n = N.Notifier(root, recipient="r", send=slow, clock=clock, budget=timedelta(seconds=90), max_sends=5)
    n.enqueue([CheckResult("W", "2026-10-06", Status.FAULT, "x", incident=f"I-{i}") for i in range(8)])
    report = n.flush()
    assert len(sent) == 2 and report["skipped_budget"] == 3      # 5 claimed, 2 sent within budget, 3 released
    assert sum(1 for x in notices(root) if x["status"] == "pending") == 6


# ---- the CLI ------------------------------------------------------------------------------------------------------

def test_the_cli_sends_through_the_dm_transport_and_fails_loudly(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import bts.dm
    import bts.watchdog.cli as wcli
    from bts.cli import cli
    (tmp_path / "data").mkdir()
    sent = []
    monkeypatch.setattr(bts.dm, "send_dm", lambda h, m: (sent.append((h, m)), "dm-1")[1])
    monkeypatch.setitem(wcli.JOBS, "t", [faulty_check, ok_check])
    res = CliRunner().invoke(cli, ["watchdog", "run", "t", "--data-dir", str(tmp_path / "data"),
                                   "--dm-recipient", "watchdog-test.invalid"])
    assert res.exit_code == 0, res.output
    assert len(sent) == 1 and sent[0][0] == "watchdog-test.invalid" and "I-test" in sent[0][1]
    OwnedRoot.under(tmp_path / "data").write_atomic(N.STATE, b"[]")      # corrupt state: a loud failure
    res = CliRunner().invoke(cli, ["watchdog", "run", "t", "--data-dir", str(tmp_path / "data"),
                                   "--dm-recipient", "watchdog-test.invalid"])
    assert res.exit_code == 1 and "NOTIFICATION FAILURE" in res.output


def test_the_cli_refuses_unknown_jobs_and_a_missing_recipient_and_queues_in_no_send(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import bts.watchdog.cli as wcli
    from bts.cli import cli
    (tmp_path / "data").mkdir()
    res = CliRunner().invoke(cli, ["watchdog", "run", "nope", "--data-dir", str(tmp_path / "data"), "--no-send"])
    assert res.exit_code != 0 and "unknown watchdog job" in res.output
    monkeypatch.setitem(wcli.JOBS, "t", [faulty_check])
    res = CliRunner().invoke(cli, ["watchdog", "run", "t", "--data-dir", str(tmp_path / "data")])
    assert res.exit_code != 0 and "--dm-recipient is required" in res.output
    res = CliRunner().invoke(cli, ["watchdog", "run", "t", "--data-dir", str(tmp_path / "data"), "--no-send"])
    assert res.exit_code == 0 and "fault" in res.output
    assert [n["status"] for n in N.load_state(OwnedRoot.under(tmp_path / "data"))["notices"].values()] == ["pending"]


# ---- gate 2: kernel-enforced write confinement ---------------------------------------------------------------------

GATE_CHILD = r'''
import bts.dm
from bts.cli import cli
import bts.watchdog.cli as wcli
from bts.watchdog.result import CheckResult, Status
D = {data!r}
sel = "802415@822934"
mode = {{"n": 0}}
def check(ctx):
    mode["n"] += 1
    st = Status.VERIFIED if mode["n"] == 3 else Status.FAULT
    return [CheckResult("W-gate", ctx.et_date, st, "gate", incident="I-gate", selection=sel)]
fails = {{"n": 1}}
def send(h, m):
    if fails["n"]:
        fails["n"] -= 1
        raise RuntimeError("first send fails")
    return "dm-ok"
bts.dm.send_dm = send
wcli.JOBS["gate"] = [check]
for _ in range(4):                  # fault (send fails), fault (retry), recovery, recurrence
    try:
        cli(["watchdog", "run", "gate", "--data-dir", D, "--dm-recipient", "watchdog-test.invalid"], standalone_mode=False)
    except SystemExit as e:
        if e.code:
            raise
cli(["watchdog", "run", "gate", "--data-dir", D, "--no-send"], standalone_mode=False)
print("GATE-OK")
'''


@needs_sandbox
def test_gate2_the_real_cli_writes_only_beneath_its_root(data):
    """The actual `bts watchdog run` CLI path (fault, failed send, retry, recovery, recurrence, queue-only) in a
    child that the kernel lets write only beneath data/watchdog. Any other write would fail with EPERM."""
    (data / "picks").mkdir()
    (data / "picks" / "2026-10-06.json").write_text("{}")
    OwnedRoot.under(data).close()
    res = run_confined(GATE_CHILD.format(data=str(data)), data / "watchdog")
    assert res.returncode == 0 and "GATE-OK" in res.stdout, res.stderr[-3000:]
    assert (data / "picks" / "2026-10-06.json").read_text() == "{}"
    assert list((data / "watchdog" / "results").rglob("*.json"))


RED = {
    "same_byte": "open(P, 'w').write(open(P).read())",
    "write_undo": "import os; os.replace(P, P + '.bak'); os.replace(P + '.bak', P)",
    "thread": "import threading\nerr = []\n"
              "def w():\n    try:\n        open(P, 'w').write('x')\n    except PermissionError as e:\n        err.append(e)\n"
              "t = threading.Thread(target=w); t.start(); t.join()\nraise SystemExit(0 if err else 3)",
    "subprocess": "import subprocess, sys\nr = subprocess.run([sys.executable, '-c', 'open(%r, \"w\").write(\"x\")' % P])\n"
                  "raise SystemExit(0 if r.returncode else 4)",
    "leaf_symlink": "import os; os.symlink(P, ROOT + '/link'); open(ROOT + '/link', 'w').write('x')",
    "dir_fd_relative": "import os; fd = os.open(os.path.dirname(P), os.O_RDONLY); "
                       "os.open(os.path.basename(P), os.O_WRONLY | os.O_TRUNC, dir_fd=fd)",
}


@needs_sandbox
@pytest.mark.parametrize("kind", sorted(RED))
def test_gate2_red_controls_are_refused_by_the_kernel(data, kind):
    """The gate's own red controls: each class the round-1 audit-hook tracer missed or could miss is refused with
    EPERM, and the outside file keeps its bytes."""
    target = data / "picks" / "x.json"
    target.parent.mkdir()
    target.write_text("{}")
    OwnedRoot.under(data).close()
    code = f"P = {str(target)!r}; ROOT = {str(data / 'watchdog')!r}\n" + RED[kind]
    res = run_confined(code, data / "watchdog")
    if kind in ("thread", "subprocess"):
        assert res.returncode == 0, res.stderr[-1000:]           # the child itself verified the refusal
    else:
        assert res.returncode != 0 and ("Operation not permitted" in res.stderr or "PermissionError" in res.stderr), \
            res.stderr[-1000:]
    assert target.read_text() == "{}" and not (data / "picks" / "x.json.bak").exists()
