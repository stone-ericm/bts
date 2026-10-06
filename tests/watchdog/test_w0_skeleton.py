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
    out = run_job("t", [("raising", raising_check), ("ok", ok_check), ("faulty", faulty_check)], root=root, clock=FixedClock(T0), notifier=None)
    by = {r.check: r for r in out.results}
    assert by["t/raising"].status is Status.CHECKER_FAILURE and "RuntimeError" in by["t/raising"].detail
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
    checks = [("unnamed", Unnamed()), ("bad-str", bad_str_check), ("partial", partial_generator),
              ("not-a-result", not_a_result), ("unserializable", unserializable), ("bad-date", bad_date), ("ok", ok_check)]
    out = run_job("t", checks, root=root, clock=FixedClock(T0), notifier=None)
    statuses = [r.status for r in out.results]
    assert statuses == [Status.CHECKER_FAILURE] * 6 + [Status.VERIFIED] and out.ok
    assert out.results[0].check == "t/unnamed" and "<unprintable>" in out.results[1].detail
    assert "1 partial result(s) discarded" in out.results[2].detail                  # transactional, stated
    assert json.loads(out.path.read_text())["results"][-1]["check"] == "W-ok"


def test_a_check_returning_nothing_is_a_checker_failure_not_silence(root):
    out = run_job("t", [("empty", lambda ctx: [])], root=root, clock=FixedClock(T0), notifier=None)
    assert [r.status for r in out.results] == [Status.CHECKER_FAILURE]


def test_a_job_is_a_singleton(root):
    with root.job_lock("t"):
        with pytest.raises(JobBusy):
            run_job("t", [("ok", ok_check)], root=root, clock=FixedClock(T0), notifier=None)
    assert run_job("u", [("ok", ok_check)], root=root, clock=FixedClock(T0), notifier=None).results


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
    out = run_job("t", [("faulty", faulty_check)], root=root, clock=FixedClock(T0), notifier=notifier(root, t))
    assert out.persist_error and "disk full" in out.persist_error and not out.ok
    assert len(t.sent) == 1 and out.results[0].status is Status.FAULT


def test_a_corrupt_notification_state_is_reported_with_results_kept(root):
    root.write_atomic(N.STATE, b'{"bad": {}}')
    out = run_job("t", [("faulty", faulty_check)], root=root, clock=FixedClock(T0), notifier=notifier(root, Transport()))
    assert out.notify_error and "NotifyStateError" in out.notify_error and out.path.exists() and not out.ok


def test_a_fault_is_sent_once_per_episode_across_restarts(root):
    t = Transport()
    for _ in range(3):                                          # repeated polling, each a new Notifier (restart)
        run_job("t", [("faulty", faulty_check)], root=root, clock=FixedClock(T0), notifier=notifier(root, t))
    assert len(t.sent) == 1 and "episode 1" in t.sent[0][1]
    (n,) = notices(root)
    assert n["status"] == "sent" and n["message_id"] == "msg-1" and n["recipient"] == "watchdog-test.invalid"


def test_fault_recovery_and_recurrence_through_the_real_path(root):
    """Round-1 B3: fault, then verified (recovery), then the same fault again is a new episode, through
    enqueue/flush with restarts in between."""
    t = Transport()
    for check in (faulty_check, faulty_check, recovered_check, recovered_check, faulty_check, faulty_check):
        run_job("t", [("w", check)], root=root, clock=FixedClock(T0), notifier=notifier(root, t))
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
    run_job("t", [("faulty", faulty_check)], root=root, clock=FixedClock(T0), notifier=notifier(root, Transport(fail=1)))
    (n,) = notices(root)
    assert n["status"] == "pending" and n["attempts"] == 1 and n["last_error"] == "RuntimeError"
    run_job("t", [("faulty", faulty_check)], root=root, clock=FixedClock(T0), notifier=notifier(root, Transport(uncertain=True)))
    (n,) = notices(root)
    assert n["status"] == "pending" and n["attempts"] == 2 and n["last_error"] == "no message id"
    ok = Transport()
    run_job("t", [("faulty", faulty_check)], root=root, clock=FixedClock(T0), notifier=notifier(root, ok))
    (n,) = notices(root)
    assert n["status"] == "sent" and len(ok.sent) == 1


def test_queue_only_mode_keeps_the_alert_for_a_later_send(root):
    """Round-1 B7: --no-send queues; a later configured flush sends that same notice exactly once."""
    run_job("t", [("faulty", faulty_check)], root=root, clock=FixedClock(T0), notifier=notifier(root, None, recipient=None))
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
    errors = []

    def run1():
        try:
            n1.flush()
        except Exception as exc:  # noqa: BLE001 - captured: the stale completion must finish cleanly (r1 B4)
            errors.append(exc)
    th = threading.Thread(target=run1)
    th.start()
    try:
        assert started.wait(10)
        clock.at = T0 + timedelta(minutes=11)
        second = Transport()
        N.Notifier(root, recipient="r", send=second, clock=clock).flush()
    finally:
        release.set()
        th.join(10)
    assert not th.is_alive() and errors == []
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


def test_the_flush_is_bounded_by_elapsed_monotonic_time_even_if_the_wall_clock_rolls_back(root):
    """Round-2 N4: the budget is elapsed monotonic time; a wall clock stepping back after every send cannot extend it."""
    clock = FixedClock(T0)
    mono = {"t": 1000.0}
    sent = []

    def slow(recipient, text):
        sent.append(text)
        mono["t"] += 60.0
        clock.at = clock.at - timedelta(hours=1)                  # the wall clock rolls back each time
        return f"m{len(sent)}"
    n = N.Notifier(root, recipient="r", send=slow, clock=clock, budget_s=90.0, max_sends=5,
                   monotonic=lambda: mono["t"])
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
    monkeypatch.setitem(wcli.JOBS, "t", (("faulty", faulty_check), ("ok", ok_check)))
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
    monkeypatch.setitem(wcli.JOBS, "t", (("faulty", faulty_check),))
    res = CliRunner().invoke(cli, ["watchdog", "run", "t", "--data-dir", str(tmp_path / "data")])
    assert res.exit_code != 0 and "--dm-recipient is required" in res.output
    res = CliRunner().invoke(cli, ["watchdog", "run", "t", "--data-dir", str(tmp_path / "data"), "--no-send"])
    assert res.exit_code == 0 and "fault" in res.output
    assert [n["status"] for n in N.load_state(OwnedRoot.under(tmp_path / "data"))["notices"].values()] == ["pending"]


# ---- gate 2: kernel-enforced write confinement ---------------------------------------------------------------------

GATE_CHILD = r"""
import json
import bts.dm
from bts.cli import cli
import bts.watchdog.cli as wcli
import bts.watchdog.notify as N
from bts.watchdog.result import CheckResult, Status
from bts.watchdog.root import OwnedRoot
D = {data!r}
sel = "802415@822934"
mode = {{"n": 0}}
def check(ctx):
    mode["n"] += 1
    st = Status.VERIFIED if mode["n"] == 3 else Status.FAULT
    return [CheckResult("W-gate", ctx.et_date, st, "gate", incident="I-gate" if st is Status.FAULT else None,
                        selection=sel)]
fails = {{"n": 1}}
def send(h, m):
    if fails["n"]:
        fails["n"] -= 1
        raise RuntimeError("first send fails")
    return "dm-ok"
bts.dm.send_dm = send
{extra}
wcli.register("gate", [("gate", check)])
for _ in range(4):                  # fault (send fails), fault (retry), recovery, recurrence
    try:
        cli(["watchdog", "run", "gate", "--data-dir", D, "--dm-recipient", "watchdog-test.invalid"], standalone_mode=False)
    except SystemExit as e:
        if e.code:
            raise
cli(["watchdog", "run", "gate", "--data-dir", D, "--no-send"], standalone_mode=False)
st = N.load_state(OwnedRoot.under(D))
notices = sorted(st["notices"].values(), key=lambda n: n["seq"])
print("SUMMARY " + json.dumps([[n["state"], n["episode"], n["status"], n["message_id"], n["attempts"]] for n in notices]))
print("GATE-OK")
"""
EXPECTED_NOTICES = [["fault", 1, "sent", "dm-ok", 2], ["recovered", 1, "sent", "dm-ok", 1],
                    ["fault", 2, "sent", "dm-ok", 1]]


def _summary(stdout):
    (line,) = [x for x in stdout.splitlines() if x.startswith("SUMMARY ")]
    return json.loads(line[len("SUMMARY "):])


@needs_sandbox
def test_gate2_the_real_cli_writes_only_beneath_its_root(data):
    """The actual `bts watchdog run` CLI path (fault, failed send, retry, recovery, recurrence, queue-only) in a
    child that the kernel lets write only beneath data/watchdog, and kills on any other write. The exact notices
    and confirmed fake deliveries are asserted, not just a clean exit."""
    (data / "picks").mkdir()
    (data / "picks" / "2026-10-06.json").write_text("{}")
    OwnedRoot.under(data).close()
    res = run_confined(GATE_CHILD.format(data=str(data), extra=""), data / "watchdog")
    assert res.returncode == 0 and "GATE-OK" in res.stdout, (res.returncode, res.stderr[-3000:])
    assert _summary(res.stdout) == EXPECTED_NOTICES
    assert (data / "picks" / "2026-10-06.json").read_text() == "{}"
    assert len(list((data / "watchdog" / "results").rglob("*.json"))) == 5


SWALLOWED_OUTSIDE_WRITE = """
_real_flush = N.Notifier.flush
def _leaky_flush(self):
    try:
        open(D + "/picks/2026-10-06.json", "w").write("outside-mutant")
    except Exception:
        pass
    return _real_flush(self)
N.Notifier.flush = _leaky_flush
"""


@needs_sandbox
def test_gate2_fails_on_an_outside_write_even_when_the_code_swallows_the_refusal(data):
    """Round-2 G1: the reviewer's caught-write mutant. Under the kill profile the child dies on the denied open, so
    the gate cannot stay green."""
    (data / "picks").mkdir()
    (data / "picks" / "2026-10-06.json").write_text("{}")
    OwnedRoot.under(data).close()
    res = run_confined(GATE_CHILD.format(data=str(data), extra=SWALLOWED_OUTSIDE_WRITE), data / "watchdog")
    assert res.returncode in (-9, 137) and "GATE-OK" not in res.stdout
    assert (data / "picks" / "2026-10-06.json").read_text() == "{}"


RED = {
    "same_byte": "b = open(P, 'rb').read()\nopen(P, 'wb').write(b)",
    "write_undo": "import os\nos.replace(P, P + '.bak')\nos.replace(P + '.bak', P)",
    "leaf_symlink": "import os\nos.symlink(P, ROOT + '/link')\nopen(ROOT + '/link', 'w').write('x')",
    "dir_fd_relative": "import os\nfd = os.open(os.path.dirname(P), os.O_RDONLY)\n"
                       "os.open(os.path.basename(P), os.O_WRONLY | os.O_TRUNC, dir_fd=fd)",
    "thread": "import threading\nerr = []\n"
              "def w():\n    try:\n        open(P, 'w').write('x')\n    except PermissionError:\n        err.append(1)\n"
              "t = threading.Thread(target=w)\nt.start()\nt.join()\nif not err:\n    raise SystemExit(3)",
    "subprocess": "import subprocess, sys\n"
                  "g = \"print('GRANDCHILD-STARTED', flush=True)\\ntry:\\n    open(%r, 'w').write('x')\\n"
                  "except PermissionError:\\n    print('GRANDCHILD-EPERM')\\n\" % P\n"
                  "r = subprocess.run([sys.executable, '-B', '-c', g], capture_output=True, text=True)\n"
                  "if 'GRANDCHILD-STARTED' not in r.stdout or 'GRANDCHILD-EPERM' not in r.stdout:\n    raise SystemExit(4)",
}


def _red_program(kind, target, root_dir):
    body = RED[kind]
    if kind not in ("thread", "subprocess"):                # wrap: the specific operation must be the refusal
        body = "try:\n" + "\n".join("    " + line for line in body.splitlines()) + \
               "\nexcept PermissionError:\n    print('EPERM-WITNESS', flush=True)\n    raise SystemExit(0)\n" \
               "raise SystemExit(5)"
    else:
        body += "\nprint('EPERM-WITNESS', flush=True)"
    return f"P = {str(target)!r}; ROOT = {str(root_dir)!r}\nprint('CHILD-STARTED', flush=True)\n" + body


@needs_sandbox
@pytest.mark.parametrize("kind", sorted(RED))
def test_gate2_red_controls_are_refused_by_the_kernel(data, kind):
    """Round-2 G3: each red control proves its child started and that the intended operation was the one refused
    (EPERM), and the outside file keeps its bytes."""
    target = data / "picks" / "x.json"
    target.parent.mkdir()
    target.write_text("{}")
    OwnedRoot.under(data).close()
    res = run_confined(_red_program(kind, target, data / "watchdog"), data / "watchdog", kill=False)
    assert res.returncode == 0, (res.returncode, res.stderr[-1000:])
    assert "CHILD-STARTED" in res.stdout and "EPERM-WITNESS" in res.stdout
    assert target.read_text() == "{}" and not (data / "picks" / "x.json.bak").exists()


@needs_sandbox
def test_gate2_kill_profile_kills_after_start_on_an_outside_write(data):
    """The kill profile's own red control: the child starts, then dies at its first outside write."""
    target = data / "picks" / "x.json"
    target.parent.mkdir()
    target.write_text("{}")
    OwnedRoot.under(data).close()
    code = (f"print('CHILD-STARTED', flush=True)\ntry:\n    open({str(target)!r}, 'w').write('x')\nexcept Exception:\n"
            "    pass\nprint('SURVIVED', flush=True)")
    res = run_confined(code, data / "watchdog")
    assert res.returncode in (-9, 137) and "CHILD-STARTED" in res.stdout and "SURVIVED" not in res.stdout
    assert target.read_text() == "{}"


# ---- W0 review r2: N1-N3 ------------------------------------------------------------------------------------------

def test_n1_checker_failure_recovery_and_recurrence_through_run_job(root):
    """The same callable fails, runs successfully (with a business fault), then fails again: the checker episode
    recovers and recurs as episode 2."""
    calls = {"n": 0}

    def transient(ctx):
        calls["n"] += 1
        if calls["n"] == 2:
            return [CheckResult("W-test", ctx.et_date, Status.FAULT, "business fault", incident="I-b", selection="s")]
        raise RuntimeError("down")
    t = Transport()
    for _ in range(3):
        run_job("t", [("transient", transient)], root=root, clock=FixedClock(T0), notifier=notifier(root, t))
    texts = [m for _, m in t.sent]
    assert len(texts) == 4
    assert "checker:t/transient" in texts[0] and "episode 1" in texts[0]
    assert any("RECOVERED [checker:t/transient]" in x for x in texts)
    assert any("[I-b]" in x and "fault" in x for x in texts)
    assert "checker:t/transient" in texts[-1] and "episode 2" in texts[-1]


def test_n2_an_unrelated_all_clear_in_the_same_batch_does_not_recover_a_live_fault(root):
    t = Transport()
    n = notifier(root, t)
    batch = [CheckResult("W-x", "2026-10-06", Status.FAULT, "bad", incident="I-bad", selection="s"),
             CheckResult("W-x", "2026-10-06", Status.VERIFIED, "other ok", incident="I-other", selection="s")]
    for _ in range(2):
        n.enqueue(batch)
        n.flush()
    assert len(t.sent) == 1 and "[I-bad]" in t.sent[0][1]
    (tgt,) = [x for x in N.load_state(root)["targets"].values() if x["incident"] == "I-bad"]
    assert tgt["open"] is True and tgt["episode"] == 1


def test_n2_an_all_clear_in_the_same_batch_as_a_fault_recovers_nothing(root):
    """The alerting guard: an incident-less all-clear for the same (check, date, selection) in the same observation
    as a live fault must not recover it."""
    t = Transport()
    n = notifier(root, t)
    batch = [CheckResult("W-x", "2026-10-06", Status.FAULT, "bad", incident="I-bad", selection="s"),
             CheckResult("W-x", "2026-10-06", Status.VERIFIED, "all clear?", selection="s")]
    for _ in range(2):
        n.enqueue(batch)
        n.flush()
    assert len(t.sent) == 1 and not any("RECOVERED" in m for _, m in t.sent)


def test_n2_incident_specific_recovery_and_its_identity(root):
    t = Transport()
    n = notifier(root, t)
    n.enqueue([CheckResult("W-x", "2026-10-06", Status.FAULT, "a", incident="I-a", selection="s"),
               CheckResult("W-x", "2026-10-06", Status.FAULT, "b", incident="I-b", selection="s")])
    n.enqueue([CheckResult("W-x", "2026-10-06", Status.VERIFIED, "a fixed", incident="I-a", selection="s")])
    n.flush()
    rec = [m for _, m in t.sent if "RECOVERED" in m]
    assert rec == [rec[0]] and "[I-a]" in rec[0]
    open_ = {x["incident"] for x in N.load_state(root)["targets"].values() if x["open"]}
    assert open_ == {"I-b"}


def test_n2_a_queue_only_backlog_is_sent_in_causal_order(root):
    q = notifier(root, None, recipient=None)
    for status in (Status.FAULT, Status.VERIFIED, Status.FAULT):
        q.enqueue([CheckResult("W-x", "2026-10-06", status, "x", incident="I-1" if status is Status.FAULT else None,
                               selection="s")])
    later = Transport()
    notifier(root, later).flush()
    texts = [m for _, m in later.sent]
    assert len(texts) == 3 and "episode 1" in texts[0] and "RECOVERED" in texts[1] and "episode 2" in texts[2]


def _generated_state(root):
    n = notifier(root, None, recipient=None)
    n.enqueue([CheckResult("W-x", "2026-10-06", Status.FAULT, "x", incident="I-1", selection="s")])
    return json.loads(root.read_bytes(N.STATE))


@pytest.mark.parametrize("corrupt", ["notice_state", "notice_episode", "notice_episode_rekeyed", "target_contradiction",
                                     "missing_queued_at", "targets_list", "bool_attempts", "bad_key",
                                     "claim_outside_sending"])
def test_n3_semantic_corruption_is_refused_and_preserved(root, corrupt):
    st = _generated_state(root)
    (nk,) = st["notices"]
    (tk,) = st["targets"]
    n, t = st["notices"][nk], st["targets"][tk]
    if corrupt == "notice_state":
        n["state"] = "unknown-state"
    elif corrupt == "notice_episode":
        n["episode"] = 999
    elif corrupt == "notice_episode_rekeyed":                     # consistent key, impossible episode
        n["episode"] = 999
        st["notices"][N.notice_key(n["target"], 999, n["state"])] = st["notices"].pop(nk)
    elif corrupt == "target_contradiction":
        t.update(open=True, state="recovered", et_date="not-a-date")
    elif corrupt == "missing_queued_at":
        del n["queued_at"]
    elif corrupt == "targets_list":
        st["targets"] = []
    elif corrupt == "bool_attempts":
        n["attempts"] = True
    elif corrupt == "bad_key":
        st["notices"]["0" * 64] = st["notices"].pop(nk)
    elif corrupt == "claim_outside_sending":
        n["claim_token"] = "x"
    raw = json.dumps(st).encode()
    root.write_atomic(N.STATE, raw)
    with pytest.raises(N.NotifyStateError):
        notifier(root, Transport()).flush()
    assert root.read_bytes(N.STATE) == raw
