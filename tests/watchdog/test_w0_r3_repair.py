"""C2 (a) unit 1: W0 review r3's required changes R3-1 to R3-4 (`docs/audit/2026-10-06-c1-r2-watchdog-w0-codex-r3.md`;
design `docs/superpowers/specs/2026-10-06-c2-w0-repair-design.md`)."""
import json
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from bts.watchdog import notify as N
from bts.watchdog.clock import FixedClock
from bts.watchdog.result import CheckResult, Status
from bts.watchdog.root import OwnedRoot
from bts.watchdog.runner import RegistrationError, run_job
from tests.watchdog.conftest import needs_sandbox, run_confined
from tests.watchdog.test_w0_skeleton import GATE_CHILD

ET = ZoneInfo("America/New_York")
T0 = datetime(2026, 10, 6, 12, 0, tzinfo=ET)
D = "2026-10-06"


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def root(data):
    return OwnedRoot.under(data)


class Recorder:
    """A transport that records each accepted text in order; `fail_when(text)` makes a send raise."""

    def __init__(self, fail_when=lambda text: False):
        self.accepted, self.fail_when = [], fail_when

    def __call__(self, recipient, text):
        if self.fail_when(text):
            raise RuntimeError("send failed")
        self.accepted.append(text)
        return f"msg-{len(self.accepted)}"


def notifier(root, send, clock=None, recipient="watchdog-test.invalid", **kw):
    return N.Notifier(root, recipient=recipient, send=send, clock=clock or FixedClock(T0), **kw)


def queue_only(root):
    return N.Notifier(root, recipient=None, send=None, clock=FixedClock(T0))


def job(name, checks, root, send=None):
    return run_job(name, checks, root=root, clock=FixedClock(T0),
                   notifier=notifier(root, send) if send is not None else queue_only(root))


def targets(root, incident):
    return [t for t in N.load_state(root)["targets"].values() if t["incident"] == incident]


def kinds(texts):
    """'fault' / 'recovered' / 'checker_failure' per accepted text, from the notice heading."""
    out = []
    for x in texts:
        out.append("recovered" if "RECOVERED" in x else "checker_failure" if "checker_failure" in x else "fault")
    return out


def broken():
    def check(ctx):
        raise RuntimeError("down")
    return check


def healthy():
    def check(ctx):
        return [CheckResult("W-ok", ctx.et_date, Status.VERIFIED, "fine")]
    return check


# ---- R3-1: a registered, job-scoped checker identity -----------------------------------------------------------------

def test_r31_a_same_named_success_in_the_same_job_never_recovers_a_broken_checker(root):
    bad, good = broken(), healthy()
    assert bad.__name__ == good.__name__ == "check"
    t = Recorder()
    for _ in range(2):
        job("j", [("broken", bad), ("good", good)], root, t)
    assert len(t.accepted) == 1 and "[checker:j/broken]" in t.accepted[0] and "episode 1" in t.accepted[0]
    (tgt,) = targets(root, "checker:j/broken")
    assert tgt["open"] is True and tgt["episode"] == 1 and tgt["check"] == "j/broken"


def test_r31_a_same_named_success_in_another_job_never_recovers_a_broken_checker(root):
    t = Recorder()
    job("broken-job", [("check", broken())], root, t)
    for _ in range(2):
        job("other-job", [("check", healthy())], root, t)
    assert len(t.accepted) == 1 and not any("RECOVERED" in x for x in t.accepted)
    (tgt,) = targets(root, "checker:broken-job/check")
    assert tgt["open"] is True and tgt["episode"] == 1


def test_r31_genuine_success_recovers_and_a_later_failure_recurs(root):
    calls = {"n": 0}

    def transient(ctx):
        calls["n"] += 1
        if calls["n"] == 2:
            return [CheckResult("W-x", ctx.et_date, Status.VERIFIED, "fine")]
        raise RuntimeError("down")
    t = Recorder()
    for _ in range(3):
        job("j", [("transient", transient)], root, t)
    assert kinds(t.accepted) == ["checker_failure", "recovered", "checker_failure"]
    assert "[checker:j/transient]" in t.accepted[1] and "episode 2" in t.accepted[2]


def test_r31_a_business_result_using_the_checker_namespace_is_a_checker_failure(root):
    def forger(ctx):
        return [CheckResult("j/victim", ctx.et_date, Status.FAULT, "forged", incident="checker:j/victim")]
    out = job("j", [("forger", forger)], root)
    (r,) = out.results
    assert r.status is Status.CHECKER_FAILURE and r.check == "j/forger" and r.incident == "checker:j/forger"
    assert targets(root, "checker:j/victim") == []


def test_r31_a_business_all_clear_never_closes_a_checker_target(root):
    t = Recorder()
    job("j", [("victim", broken())], root, t)

    def all_clear(ctx):                         # same (check, date, no selection) group as the checker target
        return [CheckResult("j/victim", ctx.et_date, Status.VERIFIED, "all clear?")]
    job("k", [("clear", all_clear)], root, t)
    (tgt,) = targets(root, "checker:j/victim")
    assert tgt["open"] is True and not any("RECOVERED" in x for x in t.accepted)


@pytest.mark.parametrize("field", [{"incident": ""}, {"selection": ""}, {"incident": 7}, {"selection": ["s"]}])
def test_a_result_the_notification_state_cannot_hold_is_a_checker_failure_and_silences_nothing(root, field):
    """Found while reading `_valid` (C2 unit 1): an empty or non-string incident/selection passed the runner, then made
    the notification state fail validation, which dropped every other check's alert in that run."""
    def odd(ctx):
        return [CheckResult("W-odd", ctx.et_date, Status.FAULT, "x", **field)]

    def fault(ctx):
        return [FAULT]
    t = Recorder()
    out = job("j", [("odd", odd), ("fault", fault)], root, t)
    assert out.ok and [r.status for r in out.results] == [Status.CHECKER_FAILURE, Status.FAULT]
    assert sorted(kinds(t.accepted)) == ["checker_failure", "fault"]


def _ran(log):
    def first(ctx):
        log.append("ran")
        return [CheckResult("W-x", ctx.et_date, Status.VERIFIED, "fine")]
    return first


@pytest.mark.parametrize("bad", [
    [("dup", lambda ctx: []), ("dup", lambda ctx: [])],          # a duplicate id
    [("Upper", lambda ctx: [])],                                 # not [a-z0-9][a-z0-9_-]*
    [("", lambda ctx: [])],
    [("a/b", lambda ctx: [])],
    [("checker:x", lambda ctx: [])],
    [("ok", "not callable")],
    [lambda ctx: []],                                            # a bare callable has no registered id
    [("ok", lambda ctx: [], "extra")],
])
def test_r31_an_invalid_registration_refuses_before_any_check_runs(root, bad):
    log = []
    with pytest.raises(RegistrationError):
        run_job("j", [("first", _ran(log))] + bad, root=root, clock=FixedClock(T0), notifier=queue_only(root))
    assert log == [] and root.list_dir(("results",)) == []


def test_r31_a_job_with_no_checks_refuses(root):
    with pytest.raises(RegistrationError):
        run_job("j", [], root=root, clock=FixedClock(T0), notifier=queue_only(root))
    assert root.list_dir(("results",)) == []


@pytest.mark.parametrize("name", ["Bad", "", "a/b", "-x"])
def test_r31_an_invalid_job_name_refuses(root, name):
    with pytest.raises(RegistrationError):
        run_job(name, [("ok", _ran([]))], root=root, clock=FixedClock(T0), notifier=queue_only(root))


def test_r31_register_validates_and_refuses_a_second_registration(monkeypatch):
    import bts.watchdog.cli as wcli
    monkeypatch.setattr(wcli, "JOBS", {})
    wcli.register("fast", [("deliver", lambda ctx: [])])
    assert list(wcli.JOBS) == ["fast"] and [cid for cid, _ in wcli.JOBS["fast"]] == ["deliver"]
    with pytest.raises(RegistrationError):
        wcli.register("fast", [("other", lambda ctx: [])])
    with pytest.raises(RegistrationError):
        wcli.register("slow", [("x", lambda ctx: []), ("x", lambda ctx: [])])
    assert list(wcli.JOBS) == ["fast"]


# ---- R3-2: a target's notices are accepted in order -----------------------------------------------------------------

FAULT = CheckResult("W-x", D, Status.FAULT, "bad", incident="I-1", selection="s")
CLEAR = CheckResult("W-x", D, Status.VERIFIED, "fixed", selection="s")


def test_r32_a_recovery_never_overtakes_a_failed_fault_send(root):
    q = queue_only(root)
    q.enqueue([FAULT])
    q.enqueue([CLEAR])
    state = {"first": True}

    def fail_first(text):
        if state["first"]:
            state["first"] = False
            return True
        return False
    t = Recorder(fail_first)
    notifier(root, t).flush()
    assert t.accepted == []                                   # the recovery waited behind the failed fault
    rec = [n for n in N.load_state(root)["notices"].values() if n["state"] == "recovered"]
    assert rec[0]["status"] == "pending" and rec[0]["attempts"] == 0
    notifier(root, t).flush()
    assert kinds(t.accepted) == ["fault", "recovered"]


def test_r32_a_recovery_never_overtakes_a_live_claimed_fault(root):
    accepted, started, release, errors = [], threading.Event(), threading.Event(), []

    def slow(recipient, text):
        started.set()
        if not release.wait(10):
            raise AssertionError("never released")
        accepted.append(text)
        return "msg-slow"
    n1 = notifier(root, slow)
    n1.enqueue([FAULT])

    def run1():
        try:
            n1.flush()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    th = threading.Thread(target=run1)
    th.start()
    fast = Recorder()
    try:
        assert started.wait(10)
        n2 = notifier(root, fast)
        n2.enqueue([CLEAR])                                   # the recovery is queued while the fault is in flight
        n2.flush()
        assert fast.accepted == [] and accepted == []
    finally:
        release.set()
        th.join(10)
    assert not th.is_alive() and errors == []
    notifier(root, fast).flush()
    assert kinds(accepted + fast.accepted) == ["fault", "recovered"]


def test_r32_an_uncertain_fault_send_holds_its_recovery(root):
    q = queue_only(root)
    q.enqueue([FAULT])
    q.enqueue([CLEAR])
    sent = []

    def uncertain(recipient, text):
        sent.append(text)
        return ""                                             # accepted? unknown: no message id
    notifier(root, uncertain).flush()
    assert kinds(sent) == ["fault"]
    by = {n["state"]: n for n in N.load_state(root)["notices"].values()}
    assert by["fault"]["status"] == "pending" and by["recovered"]["attempts"] == 0


def test_r32_a_blocked_target_does_not_block_other_targets(root):
    q = queue_only(root)
    q.enqueue([CheckResult("W-x", D, Status.FAULT, "a", incident="I-a", selection="a")])
    q.enqueue([CheckResult("W-x", D, Status.VERIFIED, "a fixed", selection="a")])
    q.enqueue([CheckResult("W-x", D, Status.FAULT, "b", incident="I-b", selection="b")])
    t = Recorder(lambda text: "(a)" in text and "RECOVERED" not in text)
    notifier(root, t).flush()
    assert len(t.accepted) == 1 and "[I-b]" in t.accepted[0]
    (rec,) = [n for n in N.load_state(root)["notices"].values() if n["state"] == "recovered"]
    assert rec["status"] == "pending" and rec["attempts"] == 0


def test_r32_an_exhausted_budget_mid_chain_releases_the_rest_unattempted(root):
    q = queue_only(root)
    q.enqueue([FAULT])
    q.enqueue([CLEAR])
    mono = {"t": 0.0}

    def slow(recipient, text):
        mono["t"] += 100.0
        return "m1"
    notifier(root, slow, budget_s=90.0, monotonic=lambda: mono["t"]).flush()
    by = {n["state"]: n for n in N.load_state(root)["notices"].values()}
    assert by["fault"]["status"] == "sent"
    assert by["recovered"]["status"] == "pending" and by["recovered"]["attempts"] == 0


# ---- R3-3: lifecycle completeness and the sequence, validated before acting ----------------------------------------

def _state(root, *batches):
    q = queue_only(root)
    for b in batches:
        q.enqueue(b)
    return json.loads(root.read_bytes(N.STATE))


def _by_seq(st):
    return {n["seq"]: k for k, n in st["notices"].items()}


def _rekey(st, k):
    n = st["notices"].pop(k)
    st["notices"][N.notice_key(n["target"], n["episode"], n["state"])] = n


def _corrupt(st, kind):
    seq = _by_seq(st)
    (tk,) = st["targets"]
    t = st["targets"][tk]
    if kind == "deleted_notice_seq_kept":                     # the reviewer's probe
        del st["notices"][seq[1]]
    elif kind == "deleted_notice_seq_decremented":
        del st["notices"][seq[1]]
        st["seq"] = 0
    elif kind == "seq_hole":
        st["notices"][seq[3]]["seq"] = 4
        st["seq"] = 4
    elif kind == "missing_earlier_recovery":
        del st["notices"][seq[2]]
        st["notices"][seq[3]]["seq"] = 2
        st["seq"] = 2
    elif kind == "recovery_not_last":                        # an open target escalated after its own recovery
        st["seq"] = 3
        st["notices"]["x"] = dict(st["notices"][seq[1]], state="checker_failure", seq=3)
        _rekey(st, "x")
        t.update(open=True, state="checker_failure")
        del t["closed_at"]
    elif kind == "open_without_current_state_notice":
        t["state"] = "checker_failure"
    elif kind == "episodes_out_of_order":
        st["notices"][seq[2]]["seq"], st["notices"][seq[3]]["seq"] = 3, 2
    elif kind == "episode_gap":
        t["episode"] = 2
    elif kind == "episode_opens_with_recovery":
        del st["notices"][seq[1]]
        st["notices"][seq[2]]["seq"] = 1
        st["seq"] = 1
    elif kind == "closed_without_recovery":
        del st["notices"][seq[2]]
        st["seq"] = 1
    elif kind == "reopened_without_new_episode":
        t.update(open=True, state="fault")
        del t["closed_at"]
    elif kind == "sending_behind_pending":
        st["notices"][seq[2]].update(status="sending", claim_token="tok", claimed_at="2026-10-06T16:00:00+00:00",
                                     lease_until="2026-10-06T16:10:00+00:00")
    return st


CASES = {                         # corruption -> the generated state it starts from
    "deleted_notice_seq_kept": "one_fault",
    "deleted_notice_seq_decremented": "one_fault",
    "open_without_current_state_notice": "one_fault",
    "episode_gap": "one_fault",
    "recovery_not_last": "recovered",
    "sending_behind_pending": "recovered",
    "episode_opens_with_recovery": "recovered",
    "closed_without_recovery": "recovered",
    "reopened_without_new_episode": "recovered",
    "seq_hole": "two_episodes",
    "missing_earlier_recovery": "two_episodes",
    "episodes_out_of_order": "two_episodes",
}


def _generate(root, base):
    if base == "one_fault":
        return _state(root, [FAULT])
    if base == "recovered":
        return _state(root, [FAULT], [CLEAR])
    return _state(root, [FAULT], [CLEAR], [FAULT])


@pytest.mark.parametrize("base", ["one_fault", "recovered", "two_episodes"])
def test_r33_every_generated_lifecycle_state_is_accepted(root, base):
    raw = json.dumps(_generate(root, base)).encode()
    root.write_atomic(N.STATE, raw)
    assert N.load_state(root)["seq"] == {"one_fault": 1, "recovered": 2, "two_episodes": 3}[base]


@pytest.mark.parametrize("kind", sorted(CASES))
@pytest.mark.parametrize("via", ["enqueue", "flush"])
def test_r33_an_impossible_lifecycle_state_refuses_untouched_and_unsent(root, kind, via):
    st = _corrupt(_generate(root, CASES[kind]), kind)
    raw = json.dumps(st).encode()
    root.write_atomic(N.STATE, raw)
    t = Recorder()
    with pytest.raises(N.NotifyStateError):
        if via == "enqueue":
            notifier(root, t).enqueue([FAULT])
        else:
            notifier(root, t).flush()
    assert root.read_bytes(N.STATE) == raw and t.accepted == []


def test_r33_the_deleted_notice_probe_through_run_job_reports_and_never_silences(root):
    """The reviewer's sequence: a real queue-only fault, its notice deleted, then the identical fault again."""
    def fault(ctx):
        return [FAULT]
    job("j", [("fault", fault)], root)
    st = json.loads(root.read_bytes(N.STATE))
    st["notices"] = {}
    raw = json.dumps(st).encode()
    root.write_atomic(N.STATE, raw)
    t = Recorder()
    out = job("j", [("fault", fault)], root, t)
    assert out.notify_error and "NotifyStateError" in out.notify_error and not out.ok
    assert root.read_bytes(N.STATE) == raw and t.accepted == []


# ---- R3-4: no descendant process inside the confinement gate ----------------------------------------------------------

DESCENDANT_SELF_KILL = """
import subprocess, sys
_real_flush = N.Notifier.flush
def _flush(self):
    subprocess.run([sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"], check=False)
    return _real_flush(self)
N.Notifier.flush = _flush
"""

DESCENDANT_OUTSIDE_WRITE = """
import subprocess, sys
_real_flush = N.Notifier.flush
def _flush(self):
    subprocess.run([sys.executable, "-c", "open(%r, 'w').write('outside-mutant')" % (D + "/picks/2026-10-06.json")],
                   check=False)
    return _real_flush(self)
N.Notifier.flush = _flush
"""


def _gate_data(data):
    (data / "picks").mkdir()
    (data / "picks" / "2026-10-06.json").write_text("{}")
    OwnedRoot.under(data).close()


@needs_sandbox
@pytest.mark.parametrize("extra", [DESCENDANT_SELF_KILL, DESCENDANT_OUTSIDE_WRITE], ids=["self_kill", "outside_write"])
def test_r34_a_parent_continues_descendant_mutant_fails_the_gate(data, extra):
    _gate_data(data)
    res = run_confined(GATE_CHILD.format(data=str(data), extra=extra), data / "watchdog")
    assert res.returncode in (-9, 137) and "GATE-OK" not in res.stdout, (res.returncode, res.stdout[-500:])
    assert (data / "picks" / "2026-10-06.json").read_text() == "{}"


@needs_sandbox
def test_r34_the_kill_profile_kills_after_start_on_any_process_creation(data):
    _gate_data(data)
    code = ("print('CHILD-STARTED', flush=True)\nimport subprocess, sys\ntry:\n"
            "    subprocess.run([sys.executable, '-c', 'pass'], check=False)\nexcept Exception:\n    pass\n"
            "print('SURVIVED', flush=True)")
    res = run_confined(code, data / "watchdog")
    assert res.returncode in (-9, 137) and "CHILD-STARTED" in res.stdout and "SURVIVED" not in res.stdout


@needs_sandbox
def test_r34_the_fork_refusal_itself_is_witnessed_under_eperm(data):
    _gate_data(data)
    code = ("print('CHILD-STARTED', flush=True)\nimport subprocess, sys\ntry:\n"
            "    subprocess.run([sys.executable, '-c', 'pass'], check=False)\nexcept PermissionError:\n"
            "    print('FORK-EPERM-WITNESS', flush=True)\n    raise SystemExit(0)\nraise SystemExit(5)")
    res = run_confined(code, data / "watchdog", kill=False, deny_fork=True)
    assert res.returncode == 0 and "CHILD-STARTED" in res.stdout and "FORK-EPERM-WITNESS" in res.stdout


def _plant_stale_chain(root):
    """A [fault, recovery] chain claimed together by a flusher that then stalled past its lease."""
    q = queue_only(root)
    q.enqueue([FAULT])
    q.enqueue([CLEAR])
    with N.state_lock(root):
        st = N.load_state(root)
        for i, n in enumerate(sorted(st["notices"].values(), key=lambda n: n["seq"])):
            n.update(status="sending", claim_token=f"stale-{i}",
                     claimed_at=(T0 - timedelta(minutes=20)).astimezone(timezone.utc).isoformat(),
                     lease_until=(T0 - timedelta(minutes=10)).astimezone(timezone.utc).isoformat())
        N.save_state(root, st)


def test_r32_a_reclaim_under_max_sends_takes_the_whole_chain_and_stays_valid(root):
    """False-refusal check (author, C2 unit 1): a stalled flusher's expired chain re-claimed with max_sends=1. A
    partial re-claim would leave the recovery on the stale claim behind a released fault, a state the validator
    rejects, so the failing flush could never record its outcome."""
    _plant_stale_chain(root)
    t = Recorder(lambda text: True)                           # every send fails
    report = notifier(root, t, max_sends=1).flush()
    by = {n["state"]: n for n in N.load_state(root)["notices"].values()}
    assert by["fault"]["status"] == "pending" and by["fault"]["attempts"] == 1
    assert by["recovered"]["status"] == "pending" and by["recovered"]["attempts"] == 0
    assert report["failed"] == 1


def test_r32_max_sends_bounds_the_sends_of_a_long_chain(root):
    q = queue_only(root)
    q.enqueue([FAULT])
    q.enqueue([CLEAR])
    q.enqueue([FAULT])                                        # episode 2: three notices for one target
    t = Recorder()
    notifier(root, t, max_sends=2).flush()
    assert kinds(t.accepted) == ["fault", "recovered"]
    last = max(N.load_state(root)["notices"].values(), key=lambda n: n["seq"])
    assert last["status"] == "pending" and last["attempts"] == 0
    notifier(root, t, max_sends=2).flush()
    assert kinds(t.accepted) == ["fault", "recovered", "fault"]


# ---- C2 unit 1 review r1 (docs/audit/2026-10-06-c2-w0-codex-r1.md): B1-B4 ------------------------------------------

def _seq_sorted(st):
    return sorted(st["notices"].values(), key=lambda n: n["seq"])


B_FAULT = CheckResult("W-y", D, Status.FAULT, "b", incident="I-b", selection="b")


@pytest.mark.parametrize("cycles, max_sends", [(10, 20), (1, 2)])
def test_b1_a_blocked_long_chain_never_starves_an_independent_alert(root, cycles, max_sends):
    """The reviewer's case: target A holds a long chain (2*cycles+1 notices) whose head keeps failing; independent
    target B's alert must still be attempted and accepted, within the send bound, across repeated flushes."""
    q = queue_only(root)
    for _ in range(cycles):
        q.enqueue([FAULT])
        q.enqueue([CLEAR])
    q.enqueue([FAULT])
    q.enqueue([B_FAULT])
    per_flush = []
    for _ in range(3):
        calls = []

        def send(recipient, text):
            calls.append(text)
            if "[I-b]" not in text:
                raise RuntimeError("A's head fails")
            return "msg-b"
        notifier(root, send, max_sends=max_sends).flush()
        per_flush.append(len(calls))
    assert per_flush == [2, 1, 1]                             # A's head and B, then A's head alone; within the bound
    a = [n for n in _seq_sorted(N.load_state(root)) if "[I-b]" not in n["text"]]
    (b,) = [n for n in N.load_state(root)["notices"].values() if "[I-b]" in n["text"]]
    assert b["status"] == "sent" and b["attempts"] == 1 and len(a) == 2 * cycles + 1
    assert a[0]["attempts"] == 3 and all(n["status"] == "pending" and n["attempts"] == 0 for n in a[1:])


def test_b1_a_fresh_alert_is_tried_before_a_repeatedly_failing_one(root):
    q = queue_only(root)
    q.enqueue([FAULT])
    for _ in range(2):
        notifier(root, Recorder(lambda text: True)).flush()  # A's only notice fails twice
    q.enqueue([B_FAULT])
    t = Recorder(lambda text: "[I-b]" not in text)
    notifier(root, t, max_sends=1).flush()
    assert len(t.accepted) == 1 and "[I-b]" in t.accepted[0]


def _one_fault_state(root):
    return _state(root, [FAULT])


META = {
    "sent_without_attempt": dict(status="sent", message_id="fabricated-id"),          # the reviewer's B2 case
    "sent_without_recipient": dict(status="sent", message_id="m", attempts=1, last_attempt_at="2026-10-06T16:00:00+00:00"),
    "sent_with_error": dict(status="sent", message_id="m", attempts=1, recipient="r", last_error="x",
                            last_attempt_at="2026-10-06T16:00:00+00:00"),
    "sent_without_attempt_time": dict(status="sent", message_id="m", attempts=1, recipient="r"),
    "pending_with_recipient": dict(recipient="r"),
    "attempts_without_time": dict(attempts=1),
    "time_without_attempts": dict(last_attempt_at="2026-10-06T16:00:00+00:00"),
}


@pytest.mark.parametrize("kind", sorted(META))
@pytest.mark.parametrize("via", ["enqueue", "flush"])
def test_b2_impossible_delivery_metadata_refuses_untouched_and_unsent(root, kind, via):
    st = _one_fault_state(root)
    (n,) = st["notices"].values()
    n.update(META[kind])
    raw = json.dumps(st).encode()
    root.write_atomic(N.STATE, raw)
    t = Recorder()
    with pytest.raises(N.NotifyStateError):
        if via == "enqueue":
            notifier(root, t).enqueue([FAULT])
        else:
            notifier(root, t).flush()
    assert root.read_bytes(N.STATE) == raw and t.accepted == []


def _split_lease_state(root):
    """The reviewer's B3 corruption: a fault/recovery chain whose two sending records carry different claims."""
    st = _state(root, [FAULT], [CLEAR])
    fault, rec = _seq_sorted(st)
    fault.update(status="sending", claim_token="t-a", claimed_at="2026-10-06T15:40:00+00:00",
                 lease_until="2026-10-06T15:50:00+00:00")
    rec.update(status="sending", claim_token="t-b", claimed_at="2026-10-06T16:00:00+00:00",
               lease_until="2026-10-06T16:10:00+00:00")
    return st


@pytest.mark.parametrize("via", ["enqueue", "flush"])
def test_b3_a_split_claim_refuses_before_any_send(root, via):
    raw = json.dumps(_split_lease_state(root)).encode()
    root.write_atomic(N.STATE, raw)
    calls = []

    def send(recipient, text):
        calls.append(text)
        raise RuntimeError("fails")
    with pytest.raises(N.NotifyStateError):
        if via == "enqueue":
            notifier(root, send).enqueue([FAULT])
        else:
            notifier(root, send).flush()
    assert root.read_bytes(N.STATE) == raw and calls == []


def test_b3_a_legitimate_stale_chain_with_a_later_alert_reclaims_whole_and_stays_valid(root):
    _plant_stale_chain(root)                                  # fault + recovery on an expired claim
    queue_only(root).enqueue([FAULT])                         # episode 2 arrives behind the stale claim
    assert [n["status"] for n in _seq_sorted(N.load_state(root))] == ["sending", "sending", "pending"]
    notifier(root, Recorder(lambda text: True), max_sends=1).flush()
    st = N.load_state(root)
    assert [(n["status"], n["attempts"]) for n in _seq_sorted(st)] == [("pending", 1), ("pending", 0), ("pending", 0)]
    t = Recorder()
    notifier(root, t).flush()
    assert kinds(t.accepted) == ["fault", "recovered", "fault"]


def test_b3_a_chain_claimed_in_the_future_is_reclaimed_after_a_clock_rollback(root):
    q = queue_only(root)
    q.enqueue([FAULT])
    q.enqueue([CLEAR])
    with N.state_lock(root):
        st = N.load_state(root)
        for i, n in enumerate(_seq_sorted(st)):
            n.update(status="sending", claim_token=f"future-{i}",
                     claimed_at=(T0 + timedelta(minutes=30)).astimezone(timezone.utc).isoformat(),
                     lease_until=(T0 + timedelta(minutes=40)).astimezone(timezone.utc).isoformat())
        N.save_state(root, st)
    t = Recorder()
    notifier(root, t, max_sends=1).flush()                    # the clock reads before the claim: a rollback
    assert kinds(t.accepted) == ["fault"]
    notifier(root, t).flush()
    assert kinds(t.accepted) == ["fault", "recovered"]


def test_b4_a_late_success_never_overwrites_a_newer_confirmation(root):
    """Flusher A stalls on its send; after the lease B reclaims and confirms new-id; A then returns old-id. A's
    obsolete claim must not replace the newer confirmation (a duplicate send is the documented limit)."""
    started, release, errors = threading.Event(), threading.Event(), []
    clock = FixedClock(T0)

    def slow_ok(recipient, text):
        started.set()
        if not release.wait(10):
            raise AssertionError("never released")
        return "old-id"
    n1 = N.Notifier(root, recipient="r", send=slow_ok, clock=clock)
    n1.enqueue([FAULT])

    def run1():
        try:
            n1.flush()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    th = threading.Thread(target=run1)
    th.start()
    try:
        assert started.wait(10)
        clock.at = T0 + timedelta(minutes=11)
        N.Notifier(root, recipient="r", send=lambda r, t: "new-id", clock=clock).flush()
    finally:
        release.set()
        th.join(10)
    assert not th.is_alive() and errors == []
    (n,) = N.load_state(root)["notices"].values()
    assert n["status"] == "sent" and n["message_id"] == "new-id" and n["attempts"] == 1


@pytest.mark.parametrize("message_id", ["", False, 7, {"id": "m"}])
@pytest.mark.parametrize("status", ["pending", "sending"])
@pytest.mark.parametrize("via", ["enqueue", "flush"])
def test_b2_unsent_message_id_refuses_untouched_and_unsent(root, message_id, status, via):
    st = _one_fault_state(root)
    (n,) = st["notices"].values()
    n.update(status=status, message_id=message_id)
    if status == "sending":
        n.update(claim_token="t", claimed_at="2026-10-06T16:00:00+00:00",
                 lease_until="2026-10-06T16:10:00+00:00")
    raw = json.dumps(st).encode()
    root.write_atomic(N.STATE, raw)
    calls = []

    def send(recipient, text):
        calls.append(text)
        return "m"
    with pytest.raises(N.NotifyStateError):
        if via == "enqueue":
            notifier(root, send).enqueue([FAULT])
        else:
            notifier(root, send).flush()
    assert root.read_bytes(N.STATE) == raw and calls == []


@pytest.mark.parametrize("same_field", ["claimed_at", "lease_until"])
@pytest.mark.parametrize("via", ["enqueue", "flush"])
def test_b3_each_claim_component_is_required_before_action(root, same_field, via):
    st = _split_lease_state(root)
    fault, recovery = _seq_sorted(st)
    if same_field == "claimed_at":
        recovery["claimed_at"] = fault["claimed_at"]
    else:
        fault["lease_until"] = recovery["lease_until"]
    raw = json.dumps(st).encode()
    root.write_atomic(N.STATE, raw)
    calls = []

    def send(recipient, text):
        calls.append(text)
        return "m"
    with pytest.raises(N.NotifyStateError):
        if via == "enqueue":
            notifier(root, send).enqueue([FAULT])
        else:
            notifier(root, send).flush()
    assert root.read_bytes(N.STATE) == raw and calls == []
