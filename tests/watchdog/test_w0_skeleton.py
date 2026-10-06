"""W0: the watchdog skeleton (C1 rank-2 build plan W0; registration R1, R5, §2 statuses, §4.2 gate 2).

Covers the owned root and its escape refusal, the clock seam, isolated check results, the job singleton, the
notification state (dedup, pending retries, lease claims, message-id confirmation, restart persistence), and write
confinement traced at the syscall level.
"""
import json
import os
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from bts.watchdog import notify as N
from bts.watchdog.clock import FixedClock
from bts.watchdog.result import CheckResult, Status
from bts.watchdog.root import OwnedRoot, RootError
from bts.watchdog.runner import JobBusy, run_job
from tests.watchdog.conftest import outside

ET = ZoneInfo("America/New_York")
T0 = datetime(2026, 10, 6, 12, 0, tzinfo=ET)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "data").mkdir()
    return OwnedRoot.under(tmp_path / "data")


# ---- the owned root -----------------------------------------------------------------------------------------------

def test_the_root_is_data_watchdog_and_children_stay_inside(root, tmp_path):
    assert root.path == (tmp_path / "data" / "watchdog").resolve() and root.path.is_dir()
    assert root.child("notify", "state.json") == root.path / "notify" / "state.json"
    for bad in (("..", "picks", "x.json"), ("/etc/passwd",), ("a", "..", "..", "x")):
        with pytest.raises(RootError):
            root.child(*bad)


def test_a_symlinked_root_or_component_is_refused(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "data" / "watchdog").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(RootError, match="is a symlink: refused"):
        OwnedRoot.under(tmp_path / "data")
    (tmp_path / "data2").mkdir()
    r = OwnedRoot.under(tmp_path / "data2")
    (r.path / "notify").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(RootError, match="is a symlink: refused"):
        r.child("notify", "state.json")
    (r.path / "real").mkdir()
    (r.path / "alias").symlink_to(r.path / "real")       # inside the root: only the component rule catches it
    with pytest.raises(RootError, match="is a symlink: refused"):
        r.child("alias", "x.json")


# ---- results and the runner ---------------------------------------------------------------------------------------

def ok_check(ctx):
    return [CheckResult("W-ok", ctx.et_date, Status.VERIFIED, "fine")]


def faulty_check(ctx):
    return [CheckResult("W-bad", ctx.et_date, Status.FAULT, "missing delivery", incident="I-test",
                        selection="802415@822934")]


def raising_check(ctx):
    raise RuntimeError("boom")


def test_every_check_runs_and_an_exception_is_a_checker_failure(root):
    out = run_job("t", [raising_check, ok_check, faulty_check], root=root, clock=FixedClock(T0), notifier=None)
    by = {r.check: r for r in out.results}
    assert by["raising_check"].status is Status.CHECKER_FAILURE and "RuntimeError" in by["raising_check"].detail
    assert by["W-ok"].status is Status.VERIFIED and by["W-bad"].status is Status.FAULT
    saved = json.loads(out.path.read_text())
    assert saved["job"] == "t" and saved["et_date"] == "2026-10-06" and len(saved["results"]) == 3
    assert out.path.is_relative_to(root.path)


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


def notifier(root, transport, clock=None):
    return N.Notifier(root, recipient="watchdog-test.invalid", send=transport, clock=clock or FixedClock(T0))


def test_a_fault_is_sent_once_and_confirmed_only_by_a_message_id(root):
    t = Transport()
    run_job("t", [faulty_check], root=root, clock=FixedClock(T0), notifier=notifier(root, t))
    run_job("t", [faulty_check], root=root, clock=FixedClock(T0), notifier=notifier(root, t))   # a restart
    assert len(t.sent) == 1
    (e,) = N.load_state(root).values()
    assert e["status"] == "sent" and e["message_id"] == "msg-1" and e["incident"] == "I-test"


def test_failed_and_uncertain_sends_stay_pending_and_retry(root):
    t = Transport(fail=1)
    run_job("t", [faulty_check], root=root, clock=FixedClock(T0), notifier=notifier(root, t))
    (e,) = N.load_state(root).values()
    assert e["status"] == "pending" and e["attempts"] == 1 and e["last_error"] == "RuntimeError"
    u = Transport(uncertain=True)
    run_job("t", [faulty_check], root=root, clock=FixedClock(T0), notifier=notifier(root, u))
    (e,) = N.load_state(root).values()
    assert e["status"] == "pending" and e["attempts"] == 2 and e["last_error"] == "no message id"
    ok = Transport()
    run_job("t", [faulty_check], root=root, clock=FixedClock(T0), notifier=notifier(root, ok))
    (e,) = N.load_state(root).values()
    assert e["status"] == "sent" and len(ok.sent) == 1


def test_the_dedup_key_binds_incident_date_selection_and_state(root):
    k = N.dedup_key
    base = k("I-1", "2026-10-06", "sel", "fault")
    assert len({base, k("I-2", "2026-10-06", "sel", "fault"), k("I-1", "2026-10-07", "sel", "fault"),
                k("I-1", "2026-10-06", "other", "fault"), k("I-1", "2026-10-06", "sel", "recovered")}) == 5


def test_a_leased_send_is_not_sent_twice_by_a_concurrent_flush(root):
    """Two jobs flushing at once: the first claims the entry under the short lock and sends outside it; the second
    sees the lease and skips it."""
    started, release = threading.Event(), threading.Event()
    sent = []

    def slow(recipient, text):
        started.set()
        release.wait(5)
        sent.append(text)
        return "msg-slow"
    n1 = notifier(root, slow)
    n1.enqueue([CheckResult("W-bad", "2026-10-06", Status.FAULT, "x", incident="I-1", selection="s")])
    th = threading.Thread(target=n1.flush)
    th.start()
    assert started.wait(5)
    other = Transport()
    notifier(root, other).flush()                     # the lease is held: nothing to send
    release.set()
    th.join(5)
    assert other.sent == [] and len(sent) == 1
    (e,) = N.load_state(root).values()
    assert e["status"] == "sent" and e["message_id"] == "msg-slow"


def test_an_expired_lease_is_retried(root):
    n = notifier(root, Transport(), clock=FixedClock(T0))
    n.enqueue([CheckResult("W-bad", "2026-10-06", Status.FAULT, "x", incident="I-1", selection="s")])
    with N.state_lock(root):                           # a crashed sender left a claim behind
        st = N.load_state(root)
        (key,) = st
        st[key].update(status="sending", lease_until=T0.isoformat())
        N.save_state(root, st)
    later = Transport()
    notifier(root, later, clock=FixedClock(datetime(2026, 10, 6, 12, 30, tzinfo=ET))).flush()
    assert len(later.sent) == 1


def test_verified_unverifiable_and_pending_results_do_not_notify(root):
    t = Transport()
    n = notifier(root, t)
    n.enqueue([CheckResult("a", "2026-10-06", s, "x") for s in (Status.VERIFIED, Status.PENDING, Status.UNVERIFIABLE)])
    n.flush()
    assert t.sent == [] and N.load_state(root) == {}


def test_a_checker_failure_notifies(root):
    t = Transport()
    run_job("t", [raising_check], root=root, clock=FixedClock(T0), notifier=notifier(root, t))
    assert len(t.sent) == 1 and "raising_check" in t.sent[0][1]


# ---- gate 2: write confinement -------------------------------------------------------------------------------------

def test_every_watchdog_write_stays_inside_its_root(root, write_trace, tmp_path):
    """Run a job with a fault, a failed send, a retry and a sent notification while tracing every write-type syscall.
    Every target resolves beneath data/watchdog."""
    picks = tmp_path / "data" / "picks"
    picks.mkdir()
    (picks / "2026-10-06.json").write_text("{}")
    with write_trace() as sink:
        run_job("t", [faulty_check, ok_check], root=root, clock=FixedClock(T0), notifier=notifier(root, Transport(fail=1)))
        run_job("t", [faulty_check], root=root, clock=FixedClock(T0), notifier=notifier(root, Transport()))
    assert sink, "the trace saw no writes at all"
    assert outside(sink, root.path) == []


def test_the_trace_catches_same_byte_and_undone_writes_outside_the_root(write_trace, tmp_path):
    """The gate's own red control: a same-byte rewrite and a write-then-restore outside the root are both caught."""
    f = tmp_path / "data" / "picks" / "x.json"
    f.parent.mkdir(parents=True)
    f.write_text("{}")
    with write_trace() as sink:
        f.write_text("{}")                             # same bytes
        os.replace(f, f.with_suffix(".bak"))           # write ...
        os.replace(f.with_suffix(".bak"), f)           # ... and undo
    assert len(outside(sink, tmp_path / "data" / "watchdog")) >= 3


# ---- the CLI ------------------------------------------------------------------------------------------------------

def test_the_cli_runs_a_registered_job_and_sends_through_the_dm_transport(tmp_path, monkeypatch):
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
    assert sent == [("watchdog-test.invalid", sent[0][1])] and "I-test" in sent[0][1]
    assert list((tmp_path / "data" / "watchdog" / "results").rglob("*.json"))


def test_the_cli_refuses_unknown_jobs_and_a_missing_recipient(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import bts.watchdog.cli as wcli
    from bts.cli import cli
    (tmp_path / "data").mkdir()
    res = CliRunner().invoke(cli, ["watchdog", "run", "nope", "--data-dir", str(tmp_path / "data"), "--no-send"])
    assert res.exit_code != 0 and "unknown watchdog job" in res.output
    monkeypatch.setitem(wcli.JOBS, "t", [ok_check])
    res = CliRunner().invoke(cli, ["watchdog", "run", "t", "--data-dir", str(tmp_path / "data")])
    assert res.exit_code != 0 and "--dm-recipient is required" in res.output
    res = CliRunner().invoke(cli, ["watchdog", "run", "t", "--data-dir", str(tmp_path / "data"), "--no-send"])
    assert res.exit_code == 0 and "verified" in res.output
