"""C1 cycle guard (docs/audit/2026-10-04-d4-cycle-proposal.md, approved 10/04): compute ledger + launcher rules."""
import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

from scripts.audit.c1 import launch, ledger

UTC = timezone.utc


def jline(unit, inv, nsec, ts_us=1791135113244579, msg=None):
    d = {"USER_UNIT": unit, "USER_INVOCATION_ID": inv, "CPU_USAGE_NSEC": str(nsec), "__REALTIME_TIMESTAMP": str(ts_us),
         "MESSAGE": msg or f"{unit}: Consumed CPU time."}
    return json.dumps(d)


# ---------- ledger ----------
def test_parse_journal_keeps_only_c1_service_cpu_records():
    lines = [jline("c1-r2-build-x.service", "a1", 3_600_000_000_000),
             jline("w23-mlb.service", "b2", 1_539_891_454_000),              # not a C1 unit
             json.dumps({"MESSAGE": "Started c1-r2-build-x.service", "USER_UNIT": "c1-r2-build-x.service"}),  # no CPU
             ""]
    rows = ledger.parse_journal(lines)
    assert rows == [{"invocation": "a1", "unit": "c1-r2-build-x.service",
                     "stopped_at": "2026-10-04T17:31:53.244579+00:00", "cpu_seconds": 3600.0}]


def test_parse_journal_refuses_a_c1_record_without_invocation_id():
    bad = json.dumps({"USER_UNIT": "c1-x.service", "CPU_USAGE_NSEC": "5", "__REALTIME_TIMESTAMP": "1"})
    with pytest.raises(ValueError):
        ledger.parse_journal([bad])


def test_merge_dedupes_by_invocation_and_refuses_conflicts():
    a = {"invocation": "a1", "unit": "c1-x.service", "stopped_at": "t", "cpu_seconds": 10.0}
    assert ledger.merge([a], [dict(a)]) == [a]
    with pytest.raises(ledger.LedgerConflict):
        ledger.merge([a], [{**a, "cpu_seconds": 11.0}])


def test_tsv_round_trip(tmp_path):
    rows = [{"invocation": "a1", "unit": "c1-x.service", "stopped_at": "2026-10-04T00:00:00+00:00", "cpu_seconds": 7.5}]
    p = tmp_path / "c1" / "compute_ledger.tsv"
    ledger.write_tsv(p, rows)
    assert ledger.read_tsv(p) == rows
    assert ledger.read_tsv(tmp_path / "missing.tsv") == []


@pytest.mark.parametrize("total,declared,acked,expect", [
    (0.0, 10.0, False, "ok"),
    (49.0, 10.0, False, "ok"),        # crossing 50 during a job is allowed; the next launch stops
    (50.0, 1.0, False, "checkpoint"),  # at 50: stop and report until Eric's acknowledgement exists
    (50.0, 1.0, True, "ok"),
    (95.0, 6.0, True, "over_cap"),     # the declared budget may not cross 100
    (100.0, 0.5, True, "stop"),
])
def test_gate(total, declared, acked, expect):
    assert ledger.gate(total, declared, acked) == expect


# ---------- scheduler sleep window ----------
def sched(ts, msg):
    return json.dumps({"__REALTIME_TIMESTAMP": str(int(ts.timestamp() * 1_000_000)), "MESSAGE": msg})


T0 = datetime(2027, 5, 1, 14, 0, tzinfo=UTC)


def test_scheduler_wake_from_sleep_and_idle_lines():
    assert launch.scheduler_wake([sched(T0, "  Sleeping until 11:05 ET (57 min)...")]) == T0 + timedelta(minutes=57)
    assert launch.scheduler_wake([sched(T0, "  Idle until tomorrow's wakeup 10:00 ET (24.0h)...")]) == T0 + timedelta(hours=24)


def test_scheduler_wake_is_none_when_the_latest_line_is_not_a_sleep():
    lines = [sched(T0, "  Sleeping until 11:05 ET (57 min)..."), sched(T0 + timedelta(minutes=58), "Lineup check for ...")]
    assert launch.scheduler_wake(lines) is None
    assert launch.scheduler_wake([]) is None


def test_season_guard_is_off_before_march_2027_and_fails_closed_after():
    off = datetime(2026, 10, 5, 12, tzinfo=UTC)
    assert launch.season_guard(off, [], max_hours=48)[0] is True
    on = datetime(2027, 3, 1, 12, tzinfo=UTC)
    assert launch.season_guard(on, [], max_hours=1)[0] is False                      # no sleep line: refuse
    sleeping = [sched(on - timedelta(minutes=5), "  Sleeping until 15:00 ET (120 min)...")]
    assert launch.season_guard(on, sleeping, max_hours=1)[0] is True                  # wakes in 115 min
    assert launch.season_guard(on, sleeping, max_hours=2)[0] is False                 # would overrun the wake


# ---------- launch planning ----------
NOW = datetime(2026, 10, 5, 12, tzinfo=UTC)


def plan(**kw):
    base = dict(name="r2-build", cpu_hours=2.0, max_hours=3.0, rows=[], acked=False, active_units=[], sched_lines=[],
                now=NOW, cwd="/home/bts/projects/bts-c1", env_file="/home/bts/projects/bts/.env", command=["echo", "hi"])
    base.update(kw)
    return launch.plan_launch(**base)


def test_plan_ok_builds_a_capped_transient_unit():
    p = plan()
    assert p["ok"] is True and p["reasons"] == []
    argv = p["argv"]
    assert argv[:3] == ["systemd-run", "--user", f"--unit={p['unit']}"]
    assert p["unit"].startswith("c1-r2-build-20261005T120000Z")
    for prop in ("--collect", "--nice=10", "-p", "MemoryMax=12G", "OOMScoreAdjust=1000", "LimitCPU=7200",
                 "RuntimeMaxSec=10800",
                 "--working-directory=/home/bts/projects/bts-c1"):
        assert prop in argv
    assert argv[-2:] == ["echo", "hi"]
    assert "EnvironmentFile=/home/bts/projects/bts/.env" in argv      # the production .env, not the worktree's


def test_ledger_paths_live_in_the_production_data_root_not_the_worktree(tmp_path):
    paths = launch.c1_paths(tmp_path / "data")
    assert paths["ledger"] == tmp_path / "data" / "hetzner_results" / "c1" / "compute_ledger.tsv"
    assert paths["ack"] == tmp_path / "data" / "hetzner_results" / "c1" / "CHECKPOINT_50_ACK.json"
    assert launch.DATA_ROOT == Path.home() / "projects" / "bts" / "data"


def test_plan_limits_are_the_jobs_own_declared_budgets():
    """Code review r1 F10: the per-job CPU and wall limits, not the whole remaining cycle budget."""
    rows = [{"invocation": "a", "unit": "c1-x.service", "stopped_at": "t", "cpu_seconds": 40 * 3600.0}]
    argv = plan(rows=rows, cpu_hours=4.0, max_hours=3.0)["argv"]
    assert "LimitCPU=14400" in argv and "RuntimeMaxSec=10800" in argv


def test_a_failed_earlier_unit_of_the_same_job_blocks_an_automatic_retry():
    failed = ["c1-r2-build-20261005T010000Z"]
    p = plan(failed_units=failed)
    assert p["ok"] is False and any("no automatic retry" in r for r in p["reasons"])
    assert plan(failed_units=failed, acked_failures=failed)["ok"] is True
    assert plan(failed_units=["c1-other-20261005T010000Z"])["ok"] is True


def test_failed_units_are_read_from_systemd_failure_messages():
    lines = [json.dumps({"USER_UNIT": "c1-r4b-run-20261005T010000Z.service",
                         "MESSAGE": "c1-r4b-run-20261005T010000Z.service: Failed with result 'timeout'."}),
             json.dumps({"USER_UNIT": "c1-ok-20261005T020000Z.service", "MESSAGE": "Deactivated successfully."}),
             json.dumps({"USER_UNIT": "w23-mlb.service", "MESSAGE": "w23-mlb.service: Failed with result 'signal'."})]
    assert launch.failed_units(lines) == ["c1-r4b-run-20261005T010000Z"]


@pytest.mark.parametrize("kw,reason", [
    (dict(name="R2 build"), "name"),
    (dict(active_units=["c1-r3-fit-20261005T110000Z.service"]), "another C1 job is active"),
    (dict(rows=[{"invocation": "a", "unit": "c1-x.service", "stopped_at": "t", "cpu_seconds": 50 * 3600.0}]), "checkpoint"),
    (dict(cpu_hours=101.0), "over_cap"),
    (dict(now=datetime(2027, 4, 1, 12, tzinfo=UTC)), "sleep window"),
    (dict(command=[]), "command"),
])
def test_plan_refusals(kw, reason):
    p = plan(**kw)
    assert p["ok"] is False and any(reason in r for r in p["reasons"]), p["reasons"]
    assert "argv" not in p


def test_rate_limit_stop_markers_are_found_anywhere_under_c1_and_in_the_capture(tmp_path):
    root = tmp_path / "data"
    assert launch.rate_limit_stops(root) == []
    (root / "hetzner_results" / "c1" / "r3").mkdir(parents=True)
    (root / "hetzner_results" / "c1" / "r3" / "STOP_403_429.json").write_text("{}")
    (root / "leaderboard" / "static_snapshots" / "_receipts").mkdir(parents=True)
    (root / "leaderboard" / "static_snapshots" / "_receipts" / "STOP_403_429.json").write_text("{}")
    assert launch.rate_limit_stops(root) == ["hetzner_results/c1/r3/STOP_403_429.json",
                                             "leaderboard/static_snapshots/_receipts/STOP_403_429.json"]


# ---------- Eric 2026-10-05 (row C1-4b-deferral; code review r3 B4/B5): C1-wide box limits ----------
import importlib.util as _ilu

_gspec = _ilu.spec_from_file_location("c1_guard", Path(launch.__file__).with_name("guard.py"))
guard = _ilu.module_from_spec(_gspec)
_gspec.loader.exec_module(guard)


class FakeProc:
    def __init__(self, finish_after=None, rc=0):
        self.polls, self.finish_after, self.rc = 0, finish_after, rc

    def poll(self):
        self.polls += 1
        return self.rc if self.finish_after is not None and self.polls >= self.finish_after else None


def test_the_guard_kills_the_whole_job_before_its_cumulative_cpu_reaches_the_budget():
    """The cap holds across child processes: the cgroup's total, not any one process, is compared with the budget.
    Usage grows by at most cores x poll per interval, so acting at budget - cores x poll keeps the total under it."""
    usage, events = [0.0], []

    def read():
        usage[0] += 8 * 2.0          # every core busy (e.g. several children) for a whole poll interval
        return usage[0]
    rc = guard.watch(FakeProc(), read, budget=100.0, ncpu=8, poll=2.0,
                     on_overrun=lambda u: events.append(("marker", u)), kill=lambda: events.append(("kill",)),
                     sleep=lambda s: None)
    assert rc == guard.OVERRUN_EXIT
    assert events[0][0] == "marker" and events[1] == ("kill",)       # the durable pause marker comes first
    assert events[0][1] < 100.0                                       # caught before the budget was crossed


def test_the_guard_returns_the_jobs_own_exit_code_within_budget():
    rc = guard.watch(FakeProc(finish_after=3, rc=5), lambda: 1.0, budget=100.0, ncpu=8, poll=2.0,
                     on_overrun=lambda u: pytest.fail("no overrun"), kill=lambda: pytest.fail("no kill"),
                     sleep=lambda s: None)
    assert rc == 5


def test_the_guard_records_an_overrun_seen_only_at_exit():
    seen = []
    rc = guard.watch(FakeProc(finish_after=1), iter([1.0, 150.0]).__next__, budget=100.0, ncpu=1, poll=2.0,
                     on_overrun=seen.append, kill=lambda: None, sleep=lambda s: None)
    assert rc == guard.OVERRUN_EXIT and seen == [150.0]


def test_the_guard_reads_its_own_cgroup_v2_cpu_total(tmp_path):
    cg = guard.own_cgroup("0::/user.slice/user-1000.slice/user@1000.service/app.slice/c1-x.service\n", root=tmp_path)
    assert cg == tmp_path / "user.slice/user-1000.slice/user@1000.service/app.slice/c1-x.service"
    cg.mkdir(parents=True)
    (cg / "cpu.stat").write_text("usage_usec 2500000\nuser_usec 2000000\nsystem_usec 500000\n")
    assert guard.cgroup_cpu_seconds(cg) == 2.5
    with pytest.raises(RuntimeError):
        guard.own_cgroup("12:cpu:/x\n", root=tmp_path)          # cgroup v1 only: refuse rather than guess


def test_the_guard_writes_a_durable_pause_marker(tmp_path):
    guard.write_marker(tmp_path / "OVERRUN_c1-x-1.json", {"unit": "c1-x-1", "cpu_seconds": 99.0})
    assert json.loads((tmp_path / "OVERRUN_c1-x-1.json").read_text())["unit"] == "c1-x-1"
    assert not list(tmp_path.glob("*.tmp"))


def test_every_job_runs_under_the_guard_with_its_cpu_budget():
    p = plan(cpu_hours=2.0)
    argv = p["argv"]
    g = argv.index(str(launch.GUARD))
    assert argv[g - 1] == launch.PYTHON and argv[g + 1:g + 3] == ["--cpu-seconds", "7200"]
    assert argv[g + 3] == "--pause-dir" and argv[argv.index("--unit") + 1] == p["unit"]
    assert argv[argv.index("--", g) + 1:] == ["echo", "hi"]


@pytest.mark.parametrize("kw", [dict(cpu_hours=float("nan")), dict(cpu_hours=0.0), dict(cpu_hours=-1.0),
                                dict(cpu_hours=float("inf")), dict(max_hours=float("nan")), dict(max_hours=0.0)])
def test_invalid_budgets_are_refused(kw):
    p = plan(**kw)
    assert p["ok"] is False and any("invalid" in r for r in p["reasons"])


def test_a_non_finite_or_negative_ledger_total_stops():
    assert ledger.gate(float("nan"), 4, False) == "stop"
    assert ledger.gate(-1.0, 4, False) == "stop"


@pytest.mark.parametrize("bad", ["nan", "inf", "-3.0"])
def test_a_corrupt_ledger_row_is_refused(tmp_path, bad):
    p = tmp_path / "compute_ledger.tsv"
    p.write_text("invocation\tunit\tstopped_at\tcpu_seconds\na\tc1-x.service\tt\t" + bad + "\n")
    with pytest.raises(ValueError):
        ledger.read_tsv(p)


def test_ledger_writes_use_a_unique_temporary_file(tmp_path, monkeypatch):
    srcs, real = [], os.replace
    monkeypatch.setattr(ledger.os, "replace", lambda a, b: (srcs.append(str(a)), real(a, b)))
    row = [{"invocation": "a", "unit": "c1-x.service", "stopped_at": "t", "cpu_seconds": 1.0}]
    ledger.write_tsv(tmp_path / "l.tsv", row)
    ledger.write_tsv(tmp_path / "l.tsv", row)
    assert len(set(srcs)) == 2 and ledger.read_tsv(tmp_path / "l.tsv") == row


REG = ("| C1-resume-x | resume after c1-r4b-run-20261005T010000Z timed out | **RULED 2026-10-06: resume** | Eric |\n"
       "| C1-other | an unrelated decision | **RULED 2026-10-06: something** | Eric |\n")


def test_a_timeout_in_one_job_pauses_every_other_c1_job(tmp_path):
    """r3 B5: another candidate's overrun used to be ignored; now it is a durable cycle-wide pause."""
    c1 = tmp_path / "hetzner_results" / "c1"
    lines = [json.dumps({"USER_UNIT": "c1-r4b-run-20261005T010000Z.service",
                         "MESSAGE": "c1-r4b-run-20261005T010000Z.service: Failed with result 'timeout'."}),
             json.dumps({"USER_UNIT": "c1-r2-x-20261005T020000Z.service",
                         "MESSAGE": "c1-r2-x-20261005T020000Z.service: Failed with result 'exit-code'."}),
             json.dumps({"USER_UNIT": "c1-burner-20261004T190543Z.service",
                         "MESSAGE": "c1-burner-20261004T190543Z.service: Failed with result 'signal'."})]
    launch.record_overruns(c1, lines)
    assert sorted(p.name for p in c1.glob("OVERRUN_*.json")) == ["OVERRUN_c1-r4b-run-20261005T010000Z.json"]
    stops = launch.overrun_stops(c1, REG)
    assert stops and "c1-r4b-run-20261005T010000Z" in stops[0]
    assert plan(overrun_stops=stops)["ok"] is False                     # a different candidate is refused too
    assert any("overrun" in r for r in plan(overrun_stops=stops)["reasons"])


def test_an_overrun_is_released_only_by_a_matching_owner_decision(tmp_path):
    c1 = tmp_path / "hetzner_results" / "c1"
    unit = "c1-r4b-run-20261005T010000Z"
    guard.write_marker(c1 / f"OVERRUN_{unit}.json", {"unit": unit, "source": "guard", "cpu_seconds": 14000.0})
    for rec in ({}, {"unit": unit}, {"unit": unit, "register_row": "C1-other", "approved_by": "Eric", "reason": "go"},
                {"unit": unit, "register_row": "C1-resume-x", "approved_by": "someone", "reason": "go"},
                {"unit": "c1-y-1", "register_row": "C1-resume-x", "approved_by": "Eric", "reason": "go"}):
        (c1 / f"RESUME_{unit}.json").write_text(json.dumps(rec))
        assert launch.overrun_stops(c1, REG), rec                       # empty, partial, unrelated row, wrong unit
    (c1 / f"RESUME_{unit}.json").write_text(json.dumps({"unit": unit, "register_row": "C1-resume-x",
                                                        "approved_by": "Eric", "reason": "budget raised"}))
    assert launch.overrun_stops(c1, REG) == []


def test_two_launches_cannot_race(tmp_path, monkeypatch):
    """r3 B5: sweep, admission and start run under one exclusive launch lock."""
    import fcntl
    started = []
    monkeypatch.setattr(launch, "_journal", lambda: [])
    monkeypatch.setattr(launch, "_run", lambda argv: "")
    monkeypatch.setattr(launch.subprocess, "run", lambda argv, check: started.append(argv))
    c1 = tmp_path / "hetzner_results" / "c1"
    c1.mkdir(parents=True)
    argv = ["--data-root", str(tmp_path), "run", "--name", "r3-fit", "--cpu-hours", "1", "--max-hours", "1", "--", "true"]
    with open(c1 / ".launch.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert launch.main(argv) == 2 and started == []                 # a concurrent launcher is refused
    assert launch.main(argv) == 0 and len(started) == 1
