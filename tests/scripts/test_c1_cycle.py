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
    assert "--nice=10" not in argv                                   # the guard niceness-shifts the job only
    for prop in ("--collect", "Delegate=yes", "-p", "MemoryMax=12G", "OOMScoreAdjust=1000", "LimitCPU=7200",
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


# ---------- C1 box limits (Eric 2026-10-05, row C1-4b-deferral; C1 infrastructure review r1 B1-B7) ----------
import hashlib
import importlib.util as _ilu
import subprocess
import sys

_gspec = _ilu.spec_from_file_location("c1_guard", Path(launch.__file__).with_name("guard.py"))
guard = _ilu.module_from_spec(_gspec)
_gspec.loader.exec_module(guard)


class FakeProc:
    def __init__(self, finish_after=None, rc=0):
        self.polls, self.finish_after, self.rc = 0, finish_after, rc

    def poll(self):
        self.polls += 1
        return self.rc if self.finish_after is not None and self.polls >= self.finish_after else None


def test_the_guard_kills_before_it_returns_and_writes_nothing_first():
    usage, events = [0.0], []

    def read():
        usage[0] += 8 * guard.POLL_S
        return usage[0]
    kind, rc = guard.watch(FakeProc(), read, threshold=guard.act_at(400.0, 8), poll=guard.POLL_S,
                           kill=lambda: events.append("kill"), sleep=lambda s: None)
    assert (kind, rc, events) == ("overrun", None, ["kill"])
    assert usage[0] >= guard.act_at(400.0, 8) and usage[0] < 400.0


def test_the_guard_returns_the_jobs_own_exit_code_within_budget():
    assert guard.watch(FakeProc(finish_after=3, rc=5), lambda: 1.0, threshold=300.0, poll=2.0,
                       kill=lambda: pytest.fail("no kill"), sleep=lambda s: None) == ("exit", 5)


def test_a_total_over_budget_or_an_oom_kill_is_never_a_clean_exit():
    assert guard.classify("exit", 0, final_cpu=401.0, budget=400.0, oom_before=0, oom_after=0) == "overrun"
    assert guard.classify("exit", 0, final_cpu=10.0, budget=400.0, oom_before=1, oom_after=2) == "oom"
    assert guard.classify("exit", 3, final_cpu=10.0, budget=400.0, oom_before=0, oom_after=0) == "exit"
    assert guard.classify("overrun", None, final_cpu=380.0, budget=400.0, oom_before=0, oom_after=0) == "overrun"


def test_the_action_margin_and_the_enforceable_minimum():
    assert guard.act_at(400.0, 8) == 400.0 - 8 * (guard.POLL_S + guard.SLACK_S)
    assert guard.min_budget(8) == 2 * 8 * (guard.POLL_S + guard.SLACK_S)
    assert guard.act_at(guard.min_budget(8), 8) > 0


def fake_cgroup(root, *, kill=True, usage_usec=1_000_000):
    cg = root / "c1-x.service"
    for leaf in ("", "guard", "payload"):
        (cg / leaf).mkdir(parents=True, exist_ok=True)
    (cg / "cpu.stat").write_text(f"usage_usec {usage_usec}\n")
    (cg / "memory.events").write_text("oom 0\noom_kill 0\n")
    (cg / "payload" / "cgroup.events").write_text("populated 0\nfrozen 0\n")
    (cg / "guard" / "cgroup.procs").write_text("")
    (cg / "payload" / "cgroup.procs").write_text("")
    if kill:
        (cg / "payload" / "cgroup.kill").write_text("")
    return cg


def test_prepare_refuses_without_whole_cgroup_termination(tmp_path):
    with pytest.raises(guard.GuardError, match="cgroup.kill"):
        guard.prepare(fake_cgroup(tmp_path, kill=False), 123)
    cg = fake_cgroup(tmp_path / "ok")
    assert guard.prepare(cg, 123) == cg / "payload" and (cg / "guard" / "cgroup.procs").read_text() == "123"


def run_guard(tmp_path, monkeypatch, cg, command, budget=400.0):
    monkeypatch.setattr(guard, "_proc_cgroup_text", lambda: "0::/x\n")
    monkeypatch.setattr(guard, "own_cgroup", lambda text: cg)
    rc = guard.main(["--cpu-seconds", str(budget), "--pause-dir", str(tmp_path / "c1"), "--unit", "c1-x-1", "--",
                     *command])
    return rc, json.loads((tmp_path / "c1" / "TERMINAL_c1-x-1.json").read_text())


def test_the_guard_runs_the_job_in_the_payload_leaf_and_writes_a_terminal_receipt(tmp_path, monkeypatch):
    cg = fake_cgroup(tmp_path)
    rc, rec = run_guard(tmp_path, monkeypatch, cg, [sys.executable, "-c", "import os; print(os.getpid())"])
    assert rc == 0 and rec["result"] == "exit" and rec["rc"] == 0 and rec["cpu_seconds"] == 1.0
    assert (cg / "payload" / "cgroup.procs").read_text().strip().isdigit()      # the job wrote itself there


def test_the_guard_reports_an_overrun_after_killing_the_payload(tmp_path, monkeypatch):
    cg = fake_cgroup(tmp_path, usage_usec=399_000_000)
    rc, rec = run_guard(tmp_path, monkeypatch, cg, [sys.executable, "-c", "import time; time.sleep(0.2)"])
    assert rc == guard.OVERRUN_EXIT and rec["result"] == "overrun"
    assert (cg / "payload" / "cgroup.kill").read_text() == "1"


def test_without_the_kill_capability_the_job_never_starts(tmp_path, monkeypatch):
    marker = tmp_path / "started"
    rc, rec = run_guard(tmp_path, monkeypatch, fake_cgroup(tmp_path, kill=False),
                        [sys.executable, "-c", f"open({str(marker)!r}, 'w')"])
    assert rc == guard.GUARD_ERROR_EXIT and rec["result"] == "guard-error" and not marker.exists()


def test_a_budget_below_the_enforceable_minimum_never_starts(tmp_path, monkeypatch):
    rc, rec = run_guard(tmp_path, monkeypatch, fake_cgroup(tmp_path), ["true"], budget=1.0)
    assert rec["result"] == "guard-error" and "--cpu-seconds" in rec["reason"]


def test_every_job_runs_under_the_guard_in_a_delegated_unit():
    p = plan(cpu_hours=2.0)
    argv = p["argv"]
    g = argv.index(str(launch.GUARD))
    assert "Delegate=yes" in argv and argv[g - 1] == launch.PYTHON
    assert argv[g + 1:g + 3] == ["--cpu-seconds", "7200"] and argv[argv.index("--unit") + 1] == p["unit"]
    assert argv[argv.index("--", g) + 1:] == ["echo", "hi"]


@pytest.mark.parametrize("kw,reason", [(dict(cpu_hours=float("nan")), "invalid cpu-hours"),
                                       (dict(cpu_hours=0.0), "invalid cpu-hours"),
                                       (dict(cpu_hours=float("inf")), "invalid cpu-hours"),
                                       (dict(max_hours=float("nan")), "invalid max-hours"),
                                       (dict(cpu_hours=1e-5), "below the guard's enforceable minimum")])
def test_invalid_or_unenforceable_budgets_are_refused(kw, reason):
    p = plan(ncpu=8, **kw)
    assert p["ok"] is False and any(reason in r for r in p["reasons"]), p["reasons"]


def test_a_non_finite_or_negative_ledger_total_stops():
    assert ledger.gate(float("nan"), 4, False) == "stop" and ledger.gate(-1.0, 4, False) == "stop"


@pytest.mark.parametrize("bad", ["nan", "inf", "-3.0"])
def test_a_corrupt_ledger_row_is_refused(tmp_path, bad):
    p = tmp_path / "compute_ledger.tsv"
    p.write_text("invocation\tunit\tstopped_at\tcpu_seconds\na\tc1-x.service\tt\t" + bad + "\n")
    with pytest.raises(ValueError):
        ledger.read_tsv(p)


def test_guard_and_reservation_rows_give_way_to_the_journal_record():
    rows = ledger.add_guard_row([], "c1-a-1", 30.0, "t1")
    rows = ledger.add_guard_row(rows, "c1-b-1", 7200.0, "reserved", reserve=True)
    assert {r["invocation"] for r in rows} == {"guard:c1-a-1", "reserve:c1-b-1"}
    journal = [{"invocation": "inv-a", "unit": "c1-a-1.service", "stopped_at": "t2", "cpu_seconds": 31.0}]
    merged = ledger.merge(rows, journal)
    assert {r["invocation"] for r in merged} == {"inv-a", "reserve:c1-b-1"}       # no double count
    assert ledger.add_guard_row(merged, "c1-a-1", 99.0, "t3") == merged          # journal already has it


# --- the launcher's admission, end to end with systemd replaced ---
@pytest.fixture
def box(tmp_path, monkeypatch):
    """The launcher on a fake canonical root: journal, active units and systemd-run are injected."""
    state = {"journal": [], "active": [], "started": [], "show": "loaded"}
    monkeypatch.setattr(launch, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(launch, "_journal", lambda: list(state["journal"]))
    monkeypatch.setattr(launch, "_active", lambda: list(state["active"]))

    def run(argv):
        if argv[:2] == ["systemctl", "--user"] and "show" in argv:
            return state["show"]
        return ""
    monkeypatch.setattr(launch, "_run", run)
    monkeypatch.setattr(launch.subprocess, "run", lambda argv, check: state["started"].append(argv))
    state["c1"] = tmp_path / "data" / "hetzner_results" / "c1"
    state["c1"].mkdir(parents=True)
    return state


ARGV = ["run", "--name", "r3-fit", "--cpu-hours", "1", "--max-hours", "1", "--", "true"]


def ledger_hours(c1):
    return ledger.total_hours(ledger.read_tsv(c1 / "compute_ledger.tsv"))


def plant_ledger(c1, hours):
    ledger.write_tsv(c1 / "compute_ledger.tsv", [{"invocation": "old", "unit": "c1-old.service", "stopped_at": "t",
                                                  "cpu_seconds": hours * 3600}])


def test_a_launch_writes_a_durable_pending_record_before_starting(box):
    assert launch.main(ARGV) == 0 and len(box["started"]) == 1
    (pend,) = box["c1"].glob("PENDING_*.json")
    assert json.loads(pend.read_text())["limit_cpu_seconds"] == 3600


def test_an_active_job_refuses_the_launch_before_any_accounting(box, monkeypatch):
    box["active"] = ["c1-r4b-run-20261005T010000Z.service"]
    swept = []
    monkeypatch.setattr(launch, "_journal", lambda: swept.append(1) or [])
    assert launch.main(ARGV) == 2 and box["started"] == [] and swept == []


def test_a_job_that_ended_just_before_admission_is_accounted_and_pauses(box):
    """r1 B2's interleaving: 49 h in the ledger; a job ended (timeout, 2 h) and its journal record is not yet
    visible. Its guard receipt is: the next launch sees 51 h and the pause."""
    plant_ledger(box["c1"], 49)
    unit = "c1-r4b-run-20261005T010000Z"
    guard.write_receipt(box["c1"] / f"PENDING_{unit}.json", {"unit": unit, "limit_cpu_seconds": 14400})
    guard.write_receipt(box["c1"] / f"TERMINAL_{unit}.json", {"unit": unit, "result": "terminated", "budget_seconds": 14400,
                                                              "cpu_seconds": 7200.0, "ended_utc": "t"})
    assert launch.main(ARGV) == 2 and box["started"] == []
    assert ledger_hours(box["c1"]) == pytest.approx(51.0)
    assert (box["c1"] / f"OVERRUN_{unit}.json").exists() and (box["c1"] / f"RECONCILED_{unit}.json").exists()
    assert (box["c1"] / f"PENDING_{unit}.json").exists()                  # evidence is never moved (r2 N1)


def test_a_job_whose_guard_left_no_receipt_reserves_its_budget_and_pauses(box):
    """r1 B4: a lost journal and a killed guard. The budget is reserved and C1 pauses, not 'nothing happened'."""
    plant_ledger(box["c1"], 49)
    unit = "c1-r4b-run-20261005T010000Z"
    guard.write_receipt(box["c1"] / f"PENDING_{unit}.json", {"unit": unit, "limit_cpu_seconds": 14400})
    assert launch.main(ARGV) == 2 and box["started"] == []
    assert ledger_hours(box["c1"]) == pytest.approx(53.0)
    assert json.loads((box["c1"] / f"OVERRUN_{unit}.json").read_text())["source"] == "unreconciled"


def test_a_clean_exit_is_accounted_without_a_pause(box):
    unit = "c1-r3-fit-20261005T010000Z"
    guard.write_receipt(box["c1"] / f"PENDING_{unit}.json", {"unit": unit, "limit_cpu_seconds": 3600})
    guard.write_receipt(box["c1"] / f"TERMINAL_{unit}.json", {"unit": unit, "result": "exit", "rc": 0, "budget_seconds": 3600,
                                                              "cpu_seconds": 90.0, "ended_utc": "t"})
    assert launch.main(ARGV) == 0 and len(box["started"]) == 1
    assert ledger_hours(box["c1"]) == pytest.approx(90 / 3600) and not list(box["c1"].glob("OVERRUN_*"))


def test_a_timeout_in_one_job_pauses_every_other_c1_job(box):
    box["journal"] = [json.dumps({"USER_UNIT": "c1-r4b-run-20261005T010000Z.service",
                                  "MESSAGE": "c1-r4b-run-20261005T010000Z.service: Failed with result 'timeout'."}),
                      json.dumps({"USER_UNIT": "c1-burner-20261004T190543Z.service",
                                  "MESSAGE": "c1-burner-20261004T190543Z.service: Failed with result 'signal'."})]
    assert launch.main(ARGV) == 2 and box["started"] == []
    assert sorted(p.name for p in box["c1"].glob("OVERRUN_*.json")) == ["OVERRUN_c1-r4b-run-20261005T010000Z.json"]


def resume_row(unit, sha, *, verb="RESUME", source="Eric 2026-10-06"):
    return f"| C1-resume-{unit} | release after the overrun | **RULED 2026-10-06: {verb} `{unit}` overrun `{sha[:16]}`** | {source} |\n"


def test_an_overrun_is_released_only_by_eric_s_exact_resume_ruling(tmp_path):
    c1 = tmp_path / "c1"
    unit = "c1-r4b-run-20261005T010000Z"
    guard.write_receipt(c1 / f"OVERRUN_{unit}.json", {"unit": unit, "source": "terminal", "result": "overrun"})
    sha = hashlib.sha256((c1 / f"OVERRUN_{unit}.json").read_bytes()).hexdigest()
    (c1 / f"RESUME_{unit}.json").write_text(json.dumps({"unit": unit, "overrun_sha256": sha}))
    negatives = {
        "no row": "",
        "denial": resume_row(unit, sha, verb="DO NOT RESUME"),
        "unrelated ruling naming the unit": f"| C1-resume-{unit} | x | **RULED 2026-10-06: keep paused; {unit}** | Eric |\n",
        "wrong source": resume_row(unit, sha, source="Codex"),
        "other row id": resume_row(unit, sha).replace(f"C1-resume-{unit} |", "C1-other |", 1),
        "prefix of another unit": resume_row(unit + "9", sha).replace(f"C1-resume-{unit}9", f"C1-resume-{unit}"),
        "another overrun's sha": resume_row(unit, "f" * 64),
    }
    for label, reg in negatives.items():
        assert launch.overrun_stops(c1, reg), label
    assert launch.overrun_stops(c1, resume_row(unit, sha)) == []
    (c1 / f"RESUME_{unit}.json").write_text(json.dumps({"unit": unit, "overrun_sha256": "0" * 64}))
    assert launch.overrun_stops(c1, resume_row(unit, sha))                   # a stale release (other instance)


def test_there_is_no_alternate_root_option():
    """r1 B3: one canonical root, lock and stop namespace for every live launch and status."""
    for argv in (["--data-root", "/tmp/x", "status"], ["status", "--data-root", "/tmp/x"]):
        with pytest.raises(SystemExit):
            launch.main(argv)


def test_two_launches_cannot_race(box):
    import fcntl
    with open(box["c1"] / ".launch.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert launch.main(ARGV) == 2 and box["started"] == []
    assert launch.main(ARGV) == 0 and len(box["started"]) == 1


def test_a_systemd_run_failure_always_keeps_the_pending_record(box, monkeypatch):
    """Infra review r2 N2: 'not-found' after a failed submission does not prove the collected unit never ran."""
    def fail(argv, check):
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(launch.subprocess, "run", fail)
    assert launch.main(ARGV) == 2 and len(list(box["c1"].glob("PENDING_*.json"))) == 1


def test_a_collected_unit_whose_submission_failed_is_still_accounted_and_paused(box, monkeypatch):
    def ran_then_failed(argv, check):           # the unit ran (overrun receipt), then the client reported failure
        unit = argv[argv.index("--unit") + 1]
        guard.write_receipt(box["c1"] / f"TERMINAL_{unit}.json", {"unit": unit, "result": "overrun",
                                                                  "budget_seconds": 3600, "cpu_seconds": 3500.0})
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(launch.subprocess, "run", ran_then_failed)
    assert launch.main(ARGV) == 2
    monkeypatch.setattr(launch.subprocess, "run", lambda argv, check: box["started"].append(argv))
    assert launch.main(ARGV) == 2 and box["started"] == []               # the overrun pauses the next launch
    assert ledger_hours(box["c1"]) == pytest.approx(3500 / 3600)



# ---------- infra review r2: N1 crash-safe reconciliation, N3, N4, N5, N6 ----------
def plant(c1, unit, limit, terminal=None):
    guard.write_receipt(c1 / f"PENDING_{unit}.json", {"unit": unit, "limit_cpu_seconds": limit})
    if terminal is not None:
        guard.write_receipt(c1 / f"TERMINAL_{unit}.json", {"unit": unit, "budget_seconds": limit, **terminal})


def test_a_crash_during_reconciliation_loses_no_accounting(box, monkeypatch):
    """r2 N1: 49 h + a clean 2 h exit; the ledger write fails once. Nothing was moved, so a retry recovers 51 h."""
    plant_ledger(box["c1"], 49)
    plant(box["c1"], "c1-x-1-aa", 14400, {"result": "exit", "rc": 0, "cpu_seconds": 7200.0})
    real, calls = ledger.write_tsv, []

    def crash_once(path, rows):
        calls.append(1)
        if len(calls) == 2:                     # the sweep's write succeeds; reconciliation's first write fails
            raise OSError("synthetic crash")
        return real(path, rows)
    monkeypatch.setattr(ledger, "write_tsv", crash_once)
    with pytest.raises(OSError):
        launch.main(ARGV)
    monkeypatch.setattr(ledger, "write_tsv", real)
    assert not (box["c1"] / "RECONCILED_c1-x-1-aa.json").exists()
    assert launch.main(["status"]) == 0 and ledger_hours(box["c1"]) == pytest.approx(51.0)
    assert launch.main(["status"]) == 0 and ledger_hours(box["c1"]) == pytest.approx(51.0)   # idempotent


def test_an_ordinary_failure_blocks_its_same_name_relaunch_after_the_journal_is_gone(box):
    """r2 N3: the receipt's non-zero rc keeps the same-name gate; another name is not blocked."""
    plant(box["c1"], "c1-r3-fit-20261005T010000Z-aa", 3600, {"result": "exit", "rc": 5, "cpu_seconds": 1.0})
    assert launch.main(ARGV) == 2 and box["started"] == []
    assert launch.main(ARGV) == 2                                        # repeated: still blocked
    assert launch.main([*ARGV[:-2], "--ack-failure", "c1-r3-fit-20261005T010000Z-aa", "--", "true"]) == 0
    unit = box["started"][-1][box["started"][-1].index("--unit") + 1]
    guard.write_receipt(box["c1"] / f"TERMINAL_{unit}.json", {"unit": unit, "result": "exit", "rc": 0,
                                                              "budget_seconds": 3600, "cpu_seconds": 2.0})
    other = ["run", "--name", "r2-x", "--cpu-hours", "1", "--max-hours", "1", "--", "true"]
    assert launch.main(other) == 0


def test_unit_names_are_never_reused():
    a, b = plan()["unit"], plan()["unit"]
    assert a != b and a.startswith("c1-r2-build-20261005T120000Z-")


@pytest.mark.parametrize("terminal", [
    {"result": "exit", "rc": 0, "cpu_seconds": 10.0, "unit": "c1-someone-else"},
    {"result": "exit", "rc": 0, "cpu_seconds": True},
    {"result": "exit", "rc": 0, "cpu_seconds": None},
    {"result": "exit", "rc": 0, "cpu_seconds": -1.0},
    {"result": "exit", "rc": 0, "cpu_seconds": float("nan")},
    {"result": "exit", "rc": "0", "cpu_seconds": 1.0},
    {"result": "exit", "rc": 0, "cpu_seconds": 4000.0},               # over its 3600 s budget, labelled exit
    {"result": "fine", "rc": 0, "cpu_seconds": 1.0},
    {"result": "exit", "rc": 0, "cpu_seconds": 1.0, "budget_seconds": 99.0},
])
def test_an_invalid_receipt_is_unreconciled_reserves_the_budget_and_pauses(box, terminal):
    """r2 N4: a receipt is trusted only if it is this invocation's and well formed."""
    plant(box["c1"], "c1-x-1-aa", 3600, {"budget_seconds": 3600, **terminal} if "budget_seconds" in terminal else terminal)
    if "unit" in terminal:
        guard.write_receipt(box["c1"] / "TERMINAL_c1-x-1-aa.json", {"budget_seconds": 3600, **terminal})
    assert launch.main(ARGV) == 2 and box["started"] == []
    assert ledger_hours(box["c1"]) == pytest.approx(1.0)
    assert json.loads((box["c1"] / "OVERRUN_c1-x-1-aa.json").read_text())["source"] == "unreconciled"


def test_the_source_must_be_eric_exactly(tmp_path):
    """r2 N6: 'Erica' and 'Ericson' are not Eric."""
    c1 = tmp_path / "c1"
    unit = "c1-x-1-aa"
    guard.write_receipt(c1 / f"OVERRUN_{unit}.json", {"unit": unit})
    sha = hashlib.sha256((c1 / f"OVERRUN_{unit}.json").read_bytes()).hexdigest()
    (c1 / f"RESUME_{unit}.json").write_text(json.dumps({"unit": unit, "overrun_sha256": sha}))
    for src in ("Erica 2026-10-06", "Ericson", "Eric's assistant"):
        assert launch.overrun_stops(c1, resume_row(unit, sha, source=src)), src
    for src in ("Eric", "Eric 2026-10-06", "Eric 2026-10-06 (manager relay)"):
        assert launch.overrun_stops(c1, resume_row(unit, sha, source=src)) == [], src


def test_a_nested_descendant_is_killed_before_the_measurement(tmp_path, monkeypatch):
    """r2 N5: the leader exits; the recursive payload is still populated (a nested cgroup). The guard kills it and
    confirms emptiness before recording."""
    cg = fake_cgroup(tmp_path)
    events = cg / "payload" / "cgroup.events"
    real_populated = guard.populated

    def populated(c):
        if (c / "cgroup.kill").read_text() == "1":          # the kill was written: the fake subtree empties
            events.write_text("populated 0\n")
        return real_populated(c)
    monkeypatch.setattr(guard, "populated", populated)
    events.write_text("populated 1\n")
    rc, rec = run_guard(tmp_path, monkeypatch, cg, [sys.executable, "-c", "pass"])
    assert rec["result"] == "exit" and rec["leftover"] is True and (cg / "payload" / "cgroup.kill").read_text() == "1"


def test_an_unconfirmed_empty_payload_is_an_enforcement_failure(tmp_path, monkeypatch):
    cg = fake_cgroup(tmp_path)
    (cg / "payload" / "cgroup.events").write_text("populated 1\n")      # never empties
    monkeypatch.setattr(guard, "CONFIRM_S", 0.4)
    monkeypatch.setattr(guard.time, "sleep", lambda s: None)
    rc, rec = run_guard(tmp_path, monkeypatch, cg, [sys.executable, "-c", "pass"])
    assert rc == guard.GUARD_ERROR_EXIT and rec["result"] == "guard-error" and "not confirmed empty" in rec["reason"]
