"""C1 cycle guard (docs/audit/2026-10-04-d4-cycle-proposal.md, approved 10/04): compute ledger + launcher rules."""
import json
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
    for prop in ("--collect", "--nice=10", "-p", "MemoryMax=12G", "OOMScoreAdjust=1000", "LimitCPU=360000",
                 "--working-directory=/home/bts/projects/bts-c1"):
        assert prop in argv
    assert argv[-2:] == ["echo", "hi"]
    assert "EnvironmentFile=/home/bts/projects/bts/.env" in argv      # the production .env, not the worktree's


def test_ledger_paths_live_in_the_production_data_root_not_the_worktree(tmp_path):
    paths = launch.c1_paths(tmp_path / "data")
    assert paths["ledger"] == tmp_path / "data" / "hetzner_results" / "c1" / "compute_ledger.tsv"
    assert paths["ack"] == tmp_path / "data" / "hetzner_results" / "c1" / "CHECKPOINT_50_ACK.json"
    assert launch.DATA_ROOT == Path.home() / "projects" / "bts" / "data"


def test_plan_limit_cpu_is_the_remaining_cycle_budget():
    rows = [{"invocation": "a", "unit": "c1-x.service", "stopped_at": "t", "cpu_seconds": 40 * 3600.0}]
    assert "LimitCPU=216000" in plan(rows=rows)["argv"]


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
