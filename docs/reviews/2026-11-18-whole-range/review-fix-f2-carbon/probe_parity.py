"""Item 1 probe: diff scope check and paired base-vs-candidate parity grid."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_SRC = Path(__file__).resolve().parent / "base_archive" / "src"
CANDIDATE_SRC = REPO_ROOT / "src"
ET = ZoneInfo("America/New_York")


RUNNER_SNIPPET = r"""
import datetime as dt
import json
import os
import sys
from collections import namedtuple
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

target_src = Path(sys.argv[1]).resolve()
work_root = Path(sys.argv[2]).resolve()

# Ensure bts imports strictly from target_src, even with editable installs.
sys.path = [
    p for p in sys.path
    if Path(p or ".").resolve() != (Path.cwd() / "src").resolve()
]
sys.meta_path = [
    finder for finder in sys.meta_path
    if "editable" not in type(finder).__name__.lower()
]
sys.path.insert(0, str(target_src))

import bts
from bts import contest_state, scheduler, strategy
from bts.health import alert, contest_state as health_cs, runner

actual_init = Path(bts.__file__).resolve()
expected_init = (target_src / "bts" / "__init__.py").resolve()
assert actual_init == expected_init, f"Wrong bts.__file__: {actual_init} != {expected_init}"

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
DAY = REAL_DT(2027, 7, 15, 22, 15, tzinfo=ET)
ACT_ID = "a" * 32


def make_clock(fixed_now):
    class PinnedDatetime(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)
    return PinnedDatetime


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def put_dated(picks: Path, eff_date: dt.date, saver: str, streak: int, best: int) -> None:
    exp = REAL_DT.combine(eff_date, dt.time.max, ET)
    write_json(picks / "account_state" / "contest_streak.manual.json", {
        "schema_version": "bts_contest_streak_manual_v3",
        "effective_date": eff_date.isoformat(),
        "season": eff_date.year,
        "expires_at": exp.isoformat(),
        "active_streak": streak,
        "best_streak": best,
        "saver_state": saver,
        "recorded_at": DAY.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "cli_dated",
    })


def put_switch(picks: Path, season: int = 2027, status: str = "automatic") -> None:
    write_json(picks / "account_state" / "season_contract.json", {
        "schema_version": "bts_season_contract_v1",
        "season": season,
        "status": status,
        "activation_id": ACT_ID,
        "switched_at": "2027-04-01T12:00:00Z" if status == "automatic" else None,
        "revoked_at": "2027-04-01T12:00:00Z" if status == "revoked" else None,
        "note": None,
    })


def put_auto(picks: Path, source_date: str | None, streak: int = 12, season: int = 2027) -> None:
    payload = {
        "schema_version": "bts_contest_streak_auto_v2",
        "season": season,
        "activation_id": ACT_ID,
        "active_streak": streak,
        "best_streak": max(streak, 15),
        "source": "mlb_bts_profile",
        "recorded_at": DAY.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if source_date is not None:
        payload["source_date"] = source_date
    write_json(picks / "account_state" / "contest_streak.json", payload)


def put_flag(picks: Path, kind: str, year: int = 2027) -> None:
    p = picks / "account_state" / "saver_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if kind == "absent":
        return
    if kind == "corrupt":
        p.write_text("{bad", encoding="utf-8")
        return
    if kind == "other_season":
        write_json(p, {"season": year - 1, "state": "active"})
        return
    write_json(p, {"season": year, "state": kind})


def setup_clean_aux(root: Path, slate: dt.date) -> tuple[Path, Path]:
    picks = root / "data" / "picks"
    models = root / "data" / "models"
    picks.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    write_json(picks / "streak.json", {"season": slate.year, "streak": 5, "saver_available": True})
    tomorrow = slate + dt.timedelta(days=1)
    (models / f"blend_{tomorrow.isoformat()}.pkl").write_text("", encoding="utf-8")
    return picks, models


SCENARIOS = [
    ("dated_active_s0", {"obs": ("dated", "active", 0, 0)}),
    ("dated_active_s14", {"obs": ("dated", "active", 14, 15)}),
    ("dated_active_s58", {"obs": ("dated", "active", 58, 60)}),
    ("dated_not_earned_s9", {"obs": ("dated", "not_earned", 9, 9)}),
    ("dated_not_earned_s12", {"obs": ("dated", "not_earned", 12, 15)}),
    ("dated_not_earned_s16", {"obs": ("dated", "not_earned", 16, 20)}),
    ("expired_dated_used_auto_ok", {"obs": ("expired_used_auto_ok",)}),
    ("expired_dated_used_refused", {"obs": ("expired_used_refused",)}),
    ("future_dated_used_auto_ok", {"obs": ("future_used_auto_ok",)}),
    ("legacy_unexpired", {"obs": ("legacy", False)}),
    ("legacy_expired", {"obs": ("legacy", True)}),
    ("auto_accepted_gap0", {"obs": ("auto", "2027-07-15", 0)}),
    ("auto_accepted_gap1", {"obs": ("auto", "2027-07-13", 1)}),
    ("auto_accepted_gap2", {"obs": ("auto", "2027-07-12", 2)}),
    ("auto_future_source", {"obs": ("auto", "2027-07-25", 0)}),
    ("auto_missing_source_date", {"obs": ("auto", None, 0)}),
    ("missing_state", {"obs": ("missing",)}),
    ("invalid_v3", {"obs": ("invalid_v3",)}),
]

FLAG_KINDS = ["active", "not_earned", "used", "uninitialized", "absent", "other_season", "corrupt"]
MODES = [False, True]
RECIPIENTS = ["ops.bsky.social", None, ""]
EXPECTED_VALS = [True, False]

_Usage = namedtuple("_Usage", ["total", "used", "free"])
clean_usage = _Usage(total=100 * 1024 ** 3, used=40 * 1024 ** 3, free=60 * 1024 ** 3)

results = []
idx = 0
for scen_name, spec in SCENARIOS:
    for flag_kind in FLAG_KINDS:
        for historical in MODES:
            # Test full recipient x expected cross-product on key scenarios, representative on rest
            recips = RECIPIENTS if flag_kind in ("active", "absent") else ["ops.bsky.social"]
            exps = EXPECTED_VALS if flag_kind in ("active", "not_earned") else [True]
            for recip in recips:
                for expected in exps:
                    idx += 1
                    case_dir = work_root / f"c_{idx}"
                    slate = DAY.date()
                    wall = DAY + dt.timedelta(hours=3) if historical else DAY
                    picks, models = setup_clean_aux(case_dir, slate)
                    put_flag(picks, flag_kind, slate.year)

                    obs = spec["obs"]
                    if obs[0] == "dated":
                        _, sv, st, bst = obs
                        put_dated(picks, slate, sv, st, bst)
                    elif obs[0] == "expired_used_auto_ok":
                        put_dated(picks, slate - dt.timedelta(days=1), "used", 14, 15)
                        put_switch(picks, slate.year, "automatic")
                        put_auto(picks, "2027-07-14", 12, slate.year)
                    elif obs[0] == "expired_used_refused":
                        put_dated(picks, slate - dt.timedelta(days=1), "used", 14, 15)
                        put_switch(picks, slate.year, "revoked")
                    elif obs[0] == "future_used_auto_ok":
                        put_dated(picks, slate + dt.timedelta(days=1), "used", 14, 15)
                        put_switch(picks, slate.year, "automatic")
                        put_auto(picks, "2027-07-14", 12, slate.year)
                    elif obs[0] == "legacy":
                        expired = obs[1]
                        exp_dt = DAY - dt.timedelta(hours=2) if expired else DAY + dt.timedelta(hours=5)
                        write_json(picks / "account_state" / "contest_streak.manual.json", {
                            "schema_version": "bts_contest_streak_manual_v2",
                            "active_streak": 12,
                            "best_streak": 15,
                            "source": "manual",
                            "source_date": slate.isoformat(),
                            "recorded_at": "2027-07-15T15:00:00Z",
                            "override_expires_at": exp_dt.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                        })
                    elif obs[0] == "auto":
                        _, src_d, gap = obs
                        put_switch(picks, slate.year, "automatic")
                        put_auto(picks, src_d, 12, slate.year)
                        for g in range(gap):
                            d_str = ( dt.date(2027, 7, 14) + dt.timedelta(days=g) ).isoformat()
                            write_json(picks / f"{d_str}.json", {"date": d_str, "result": "hit"})
                    elif obs[0] == "missing":
                        pass
                    elif obs[0] == "invalid_v3":
                        put_dated(picks, slate, "active", 12, 15)
                        mpath = picks / "account_state" / "contest_streak.manual.json"
                        bad = json.loads(mpath.read_text(encoding="utf-8"))
                        bad["saver_state"] = "bogus"
                        write_json(mpath, bad)

                    clock_cls = make_clock(wall)
                    sent_dms = []

                    def fake_send(to, body):
                        sent_dms.append([to, body])
                        return "msg-1"

                    check_now = (
                        REAL_DT.combine(slate, dt.time.max, ET).astimezone(dt.timezone.utc)
                        if historical else wall
                    )
                    with patch.object(dt, "datetime", clock_cls), \
                         patch.object(alert, "datetime", clock_cls), \
                         patch.object(health_cs, "datetime", clock_cls), \
                         patch.object(runner, "datetime", clock_cls), \
                         patch.object(alert, "send_dm", side_effect=fake_send), \
                         patch("bts.health.disk_fill.shutil.disk_usage", return_value=clean_usage), \
                         patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None), \
                         patch.object(runner.tail_policy, "check", return_value=[]), \
                         patch.object(runner.park_drag_freshness, "check", return_value=[]), \
                         patch.object(runner.unit_drift, "check", return_value=[]):
                        direct = health_cs.check(
                            picks,
                            expected=expected,
                            now=check_now,
                            historical_day=slate if historical else None,
                        )
                        eod_alerts = runner.run_all_checks(
                            picks_dir=picks,
                            models_dir=models,
                            dm_recipient=recip,
                            today=slate,
                            contest_state_expected=expected,
                            live_forward_capture_enabled=False,
                            shadow_model_enabled=False,
                        )

                    health_dir = case_dir / "data" / "health_state"
                    status_files = {}
                    if health_dir.exists():
                        for sf in sorted(health_dir.iterdir()):
                            if sf.is_file():
                                status_files[sf.name] = sf.read_text(encoding="utf-8")

                    results.append({
                        "id": f"{scen_name}|flag={flag_kind}|hist={historical}|recip={recip!r}|exp={expected}",
                        "direct": [(a.level, a.source, a.message, a.incident_key) for a in direct],
                        "eod": [(a.level, a.source, a.message, a.incident_key) for a in eod_alerts],
                        "dms": sent_dms,
                        "status_files": status_files,
                    })

# Also check health_checks.enabled = False on an in-effect dated used mismatch day!
disabled_dir = work_root / "health_disabled"
picks_d, _ = setup_clean_aux(disabled_dir, DAY.date())
put_dated(picks_d, DAY.date(), "used", 14, 15)
put_flag(picks_d, "active", DAY.year)
cfg = {
    "orchestrator": {"picks_dir": str(picks_d), "models_dir": str(disabled_dir / "data" / "models")},
    "scheduler": {"contest_state_required": True, "entry_intent": "enter"},
    "bluesky": {"pick_delivery": "dm", "dm_recipient": "ops.bsky.social"},
    "health_checks": {"enabled": False, "contest_state_expected": True},
}
sent_disabled = []
with patch.object(scheduler, "fetch_schedule", return_value=[{
         "gamePk": 1, "gameType": "R", "gameDate": "2027-07-15T23:10:00Z",
         "teams": {"away": {"team": {"name": "A"}}, "home": {"team": {"name": "B"}}},
     }]), \
     patch.object(scheduler, "_now_et", return_value=REAL_DT(2027, 7, 15, 23, 30, tzinfo=ET)), \
     patch.object(scheduler, "_idle_until_next_wakeup", return_value=None), \
     patch.object(scheduler, "notify_ready", return_value=None), \
     patch.object(scheduler, "notify_watchdog", return_value=None), \
     patch.object(alert, "send_dm", side_effect=lambda r, b: sent_disabled.append([r, b]) or "ok"):
    st = scheduler.SchedulerState(
        date="2027-07-15",
        schedule_fetched_at="2027-07-15T14:00:00Z",
        games=[],
        confirmed_game_pks=[],
        runs_completed=[],
        pick_locked=True,
        pick_locked_at="2027-07-15T22:00:00Z",
        result_status="final",
        next_wakeup="2027-07-16T14:00:00Z",
        committed_pick_written=True,
    )
    scheduler.save_state(st, picks_d)
    write_json(picks_d / "2027-07-15" / "decision.json", {"scoreable": True, "action": "single"})
    scheduler.run_day("2027-07-15", cfg)

hd_dir = disabled_dir / "data" / "health_state"
hd_files = sorted(p.name for p in hd_dir.iterdir()) if hd_dir.exists() else []

print(json.dumps({
    "bts_file": str(actual_init),
    "count": len(results),
    "results": results,
    "disabled_dms": sent_disabled,
    "disabled_health_files": hd_files,
}))
"""


def test_only_contract_files_and_authorized_cs4a_cases_changed():
    proc = subprocess.run(
        ["git", "--no-optional-locks", "diff", "--name-only", "a4e8278d", "2e20fdfc"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    changed = [
        line for line in proc.stdout.splitlines()
        if not line.startswith("docs/reviews/2026-11-18-whole-range/")
    ]
    assert changed == [
        "src/bts/health/contest_state.py",
        "src/bts/health/runner.py",
        "tests/test_contest_dated_cs4a.py",
        "tests/test_wr_f2_saver_flag.py",
    ]

    cs4a_diff = subprocess.run(
        ["git", "--no-optional-locks", "diff", "-U0", "a4e8278d", "2e20fdfc", "--", "tests/test_contest_dated_cs4a.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    ).stdout
    added = [l[1:] for l in cs4a_diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:] for l in cs4a_diff.splitlines() if l.startswith("-") and not l.startswith("---")]
    assert removed == []
    assert len(added) == 5
    assert all("saver=='used'" in line or "bounded_read" in line for line in added)


def test_paired_grid_matches_base_outside_dated_used(tmp_path):
    assert (BASE_SRC / "bts" / "__init__.py").is_file(), f"Missing base archive at {BASE_SRC}"
    outputs = {}
    for label, src_dir in [("base", BASE_SRC), ("candidate", CANDIDATE_SRC)]:
        run_dir = tmp_path / label
        run_dir.mkdir()
        proc = subprocess.run(
            [sys.executable, "-c", RUNNER_SNIPPET, str(src_dir), str(run_dir)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        payload = json.loads(proc.stdout)
        print(f"{label} bts.__file__ = {payload['bts_file']} (cases={payload['count']})")
        outputs[label] = payload

    assert outputs["base"]["bts_file"] != outputs["candidate"]["bts_file"]
    assert "base_archive/src/bts/__init__.py" in outputs["base"]["bts_file"]
    assert outputs["base"]["count"] == outputs["candidate"]["count"]
    assert outputs["base"]["count"] >= 350

    for b_row, c_row in zip(outputs["base"]["results"], outputs["candidate"]["results"]):
        assert b_row["id"] == c_row["id"]
        assert c_row["direct"] == b_row["direct"], f"Direct mismatch at {b_row['id']}"
        assert c_row["eod"] == b_row["eod"], f"EOD alerts mismatch at {b_row['id']}"
        assert c_row["dms"] == b_row["dms"], f"DM body mismatch at {b_row['id']}"
        assert c_row["status_files"] == b_row["status_files"], f"Status file mismatch at {b_row['id']}"
        assert not any(
            name.endswith(".dated_used_mismatch.json") for name in c_row["status_files"]
        ), f"Unexpected mismatch status file at {b_row['id']}"

    assert outputs["candidate"]["disabled_dms"] == outputs["base"]["disabled_dms"] == []
    assert outputs["candidate"]["disabled_health_files"] == outputs["base"]["disabled_health_files"] == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "-rA"]))
