"""Adversarial probes for S1 and S2 (streak state, saver, callers, 4 season-start stages)."""
from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from click.testing import CliRunner

ET = ZoneInfo("America/New_York")


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def probe_saver_season_mismatch(tmp_root: Path) -> None:
    print("=== 1. S1/S2 + Kickoff-A2 Saver Season Mismatch ===")
    from bts.cli import cli
    from bts.contest_state import load_decision_streak_state
    from bts.health.contest_state import check as health_check
    from bts.saver_state import load_saver_state, transition_saver_state
    from bts.strategy import select_pick
    import pandas as pd

    now_2027 = datetime(2027, 3, 24, 15, 0, tzinfo=timezone.utc)  # 11:00 ET

    # Case 1A: 2026 saver_state.json left as 'active', March 2027 fetch has recorded_at in 2027
    # and source_date = 2026-09-26 (exact Kickoff-A2 scenario).
    picks_a = tmp_root / "case_1a" / "picks"
    _write_json(picks_a / "streak.json", {"streak": 12, "saver_available": True})
    _write_json(picks_a / "2026-09-26.json", {"date": "2026-09-26", "result": "hit"})
    _write_json(
        picks_a / "account_state" / "saver_state.json",
        {"season": 2026, "state": "active", "source": "auto_earn", "updated_at": "2026-08-01T00:00:00Z"},
    )
    _write_json(
        picks_a / "account_state" / "contest_streak.json",
        {
            "schema_version": "bts_contest_streak_auto_v1",
            "source": "mlb_bts_profile",
            "active_streak": 7,
            "best_streak": 7,
            "source_date": "2026-09-26",
            "recorded_at": "2027-03-24T14:30:00Z",
        },
    )
    ds_a = load_decision_streak_state(picks_a, require_contest_state=True, now=now_2027)
    print(f"Case 1A (2026 active saver + 2027 fetch w/ 2026 source_date): "
          f"streak={ds_a.streak}, saver_available={ds_a.saver_available}, status={ds_a.status}, best_status={ds_a.best_status}")
    assert ds_a.saver_available is True, "Expected bug: 2026 saver leaked into 2027!"

    # Show real select_pick action flip at streak=7, p=0.78 on 2027-03-24
    preds = pd.DataFrame([{
        "batter_id": 1001, "batter_name": "Juan Soto", "team": "NYM",
        "lineup": 2, "pitcher_id": 2001, "pitcher_name": "Ace",
        "pitcher_team": "ATL", "p_game_hit": 0.78, "game_pk": 9001,
        "game_time": "2027-03-24T19:10:00-04:00", "projected_lineup": False,
        "flags": "",
    }])
    gsd = {9001: {"abstract": "P", "detailed": "Scheduled"}}
    sel_leaked = select_pick(
        preds, "2027-03-24", picks_a,
        streak=ds_a.streak, saver_available=ds_a.saver_available,
        game_statuses_detailed=gsd,
        best_streak=ds_a.best_streak, best_status=ds_a.best_status,
    )
    sel_clean = select_pick(
        preds, "2027-03-24", picks_a,
        streak=ds_a.streak, saver_available=False,
        game_statuses_detailed=gsd,
        best_streak=ds_a.best_streak, best_status=ds_a.best_status,
    )
    print(f"  select_pick at streak=7, p=0.78: leaked_saver->{sel_leaked.action}, clean_saver->{sel_clean.action}")

    # Case 1B: Operator properly initialized saver_state.json for 2027 ('active' or 'not_earned'),
    # then fetch-contest-streak runs on 2027-03-24 when MLB profile still has source_date=2026-09-26!
    picks_b = tmp_root / "case_1b" / "picks"
    transition_saver_state(picks_b, expected_prior="uninitialized", new_state="active",
                           season=2027, source="cli_init", force=True)
    _write_json(
        picks_b / "account_state" / "contest_streak.json",
        {
            "schema_version": "bts_contest_streak_auto_v1",
            "source": "mlb_bts_profile",
            "active_streak": 11,
            "best_streak": 11,
            "source_date": "2026-09-26",
            "recorded_at": "2027-03-24T14:30:00Z",
        },
    )
    ds_b = load_decision_streak_state(picks_b, require_contest_state=True, now=now_2027)
    alerts_b = health_check(picks_b, expected=True, now=now_2027)
    print(f"Case 1B (2027 active saver on disk + 2027 fetch w/ 2026 source_date): "
          f"ds.saver_available={ds_b.saver_available}, health={[(a.level, a.message) for a in alerts_b]}")

    # Now run real CLI `fetch-contest-streak` via CliRunner with mocked auth/profile in 2027!
    import unittest.mock as mock
    from types import SimpleNamespace
    picks_cli = tmp_root / "case_1c" / "picks"
    transition_saver_state(picks_cli, expected_prior="uninitialized", new_state="not_earned",
                           season=2027, source="cli_init", force=True)
    before_saver = json.loads((picks_cli / "account_state" / "saver_state.json").read_text())

    class Fake2027DT(datetime):
        @classmethod
        def now(cls, tz=None):
            return now_2027.astimezone(tz) if tz else now_2027.replace(tzinfo=None)

    runner = CliRunner()
    with (
        mock.patch("bts.leaderboard.auth.load_session_cookies", return_value={"c": "1"}),
        mock.patch("bts.leaderboard.auth.extract_uid", return_value="u1"),
        mock.patch("bts.leaderboard.auth.fetch_login_session", return_value=SimpleNamespace(user_id="u1", username="stonehengee", xsid="xsid")),
        mock.patch("bts.contest_fetch.fetch_profile", return_value={
            "activeStreak": 0, "seasonBestStreak": 0,
            "predictions": [{"roundId": 2026, "status": "win"}],
        }),
        mock.patch("bts.cli._fetch_rounds", return_value={2026: date(2026, 9, 26)}),
    ):
        res = runner.invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks_cli),
                                  "--expected-username", "stonehengee"])
    after_saver = json.loads((picks_cli / "account_state" / "saver_state.json").read_text())
    print(f"Case 1C (CLI fetch-contest-streak in 2027 w/ 2026 source_date): exit={res.exit_code}, "
          f"before_season={before_saver['season']}, after_season={after_saver['season']}, after_state={after_saver['state']}")

    # Case 1D: Dashboard split-brain when 2026 contest_streak.json is on disk in 2027
    import bts.web as web
    picks_web = tmp_root / "case_1d" / "picks"
    _write_json(picks_web / "streak.json", {"streak": 14, "saver_available": True})
    _write_json(picks_web / "account_state" / "saver_state.json",
                {"season": 2026, "state": "active", "source": "auto_earn"})
    _write_json(picks_web / "account_state" / "contest_streak.json", {
        "schema_version": "bts_contest_streak_auto_v1",
        "source": "mlb_bts_profile",
        "active_streak": 14,
        "best_streak": 18,
        "source_date": "2026-09-26",
        "recorded_at": "2026-09-27T06:10:00Z",
    })
    with (
        mock.patch.object(web, "PICKS_DIR", picks_web),
        mock.patch("bts.contest_state.datetime", Fake2027DT),
    ):
        dash_ds = web.load_dashboard_decision_state()
        dash_saver = web.saver_dashboard_context(picks_web, now=now_2027)
    print(f"Case 1D (Web dashboard in 2027 w/ 2026 contest_streak.json): "
          f"header_streak={dash_ds.streak}, header_saver={dash_ds.saver_available}, "
          f"saver_widget_state={dash_saver.state}, saver_widget_button={dash_saver.button}")


def probe_four_stages_of_season_start(tmp_root: Path) -> None:
    print("\n=== 2. Four Stages of Season Start (2027-03-24) ===")
    from bts.contest_state import load_decision_streak_state
    from bts.health.contest_state import check as health_check
    from bts.picks import DailyPick, Pick, reconcile_results, save_pick, update_streak

    picks_dir = tmp_root / "stages" / "picks"
    # End-of-2026 state carried into 2027:
    _write_json(picks_dir / "streak.json", {"streak": 12, "saver_available": True, "applied_dates": ["2026-09-26"]})
    _write_json(picks_dir / "2026-09-26.json", {"date": "2026-09-26", "result": "hit"})
    # Stage 1: First fetch in 2027 (profile reset: source_date=None, active_streak=0)
    _write_json(picks_dir / "account_state" / "contest_streak.json", {
        "schema_version": "bts_contest_streak_auto_v1",
        "source": "mlb_bts_profile",
        "active_streak": 0,
        "best_streak": 0,
        "source_date": None,
        "recorded_at": "2027-03-24T14:30:00Z",
    })
    t1 = datetime(2027, 3, 24, 15, 0, tzinfo=timezone.utc)
    print("Stage 1 (first fetch):",
          "contest_ds=", load_decision_streak_state(picks_dir, require_contest_state=True, now=t1).streak,
          "health=", [(a.level, a.message[:60]) for a in health_check(picks_dir, expected=True, now=t1)])

    # Stage 2: First pick delivered and unsettled
    dp = DailyPick(
        date="2027-03-24", run_time="2027-03-24T22:10:00Z",
        pick=Pick("Juan Soto", 1001, "NYM", 2, "Ace", 2001, 0.80, [], False, 9001, "2027-03-24T19:10:00-04:00"),
        double_down=None, runner_up=None, notification_sent=True, notification_id="dm-1",
    )
    save_pick(dp, picks_dir)
    t2 = datetime(2027, 3, 24, 22, 30, tzinfo=timezone.utc)
    print("Stage 2 (pick unsettled):",
          "contest_ds=", load_decision_streak_state(picks_dir, require_contest_state=True, now=t2).streak,
          "health=", [(a.level, a.message[:60]) for a in health_check(picks_dir, expected=True, now=t2)])

    # Stage 3: First settlement (~22:00 ET Opening Night, run_day Step 8 -> Step 9 BEFORE 01:10 fetch & BEFORE 02:00 reconcile!)
    new_model_streak = update_streak([True], picks_dir, date="2027-03-24")
    dp.result = "hit"
    dp.slot_results = {"pick": "hit"}
    save_pick(dp, picks_dir)
    t3 = datetime(2027, 3, 25, 2, 0, tzinfo=timezone.utc)  # 2027-03-24 22:00 ET!
    s3_health = health_check(picks_dir, expected=True, now=t3)
    s3_contest_ds = load_decision_streak_state(picks_dir, require_contest_state=True, now=t3)
    # Also check model_only fallback (e.g., research/private box without contest_streak.json) between Stage 3 and Stage 4:
    picks_model_only = tmp_root / "stages_model_only" / "picks"
    _write_json(picks_model_only / "streak.json", {"streak": 12, "saver_available": True, "applied_dates": ["2026-09-26"]})
    save_pick(DailyPick(
        date="2027-03-24", run_time="2027-03-24T22:10:00Z",
        pick=Pick("Juan Soto", 1001, "NYM", 2, "Ace", 2001, 0.80, [], False, 9001, "2027-03-24T19:10:00-04:00"),
        double_down=None, runner_up=None,
    ), picks_model_only)
    m_before = load_decision_streak_state(picks_model_only, require_contest_state=False, now=t2).streak
    update_streak([True], picks_model_only, date="2027-03-24")
    dp_m = dp
    save_pick(dp_m, picks_model_only)
    m_after_settle = load_decision_streak_state(picks_model_only, require_contest_state=False, now=t3).streak
    print(f"Stage 3 (first settlement at 22:00 ET BEFORE fetch/reconcile): "
          f"update_streak_returned={new_model_streak}, contest_ds.model_streak={s3_contest_ds.model_streak}, "
          f"model_only_streak_before={m_before} -> after_settle={m_after_settle}, "
          f"EOD_health={[(a.level, a.message) for a in s3_health]}")

    # Stage 3b: Next morning 01:10 ET fetch under Phase-1 coverage split (activeStreak=1, predictions=[] -> source_date=None)
    _write_json(picks_dir / "account_state" / "contest_streak.json", {
        "schema_version": "bts_contest_streak_auto_v1",
        "source": "mlb_bts_profile",
        "active_streak": 1,
        "best_streak": 1,
        "source_date": None,
        "recorded_at": "2027-03-25T05:10:00Z",
    })
    t3b = datetime(2027, 3, 25, 12, 0, tzinfo=timezone.utc)
    print("Stage 3b (01:10 fetch w/ activeStreak=1, source_date=None):",
          "health=", [(a.level, a.message) for a in health_check(picks_dir, expected=True, now=t3b)])

    # Stage 4: First reconcile (02:00 ET on 2027-03-25)
    import unittest.mock as mock
    with mock.patch("bts.picks.resolve_daily_slot_results", return_value={"pick": "hit"}):
        reconcile_results(picks_model_only, lookback_days=3, clock=lambda: datetime(2027, 3, 25, 6, 0, tzinfo=timezone.utc))
    m_after_reconcile = load_decision_streak_state(picks_model_only, require_contest_state=False, now=t3b).streak
    print(f"Stage 4 (after first reconcile): model_only_streak={m_after_reconcile}")


def probe_unexpired_manual_override_health(tmp_root: Path) -> None:
    print("\n=== 3. S2 Unexpired Manual Override Across Year Boundary ===")
    from bts.contest_state import load_decision_streak_state
    from bts.health.contest_state import check as health_check

    picks_dir = tmp_root / "manual_ny" / "picks"
    now = datetime(2027, 1, 1, 14, 0, tzinfo=timezone.utc)  # 09:00 ET Jan 1 2027
    _write_json(picks_dir / "account_state" / "contest_streak.manual.json", {
        "schema_version": "bts_contest_streak_manual_v1",
        "source": "operator",
        "active_streak": 0,
        "best_streak": 0,
        "source_date": None,
        "recorded_at": "2026-12-31T23:30:00-05:00",
        "override_expires_at": "2027-01-02T00:00:00Z",
    })
    ds = load_decision_streak_state(picks_dir, require_contest_state=True, now=now)
    alerts = health_check(picks_dir, expected=True, now=now)
    print(f"Unexpired manual override recorded 2026-12-31: S1 uses={ds.source}/{ds.streak}, "
          f"S2 alerts={[(a.level, a.message) for a in alerts]}")


if __name__ == "__main__":
    scratch = Path(".codex-review/sr-review/scratch/tmp_s1_s2")
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    probe_saver_season_mismatch(scratch)
    probe_four_stages_of_season_start(scratch)
    probe_unexpired_manual_override_health(scratch)
