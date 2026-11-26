"""Probe Requirement 4: season-start health checks, scoring, streak, and cron behavior.

Runs real functions against isolated temp dirs (cwd=tmp) representing:
  (A) brand-new 2027 state (no 2026 files, no streak.json, no pa_2027)
  (B) carried-over post-2026 state on Opening Day 2027-03-24
      (2026 picks + 2026 streak.json + 2026 contest_streak.json + 2026 saver_state.json + stale park_drag + missing blend_2027-03-25)
  (C) fresh 2027 contest fetch before any 2027 settled pick (source_date=None)
  (D) cron paths: check-results, reconcile, skip-policy-shadow-update, park-drag-refresh, day-outcome
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

REPO = Path("/Users/eric/projects/bts-agy-g3")


def setup_base_tmp(tmp: Path):
    for rel in ("data/picks/account_state", "data/models", "data/processed", "data/health_state", "data/external/park_drag/producer"):
        (tmp / rel).mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "data/models/mdp_policy.npz", tmp / "data/models/mdp_policy.npz")
    shutil.copy(REPO / "data/models/mdp_tail_policy.npz", tmp / "data/models/mdp_tail_policy.npz")


def run_health_case(label: str, picks_dir: Path, models_dir: Path, data_dir: Path, today: date, *, contest_expected: bool = True, shadow: bool = False):
    from bts.health.runner import run_all_checks
    with patch("bts.health.alert.dispatch_dm_for_health_alerts"):
        alerts = run_all_checks(
            picks_dir=picks_dir,
            models_dir=models_dir,
            dm_recipient=None,
            today=today,
            data_dir=data_dir,
            leaderboard_dir=picks_dir.parent / "leaderboard",
            shadow_model_enabled=shadow,
            live_forward_capture_enabled=False,
            contest_state_expected=contest_expected,
        )
    print(f"\n=== HEALTH: {label} (today={today}, contest_expected={contest_expected}) ===")
    for a in alerts:
        print(f"  [{a.level}] {a.source}: {a.message}")
    if not alerts:
        print("  (no alerts)")


def main():
    from zoneinfo import ZoneInfo
    from click.testing import CliRunner
    from bts.cli import cli
    from bts.contest_state import load_decision_streak_state
    from bts.picks import (
        DailyPick, Pick, save_pick, update_streak, reconcile_results, load_streak,
    )
    from bts.features.park_drag_producer import refresh as park_drag_refresh

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        old_cwd = Path.cwd()
        os.chdir(tmp)
        try:
            setup_base_tmp(tmp)
            picks = tmp / "data/picks"
            models = tmp / "data/models"
            proc = tmp / "data/processed"

            # 1) Brand-new directory on 2027-03-24 (no streak.json, no contest_streak.json, no pa_2027)
            run_health_case("A1_brand_new_no_files", picks, models, proc, date(2027, 3, 24), contest_expected=True)
            run_health_case("A2_brand_new_contest_not_expected", picks, models, proc, date(2027, 3, 24), contest_expected=False)

            # 2) Carried-over 2026 state on 2027-03-24
            (picks / "streak.json").write_text(json.dumps({
                "streak": 12, "saver_available": False, "updated": "2026-09-28T05:00:00+00:00",
                "applied_dates": ["2026-09-27"],
            }))
            (picks / "2026-09-27.json").write_text(json.dumps({
                "date": "2026-09-27", "result": "hit",
                "pick": {"batter_id": 101, "batter_name": "Vet", "team": "LAD", "game_pk": 1, "p_game_hit": 0.76, "game_time": "2026-09-27T19:00:00Z"},
            }))
            (picks / "account_state/saver_state.json").write_text(json.dumps({
                "season": 2026, "state": "active", "source": "manual", "updated_at": "2026-09-01T00:00:00Z",
            }))
            (picks / "account_state/contest_streak.json").write_text(json.dumps({
                "schema_version": "bts_contest_streak_auto_v1",
                "active_streak": 9,
                "best_streak": 19,
                "source": "mlb_bts_profile",
                "source_date": "2026-09-27",
                "recorded_at": "2026-09-28T10:30:00Z",
                "saver_available": None,
            }))
            # Stale park_drag artifacts from 2026
            ext_drag = tmp / "data/external/park_drag"
            pd.DataFrame([{"venue_id": 10, "date": "2026-09-27", "park_drag_delta": 0.01}]).to_csv(ext_drag / "park_drag_export.csv", index=False)
            (ext_drag / "park_drag_manifest.json").write_text(json.dumps({"max_source_game_date": "2026-09-27", "generated_at": "2026-09-28T07:45:00"}))
            pd.DataFrame([{"game_pk": 1, "game_date": "2026-09-27", "home_team": "LAD", "release_speed": 94, "vx0": 1, "vy0": -130, "vz0": -5, "ax": 1, "ay": 25, "az": -14}]).to_csv(ext_drag / "producer/pitches_current.csv.gz", index=False, compression="gzip")
            pd.DataFrame([{"game_pk": 1}]).to_csv(ext_drag / "producer/games_meta.csv", index=False)
            pd.DataFrame([{"game_pk": 1, "game_date": "2025-09-27"}]).to_csv(ext_drag / "producer/game_level_static.csv", index=False)

            park_summary = park_drag_refresh(ext_drag, today=date(2027, 3, 24))
            print("\n=== PARK DRAG REFRESH ON 2027-03-24 ===")
            print(park_summary)

            # Also seed blend_2027-03-25.pkl so blend check shows cleanly, or leave absent to inspect
            ds_2026 = load_decision_streak_state(picks, now=datetime(2027, 3, 24, 18, 0, tzinfo=timezone.utc))
            print("\n=== DECISION STREAK STATE WITH 2026 CARRYOVER ON 2027-03-24 ===")
            print(ds_2026)

            run_health_case("B1_2026_carryover_on_2027-03-24", picks, models, proc, date(2027, 3, 24), contest_expected=True)

            # 3) Check what happens when 2027 fetch-contest-streak gets active_streak=0, source_date=None on 2027-03-24
            (picks / "account_state/contest_streak.json").write_text(json.dumps({
                "schema_version": "bts_contest_streak_auto_v1",
                "active_streak": 0,
                "best_streak": 0,
                "source": "mlb_bts_profile",
                "source_date": None,
                "recorded_at": "2027-03-24T13:30:00Z",
                "saver_available": None,
            }))
            ds_none = load_decision_streak_state(picks, now=datetime(2027, 3, 24, 18, 0, tzinfo=timezone.utc))
            print("\n=== DECISION STREAK STATE WITH 2027 PRE-SETTLE source_date=None ===")
            print(ds_none)
            run_health_case("C_2027_source_date_None", picks, models, proc, date(2027, 3, 24), contest_expected=True)

            # 4) Check streak carryover bug in update_streak vs reconcile_results across 2026 -> 2027
            # Suppose 2026 ended with streak.json=12, and 2027-03-24 Opening Night pick hits:
            save_pick(DailyPick(
                date="2027-03-24",
                run_time="2027-03-24T22:30:00+00:00",
                pick=Pick(batter_name="Vet", batter_id=101, team="LAD", lineup_position=1,
                          pitcher_name="P", pitcher_id=6001, p_game_hit=0.76, flags=[],
                          projected_lineup=False, game_pk=910001, game_time="2027-03-24T23:10:00Z"),
                double_down=None,
                runner_up=None,
                bluesky_posted=False,
                bluesky_uri=None,
                result="hit",
                slot_results={"pick": "hit"},
            ), picks)
            # Clear applied_dates / ensure 2027-03-24 isn't already there
            s_after_update = update_streak([True], picks, date="2027-03-24")
            et = ZoneInfo("America/New_York")
            with patch("bts.picks.resolve_daily_slot_results", return_value={"pick": "hit"}):
                reconcile_results(picks, lookback_days=8, clock=lambda: datetime(2027, 3, 25, 2, 0, tzinfo=et))
            s_after_reconcile = load_streak(picks)
            print(f"\n=== STREAK CARRYOVER: 2026=12 -> Opening Night hit update_streak={s_after_update} -> reconcile={s_after_reconcile} ===")

            # Also test reconcile_results if 2027-03-24 was unresolved vs no pick at 02:00 2027-03-25
            picks_empty_27 = tmp / "picks_empty_27"
            picks_empty_27.mkdir()
            (picks_empty_27 / "streak.json").write_text(json.dumps({
                "streak": 12, "saver_available": False, "updated": "2026-09-28T05:00:00+00:00",
                "applied_dates": ["2026-09-27"],
            }))
            reconcile_results(picks_empty_27, lookback_days=8, clock=lambda: datetime(2027, 3, 25, 2, 0, tzinfo=et))
            print(f"reconcile_results on 2027-03-25 with 0 2027 picks resets 2026 streak 12 -> {load_streak(picks_empty_27)}")

            # 5) Check CLI cron commands on empty/Opening Night state (no 2027-03-23 pick)
            runner = CliRunner()
            cfg = tmp / "orch.toml"
            cfg.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
            res_do = runner.invoke(cli, ["day-outcome", "--date", "2027-03-23", "--config", str(cfg), "--picks-dir", str(picks), "--expected-username", "stonehengee", "--dm-recipient", "me.bsky.social"])
            print(f"\n=== CRON: day-outcome with no pick (2027-03-23) exit={res_do.exit_code} ===")
            print(res_do.output.strip())

            res_cr = runner.invoke(cli, ["check-results", "--date", "2027-03-23", "--picks-dir", str(picks)])
            print(f"\n=== CRON: check-results with no pick (2027-03-23) exit={res_cr.exit_code} ===")
            print(res_cr.output.strip())

            res_sps = runner.invoke(cli, ["skip-policy-shadow-update", "--picks-dir", str(picks), "--status-output", str(tmp / "data/validation/sps.json")])
            print(f"\n=== CRON: skip-policy-shadow-update empty exit={res_sps.exit_code} ===")
            print(res_sps.output.strip())

        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    main()
