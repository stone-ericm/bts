from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
from click.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[3]
ET = ZoneInfo("America/New_York")


def _pred_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _cand_row(batter_id: int, name: str, team: str, opp: str, p_hit: float, game_pk: int,
              pos: int = 1, projected: bool = False) -> dict:
    return {
        "batter_id": batter_id,
        "batter_name": name,
        "team": team,
        "opponent": opp,
        "pitcher_id": 500 + game_pk % 100,
        "pitcher_name": "Pitcher X",
        "pitcher_team": opp,
        "p_game_hit": p_hit,
        "lineup": pos,
        "lineup_position": pos,
        "flags": "PROJECTED" if projected else "",
        "projected_lineup": projected,
        "game_pk": game_pk,
        "game_time": "2027-03-24T23:05:00Z",
    }


def probe_finding_1_saver_not_earned() -> dict:
    from bts.contest_state import load_decision_streak_state
    from bts.saver_state import load_saver_state, maybe_auto_earn_saver
    from bts.simulate.mdp import load_policy
    from bts.strategy import select_pick

    policy_path = REPO_ROOT / "data" / "models" / "mdp_policy.npz"
    pt, bounds, _ = load_policy(policy_path)
    diff_0_9 = int((pt[:10, 1:, 0, :] != pt[:10, 1:, 1, :]).sum())
    total_0_9 = int(pt[:10, 1:, 0, :].size)
    actions = {0: "skip", 1: "single", 2: "double"}
    d180_flips = []
    for s in range(10):
        for q in range(pt.shape[3]):
            a0, a1 = int(pt[s, 180, 0, q]), int(pt[s, 180, 1, q])
            if a0 != a1:
                d180_flips.append(
                    f"s={s},q={q}:saver0={actions[a0]} vs saver1={actions[a1]}"
                )

    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        (picks_dir / "account_state").mkdir(parents=True)
        (picks_dir / "account_state" / "contest_streak.json").write_text(json.dumps({
            "active_streak": 0,
            "season_best_streak": 0,
            "saver_available": True,
            "source_date": None,
            "recorded_at": "2027-03-23T17:30:00Z",
        }))
        maybe_auto_earn_saver(picks_dir, season=2027, best_streak=0)
        saver = load_saver_state(picks_dir, season=2027)
        ds0 = load_decision_streak_state(picks_dir, now=datetime(2027, 3, 24, 15, 0, tzinfo=ET))

        # Day 6 of 2027: 2027-03-29, streak=5, best_streak=5 -> saver_state still "not_earned"
        (picks_dir / "account_state" / "contest_streak.json").write_text(json.dumps({
            "active_streak": 5,
            "season_best_streak": 5,
            "saver_available": True,
            "source_date": "2027-03-28",
            "recorded_at": "2027-03-29T06:10:00Z",
        }))
        maybe_auto_earn_saver(picks_dir, season=2027, best_streak=5)
        ds5 = load_decision_streak_state(picks_dir, now=datetime(2027, 3, 29, 15, 0, tzinfo=ET))

        p_q0 = float(bounds[0]) - 0.005
        preds = _pred_df([
            _cand_row(1, "Batter A", "NYY", "BOS", p_q0, 1001, pos=1),
            _cand_row(2, "Batter B", "LAD", "SD", p_q0 - 0.002, 1002, pos=1),
        ])
        statuses = {
            1001: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
            1002: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
        }
        sel_bug = select_pick(
            preds, "2027-03-29", picks_dir,
            streak=ds5.streak, saver_available=ds5.saver_available,
            game_statuses_detailed=statuses,
        )
        sel_true = select_pick(
            preds, "2027-03-29", picks_dir,
            streak=ds5.streak, saver_available=True,
            game_statuses_detailed=statuses,
        )
        return {
            "saver_state_at_0": saver.state,
            "b8_decision_saver_available_at_0": ds0.saver_available,
            "diff_cells_streaks_0_9": f"{diff_0_9}/{total_0_9}",
            "d180_flips": d180_flips,
            "s5_q0_action_with_bug": sel_bug.action,
            "s5_q0_action_with_saver_true": sel_true.action,
        }


def probe_finding_2_partial_rollover() -> dict:
    from bts.contest_fetch import derive_source_date
    from bts.contest_state import load_decision_streak_state
    from bts.saver_state import load_saver_state, maybe_auto_earn_saver, season_for
    from bts.strategy import select_pick

    rounds_2027 = {"9001": date(2027, 3, 24), "9002": date(2027, 3, 25)}
    stale_2026_predictions = [
        {"roundId": "8120", "streakResult": 18, "picks": [{"playerId": "660271", "status": "HIT"}]},
    ]
    derived = derive_source_date(stale_2026_predictions, rounds_2027)
    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        (picks_dir / "account_state").mkdir(parents=True)
        maybe_auto_earn_saver(picks_dir, season=2027, best_streak=0)
        before_saver = load_saver_state(picks_dir, season=2027).state
        inferred_season = season_for(derived, now_year=2027)
        maybe_auto_earn_saver(picks_dir, season=inferred_season, best_streak=18)
        after_saver = load_saver_state(picks_dir, season=2027).state
        (picks_dir / "account_state" / "contest_streak.json").write_text(json.dumps({
            "active_streak": 18,
            "season_best_streak": 18,
            "saver_available": False,
            "source_date": derived,
            "recorded_at": "2027-03-20T14:30:00Z",
        }))
        ds = load_decision_streak_state(picks_dir, now=datetime(2027, 3, 24, 15, 0, tzinfo=ET))
        preds = _pred_df([_cand_row(1, "Batter A", "NYY", "SF", 0.77, 1001, pos=1)])
        statuses = {1001: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}
        sel_s18 = select_pick(
            preds, "2027-03-24", picks_dir,
            streak=ds.streak, saver_available=ds.saver_available,
            game_statuses_detailed=statuses,
        )
        sel_s0 = select_pick(
            preds, "2027-03-24", picks_dir,
            streak=0, saver_available=True,
            game_statuses_detailed=statuses,
        )
        return {
            "derived_source_date": derived,
            "inferred_season": inferred_season,
            "saver_2027_before_vs_after": f"{before_saver} -> {after_saver}",
            "decision_streak_on_opening_night": ds.streak,
            "opening_night_action_s18_vs_s0": f"{sel_s18.action} vs {sel_s0.action}",
        }


def probe_finding_3_dashboard_saver_season() -> dict:
    from bts.health import contest_state as health_cs
    from bts.saver_state import load_saver_state
    from bts.web import saver_dashboard_context, saver_transition_response

    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        (picks_dir / "account_state").mkdir(parents=True)
        (picks_dir / "account_state" / "contest_streak.json").write_text(json.dumps({
            "active_streak": 12,
            "season_best_streak": 18,
            "saver_available": True,
            "source_date": "2026-09-27",
            "recorded_at": "2026-09-28T06:10:00Z",
        }))
        (picks_dir / "account_state" / "saver_state.json").write_text(json.dumps({
            "season": 2026,
            "state": "active",
            "updated_at": "2026-09-10T12:00:00Z",
            "source": "auto_earn",
        }))
        now = datetime(2027, 3, 10, 12, 0, tzinfo=ET)
        ctx = saver_dashboard_context(picks_dir, now=now)
        code, msg = saver_transition_response(
            picks_dir, expected_prior="active", new_state="used", same_origin=True, now=now,
        )
        raw_after = json.loads((picks_dir / "account_state" / "saver_state.json").read_text())
        s2027 = load_saver_state(picks_dir, season=2027).state
        h_alerts = health_cs.check(picks_dir, expected=False, now=now)
        return {
            "dashboard_ctx_on_2027_03_10": {"state": ctx.state, "button": ctx.button, "warning": ctx.warning},
            "post_transition_result": (code, msg),
            "persisted_saver_json": raw_after,
            "saver_2027_state": s2027,
            "health_contest_state_alerts_at_s12": [a.message for a in h_alerts],
        }


def probe_finding_4_live_forward_9_rows() -> dict:
    from bts.daily_decision import write_decision
    from bts.experiment.artifacts import (
        materialize_live_candidate_profile_pair,
        verify_candidate_artifact_pair,
    )
    from bts.experiment.registry import get_experiment, load_all_experiments
    from bts.health import analytics_artifacts_missing
    from bts.model.predict import _fetch_prior_lineup
    from bts.picks import DailyPick, Pick, save_pick
    from bts.scheduler import _lock_decision_from_predictions

    load_all_experiments()

    spec = importlib.util.spec_from_file_location(
        "live_forward_capture_once", REPO_ROOT / "scripts" / "live_forward_capture_once.py"
    )
    lfc = importlib.util.module_from_spec(spec)
    sys.modules["live_forward_capture_once"] = lfc
    spec.loader.exec_module(lfc)

    class FakeUrlResp:
        def __init__(self, body: dict):
            self._raw = json.dumps(body).encode()
        def read(self) -> bytes:
            return self._raw

    with patch("bts.model.predict.urlopen", return_value=FakeUrlResp({"dates": []})):
        prior_lu = _fetch_prior_lineup(137, 2027)

    cands_9 = _pred_df([
        _cand_row(100 + i, f"Player {i}", "NYY", "SF", 0.80 - i * 0.01, 777001, pos=i + 1)
        for i in range(9)
    ])
    pick_slot = Pick(
        batter_name="Player 0", batter_id=100, team="NYY", lineup_position=1,
        pitcher_name="Webb", pitcher_id=500, p_game_hit=0.80, flags=[],
        projected_lineup=False, game_pk=777001, game_time="2027-03-24T23:05:00Z",
        pitcher_team="SF",
    )
    daily_9 = DailyPick(
        date="2027-03-24", run_time="2027-03-24T17:05:00-04:00",
        pick=pick_slot, double_down=None, runner_up=None,
        notification_sent=True, notification_id="dm-1",
        delivered_at="2027-03-24T17:05:05-04:00",
    )
    statuses = {777001: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}
    with patch("bts.picks.get_game_statuses_detailed", return_value=statuses):
        lock_dec = _lock_decision_from_predictions(cands_9, daily_9, "2027-03-24", early_lock_gap=0.02)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        picks_dir = root / "data" / "picks"
        data_dir = root / "data" / "processed"
        picks_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)
        save_pick(daily_9, picks_dir)
        write_decision(
            "2027-03-24", picks_dir, action="single", source="mdp",
            primary={"batter_id": 100, "batter_name": "Player 0", "p_game_hit": 0.80},
            double_down=None, streak=0, saver_available=True,
            delivery_status="delivered", scoreable=True,
        )

        commit_sha = "deadbeef" * 5
        export_calls = {"n": 0, "n_rows_available": 9}
        cand_exp = get_experiment("decision_weighted_lgbm_v0")

        def fake_export(config, *, pick_path, artifact_dir):
            export_calls["n"] += 1
            n = export_calls["n_rows_available"]
            fake_preds = _pred_df([
                _cand_row(100 + i, f"Player {i}", "NYY", "SF", 0.80 - i * 0.005, 777001, pos=(i % 9) + 1)
                for i in range(n)
            ])
            with patch("bts.model.predict.run_pipeline", return_value=fake_preds):
                materialize_live_candidate_profile_pair(
                    date=config.date,
                    candidate=cand_exp,
                    output_dir=artifact_dir,
                    data_dir=data_dir,
                    top_n=config.top_n,
                    refresh_data=False,
                    production_pick_file=pick_path,
                    git_commit=commit_sha,
                )
            return subprocess.CompletedProcess(["bts"], 0, stdout="exported", stderr="")

        def fake_verify(config, *, artifact_dir, verification_path, live_forward_head, require_pick_snapshot=True):
            rep = verify_candidate_artifact_pair(
                artifact_dir=artifact_dir,
                expected_run_kind="live_forward_preoutcome",
                expected_candidate=config.candidate,
                expected_date=config.date,
                expected_git_commit=live_forward_head,
                expected_top_n=config.top_n,
                require_live_preoutcome=True,
                require_production_pick_snapshot=require_pick_snapshot,
                save_path=verification_path,
            )
            ok = bool(rep["ok"])
            failed = [(c["name"], c.get("detail")) for c in rep["checks"] if c["status"] != "pass"]
            return subprocess.CompletedProcess(
                ["bts"], 0 if ok else 1,
                stdout="", stderr=f"failed_checks={failed}" if failed else "",
            )

        cfg = lfc.CaptureConfig(
            date="2027-03-24",
            production_root=root,
            live_forward_root=root,
            python=Path(sys.executable),
            candidate="decision_weighted_lgbm_v0",
            artifact_root=Path("data/validation/decision_weighted_lgbm_v0_live_forward"),
            data_dir=Path("data/processed"),
            picks_dir=Path("data/picks"),
            top_n=10,
            overwrite=False,
            fail_on_pending=False,
            auto_recapture_on_snapshot_drift=True,
            now=datetime(2027, 3, 24, 17, 15, tzinfo=ET),
        )
        with patch.object(lfc, "git_head", return_value=commit_sha), \
             patch.object(lfc, "export_artifact", side_effect=fake_export), \
             patch.object(lfc, "verify_artifact", side_effect=fake_verify):
            rc1, p1 = lfc.capture_once(cfg)
            export_calls["n_rows_available"] = 18
            rc2, p2 = lfc.capture_once(cfg)

        with patch.object(analytics_artifacts_missing, "read_systemd_unit_summary", return_value=None):
            eod_alerts = analytics_artifacts_missing.check(
                picks_dir, today=date(2027, 3, 24), shadow_expected=False, capture_expected=True,
            )
        return {
            "prior_lineup_2027_len": len(prior_lu),
            "should_lock_9_batters": lock_dec.should_lock,
            "tick1_rc_status": (rc1, p1["status"], p1["message"]),
            "tick2_rc_status_after_18_avail": (rc2, p2["status"]),
            "export_call_count": export_calls["n"],
            "eod_health_alerts": [(a.level, a.message) for a in eod_alerts],
        }


def probe_finding_5_opening_night_eod_contest_state() -> dict:
    from bts.contest_state import latest_resolved_pick_date
    from bts.health import contest_state as health_cs
    from bts.picks import DailyPick, Pick, save_pick

    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        (picks_dir / "account_state").mkdir(parents=True)
        (picks_dir / "account_state" / "contest_streak.json").write_text(json.dumps({
            "active_streak": 0,
            "season_best_streak": 0,
            "saver_available": True,
            "source_date": None,
            "recorded_at": "2027-03-24T17:30:00Z",
        }))
        (picks_dir / "account_state" / "saver_state.json").write_text(json.dumps({
            "season": 2027, "state": "not_earned", "updated_at": "2027-03-24T17:30:00Z", "source": "auto_earn",
        }))
        slot = Pick(
            batter_name="Judge", batter_id=100, team="NYY", lineup_position=2,
            pitcher_name="Webb", pitcher_id=500, p_game_hit=0.80, flags=[],
            projected_lineup=False, game_pk=777001, game_time="2027-03-24T23:05:00Z",
        )
        save_pick(DailyPick(
            date="2027-03-24", run_time="2027-03-24T17:05:00-04:00",
            pick=slot, double_down=None, runner_up=None,
            notification_sent=True, delivered_at="2027-03-24T17:05:05-04:00",
            result="hit", slot_results={"pick": "hit"},
        ), picks_dir)
        now_eod1 = datetime(2027, 3, 24, 22, 15, tzinfo=ET)
        latest = latest_resolved_pick_date(picks_dir)
        s2_suppressed = not (latest is not None and latest.year == 2027)
        alerts_day1 = health_cs.check(picks_dir, expected=True, now=now_eod1)

        # Day 2 (2027-03-25): MLB per-round predictions lags 1 settled pick -> activeStreak=1, predictions=[]
        (picks_dir / "account_state" / "contest_streak.json").write_text(json.dumps({
            "active_streak": 1,
            "season_best_streak": 1,
            "saver_available": True,
            "source_date": None,
            "recorded_at": "2027-03-25T17:30:00Z",
        }))
        save_pick(DailyPick(
            date="2027-03-25", run_time="2027-03-25T18:05:00-04:00",
            pick=slot, double_down=None, runner_up=None,
            notification_sent=True, delivered_at="2027-03-25T18:05:05-04:00",
            result="hit", slot_results={"pick": "hit"},
        ), picks_dir)
        alerts_day2 = health_cs.check(picks_dir, expected=True, now=datetime(2027, 3, 25, 22, 15, tzinfo=ET))
        return {
            "latest_resolved_at_day1_eod": str(latest),
            "kickoff_s2_suppression_applies_at_day1_eod": s2_suppressed,
            "day1_eod_alerts": [(a.level, a.message) for a in alerts_day1],
            "day2_eod_alerts": [(a.level, a.message) for a in alerts_day2],
        }


def probe_finding_6_park_drag() -> dict:
    from bts.features import park_drag_producer
    from bts.health import park_drag_freshness

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "park_drag"
        prod = root / "producer"
        prod.mkdir(parents=True)
        pd.DataFrame(columns=[
            "game_pk", "game_date", "vx0", "vy0", "vz0", "ax", "ay", "az", "release_speed"
        ]).to_csv(prod / "pitches_current.csv.gz", index=False, compression="gzip")
        pd.DataFrame(columns=["game_pk", "venue_id", "venue", "home_team", "temp_f"]).to_csv(
            prod / "games_meta.csv", index=False
        )
        pd.DataFrame([{
            "game_pk": 1, "season": 2026, "game_date": "2026-09-27",
            "venue_id": 3313, "venue": "Yankee Stadium", "n_pitch": 60, "cd_trim": 0.34,
        }]).to_csv(prod / "game_level_static.csv", index=False)

        refresh_summary = park_drag_producer.refresh(
            root, today=date(2027, 3, 25), http_get=lambda *a, **kw: ""
        )

        (root / "park_drag_export.csv").write_text("venue_id,date,park_drag_delta\n3313,2026-09-28,0.0\n")
        (root / "park_drag_manifest.json").write_text(json.dumps({
            "generated_at": "2027-03-24T07:45:05",
            "max_source_game_date": "2026-09-27",
        }))
        (root / "producer_status.json").write_text(json.dumps({"ok": True}))
        fresh_alerts = park_drag_freshness.check(root, today=date(2027, 3, 24))
        return {
            "empty_store_refresh_summary": refresh_summary,
            "opening_night_freshness_alerts": [(a.level, a.message) for a in fresh_alerts],
        }


def probe_finding_7_research_cron_auth_dm() -> dict:
    from bts.cli import cli
    from bts.leaderboard.auth import AuthError

    dms_sent = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        picks_dir = root / "data" / "picks"
        picks_dir.mkdir(parents=True)
        runner = CliRunner()
        with patch("bts.leaderboard.auth.load_session_cookies", return_value={"ipid": "expired"}), \
             patch("bts.leaderboard.auth.extract_uid", return_value="12345"), \
             patch("bts.leaderboard.auth.fetch_login_session", side_effect=AuthError("401 expired cookie")), \
             patch("bts.dm.send_dm", side_effect=lambda r, m: dms_sent.append((r, m))):
            res = runner.invoke(cli, [
                "fetch-contest-streak",
                "--picks-dir", str(picks_dir),
                "--expected-username", "stonehengee",
                "--dm-recipient", "stonehengee.bsky.social",
            ])
        svc = (REPO_ROOT / "deploy" / "systemd" / "bts-leaderboard.service").read_text()
        has_wrong_handle = "stoneericm.bsky.social" in svc
        return {
            "cli_exit_code": res.exit_code,
            "dms_sent": dms_sent,
            "leaderboard_service_has_stoneericm": has_wrong_handle,
        }


def probe_finding_8_shadow_cross_season() -> dict:
    from bts.picks import DailyPick, Pick, save_pick, save_shadow_pick
    from bts.shadow_eval import build_shadow_cycle_status
    from bts.skip_policy_shadow import build_skip_policy_shadow_status, load_decision_records

    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir(parents=True)
        for i in range(1, 30):
            d = f"2026-08-{i:02d}"
            (picks_dir / f"{d}.policy_shadow.json").write_text(json.dumps({
                "date": d, "divergent": True, "shadow_pick_result": "hit",
                "rank1": {"batter_name": "Judge", "p_game_hit": 0.785}, "streak": 10,
            }))
        (picks_dir / "2027-03-24.policy_shadow.json").write_text(json.dumps({
            "date": "2027-03-24", "divergent": True, "shadow_pick_result": "hit",
            "rank1": {"batter_name": "Judge", "p_game_hit": 0.785}, "streak": 0,
        }))
        recs = load_decision_records(picks_dir)
        status = build_skip_policy_shadow_status(recs, as_of=date(2027, 4, 2))

        slot = Pick(
            batter_name="Judge", batter_id=100, team="NYY", lineup_position=2,
            pitcher_name="Webb", pitcher_id=500, p_game_hit=0.80, flags=[],
            projected_lineup=False, game_pk=777001, game_time="2027-03-24T23:05:00Z",
        )
        for d in ("2026-09-27", "2027-03-24"):
            sp = DailyPick(
                date=d, run_time=f"{d}T12:00:00-04:00",
                pick=slot, double_down=None, runner_up=None, result="hit",
                shadow_model_version="context_stack_shadow_v2",
            )
            save_pick(sp, picks_dir)
            save_shadow_pick(sp, picks_dir)
        cycle = build_shadow_cycle_status(picks_dir)
        return {
            "skip_policy_shadow_counts": status["counts"],
            "skip_policy_shadow_verdict": (
                status["shadow_band_hit_rate"]["verdict"],
                status["shadow_band_hit_rate"]["verdict_basis"]["checkpoint"],
            ),
            "shadow_eval_counts": cycle["counts"],
        }


def probe_checked_and_clean() -> dict:
    from bts.contest_state import ContestStreakState, classify_best_streak
    from bts.data import lineup_collect
    from bts.data.sync import sync_to_r2
    from bts.day_outcome import gather as day_outcome_gather
    from bts.health import (
        backup_freshness,
        blend_training,
        calibration,
        mdp_policy_alignment,
        predicted_vs_realized,
        realized_calibration,
        slate_auc,
        tail_policy as health_tail,
    )
    from bts.scheduler import fetch_schedule
    from bts.picks import DailyPick, Pick, load_saver_available, load_streak, reconcile_results, save_pick
    from bts.strategy import season_end_date

    out = {}

    st_payload = {
        "dates": [{
            "games": [
                {"gamePk": 111, "gameType": "S", "gameDate": "2027-03-10T17:05:00Z",
                 "status": {"detailedState": "Scheduled"},
                 "teams": {"away": {"team": {"id": 147, "name": "Yankees", "abbreviation": "NYY"}},
                           "home": {"team": {"id": 111, "name": "Red Sox", "abbreviation": "BOS"}}}},
            ]
        }]
    }
    class FakeUrlResp:
        def read(self): return json.dumps(st_payload).encode()

    with patch("bts.util.urlopen", return_value=FakeUrlResp()):
        sched = fetch_schedule("2027-03-10")
    out["fetch_schedule_spring_training_count"] = len(sched)

    with tempfile.TemporaryDirectory() as tmp:
        troot = Path(tmp)
        with patch("bts.util.urlopen", return_value=FakeUrlResp()):
            lu_state = lineup_collect.collect_for_date("2027-03-10", troot / "lu")
        out["lineup_collect_spring_training_games"] = len(lu_state.games)

        picks_dir = troot / "picks"
        picks_dir.mkdir()
        do_res = day_outcome_gather(
            "2027-03-10", picks_dir,
            {"scheduler": {"entry_intent": "research", "pick_delivery": "private"}},
            expected_username="stonehengee",
            fetch_schedule=lambda _d: [],
        )
        out["day_outcome_2027_03_10"] = (do_res.state, do_res.ok)

        slot = Pick(
            batter_name="Judge", batter_id=100, team="NYY", lineup_position=2,
            pitcher_name="Sale", pitcher_id=500, p_game_hit=0.80, flags=[],
            projected_lineup=False, game_pk=666001, game_time="2026-09-27T17:05:00Z",
        )
        save_pick(DailyPick(
            date="2026-09-27", run_time="2026-09-27T12:00:00-04:00",
            pick=slot, double_down=None, runner_up=None, result="hit",
        ), picks_dir)
        (picks_dir / "streak.json").write_text(json.dumps({
            "streak": 14, "saver_available": False, "last_updated": "2026-09-27",
        }))
        reconcile_results(picks_dir, lookback_days=7, clock=lambda: datetime(2027, 3, 4, 2, 0, tzinfo=ET))
        out["reconcile_2027_03_04_streak_saver"] = (
            load_streak(picks_dir), load_saver_available(picks_dir)
        )

        out["season_end_2027"] = season_end_date("2027-03-24")
        tp_alerts_mar = health_tail.check(
            base_path=REPO_ROOT / "data" / "models" / "mdp_policy.npz",
            tail_path=REPO_ROOT / "data" / "models" / "mdp_tail_policy.npz",
            today=date(2027, 3, 24),
        )
        tp_alerts_sep = health_tail.check(
            base_path=REPO_ROOT / "data" / "models" / "mdp_policy.npz",
            tail_path=REPO_ROOT / "data" / "models" / "mdp_tail_policy.npz",
            today=date(2027, 9, 10),
        )
        out["tail_policy_alerts_mar_sep"] = (len(tp_alerts_mar), len(tp_alerts_sep))
        st_2026 = ContestStreakState(
            streak=12, saver_available=True, source="auto",
            path=picks_dir / "contest_streak.json",
            source_date=date(2026, 9, 27), best_streak=18,
            recorded_at=datetime(2026, 9, 28, 6, 0, tzinfo=timezone.utc),
        )
        out["classify_best_streak_2026_in_2027"] = classify_best_streak(
            st_2026, now=datetime(2027, 3, 24, 15, 0, tzinfo=ET), now_year=2027
        )

        models_dir = troot / "models"
        models_dir.mkdir()
        (models_dir / "blend_2027-03-25.pkl").write_bytes(b"ok")
        proc_dir = troot / "processed"
        proc_dir.mkdir()
        out["lookback_health_alerts_on_2027_03_24"] = {
            "blend_training": len(blend_training.check(models_dir, today=date(2027, 3, 24))),
            "calibration": len(calibration.check(picks_dir, today=date(2027, 3, 24))),
            "predicted_vs_realized": len(predicted_vs_realized.check(picks_dir, today=date(2027, 3, 24))),
            "realized_calibration": len(realized_calibration.check(picks_dir, today=date(2027, 3, 24), data_dir=proc_dir)),
            "mdp_policy_alignment": len(mdp_policy_alignment.check(
                picks_dir, policy_path=REPO_ROOT / "data" / "models" / "mdp_policy.npz", today=date(2027, 3, 24)
            )),
            "slate_auc": len(slate_auc.check(picks_dir, data_dir=proc_dir, today=date(2027, 3, 24))),
        }

        p2026 = proc_dir / "pa_2026.parquet"
        p2026.write_bytes(b"parquet2026")
        class FakeR2:
            def __init__(self):
                self.objs = {"manifest.json": {"files": {
                    "parquets/pa_2026.parquet": {"sha256": "x", "size": 11, "key": "objects/x/pa_2026.parquet"}
                }}}
            def get_object_json(self, k): return self.objs.get(k)
            def upload_file(self, src, k): self.objs[k] = src.read_bytes()
            def put_object_json(self, k, d): self.objs[k] = d
            def copy_object(self, s, d): self.objs[d] = self.objs[s]
            def delete_object(self, k): self.objs.pop(k, None)

        r2_manifest = sync_to_r2(FakeR2(), proc_dir, REPO_ROOT / "data" / "models")
        out["sync_to_r2_preseason_files"] = sorted(r2_manifest["files"].keys())

        bdir = troot / "health_state"
        bdir.mkdir()
        (bdir / "backup_status.json").write_text(json.dumps({
            "ops": {"ok": True, "last_success_at": "2027-03-10T09:20:00+00:00"},
            "archive": {"ok": True, "last_success_at": "2027-03-10T09:50:00+00:00"},
        }))
        out["backup_freshness_alerts"] = len(backup_freshness.check(
            bdir, now=datetime(2027, 3, 10, 12, 0, tzinfo=timezone.utc)
        ))
    return out


if __name__ == "__main__":
    results = {
        "finding_1": probe_finding_1_saver_not_earned(),
        "finding_2": probe_finding_2_partial_rollover(),
        "finding_3": probe_finding_3_dashboard_saver_season(),
        "finding_4": probe_finding_4_live_forward_9_rows(),
        "finding_5": probe_finding_5_opening_night_eod_contest_state(),
        "finding_6": probe_finding_6_park_drag(),
        "finding_7": probe_finding_7_research_cron_auth_dm(),
        "finding_8": probe_finding_8_shadow_cross_season(),
        "checked_and_clean": probe_checked_and_clean(),
    }
    print(json.dumps(results, indent=2, default=str))
