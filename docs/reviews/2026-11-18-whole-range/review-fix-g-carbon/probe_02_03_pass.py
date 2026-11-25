"""Item 2 & Item 3 regression probe: all production entry points, offline callers, and double clamps."""
from __future__ import annotations

import ast
import io
import json
import sys
import tempfile
from contextlib import redirect_stderr
from datetime import date
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
from click.testing import CliRunner

from bts import cli, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy, web
from bts.simulate import exact, mdp, quality_bins
from scripts.audit.season_walk import walk

REVIEW_DIR = REPO / ".codex-review" / "wr-g"

GUARD_SINGLE = {"rule": "monotone_skip_v1", "bin": 4, "from_bin": 3, "table_action": "skip"}
GUARD_DOUBLE = {"rule": "monotone_skip_v1", "bin": 1, "from_bin": 0, "table_action": "skip"}


def make_candidates(day_iso: str, top_p: float, *, include_same_game: bool = True, include_diff_game: bool = True) -> pd.DataFrame:
    rows = [
        {
            "batter_name": "Primary Star", "batter_id": 101, "team": "NYY", "lineup": 1,
            "pitcher_name": "Ace One", "pitcher_id": 501, "p_game_hit": top_p, "p_hit_pa": 0.32,
            "flags": "", "game_pk": 70001, "game_time": f"{day_iso}T23:05:00Z", "pitcher_team": "BOS",
        }
    ]
    if include_same_game:
        rows.append({
            "batter_name": "SameGame Teammate", "batter_id": 102, "team": "NYY", "lineup": 2,
            "pitcher_name": "Ace One", "pitcher_id": 501, "p_game_hit": round(top_p - 0.002, 6), "p_hit_pa": 0.31,
            "flags": "", "game_pk": 70001, "game_time": f"{day_iso}T23:05:00Z", "pitcher_team": "BOS",
        })
    if include_diff_game:
        rows.append({
            "batter_name": "DiffGame Second", "batter_id": 201, "team": "LAD", "lineup": 1,
            "pitcher_name": "Ace Two", "pitcher_id": 601, "p_game_hit": round(top_p - 0.02, 6), "p_hit_pa": 0.29,
            "flags": "", "game_pk": 70002, "game_time": f"{day_iso}T23:35:00Z", "pitcher_team": "SF",
        })
        rows.append({
            "batter_name": "DiffGame Third", "batter_id": 301, "team": "HOU", "lineup": 1,
            "pitcher_name": "Ace Three", "pitcher_id": 701, "p_game_hit": round(top_p - 0.05, 6), "p_hit_pa": 0.27,
            "flags": "", "game_pk": 70003, "game_time": f"{day_iso}T23:45:00Z", "pitcher_team": "SEA",
        })
    return pd.DataFrame(rows)


def make_schedule(day_iso: str, game_pks=(70001, 70002, 70003)) -> list[dict]:
    times = {70001: "23:05:00Z", 70002: "23:35:00Z", 70003: "23:45:00Z"}
    out = []
    for idx, pk in enumerate(game_pks):
        out.append({
            "gamePk": pk,
            "gameType": "R",
            "officialDate": day_iso,
            "gameDate": f"{day_iso}T{times[pk]}",
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled", "statusCode": "S"},
            "teams": {
                "away": {"team": {"name": f"Away{idx}"}},
                "home": {"team": {"name": f"Home{idx}"}},
            },
        })
    return out


def scan_lookup_callers() -> dict:
    raw_callers = []
    guarded_callers = []
    for root_name in ("src", "scripts"):
        for py_path in sorted((REPO / root_name).rglob("*.py")):
            rel = str(py_path.relative_to(REPO))
            tree = ast.parse(py_path.read_text(), filename=rel)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
                    if name == "lookup_action":
                        raw_callers.append((rel, node.lineno))
                    elif name == "lookup_action_guarded":
                        guarded_callers.append((rel, node.lineno))
    raw_callers.sort()
    guarded_callers.sort()
    return {
        "lookup_action": [f"{r}:{ln}" for r, ln in raw_callers],
        "lookup_action_guarded": [f"{r}:{ln}" for r, ln in guarded_callers],
    }


def test_item_2_production_paths_and_callers() -> dict:
    callers = scan_lookup_callers()
    assert callers["lookup_action_guarded"] == ["src/bts/strategy.py:138"], callers
    expected_raw = [
        "scripts/audit/boundary_shadow_census.py:243",
        "scripts/audit_mdp_live_aggressiveness.py:125",
        "scripts/evaluate_mdp_dd_guardrail.py:348",
        "scripts/evaluate_mdp_dd_guardrail.py:492",
        "scripts/gate_b_production_metric_rebaseline.py:290",
        "scripts/gate_b_production_metric_rebaseline.py:299",
        "scripts/oos_validation.py:31",
        "scripts/validation/compare_policies.py:13",
        "scripts/validation/compare_policies.py:19",
        "scripts/validation/compare_policies.py:20",
    ]
    assert callers["lookup_action"] == expected_raw, callers

    table, bounds, sl = mdp.load_policy(REPO / "data" / "models" / "mdp_policy.npz")
    assert mdp.lookup_action(table, bounds, 50, 180, False, 0.96, sl) == "skip"
    assert mdp.lookup_action_guarded(table, bounds, 50, 180, False, 0.96, sl) == ("single", GUARD_SINGLE)
    qb = quality_bins.QualityBins(
        bins=[
            quality_bins.QualityBin(
                index=i, p_range=(0.7 + 0.05 * i, 0.75 + 0.05 * i),
                frequency=0.2, p_hit=0.75 + 0.02 * i, p_both=0.58 + 0.02 * i,
            )
            for i in range(5)
        ],
        boundaries=bounds,
    )
    p57_val = exact.exact_p57_policy_table(table, qb, season_length=sl)
    assert isinstance(p57_val, float) and p57_val > 0.0

    day_iso = "2027-03-24"
    day_obj = date(2027, 3, 24)
    now_et = walk.REAL_DATETIME(2027, 3, 24, 18, 25, tzinfo=walk.ET)
    frame = make_candidates(day_iso, 0.96)
    statuses = {pk: {"abstract": "P", "detailed": "Scheduled", "code": "S"} for pk in (70001, 70002, 70003)}

    results_summary = {}

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p02p_sel_") as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        with walk.offline(now_et), \
                patch.object(orchestrator, "run_cascade", lambda tiers, d: (frame.copy(), "local")), \
                patch.object(picks, "get_game_statuses_detailed", lambda d: statuses):
            assert walk.install_state(pdir, day_obj, walk.State("dated", 50, 50, "used", 0.96)) == []
            bar = strategy.effective_pick_bar(50, day_iso, False)
            assert bar is not None and 0.96 >= bar
            sel_direct = strategy.select_pick(
                frame.copy(), day_iso, pdir, streak=50, saver_available=False,
                game_statuses_detailed=statuses, require_detailed_statuses=True,
                best_streak=50, best_status="trusted",
            )
            _, sel_rap, _ = orchestrator.run_and_pick(walk._config(pdir), day_iso)
            assert sel_direct.action == sel_rap.action == "single"
            assert sel_direct.decision.guard == sel_rap.decision.guard == GUARD_SINGLE
            assert sel_rap.pick_result.daily.pick.batter_name == "Primary Star"
            assert sel_rap.pick_result.daily.double_down is None
            assert sel_rap.pick_result.daily.policy_decision["guard"] == GUARD_SINGLE
            results_summary["select_pick_and_run_and_pick"] = "single"

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p02p_sched_") as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        cfg = walk._config(pdir)
        cfg["bluesky"] = {"dm_recipient": "eric.test"}
        cfg["scheduler"]["live_forward_capture_on_lock"] = False
        cfg["scheduler"]["shadow_model"] = True
        cfg["health_checks"]["enabled"] = False
        sent_dms = []

        with walk.offline(now_et), \
                patch.object(orchestrator, "run_cascade", lambda tiers, d: (frame.copy(), "local")), \
                patch.object(scheduler, "predict_local_shadow", lambda *a, **k: frame.copy()), \
                patch.object(picks, "get_game_statuses_detailed", lambda d: statuses), \
                patch.object(scheduler, "fetch_schedule", lambda d: make_schedule(d) if d == day_iso else []), \
                patch.object(scheduler, "check_confirmed_lineups", lambda pks: {pk: {"away", "home"} for pk in pks}), \
                patch.object(scheduler, "_now_et", lambda: now_et), \
                patch.object(scheduler, "_watchdog_ping_sleep", lambda s, *a, **k: None), \
                patch.object(scheduler, "_idle_until_next_wakeup", lambda *a, **k: None), \
                patch.object(scheduler, "run_result_polling", lambda *a, **k: "final"), \
                patch.object(dm, "send_dm", lambda rec, txt: sent_dms.append((rec, txt)) or "dm-guarded-1"):
            assert walk.install_state(pdir, day_obj, walk.State("dated", 50, 50, "used", 0.96)) == []
            picks.save_streak(50, pdir, saver_available=False, season=2027)
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                scheduler.run_day(day_iso, cfg)
            journal = err_buf.getvalue()
            assert (
                "  POLICY GUARD — At streak 50 with 180 policy days left and saver input unavailable, "
                "the table skips bin 4. Bin 3 supplies single. Selected action is single."
            ) in journal, journal
            assert len(sent_dms) == 1 and "Primary Star" in sent_dms[0][1]
            saved_daily = picks.load_pick(day_iso, pdir)
            dec_rec = daily_decision.load_decision(day_iso, pdir)
            shadow_daily = picks.load_shadow_pick(day_iso, pdir)
            assert saved_daily.pick.batter_name == "Primary Star" and saved_daily.double_down is None
            assert saved_daily.policy_decision["guard"] == GUARD_SINGLE
            assert dec_rec["action"] == "single" and dec_rec["policy_guard"] == GUARD_SINGLE
            assert shadow_daily is not None and shadow_daily.pick.batter_name == "Primary Star", journal
            assert shadow_daily.policy_decision["guard"] == GUARD_SINGLE

            with patch.object(web, "PICKS_DIR", pdir), \
                    patch.object(web, "HEALTH_STATE_DIR", pdir.parent / "health_state"), \
                    patch.object(web, "VALIDATION_DIR", pdir.parent / "validation"), \
                    patch.object(web, "fetch_bluesky_posts", lambda limit=5: []), \
                    patch.object(web, "_build_live_game_data", lambda pd_doc: ([], None)):
                html = web.render_page()
            assert "TODAY'S PICK" in html and "TODAY'S PICKS" not in html
            assert "Primary Star" in html and "DM SENT" in html and "SKIP TODAY" not in html
            results_summary["scheduler_check_shadow_dashboard"] = "single"

    for fb_mode in ("in_loop", "post_loop"):
        with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix=f"tmp_p02p_{fb_mode}_") as tmp:
            pdir = Path(tmp) / "picks"
            pdir.mkdir()
            cfg = walk._config(pdir)
            cfg["bluesky"] = {"dm_recipient": "eric.test"}
            cfg["scheduler"]["live_forward_capture_on_lock"] = False
            cfg["health_checks"]["enabled"] = False
            proj_frame = frame.copy()
            proj_frame["flags"] = "PROJECTED"
            sent_dms = []

            fb_now = (
                walk.REAL_DATETIME(2027, 3, 24, 18, 40, tzinfo=walk.ET)
                if fb_mode == "post_loop"
                else walk.REAL_DATETIME(2027, 3, 24, 18, 21, tzinfo=walk.ET)
            )
            call_num = {"n": 0}

            def cascade_for_fb(tiers, d, _mode=fb_mode, _cnt=call_num, _pf=proj_frame):
                _cnt["n"] += 1
                if _mode == "post_loop" and _cnt["n"] == 1:
                    return None, None
                return _pf.copy(), "local"

            with walk.offline(fb_now), \
                    patch.object(orchestrator, "run_cascade", cascade_for_fb), \
                    patch.object(picks, "get_game_statuses_detailed", lambda d: {70001: statuses[70001]}), \
                    patch.object(scheduler, "fetch_schedule", lambda d: make_schedule(d, (70001,)) if d == day_iso else []), \
                    patch.object(scheduler, "check_confirmed_lineups", lambda pks: {70001: set()}), \
                    patch.object(scheduler, "_now_et", lambda _t=fb_now: _t), \
                    patch.object(scheduler, "_watchdog_ping_sleep", lambda s, *a, **k: None), \
                    patch.object(scheduler, "_idle_until_next_wakeup", lambda *a, **k: None), \
                    patch.object(scheduler, "run_result_polling", lambda *a, **k: "final"), \
                    patch.object(dm, "send_dm", lambda rec, txt, _s=sent_dms, _m=fb_mode: _s.append((rec, txt)) or f"dm-{_m}"):
                assert walk.install_state(pdir, day_obj, walk.State("dated", 50, 50, "used", 0.96)) == []
                if fb_mode == "post_loop":
                    from bts.contest_state import load_decision_streak_state
                    seed_sel = strategy.select_pick(
                        proj_frame.copy(), day_iso, pdir, streak=50, saver_available=False,
                        game_statuses_detailed={70001: statuses[70001]}, require_detailed_statuses=True,
                        best_streak=50, best_status="trusted",
                    )
                    seed_sel.pick_result.daily.chosen_on = picks.chosen_on_record(
                        load_decision_streak_state(pdir, require_contest_state=True), required=True
                    )
                    picks.save_pick(seed_sel.pick_result.daily, pdir)
                err_buf = io.StringIO()
                with redirect_stderr(err_buf):
                    scheduler.run_day(day_iso, cfg)
                journal = err_buf.getvalue()
                assert "FALLBACK REFRESH:" in journal, journal
                assert "POLICY GUARD —" in journal, journal
                assert len(sent_dms) == 1, journal
                dec_rec = daily_decision.load_decision(day_iso, pdir)
                assert dec_rec["action"] == "single" and dec_rec["policy_guard"] == GUARD_SINGLE
                results_summary[f"fallback_{fb_mode}"] = dec_rec["action"]

    runner = CliRunner()
    for cmd_name in ("run", "preview", "orchestrate"):
        with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix=f"tmp_p02p_{cmd_name}_") as tmp:
            pdir = Path(tmp) / "picks"
            mdir = Path(tmp) / "models"
            pdir.mkdir()
            mdir.mkdir()
            posted = []
            with walk.offline(now_et), \
                    patch("bts.model.predict.run_pipeline", lambda *a, **k: frame.copy()), \
                    patch.object(orchestrator, "run_cascade", lambda tiers, d: (frame.copy(), "local")), \
                    patch.object(picks, "get_game_statuses_detailed", lambda d: statuses), \
                    patch.object(posting, "should_post_now", lambda *a, **k: True), \
                    patch.object(posting, "post_to_bluesky", lambda txt, _p=posted: _p.append(txt) or "at://guarded/1"):
                assert walk.install_state(pdir, day_obj, walk.State("dated", 50, 50, "used", 0.96)) == []
                if cmd_name == "run":
                    res = runner.invoke(cli.cli, ["run", "--date", day_iso, "--picks-dir", str(pdir), "--models-dir", str(mdir)])
                    assert res.exit_code == 0, res.output
                    assert len(posted) == 1 and "Primary Star" in posted[0]
                elif cmd_name == "preview":
                    res = runner.invoke(cli.cli, ["preview", "--date", day_iso, "--picks-dir", str(pdir), "--models-dir", str(mdir)])
                    assert res.exit_code == 0, res.output
                    assert len(posted) == 0
                else:
                    toml_path = Path(tmp) / "orch.toml"
                    toml_path.write_text(
                        f'[orchestrator]\npicks_dir = "{pdir}"\nmodels_dir = "{mdir}"\n'
                        f'[bluesky]\ndm_recipient = "eric.test"\n'
                        f'[scheduler]\npick_delivery = "public"\nentry_intent = "enter"\ncontest_state_required = true\n'
                        f'[health_checks]\ncontest_state_expected = true\n'
                        f'[[tiers]]\nname = "local"\ntype = "local"\n'
                    )
                    res = runner.invoke(cli.cli, ["orchestrate", "--date", day_iso, "--config", str(toml_path)])
                    assert res.exit_code == 0, res.output
                    assert len(posted) == 1 and "Primary Star" in posted[0]

                saved = picks.load_pick(day_iso, pdir)
                assert saved is not None and saved.pick.batter_name == "Primary Star"
                assert saved.double_down is None
                assert saved.policy_decision["action"] == "single"
                assert saved.policy_decision["guard"] == GUARD_SINGLE
                results_summary[f"cli_{cmd_name}"] = saved.policy_decision["action"]

    return {"callers": callers, "paths": results_summary}


def test_item_3_clamps() -> dict:
    with np.load(REPO / "data" / "models" / "mdp_policy.npz") as data:
        b = data["boundaries"].tolist()
    mid_bin1 = (b[0] + b[1]) / 2.0
    mid_bin0 = b[0] / 2.0

    day_iso = "2027-08-29"
    now_et = walk.REAL_DATETIME(2027, 8, 29, 18, 25, tzinfo=walk.ET)
    statuses = {pk: {"abstract": "P", "detailed": "Scheduled", "code": "S"} for pk in (70001, 70002, 70003)}

    outcomes = {}

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p03p_allow_") as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        frame = make_candidates(day_iso, mid_bin1, include_same_game=True, include_diff_game=True)
        with walk.offline(now_et), patch.object(picks, "get_game_statuses_detailed", lambda d: statuses):
            sel = strategy.select_pick(
                frame, day_iso, pdir, streak=1, saver_available=False, allow_double=False,
                game_statuses_detailed=statuses, require_detailed_statuses=True,
                best_streak=10, best_status="trusted",
            )
            assert sel.decision.policy_action == "double" and sel.decision.action == "single"
            assert sel.decision.guard == GUARD_DOUBLE
            daily = sel.pick_result.daily
            assert daily.pick.batter_name == "Primary Star" and daily.double_down is None
            picks.save_pick(daily, pdir)

            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                scheduler._log_policy_decision(sel)
            jline = err_buf.getvalue().strip()
            assert jline == (
                "POLICY GUARD — At streak 1 with 29 policy days left and saver input unavailable, "
                "the table skips bin 1. Bin 0 supplies double. Selected action is single."
            ), jline

            st = scheduler.SchedulerState(
                date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=True, pick_locked_at=now_et.isoformat(),
                result_status=None, next_wakeup=None,
            )
            scheduler._commit_decision_for_pick(
                pdir, day_iso, daily, selection=sel, delivery_status="delivered", state=st,
            )
            rec = daily_decision.load_decision(day_iso, pdir)
            assert rec["action"] == "single" and rec["policy_guard"] == GUARD_DOUBLE
            outcomes["clamp_allow_double_off"] = {
                "policy_action": sel.decision.policy_action,
                "action": sel.decision.action,
                "legs": 1,
                "record_action": rec["action"],
                "journal": jline,
            }

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p03p_nodiff_") as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        frame = make_candidates(day_iso, mid_bin1, include_same_game=True, include_diff_game=False)
        with walk.offline(now_et), patch.object(picks, "get_game_statuses_detailed", lambda d: {70001: statuses[70001]}):
            sel = strategy.select_pick(
                frame, day_iso, pdir, streak=1, saver_available=False, allow_double=True,
                game_statuses_detailed={70001: statuses[70001]}, require_detailed_statuses=True,
                best_streak=10, best_status="trusted",
            )
            assert sel.decision.policy_action == "double" and sel.decision.action == "single"
            assert sel.decision.guard == GUARD_DOUBLE
            daily = sel.pick_result.daily
            assert daily.pick.batter_name == "Primary Star" and daily.double_down is None
            assert daily.runner_up is None
            picks.save_pick(daily, pdir)

            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                scheduler._log_policy_decision(sel)
            jline = err_buf.getvalue().strip()
            assert jline == (
                "POLICY GUARD — At streak 1 with 29 policy days left and saver input unavailable, "
                "the table skips bin 1. Bin 0 supplies double. Selected action is single."
            ), jline

            st = scheduler.SchedulerState(
                date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=True, pick_locked_at=now_et.isoformat(),
                result_status=None, next_wakeup=None,
            )
            scheduler._commit_decision_for_pick(
                pdir, day_iso, daily, selection=sel, delivery_status="delivered", state=st,
            )
            rec = daily_decision.load_decision(day_iso, pdir)
            assert rec["action"] == "single" and rec["double_down"] is None and rec["policy_guard"] == GUARD_DOUBLE
            outcomes["clamp_no_diff_game"] = {
                "policy_action": sel.decision.policy_action,
                "action": sel.decision.action,
                "legs": 1,
                "record_action": rec["action"],
                "journal": jline,
            }

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p03p_exec_") as tmp:
        pdir_g = Path(tmp) / "guarded"
        pdir_u = Path(tmp) / "unguarded"
        pdir_g.mkdir()
        pdir_u.mkdir()

        frame_g = make_candidates(day_iso, mid_bin1, include_same_game=True, include_diff_game=True)
        frame_u = make_candidates(day_iso, mid_bin0, include_same_game=True, include_diff_game=True)
        frame_g.loc[frame_g["batter_id"] == 201, "p_game_hit"] = 0.35
        frame_g.loc[frame_g["batter_id"] == 301, "p_game_hit"] = 0.32
        frame_u.loc[frame_u["batter_id"] == 201, "p_game_hit"] = 0.35
        frame_u.loc[frame_u["batter_id"] == 301, "p_game_hit"] = 0.32

        with walk.offline(now_et), patch.object(picks, "get_game_statuses_detailed", lambda d: statuses):
            sel_g = strategy.select_pick(
                frame_g, day_iso, pdir_g, streak=1, saver_available=False, allow_double=True,
                game_statuses_detailed=statuses, require_detailed_statuses=True,
                best_streak=10, best_status="trusted",
            )
            sel_u = strategy.select_pick(
                frame_u, day_iso, pdir_u, streak=1, saver_available=False, allow_double=True,
                game_statuses_detailed=statuses, require_detailed_statuses=True,
                best_streak=10, best_status="trusted",
            )
            assert sel_g.decision.action == sel_g.decision.policy_action == "double" and sel_g.decision.guard == GUARD_DOUBLE
            assert sel_u.decision.action == sel_u.decision.policy_action == "double" and sel_u.decision.guard is None

            dg = sel_g.pick_result.daily
            du = sel_u.pick_result.daily
            assert dg.double_down is not None and du.double_down is not None
            assert (dg.pick.batter_id, dg.pick.game_pk) == (du.pick.batter_id, du.pick.game_pk) == (101, 70001)
            assert (dg.double_down.batter_id, dg.double_down.game_pk) == (du.double_down.batter_id, du.double_down.game_pk) == (201, 70002)

            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                scheduler._log_policy_decision(sel_g)
            jline = err_buf.getvalue().strip()
            assert jline == (
                "POLICY GUARD — At streak 1 with 29 policy days left and saver input unavailable, "
                "the table skips bin 1. Bin 0 supplies double. Selected action is double."
            ), jline

            picks.save_pick(dg, pdir_g)
            st = scheduler.SchedulerState(
                date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=True, pick_locked_at=now_et.isoformat(),
                result_status=None, next_wakeup=None,
            )
            scheduler._commit_decision_for_pick(
                pdir_g, day_iso, dg, selection=sel_g, delivery_status="delivered", state=st,
            )
            rec_g = daily_decision.load_decision(day_iso, pdir_g)
            assert rec_g["action"] == "double" and rec_g["double_down"]["batter_id"] == 201
            assert rec_g["policy_guard"] == GUARD_DOUBLE

            with walk.offline(now_et), \
                    patch.object(web, "PICKS_DIR", pdir_g), \
                    patch.object(web, "HEALTH_STATE_DIR", pdir_g.parent / "health_state"), \
                    patch.object(web, "VALIDATION_DIR", pdir_g.parent / "validation"), \
                    patch.object(web, "fetch_bluesky_posts", lambda limit=5: []), \
                    patch.object(web, "_build_live_game_data", lambda pd_doc: ([], None)):
                html_dbl = web.render_page()
            assert "TODAY'S PICKS" in html_dbl and "DOUBLE DOWN" in html_dbl and "DiffGame Second" in html_dbl

            outcomes["executable_double"] = {
                "primary": dg.pick.batter_name,
                "double_down": dg.double_down.batter_name,
                "matches_ordinary_double": True,
                "journal": jline,
            }

    return outcomes


def main() -> None:
    res2 = test_item_2_production_paths_and_callers()
    res3 = test_item_3_clamps()
    out = {"status": "PASS", "item_2": res2, "item_3": res3}
    (REVIEW_DIR / "probe_02_03_pass.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
