"""Independent probe for Items 1, 2, and 3: 1,511-cell & 2027 grids, all production entry points & AST scan, and clamps."""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import ExitStack, nullcontext, redirect_stderr
import csv
from dataclasses import asdict
from datetime import date, timedelta
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import bts
from bts import cli as cli_mod, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy, web
from bts.simulate import mdp
from click.testing import CliRunner
from scripts.audit.season_walk import walk

BASE_SRC = HERE / "base_f83e0d0b" / "src"
CONTRACT_CSV = ROOT / "docs/reviews/2026-11-18-whole-range/contract-g/affected-states.csv"
ENUMERATE_PY = ROOT / "docs/reviews/2026-11-18-whole-range/contract-g/enumerate_states.py"
GUARD_BIN4 = {"rule": "monotone_skip_v1", "bin": 4, "from_bin": 3, "table_action": "skip"}
GUARD_BIN1 = {"rule": "monotone_skip_v1", "bin": 1, "from_bin": 0, "table_action": "skip"}


def _build_prob_list(bounds: list[float]) -> tuple[list[float], list[tuple[float, float, float]]]:
    edges = [0.0, *bounds, 1.0]
    per_bin = []
    for lo, hi in zip(edges, edges[1:]):
        mid = (lo + hi) / 2.0
        below_hi = math.nextafter(hi, -math.inf) if hi < 1.0 else 0.999
        per_bin.append((lo, mid, below_hi))
    all_ps = [p for grp in per_bin for p in grp] + [
        math.nextafter(1.0, 0.0),
        1.0,
        -0.25,
        1.25,
        float("-inf"),
        float("inf"),
        float("nan"),
    ]
    return all_ps, per_bin


def _classify_bin(bounds: list[float], p: float) -> int:
    q = 0
    for i, b in enumerate(bounds):
        if p >= b:
            q = i + 1
    return min(q, len(bounds))


def _base_grid_worker(out_pkl: str) -> None:
    bts_path = Path(bts.__file__).resolve()
    assert bts_path.is_relative_to(BASE_SRC.resolve()), bts_path

    table, bounds, sl = mdp.load_policy(ROOT / "data/models/mdp_policy.npz")
    all_ps, per_bin = _build_prob_list(bounds)
    resolver_ps = [p for grp in per_bin for p in grp] + [1.0, -0.2, 1.2, float("nan")]

    # 1. Raw lookup over all 20,996 (s, d, v) * 22 probabilities = 461,912 calls + sentinels
    raw_actions = np.empty((58, 181, 2, len(all_ps)), dtype=np.int8)
    code_map = {"skip": 0, "single": 1, "double": 2}
    for s in range(58):
        for d in range(181):
            for v in (0, 1):
                sv = bool(v)
                for pi, p in enumerate(all_ps):
                    raw_actions[s, d, v, pi] = code_map[mdp.lookup_action(table, bounds, s, d, sv, p, sl)]

    sentinels = []
    for s in (57, 58, 100):
        for d in (180, 1, 0, -1):
            for sv in (False, True):
                for p in (0.0, 0.85, 0.99):
                    sentinels.append(mdp.lookup_action(table, bounds, s, d, sv, p, sl))
    for s in (0, 25, 56):
        for d in (0, -1, -30):
            for sv in (False, True):
                for p in (0.0, 0.85, 0.99):
                    sentinels.append(mdp.lookup_action(table, bounds, s, d, sv, p, sl))

    # 2. Resolver & effective_pick_bar over 2027-03-24..2027-09-27 (188 dates) * 58 streaks * 2 savers
    strategy._mdp_cache = None
    art = strategy._load_mdp()
    bars = []
    decisions = []
    start = date(2027, 3, 24)
    for off in range(188):
        day = (start + timedelta(days=off)).isoformat()
        for s in range(58):
            for sv in (False, True):
                bars.append(strategy.effective_pick_bar(s, day, sv, art, s, "trusted"))
                row_decs = []
                for p in resolver_ps:
                    ctx = strategy.DecisionContext(p, 0.95, True, day, True, art, s, "trusted")
                    dec = strategy.resolve_policy_decision(ctx, s, sv)
                    row_decs.append((
                        dec.action, dec.policy_action, dec.source, dec.objective,
                        dec.streak, dec.days_effective, dec.best_supplied, dec.best_status,
                        dec.effective_best, dec.tail_sha256, dec.degraded_reason,
                        dec.reason, dec.pick_bar,
                    ))
                decisions.append(row_decs)

    with open(out_pkl, "wb") as f:
        pickle.dump(
            {
                "bts_file": str(bts_path),
                "raw_actions": raw_actions,
                "sentinels": sentinels,
                "bars": bars,
                "decisions": decisions,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _verify_item1() -> dict:
    bts_head = Path(bts.__file__).resolve()
    assert bts_head.is_relative_to((ROOT / "src").resolve()), bts_head

    # Regenerate CSV and compare bytes
    regen_csv = HERE / "regen-affected-states.csv"
    subprocess.run(
        [sys.executable, str(ENUMERATE_PY), str(ROOT), str(regen_csv)],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=20,
    )
    expected_bytes = CONTRACT_CSV.read_bytes()
    actual_bytes = regen_csv.read_bytes()
    assert actual_bytes == expected_bytes, "Regenerated CSV differs from committed affected-states.csv!"
    csv_sha = hashlib.sha256(actual_bytes).hexdigest()
    assert csv_sha.startswith("77ec5404"), csv_sha

    affected = {}
    by_combo = Counter()
    with CONTRACT_CSV.open() as f:
        for r in csv.DictReader(f):
            key = (int(r["streak"]), int(r["days_left"]), int(r["saver"]), int(r["bin"]))
            g = {
                "rule": "monotone_skip_v1",
                "bin": int(r["bin"]),
                "from_bin": int(r["from_bin"]),
                "table_action": "skip",
            }
            affected[key] = (r["guarded_action"], g)
            by_combo[(int(r["bin"]), int(r["from_bin"]), r["guarded_action"])] += 1
    assert len(affected) == 1511
    assert by_combo == {(4, 3, "single"): 1156, (1, 0, "single"): 171, (1, 0, "double"): 184}

    with TemporaryDirectory() as d:
        pkl_path = Path(d) / "base_grid.pkl"
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "base_grid", str(pkl_path)],
            cwd=ROOT, check=True, capture_output=True, text=True, timeout=90,
            env=os.environ | {"PYTHONPATH": f"{BASE_SRC}:{ROOT}"},
        )
        with open(pkl_path, "rb") as f:
            base_out = pickle.load(f)

    table, bounds, sl = mdp.load_policy(ROOT / "data/models/mdp_policy.npz")
    all_ps, per_bin = _build_prob_list(bounds)
    resolver_ps = [p for grp in per_bin for p in grp] + [1.0, -0.2, 1.2, float("nan")]
    actions_tuple = ("skip", "single", "double")

    raw_calls = 0
    guarded_override_calls = 0
    seen_cells = set()

    for s in range(58):
        for d_left in range(181):
            for v in (0, 1):
                sv = bool(v)
                for pi, p in enumerate(all_ps):
                    base_act = actions_tuple[int(base_out["raw_actions"][s, d_left, v, pi])]
                    head_raw = mdp.lookup_action(table, bounds, s, d_left, sv, p, sl)
                    assert head_raw == base_act, (s, d_left, v, p, head_raw, base_act)

                    g_act, g_dict = mdp.lookup_action_guarded(table, bounds, s, d_left, sv, p, sl)
                    assert type(g_act) is str
                    if s >= 57 or d_left <= 0:
                        hit = None
                    else:
                        q = _classify_bin(bounds, p)
                        hit = affected.get((min(s, 56), min(d_left, sl), v, q))
                    if hit is not None:
                        exp_act, exp_g = hit
                        assert g_act == exp_act and g_dict == exp_g, (s, d_left, v, p, g_act, g_dict, hit)
                        assert g_dict.keys() == {"rule", "bin", "from_bin", "table_action"}
                        assert type(g_dict["bin"]) is int and type(g_dict["from_bin"]) is int
                        assert type(g_dict["rule"]) is str and type(g_dict["table_action"]) is str
                        guarded_override_calls += 1
                        seen_cells.add((s, d_left, v, _classify_bin(bounds, p)))
                    else:
                        assert g_act == base_act and g_dict is None, (s, d_left, v, p, g_act, g_dict, base_act)
                    raw_calls += 1

    assert seen_cells == set(affected.keys())

    # Sentinels
    idx_sent = 0
    for s in (57, 58, 100):
        for d_left in (180, 1, 0, -1):
            for sv in (False, True):
                for p in (0.0, 0.85, 0.99):
                    assert mdp.lookup_action(table, bounds, s, d_left, sv, p, sl) == base_out["sentinels"][idx_sent] == "skip"
                    assert mdp.lookup_action_guarded(table, bounds, s, d_left, sv, p, sl) == ("skip", None)
                    idx_sent += 1
    for s in (0, 25, 56):
        for d_left in (0, -1, -30):
            for sv in (False, True):
                for p in (0.0, 0.85, 0.99):
                    assert mdp.lookup_action(table, bounds, s, d_left, sv, p, sl) == base_out["sentinels"][idx_sent] == "skip"
                    assert mdp.lookup_action_guarded(table, bounds, s, d_left, sv, p, sl) == ("skip", None)
                    idx_sent += 1

    # Resolver & bar grid comparison against archived base package
    strategy._mdp_cache = None
    art = strategy._load_mdp()
    start = date(2027, 3, 24)
    state_idx = 0
    resolver_calls = 0
    resolver_overrides = 0
    resolver_seen_csv = set()

    for off in range(188):
        day = (start + timedelta(days=off)).isoformat()
        for s in range(58):
            for sv in (False, True):
                head_bar = strategy.effective_pick_bar(s, day, sv, art, s, "trusted")
                base_bar = base_out["bars"][state_idx]
                assert head_bar == base_bar, (day, s, sv, head_bar, base_bar)

                base_row_decs = base_out["decisions"][state_idx]
                for pi, p in enumerate(resolver_ps):
                    ctx = strategy.DecisionContext(p, 0.95, True, day, True, art, s, "trusted")
                    dec = strategy.resolve_policy_decision(ctx, s, sv)
                    b_tup = base_row_decs[pi]
                    head_tup = (
                        dec.action, dec.policy_action, dec.source, dec.objective,
                        dec.streak, dec.days_effective, dec.best_supplied, dec.best_status,
                        dec.effective_best, dec.tail_sha256, dec.degraded_reason,
                        dec.reason, dec.pick_bar,
                    )
                    q = _classify_bin(bounds, p)
                    csv_key = (s, dec.days_effective, int(sv), q)
                    hit = affected.get(csv_key) if dec.objective == "reach57" and s < 57 and dec.days_effective > 0 else None
                    if hit is not None:
                        exp_act, exp_g = hit
                        assert b_tup[0] == "skip" and b_tup[1] == "skip" and b_tup[12] is not None
                        expected_tup = (exp_act, exp_act, *b_tup[2:12], None)
                        assert head_tup == expected_tup, (day, s, sv, p, head_tup, expected_tup)
                        assert dec.guard == exp_g
                        assert type(dec.guard["bin"]) is int and type(dec.guard["from_bin"]) is int
                        resolver_overrides += 1
                        resolver_seen_csv.add(csv_key)
                    else:
                        assert head_tup == b_tup, (day, s, sv, p, head_tup, b_tup)
                        assert dec.guard is None

                    if dec.objective == "reach57":
                        if head_bar is None:
                            assert dec.action == "skip"
                        elif not math.isnan(p) and p >= head_bar:
                            assert dec.action != "skip"
                    resolver_calls += 1
                state_idx += 1

    assert resolver_seen_csv == set(affected.keys())

    return {
        "csv_sha256": csv_sha,
        "base_bts_file": base_out["bts_file"],
        "head_bts_file": str(bts_head),
        "raw_lookup_calls_per_side": raw_calls,
        "sentinel_calls_per_side": idx_sent,
        "guarded_override_calls": guarded_override_calls,
        "distinct_csv_cells_seen_lookup": len(seen_cells),
        "bar_calls_per_side": state_idx,
        "resolver_calls_per_side": resolver_calls,
        "resolver_override_calls": resolver_overrides,
        "distinct_csv_cells_seen_resolver": len(resolver_seen_csv),
    }


def _verify_item2_and_item3() -> dict:
    # AST scan of src/ and scripts/ for lookup_action and lookup_action_guarded
    call_sites = {"lookup_action": [], "lookup_action_guarded": []}
    for folder in (ROOT / "src", ROOT / "scripts"):
        for py in sorted(folder.rglob("*.py")):
            tree = ast.parse(py.read_text(), filename=str(py))
            rel = str(py.relative_to(ROOT))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
                    if name in call_sites:
                        call_sites[name].append(f"{rel}:{node.lineno}")

    assert call_sites["lookup_action_guarded"] == ["src/bts/strategy.py:138"], call_sites["lookup_action_guarded"]
    assert all(s.startswith("scripts/") for s in call_sites["lookup_action"]), call_sites["lookup_action"]

    # Item 2: Trace guarded state (2027-03-24, streak=50, saver=False, p=0.96) across all production entry points
    day = "2027-03-24"
    now = walk.REAL_DATETIME(2027, 3, 24, 18, 32, tzinfo=walk.ET)
    surfaces = {}

    with TemporaryDirectory() as d, ExitStack() as stack:
        root = Path(d)
        frame = walk._predictions(date.fromisoformat(day), 0.96)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
        stack.enter_context(walk.offline(now))
        assert not walk.install_state(root, date.fromisoformat(day), walk.State("dated", 50, 50, "used", 0.96))
        walk._set_clock(now)

        # 1. effective_pick_bar
        bar = strategy.effective_pick_bar(50, day, False)
        assert bar is not None and bar <= 0.83
        surfaces["effective_pick_bar"] = bar

        # 2. select_pick
        sel_direct = strategy.select_pick(
            frame, day, root / "s_direct", streak=50, saver_available=False,
            game_statuses_detailed=statuses, require_detailed_statuses=True,
        )
        assert sel_direct.action == "single" and sel_direct.decision.guard == GUARD_BIN4
        surfaces["select_pick"] = sel_direct.pick_result.daily.pick.batter_id

        # 3. run_and_pick & 4. run_single_check
        cfg = walk._config(root)
        cfg["orchestrator"]["heartbeat_path"] = str(root / "hb")
        cfg["scheduler"].update(pick_delivery="dm", live_forward_capture_on_lock=False)
        cfg["bluesky"] = {"dm_recipient": "op"}
        cfg["health_checks"]["enabled"] = False

        stack.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *_: statuses))
        stack.enter_context(patch.object(orchestrator, "run_cascade", lambda *_: (frame.copy(), "probe")))
        stack.enter_context(patch.object(scheduler, "_now_et", lambda: now))
        stack.enter_context(patch.object(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext()))
        stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda *_: statuses))
        stack.enter_context(patch.object(scheduler, "count_new_confirmations", lambda *_: 1))
        for nm in ("_trigger_live_forward_capture_on_lock", "_alert_delivery_outcome_unknown",
                   "_idle_until_next_wakeup", "_watchdog_ping_sleep", "_morning_state_check"):
            stack.enter_context(patch.object(scheduler, nm, Mock()))
        stack.enter_context(patch.object(scheduler, "run_result_polling", Mock(return_value="hit")))

        _, sel_rap, _ = orchestrator.run_and_pick(cfg, day)
        assert sel_rap.action == "single" and sel_rap.decision.guard == GUARD_BIN4
        surfaces["run_and_pick"] = sel_rap.pick_result.daily.pick.batter_id

        chk = scheduler.run_single_check(day, list(frame.game_pk), set(), cfg, 0.03)
        assert chk["pick_name"] == "Batter 0" and chk["selection"].decision.guard == GUARD_BIN4
        surfaces["run_single_check"] = chk["pick_result"].daily.pick.batter_id

        # 5 & 6. Both run_day fallback sites (in-loop and post-loop)
        for fb_site in ("in_loop", "post_loop"):
            fb_dir = root / fb_site
            fb_dir.mkdir()
            assert not walk.install_state(fb_dir, date.fromisoformat(day), walk.State("dated", 50, 50, "used", 0.96))
            fb_now = walk.REAL_DATETIME(2027, 3, 24, 18, 25 if fb_site == "in_loop" else 18, tzinfo=walk.ET)
            if fb_site == "post_loop":
                fb_now = walk.REAL_DATETIME(2027, 3, 24, 18, 35, tzinfo=walk.ET)
            walk._set_clock(fb_now)
            proj_frame = frame.copy()
            proj_frame["flags"] = "PROJECTED"
            fb_cfg = walk._config(fb_dir)
            fb_cfg["orchestrator"]["heartbeat_path"] = str(fb_dir / "hb")
            fb_cfg["scheduler"].update(pick_delivery="dm", live_forward_capture_on_lock=False)
            fb_cfg["bluesky"] = {"dm_recipient": "op"}
            fb_cfg["health_checks"]["enabled"] = False
            games = [
                {"gamePk": int(pk), "gameDate": gt, "doubleHeader": "N", "status": {"detailedState": "Scheduled"},
                 "teams": {s: {"team": {"name": f"{s}-{pk}"}} for s in ("home", "away")}}
                for pk, gt in zip(proj_frame.game_pk, proj_frame.game_time)
            ]
            dm_spy = Mock(return_value=f"dm-{fb_site}")
            with patch.object(orchestrator, "run_cascade", lambda *_: (proj_frame.copy(), "probe")), \
                 patch.object(scheduler, "_now_et", lambda: fb_now), \
                 patch.object(scheduler, "fetch_schedule", lambda *_: games), \
                 patch.object(dm, "send_dm", dm_spy):
                if fb_site == "post_loop":
                    # Make last check return no predictions after saving projected pick so post-loop fallback fires
                    sel_init = orchestrator.run_and_pick(fb_cfg, day)[1]
                    picks.save_pick(sel_init.pick_result.daily, fb_dir)
                    calls = [0]
                    def cascade_seq(*_):
                        calls[0] += 1
                        return (None, None) if calls[0] == 1 else (proj_frame.copy(), "probe")
                    with patch.object(orchestrator, "run_cascade", cascade_seq):
                        scheduler.run_day(day, fb_cfg)
                else:
                    scheduler.run_day(day, fb_cfg)
            dm_spy.assert_called_once()
            rec_fb = daily_decision.load_decision(day, fb_dir)
            assert rec_fb["action"] == "single" and rec_fb["primary"]["batter_id"] == 100 and rec_fb["policy_guard"] == GUARD_BIN4
            surfaces[f"fallback_{fb_site}"] = rec_fb["primary"]["batter_id"]

        walk._set_clock(now)

        # 7. bts run, 8. bts orchestrate, 9. bts preview
        for cli_name in ("run", "preview", "orchestrate"):
            c_dir = root / f"cli_{cli_name}"
            c_dir.mkdir()
            assert not walk.install_state(c_dir, date.fromisoformat(day), walk.State("dated", 50, 50, "used", 0.96))
            walk._set_clock(now)
            with patch("bts.model.predict.run_pipeline", return_value=frame.copy()), \
                 patch.object(posting, "post_to_bluesky", return_value="at://post"):
                if cli_name == "orchestrate":
                    o_cfg = walk._config(c_dir)
                    o_cfg["bluesky"] = {"dm_recipient": "op"}
                    assert orchestrator.orchestrate(c_dir / "unused.toml", day, config=o_cfg) is True
                else:
                    res = CliRunner().invoke(
                        cli_mod.cli,
                        [cli_name, "--date", day, "--picks-dir", str(c_dir), "--models-dir", str(c_dir / "models")],
                    )
                    assert res.exit_code == 0, res.output
            saved_c = picks.load_pick(day, c_dir)
            assert saved_c.pick.batter_id == 100 and saved_c.policy_decision["guard"] == GUARD_BIN4
            surfaces[f"bts_{cli_name}"] = saved_c.pick.batter_id

        # 10. Shadow model pick (_run_shadow_prediction)
        sh_dir = root / "shadow"
        sh_dir.mkdir()
        picks.save_streak(50, sh_dir, saver_available=False, season=2027)
        sh_cfg = {"orchestrator": {"picks_dir": str(sh_dir), "models_dir": str(sh_dir / "m"), "data_dir": str(sh_dir / "d")}}
        with patch.object(scheduler, "predict_local_shadow", return_value=frame.copy()):
            scheduler._run_shadow_prediction(sh_cfg, day, "Batter 0")
        sh_pick = picks.load_shadow_pick(day, sh_dir)
        assert sh_pick is not None and sh_pick.pick.batter_id == 100 and sh_pick.policy_decision["guard"] == GUARD_BIN4
        surfaces["shadow_model"] = sh_pick.pick.batter_id

        # 11. Dashboard web.render_page()
        web_dir = root / "fallback_in_loop" if (root / "in_loop").exists() else root
        with patch.object(web, "PICKS_DIR", root / "in_loop"), \
             patch.object(web, "HEALTH_STATE_DIR", root / "hs"), \
             patch.object(web, "HEARTBEAT_PATH", root / "hb"), \
             patch.object(web, "fetch_bluesky_posts", return_value=[]):
            html = web.render_page()
        assert "Batter 0" in html and "SKIP TODAY" not in html
        surfaces["dashboard_rendered_batter_0"] = True

    # Item 3: Clamps and executable guarded double vs ordinary bin-0 double
    art = strategy._load_mdp()
    b0, b1 = art["boundaries"][0], art["boundaries"][1]
    p_bin1 = (b0 + b1) / 2.0
    p_bin0 = b0 - 0.01
    d_day = "2027-08-29"
    d_now = walk.REAL_DATETIME(2027, 8, 29, 18, 32, tzinfo=walk.ET)

    clamp_results = {}
    with TemporaryDirectory() as d, ExitStack() as stack:
        root = Path(d)
        stack.enter_context(walk.offline(d_now))
        start_utc = "2027-08-29T23:05:00Z"
        # Include a higher-p SAME-GAME batter (102 in 70000) to prove double leg skips same-game correlation!
        slate = pd.DataFrame([
            {"batter_name": "Primary 100", "batter_id": 100, "team": "NYY", "lineup": 1,
             "pitcher_name": "P0", "pitcher_id": 900, "p_game_hit": p_bin1, "flags": "",
             "game_pk": 70000, "game_time": start_utc, "pitcher_team": "BOS"},
            {"batter_name": "SameGame 102", "batter_id": 102, "team": "NYY", "lineup": 2,
             "pitcher_name": "P0", "pitcher_id": 900, "p_game_hit": p_bin1 - 0.001, "flags": "",
             "game_pk": 70000, "game_time": start_utc, "pitcher_team": "BOS"},
            {"batter_name": "DiffGame 201", "batter_id": 201, "team": "LAD", "lineup": 1,
             "pitcher_name": "P1", "pitcher_id": 901, "p_game_hit": p_bin0, "flags": "",
             "game_pk": 70001, "game_time": start_utc, "pitcher_team": "SF"},
        ])
        statuses = {70000: {"abstract": "P", "detailed": "Scheduled"}, 70001: {"abstract": "P", "detailed": "Scheduled"}}

        # 3a: allow_double=False
        dir_a = root / "clamp_allow"
        dir_a.mkdir()
        sel_a = strategy.select_pick(
            slate, d_day, dir_a, streak=1, saver_available=False, allow_double=False,
            game_statuses_detailed=statuses, require_detailed_statuses=True,
        )
        assert (sel_a.decision.policy_action, sel_a.decision.action, sel_a.decision.guard) == ("double", "single", GUARD_BIN1)
        assert sel_a.pick_result.daily.double_down is None
        st_a = scheduler.SchedulerState(d_day, d_now.isoformat(), [], [], [], False, None, None, None)
        scheduler._commit_decision_for_pick(dir_a, d_day, sel_a.pick_result.daily, selection=sel_a, delivery_status="private_locked", state=st_a)
        rec_a = daily_decision.load_decision(d_day, dir_a)
        assert (rec_a["action"], rec_a["policy_guard"]) == ("single", GUARD_BIN1)
        buf_a = io.StringIO()
        with redirect_stderr(buf_a):
            scheduler._log_policy_decision(sel_a)
        assert "Bin 0 supplies double. Selected action is single." in buf_a.getvalue()
        clamp_results["allow_double_false"] = "ok"

        # 3b: same-game only (no diff-game candidate)
        dir_b = root / "clamp_same_game"
        dir_b.mkdir()
        same_slate = slate.iloc[:2].copy()
        sel_b = strategy.select_pick(
            same_slate, d_day, dir_b, streak=1, saver_available=False, allow_double=True,
            game_statuses_detailed={70000: {"abstract": "P", "detailed": "Scheduled"}}, require_detailed_statuses=True,
        )
        assert (sel_b.decision.policy_action, sel_b.decision.action, sel_b.decision.guard) == ("double", "single", GUARD_BIN1)
        assert sel_b.pick_result.daily.double_down is None
        st_b = scheduler.SchedulerState(d_day, d_now.isoformat(), [], [], [], False, None, None, None)
        scheduler._commit_decision_for_pick(dir_b, d_day, sel_b.pick_result.daily, selection=sel_b, delivery_status="private_locked", state=st_b)
        rec_b = daily_decision.load_decision(d_day, dir_b)
        assert (rec_b["action"], rec_b["policy_guard"]) == ("single", GUARD_BIN1)
        buf_b = io.StringIO()
        with redirect_stderr(buf_b):
            scheduler._log_policy_decision(sel_b)
        assert "Bin 0 supplies double. Selected action is single." in buf_b.getvalue()
        clamp_results["no_diff_game"] = "ok"

        # 3c: Executable guarded double vs ordinary bin-0 raw double
        dir_c = root / "exec_double"
        dir_c.mkdir()
        sel_c = strategy.select_pick(
            slate, d_day, dir_c, streak=1, saver_available=False, allow_double=True,
            game_statuses_detailed=statuses, require_detailed_statuses=True,
        )
        raw_slate = slate.copy()
        raw_slate.loc[0, "p_game_hit"] = p_bin0
        raw_slate.loc[1, "p_game_hit"] = p_bin0 - 0.001
        raw_slate.loc[2, "p_game_hit"] = p_bin0 - 0.005
        sel_ctrl = strategy.select_pick(
            raw_slate, d_day, root / "ctrl_double", streak=1, saver_available=False, allow_double=True,
            game_statuses_detailed=statuses, require_detailed_statuses=True,
        )
        assert sel_ctrl.decision.guard is None and sel_ctrl.action == "double"
        assert sel_c.decision.guard == GUARD_BIN1 and sel_c.action == "double"
        assert sel_c.pick_result.daily.pick.batter_id == sel_ctrl.pick_result.daily.pick.batter_id == 100
        assert sel_c.pick_result.daily.double_down.batter_id == sel_ctrl.pick_result.daily.double_down.batter_id == 201
        picks.save_pick(sel_c.pick_result.daily, dir_c)
        st_c = scheduler.SchedulerState(d_day, d_now.isoformat(), [], [], [], False, None, None, None)
        scheduler._commit_decision_for_pick(dir_c, d_day, sel_c.pick_result.daily, selection=sel_c, delivery_status="private_locked", state=st_c)
        rec_c = daily_decision.load_decision(d_day, dir_c)
        assert rec_c["action"] == "double" and rec_c["primary"]["batter_id"] == 100 and rec_c["double_down"]["batter_id"] == 201
        json.dumps(asdict(sel_c.pick_result.daily), allow_nan=False)
        json.dumps(rec_c, allow_nan=False)
        clamp_results["executable_double_matches_raw_control"] = "ok"

    return {"ast_call_sites": call_sites, "surfaces": surfaces, "clamps": clamp_results}


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "base_grid":
        _base_grid_worker(sys.argv[2])
        return

    r1 = _verify_item1()
    r23 = _verify_item2_and_item3()
    out = {"item1": r1, "item2_3": r23}
    (HERE / "probe_01_02_03_results.json").write_text(json.dumps(out, indent=2))
    print("PROBE_01_02_03_PASS", json.dumps({"item1": r1, "clamps": r23["clamps"]}))


if __name__ == "__main__":
    main()
