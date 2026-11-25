"""Items 5, 6, and 7 verified regression probe: adversarial regime boundaries, standing rows, and journal rendering."""
from __future__ import annotations

import csv
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from bts import daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
from bts.simulate import mdp, tail_policy
from scripts.audit.season_walk import walk

REVIEW_DIR = REPO / ".codex-review" / "wr-g"
BASE_SRC = REVIEW_DIR / "base_f83e0d0b" / "src"
HEAD_SRC = REPO / "src"
CSV_PATH = REPO / "docs" / "reviews" / "2026-11-18-whole-range" / "contract-g" / "affected-states.csv"

GUARD_SINGLE = {"rule": "monotone_skip_v1", "bin": 4, "from_bin": 3, "table_action": "skip"}


EDGE_WORKER = """
import pickle
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
import numpy as np
from bts import strategy
from bts.simulate import mdp

out_path = Path(sys.argv[1])
has_guarded = hasattr(mdp, "lookup_action_guarded")
art = strategy._load_mdp()
table, bounds, sl = art["policy_table"], art["boundaries"], art["season_length"]

# 1. Tail boundary sweep: streak + 2*days_left in {54, 55, 56, 57, 58, 59}
tail_boundary = []
end = date(2027, 9, 27)
for d in range(1, 31):
    d_iso = (end - timedelta(days=d)).isoformat()
    for s in range(max(0, 54 - 2 * d), min(58, 60 - 2 * d)):
        for best_s, best_st in ((s, "trusted"), (56, "trusted"), (57, "trusted"), (57, "untrusted"), (None, None)):
            for v in (False, True):
                for p in (0.60, 0.72, 0.80, 0.83, 0.96):
                    for allow, diff in ((True, True), (False, True), (True, False)):
                        ctx = strategy.DecisionContext(p, 0.85 if diff else None, diff, d_iso, allow, art, best_s, best_st)
                        raw_dec = strategy.resolve_policy_decision(ctx, s, v)
                        dec = asdict(raw_dec)
                        g = dec.pop("guard", None)
                        tail_boundary.append(((s, d, best_s, best_st, v, p, allow, diff), dec, g))

# 2. Forced tail (corrupt/missing base or tail) & heuristic path (no table)
degraded_and_heuristic = []
variants = [
    ("missing_all", None),
    ("empty_dict", {}),
    ("corrupt_base", dict(art, policy_table=None, base_sha256=None, tail=None, base_error="ValueError: bad", tail_error="unverifiable")),
    ("corrupt_tail", dict(art, tail=None, tail_error="TailPolicyError: corrupt")),
    ("sha_mismatch", dict(art, base_sha256="0" * 64)),
]
for vname, mdp_inj in variants:
    for d_iso in ("2027-03-24", "2027-09-17", "2027-09-26", "2027-09-27"):
        for s in (0, 9, 12, 40, 50, 56, 57):
            for best_s in (0, 20, 56, 57):
                for v in (False, True):
                    for p in (0.50, 0.7999, 0.80, 0.96):
                        for allow, diff in ((True, True), (False, False)):
                            ctx = strategy.DecisionContext(p, 0.90 if diff else None, diff, d_iso, allow, mdp_inj, best_s, "trusted")
                            raw_dec = strategy.resolve_policy_decision(ctx, s, v)
                            d_clean = asdict(raw_dec)
                            g = d_clean.pop("guard", None)
                            degraded_and_heuristic.append(((vname, d_iso, s, best_s, v, p, allow, diff), d_clean, g))

# 3. Edge table shapes & exceptions
shape_outcomes = []
syn = np.zeros((58, 181, 2, 4), dtype=np.int8)
syn[10, 100, 0] = [0, 2, 0, 0]
syn4 = np.zeros((58, 181, 2, 4), dtype=np.int8)
syn4[10, 100, 0] = [2, 1, 0, 0]
shape_inputs = [
    ("tied_hit", syn, [0.8, 0.8], 10, 100, False, 0.8, 180),
    ("tied_below", syn, [0.8, 0.8], 10, 100, False, 0.79, 180),
    ("empty_bounds", syn, [], 10, 100, False, 0.99, 180),
    ("none_bounds_live", syn, None, 10, 100, False, 0.99, 180),
    ("none_bounds_sentinel_s", syn, None, 57, 100, False, 0.99, 180),
    ("none_bounds_sentinel_d", syn, None, 10, 0, False, 0.99, 180),
    ("narrow_below", syn[..., :2], [0.8, 0.8], 10, 100, False, 0.79, 180),
    ("narrow_oob", syn[..., :2], [0.8, 0.8], 10, 100, False, 0.8, 180),
    ("fourbin_q2_first", syn4, [0.7, 0.8, 0.9], 10, 105, False, 0.85, 100),
    ("fourbin_q3_second", syn4, [0.7, 0.8, 0.9], 10, 105, False, 0.95, 100),
    ("fourbin_q2_again", syn4, [0.7, 0.8, 0.9], 10, 105, False, 0.85, 100),
    ("fourbin_capped_play", syn4, [0.7], 10, 100, False, 0.95, 100),
]
for label, tbl, bnd, s, d, v, p, slen in shape_inputs:
    try:
        r = ("ok", mdp.lookup_action(tbl, bnd, s, d, v, p, slen))
    except Exception as exc:
        r = ("err", type(exc).__name__)
    if has_guarded:
        try:
            g = ("ok", mdp.lookup_action_guarded(tbl, bnd, s, d, v, p, slen))
        except Exception as exc:
            g = ("err", type(exc).__name__)
    else:
        g = None
    shape_outcomes.append((label, r, g))

# 4. Non-bool saver inputs
saver_outcomes = []
saver_vals = [False, True, 0, 1, 2, -1, -2, -3, None, "0", "1", "bad", np.bool_(False), np.bool_(True), np.int64(0), np.int64(1)]
for sv in saver_vals:
    for s, d, p in ((50, 180, 0.96), (50, 180, 0.83), (10, 10, 0.96)):
        try:
            r = ("ok", mdp.lookup_action(table, bounds, s, d, sv, p, sl))
        except Exception as exc:
            r = ("err", type(exc).__name__)
        if has_guarded:
            try:
                g = ("ok", mdp.lookup_action_guarded(table, bounds, s, d, sv, p, sl))
            except Exception as exc:
                g = ("err", type(exc).__name__)
        else:
            g = None
        saver_outcomes.append((repr(sv), s, d, p, r, g))

out_path.write_bytes(pickle.dumps({
    "tail_boundary": tail_boundary,
    "degraded_and_heuristic": degraded_and_heuristic,
    "shape_outcomes": shape_outcomes,
    "saver_outcomes": saver_outcomes,
}))
"""


def test_item_5_edge_regimes() -> dict:
    affected = {}
    with CSV_PATH.open() as fh:
        for row in csv.DictReader(fh):
            affected[(int(row["streak"]), int(row["days_left"]), int(row["saver"]), int(row["bin"]))] = row["guarded_action"]

    with np.load(REPO / "data" / "models" / "mdp_policy.npz") as data:
        bounds = data["boundaries"].tolist()

    def bin_of(p: float) -> int:
        q = 0
        for i, b in enumerate(bounds):
            if p >= b:
                q = i + 1
        return min(q, len(bounds))

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p05v_") as tmp:
        tmp_dir = Path(tmp)
        worker = tmp_dir / "worker5.py"
        worker.write_text(EDGE_WORKER)
        base_pkl = tmp_dir / "base5.pkl"
        head_pkl = tmp_dir / "head5.pkl"
        for src_dir, out_f in ((BASE_SRC, base_pkl), (HEAD_SRC, head_pkl)):
            env = dict(os.environ, PYTHONPATH=f"{src_dir}:{REPO}")
            subprocess.run([sys.executable, str(worker), str(out_f)], cwd=REPO, env=env, check=True, timeout=60)
        b_res = pickle.loads(base_pkl.read_bytes())
        h_res = pickle.loads(head_pkl.read_bytes())

    assert len(b_res["tail_boundary"]) == len(h_res["tail_boundary"])
    tail_checked = 0
    reach_overrides = 0
    for (key, b_dec, _), (key2, h_dec, h_guard) in zip(b_res["tail_boundary"], h_res["tail_boundary"]):
        assert key == key2
        s, d, best_s, best_st, v, p, allow, diff = key
        tail_checked += 1
        if b_dec["objective"] == "emax_season_best":
            assert s + 2 * d < 57
            assert h_dec == b_dec and h_guard is None, f"tail changed at {key}"
        else:
            assert (d <= 0 or s >= 57 or s + 2 * d >= 57)
            csv_hit = affected.get((s, d, int(v), bin_of(p))) if (0 <= s < 57 and d > 0) else None
            if csv_hit is None:
                assert h_dec == b_dec and h_guard is None, f"unexpected reach57 change at {key}"
            else:
                exp_pol = csv_hit
                exp_act = "single" if (exp_pol == "double" and (not allow or not diff)) else exp_pol
                assert h_dec["policy_action"] == exp_pol and h_dec["action"] == exp_act
                assert h_guard is not None
                reach_overrides += 1

    assert len(b_res["degraded_and_heuristic"]) == len(h_res["degraded_and_heuristic"])
    degraded_or_heur_count = 0
    reach57_intact_count = 0
    for (k1, b_dec, _), (k2, h_dec, h_guard) in zip(b_res["degraded_and_heuristic"], h_res["degraded_and_heuristic"]):
        assert k1 == k2
        vname, d_iso, s, best_s, v, p, allow, diff = k1
        if b_dec["objective"] != "reach57" or b_dec["source"] != "mdp":
            assert h_dec == b_dec, f"degraded/heuristic mismatch at {k1}: {h_dec} != {b_dec}"
            assert h_guard is None, f"unexpected guard on {k1}: {h_guard}"
            degraded_or_heur_count += 1
        else:
            assert vname in ("corrupt_tail", "sha_mismatch")
            d_eff = b_dec["days_effective"]
            csv_hit = affected.get((s, d_eff, int(v), bin_of(p))) if (0 <= s < 57 and d_eff > 0) else None
            if csv_hit is None:
                assert h_dec == b_dec and h_guard is None, f"unexpected reach57 change at {k1}"
            else:
                exp_pol = csv_hit
                exp_act = "single" if (exp_pol == "double" and (not allow or not diff)) else exp_pol
                assert h_dec["policy_action"] == exp_pol and h_dec["action"] == exp_act
                assert h_guard is not None
            reach57_intact_count += 1
    assert degraded_or_heur_count > 0

    expected_shape_guards = {
        "tied_hit": ("double", {"rule": "monotone_skip_v1", "bin": 2, "from_bin": 1, "table_action": "skip"}),
        "fourbin_q2_first": ("single", {"rule": "monotone_skip_v1", "bin": 2, "from_bin": 1, "table_action": "skip"}),
        "fourbin_q3_second": ("single", {"rule": "monotone_skip_v1", "bin": 3, "from_bin": 1, "table_action": "skip"}),
        "fourbin_q2_again": ("single", {"rule": "monotone_skip_v1", "bin": 2, "from_bin": 1, "table_action": "skip"}),
    }
    for (lbl1, b_r, _), (lbl2, h_r, h_g) in zip(b_res["shape_outcomes"], h_res["shape_outcomes"]):
        assert lbl1 == lbl2 and b_r == h_r
        if b_r[0] == "err":
            assert h_g == b_r, f"exception mismatch on {lbl1}: {h_g} != {b_r}"
        elif lbl1 in expected_shape_guards:
            assert h_g == ("ok", expected_shape_guards[lbl1])
        else:
            assert h_g == ("ok", (b_r[1], None))

    for (sv1, s, d, p, b_r, _), (sv2, _, _, _, h_r, h_g) in zip(b_res["saver_outcomes"], h_res["saver_outcomes"]):
        assert sv1 == sv2 and b_r == h_r
        if b_r[0] == "err":
            assert h_g == b_r, f"saver exception mismatch for {sv1}: {h_g} != {b_r}"
        else:
            g_act, g_meta = h_g[1]
            if g_meta is None:
                assert g_act == b_r[1]
            else:
                assert b_r[1] == "skip" and ((s, d, 0, bin_of(p)) in affected or (s, d, 1, bin_of(p)) in affected)

    return {
        "tail_boundary_cases": tail_checked,
        "tail_boundary_reach57_csv_hits": reach_overrides,
        "degraded_and_heuristic_cases": degraded_or_heur_count,
        "degraded_tail_intact_reach57_cases": reach57_intact_count,
        "shape_cases": len(h_res["shape_outcomes"]),
        "non_bool_saver_cases": len(h_res["saver_outcomes"]),
    }


def test_item_6_standing_rows() -> dict:
    outcomes = {}

    class ProbeInterrupt(BaseException):
        pass

    day_iso = "2027-03-24"
    day_obj = date(2027, 3, 24)
    now_et = walk.REAL_DATETIME(2027, 3, 24, 17, 0, tzinfo=walk.ET)
    frame = walk._predictions(day_obj, 0.96)
    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled", "code": "S"} for pk in frame["game_pk"]}

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p06v_int_") as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        with walk.offline(now_et), patch.object(picks, "get_game_statuses_detailed", lambda d: statuses):
            sel = strategy.select_pick(frame, day_iso, pdir, streak=50, saver_available=False,
                                       game_statuses_detailed=statuses, require_detailed_statuses=True)
            assert sel.decision.guard == GUARD_SINGLE
            assert picks.load_pick(day_iso, pdir) is None
            assert daily_decision.load_decision(day_iso, pdir) is None

            p_path = picks.save_pick(sel.pick_result.daily, pdir)
            orig_bytes = p_path.read_bytes()
            with patch("bts.util.os.replace", side_effect=ProbeInterrupt("crash at replace")):
                try:
                    sel.pick_result.daily.run_time = "2027-03-24T99:99:99Z"
                    picks.save_pick(sel.pick_result.daily, pdir)
                except ProbeInterrupt:
                    pass
            assert p_path.read_bytes() == orig_bytes
            assert list(pdir.glob("*.tmp")) == []
            outcomes["atomic_pick_interrupt"] = "preserved_exact_bytes"

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p06v_art_") as tmp:
        tdir = Path(tmp)
        corrupt = tdir / "corrupt.npz"
        corrupt.write_bytes(b"not-a-zip")
        oversize = tdir / "oversize.npz"
        oversize.write_bytes(b"x" * (tail_policy.MAX_ARTIFACT_BYTES + 10))
        adir = tdir / "dir_policy"
        adir.mkdir()
        fifo = tdir / "fifo_policy.npz"
        os.mkfifo(fifo)

        for bad_path, expected_substr in (
            (corrupt, "ValueError"),
            (oversize, "too large"),
            (adir, "IsADirectoryError"),
        ):
            with patch.object(mdp, "DEFAULT_POLICY_PATH", bad_path), \
                    patch.object(strategy, "_mdp_cache", None):
                loaded = strategy._load_mdp()
                assert expected_substr in loaded["base_error"], loaded
                assert loaded["tail"] is None and "unverifiable" in loaded["tail_error"]
                ctx = strategy.DecisionContext(0.96, 0.90, True, "2027-09-20", True, loaded, 10, "trusted")
                dec = strategy.resolve_policy_decision(ctx, 0, False)
                assert dec.objective == "emax_season_best" and dec.action == "single"
                assert dec.degraded_reason is not None and dec.guard is None

        fifo_code = (
            "import sys; from pathlib import Path; from bts import strategy; from bts.simulate import mdp; "
            "mdp.DEFAULT_POLICY_PATH = Path(sys.argv[1]); strategy._mdp_cache = None; strategy._load_mdp()"
        )
        timed_out = False
        try:
            subprocess.run(
                [sys.executable, "-c", fifo_code, str(fifo)],
                cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{HEAD_SRC}:{REPO}"),
                capture_output=True, timeout=0.35,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
        assert timed_out, "unwritten FIFO at DEFAULT_POLICY_PATH did not block as documented"
        outcomes["corrupt_oversize_dir_fifo"] = "verified_and_bounded"

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p06v_yr_") as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        assert strategy._days_remaining("2026-12-31") == 0
        assert strategy._days_remaining("2027-01-01") == 269
        art = strategy._load_mdp()
        dec_dec31 = strategy.resolve_policy_decision(
            strategy.DecisionContext(0.96, 0.91, True, "2026-12-31", True, art, 50, "trusted"), 50, False
        )
        dec_jan01 = strategy.resolve_policy_decision(
            strategy.DecisionContext(0.96, 0.91, True, "2027-01-01", True, art, 50, "trusted"), 50, False
        )
        assert dec_dec31.days_effective == 0 and dec_dec31.action == "skip" and dec_dec31.guard is None
        assert dec_jan01.days_effective == 180 and dec_jan01.action == "single" and dec_jan01.guard == GUARD_SINGLE

        for d_str, dt_et in (
            ("2026-12-31", walk.REAL_DATETIME(2026, 12, 31, 23, 59, tzinfo=walk.ET)),
            ("2027-01-01", walk.REAL_DATETIME(2027, 1, 1, 0, 1, tzinfo=walk.ET)),
        ):
            with walk.offline(dt_et):
                empty_sel = strategy.select_pick(pd.DataFrame(), d_str, pdir, streak=50, saver_available=False)
                assert empty_sel.pick_result is None
                f_syn = walk._predictions(date.fromisoformat(d_str), 0.96)
                st_syn = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in f_syn["game_pk"]}
                syn_sel = strategy.select_pick(f_syn, d_str, pdir, streak=50, saver_available=False,
                                               game_statuses_detailed=st_syn, require_detailed_statuses=True)
                if d_str == "2026-12-31":
                    assert syn_sel.action == "skip" and syn_sel.pick_result is None
                else:
                    assert syn_sel.action == "single" and syn_sel.decision.guard == GUARD_SINGLE

        b_chk = walk.boundary_checks()
        assert b_chk["problems"] == [], b_chk
        assert b_chk["switch_2027_at_2027_12_31_2359"] == "accepted"
        assert b_chk["switch_2027_at_2028_01_01_0001"].startswith("refused")
        outcomes["year_boundary_and_midnight_et"] = "PASS"

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_p06v_deliv_") as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        cfg = walk._config(pdir)
        cfg["bluesky"] = {"dm_recipient": "eric.test"}
        cfg["scheduler"]["live_forward_capture_on_lock"] = False

        with walk.offline(now_et), \
                patch.object(orchestrator, "run_cascade", lambda tiers, d: (frame.copy(), "local")), \
                patch.object(picks, "get_game_statuses_detailed", lambda d: statuses), \
                patch.object(scheduler, "_now_et", lambda: now_et), \
                patch.object(scheduler, "_void_committed_games", lambda *a: {}), \
                patch.object(scheduler, "_trigger_live_forward_capture_on_lock", lambda *a, **k: None):
            assert walk.install_state(pdir, day_obj, walk.State("dated", 50, 50, "used", 0.96)) == []
            _, sel, _ = orchestrator.run_and_pick(cfg, day_iso)
            daily = sel.pick_result.daily
            picks.save_pick(daily, pdir)
            st = scheduler.SchedulerState(
                date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
            )

            with patch.object(dm, "send_dm", side_effect=RuntimeError("network down")):
                assert scheduler._deliver_and_lock_pick(daily, cfg, pdir, st, day_iso, "lineup", selection=sel) is False
            reloaded = picks.load_pick(day_iso, pdir)
            assert reloaded.delivery_attempted is False
            assert daily_decision.load_decision(day_iso, pdir) is None

            cfg["scheduler"]["pick_delivery"] = "public"
            with patch.object(posting, "post_to_bluesky", side_effect=RuntimeError("bsky 503")):
                assert scheduler._deliver_and_lock_pick(reloaded, cfg, pdir, st, day_iso, "lineup", selection=sel) is False
            reloaded2 = picks.load_pick(day_iso, pdir)
            assert reloaded2.delivery_attempted is False
            assert daily_decision.load_decision(day_iso, pdir) is None

            cfg["scheduler"]["pick_delivery"] = "private"
            with patch.object(scheduler, "_trigger_live_forward_capture_on_lock", side_effect=ProbeInterrupt("crash pre-decision")):
                try:
                    scheduler._deliver_and_lock_pick(reloaded2, cfg, pdir, st, day_iso, "lineup", selection=sel)
                except ProbeInterrupt:
                    pass
            priv_saved = picks.load_pick(day_iso, pdir)
            assert priv_saved.policy_decision["guard"] == GUARD_SINGLE
            assert daily_decision.load_decision(day_iso, pdir) is None
            scheduler._write_classification_decision(
                pdir, day_iso, action="single", delivered=False, attempted=False,
                primary=scheduler._row_from_daily(priv_saved.pick), double_down=None, state=st, daily=priv_saved,
            )
            assert daily_decision.load_decision(day_iso, pdir) is None

            cfg["scheduler"]["pick_delivery"] = "dm"
            with patch.object(dm, "send_dm", return_value=""), \
                    patch.object(scheduler, "save_state", side_effect=[None, ProbeInterrupt("crash before flag save")]):
                try:
                    scheduler._deliver_and_lock_pick(priv_saved, cfg, pdir, st, day_iso, "lineup", selection=None)
                except ProbeInterrupt:
                    pass
            after_empty = picks.load_pick(day_iso, pdir)
            assert picks.pick_was_delivered(after_empty) is False
            assert scheduler.load_state(day_iso, pdir).committed_pick_written is False
            dec_path = daily_decision.decision_path(day_iso, pdir)
            raw_bytes = dec_path.read_bytes()
            dec_empty = daily_decision.load_decision(day_iso, pdir)
            assert dec_empty["delivery_status"] == "delivered" and dec_empty["scoreable"] is True
            assert dec_empty["policy_guard"] == GUARD_SINGLE

            st_fresh = scheduler.SchedulerState(
                date=day_iso, schedule_fetched_at=now_et.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
                final_skip_candidate={"source": "mdp", "objective": "reach57"},
            )
            scheduler._write_endofday_skip(pdir, day_iso, st_fresh)
            assert dec_path.read_bytes() == raw_bytes
            scheduler._write_classification_decision(
                pdir, day_iso, action="single", delivered=True, attempted=True,
                primary=scheduler._row_from_daily(after_empty.pick), double_down=None, state=st_fresh, daily=after_empty,
            )
            assert dec_path.read_bytes() == raw_bytes
            outcomes["delivery_failures_interrupts_empty_dm"] = "PASS"

    return outcomes


def test_item_7_journal_lines() -> dict:
    art = strategy._load_mdp()
    with np.load(REPO / "data" / "models" / "mdp_policy.npz") as data:
        b = data["boundaries"].tolist()
    mid_bin1 = (b[0] + b[1]) / 2.0

    cases = [
        ("unguarded_single", "2027-03-24", 50, False, 0.83, True, True, art, []),
        ("all_skip", "2027-03-24", 56, False, 0.96, True, True, art, []),
        ("tail_play", "2027-09-20", 10, False, 0.96, True, True, art, ["  Policy: objective=emax_season_best"]),
        ("heuristic_play", "2027-03-24", 10, True, 0.96, True, True, None, []),
        (
            "guarded_b4_b3_saver_unavail", "2027-03-24", 50, False, 0.96, True, True, art,
            ["  POLICY GUARD — At streak 50 with 180 policy days left and saver input unavailable, the table skips bin 4. Bin 3 supplies single. Selected action is single."],
        ),
        (
            "guarded_b4_b3_saver_avail", "2027-09-01", 53, True, 0.96, True, True, art,
            ["  POLICY GUARD — At streak 53 with 26 policy days left and saver input available, the table skips bin 4. Bin 3 supplies single. Selected action is single."],
        ),
    ]

    with CSV_PATH.open() as fh:
        b1_single_saver1 = next(
            row for row in csv.DictReader(fh)
            if int(row["bin"]) == 1 and row["guarded_action"] == "single" and int(row["saver"]) == 1
        )
    s_b1 = int(b1_single_saver1["streak"])
    d_b1 = int(b1_single_saver1["days_left"])
    day_b1 = (date(2027, 9, 27) - timedelta(days=d_b1)).isoformat()
    cases.append((
        "guarded_b1_b0_single_saver_avail", day_b1, s_b1, True, mid_bin1, True, True, art,
        [f"  POLICY GUARD — At streak {s_b1} with {d_b1} policy days left and saver input available, the table skips bin 1. Bin 0 supplies single. Selected action is single."],
    ))

    for cname, allow, diff, exp_sel in (
        ("guarded_double_exec", True, True, "double"),
        ("guarded_double_clamp_allow", False, True, "single"),
        ("guarded_double_clamp_nodiff", True, False, "single"),
    ):
        cases.append((
            cname, "2027-08-29", 1, False, mid_bin1, allow, diff, art,
            [f"  POLICY GUARD — At streak 1 with 29 policy days left and saver input unavailable, the table skips bin 1. Bin 0 supplies double. Selected action is {exp_sel}."],
        ))

    rendered = {}
    for name, d_iso, s, v, p, allow, diff, mdp_arg, exp_lines in cases:
        ctx = strategy.DecisionContext(p, 0.85 if diff else None, diff, d_iso, allow, mdp_arg, max(s, 20), "trusted")
        dec = strategy.resolve_policy_decision(ctx, s, v)
        sel = SimpleNamespace(decision=dec, saver_available=v)
        buf = io.StringIO()
        with redirect_stderr(buf):
            scheduler._log_policy_decision(sel)
        lines = buf.getvalue().splitlines()
        if name == "tail_play":
            assert len(lines) == 1 and lines[0].startswith(exp_lines[0]) and "POLICY GUARD" not in lines[0], lines
        else:
            assert lines == exp_lines, f"{name}: {lines} != {exp_lines}"
        rendered[name] = lines

    return rendered


def main() -> None:
    r5 = test_item_5_edge_regimes()
    r6 = test_item_6_standing_rows()
    r7 = test_item_7_journal_lines()
    out = {"status": "PASS", "item_5": r5, "item_6": r6, "item_7": r7}
    (REVIEW_DIR / "probe_05_06_07_verified.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
