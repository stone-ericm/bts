"""Adversarial probe for Items 5, 6, and 7: regime/table boundaries, standing rows, and journal rendering."""
from __future__ import annotations

from contextlib import ExitStack, nullcontext, redirect_stderr
from dataclasses import fields, replace
from datetime import date, timedelta
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import bts
from bts import daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
from bts.simulate import mdp, tail_policy
from scripts.audit.season_walk import walk

BASE_SRC = HERE / "base_f83e0d0b" / "src"
END_2027 = date(2027, 9, 27)
GUARD_BIN4 = {"rule": "monotone_skip_v1", "bin": 4, "from_bin": 3, "table_action": "skip"}
GUARD_BIN1 = {"rule": "monotone_skip_v1", "bin": 1, "from_bin": 0, "table_action": "skip"}


class ProbeInterrupt(BaseException):
    pass


def _fifo_worker(fifo_path: str) -> None:
    mdp.DEFAULT_POLICY_PATH = Path(fifo_path)
    strategy._mdp_cache = None
    strategy._load_mdp()


def _item5_side_snapshot(side: str) -> dict:
    expected_prefix = BASE_SRC if side == "base" else (ROOT / "src")
    bts_file = str(Path(bts.__file__).resolve())
    assert Path(bts_file).is_relative_to(expected_prefix.resolve()), (side, bts_file)

    strategy._mdp_cache = None
    art = strategy._load_mdp()
    b0, b1 = art["boundaries"][0], art["boundaries"][1]
    ps = [0.0, b0 - 0.01, (b0 + b1) / 2.0, 0.83, 0.96, 1.0]

    shared_fields = [
        "action", "policy_action", "source", "objective", "streak", "days_effective",
        "best_supplied", "best_status", "effective_best", "tail_sha256",
        "degraded_reason", "reason", "pick_bar",
    ]
    out = {"bts_file": bts_file, "tail_boundary": [], "degraded_and_heuristic": [], "early_returns": [], "edge_tables": [], "savers": []}

    # 1. Tail boundary: streak + 2*days_left in 54..60 across streaks 0..57
    for s in range(58):
        for total in (54, 55, 56, 57, 58, 59, 60):
            rem = total - s
            if rem < 0 or rem % 2 != 0:
                continue
            d = rem // 2
            if not (0 <= d <= 180):
                continue
            day = (END_2027 - timedelta(days=d)).isoformat()
            for saver in (False, True):
                for best, bstat in ((None, None), (s, "trusted"), (56, "trusted"), (56, "untrusted")):
                    for p in (0.72, (b0 + b1) / 2.0, 0.83, 0.96):
                        ctx = strategy.DecisionContext(p, 0.85, True, day, True, art, best, bstat)
                        dec = strategy.resolve_policy_decision(ctx, s, saver)
                        row = {k: getattr(dec, k) for k in shared_fields}
                        if side == "head":
                            row["guard"] = dec.guard
                        out["tail_boundary"].append(((s, d, saver, best, bstat, round(p, 6)), row))

    # 2. Forced tail & heuristic & corrupt/missing/mismatched artifacts
    degraded_fixtures = [
        ("no_mdp", None),
        ("empty_mdp", {}),
        ("corrupt_base", dict(art, policy_table=None, boundaries=None, base_sha256=None, tail=None,
                              base_error="ValueError: bad", tail_error="unverifiable")),
        ("corrupt_tail", dict(art, tail=None, tail_error="TailPolicyError: bad")),
        ("sha_mismatch", dict(art, base_sha256="0" * 64)),
    ]
    for label, inj in degraded_fixtures:
        for day, s in (("2027-09-20", 10), ("2027-09-20", 0), ("2027-03-24", 1), ("2027-03-24", 50)):
            for p in ps:
                for allow in (True, False):
                    for diff_g in (True, False):
                        ctx = strategy.DecisionContext(p, 0.85 if diff_g else None, diff_g, day, allow, inj, 30, "trusted")
                        dec = strategy.resolve_policy_decision(ctx, s, False)
                        row = {k: getattr(dec, k) for k in shared_fields}
                        if side == "head":
                            assert dec.guard is None, (label, day, s, p, dec)
                        out["degraded_and_heuristic"].append(((label, day, s, round(p, 6), allow, diff_g), row))

    # 3. Streak >= 57 and days <= 0
    for s in (0, 50, 56, 57, 58, 100):
        for day in ("2026-12-31", "2027-09-27", "2027-09-28", "2027-10-05", "2027-03-24"):
            if s < 57 and day == "2027-03-24":
                continue
            for saver in (False, True):
                for p in (0.50, 0.80, 0.96):
                    ctx = strategy.DecisionContext(p, 0.90, True, day, True, art, s, "trusted")
                    dec = strategy.resolve_policy_decision(ctx, s, saver)
                    row = {k: getattr(dec, k) for k in shared_fields}
                    if side == "head":
                        assert dec.guard is None
                    out["early_returns"].append(((s, day, saver, p), row))

    # 4. Synthetic table shapes & exceptions on lookup_action (and lookup_action_guarded on head)
    t4 = np.zeros((58, 181, 2, 4), dtype=np.int8)
    t4[10, 100, 0] = [0, 2, 0, 0]
    edge_calls = [
        ("tied_at_edge", t4, [0.8, 0.8], 10, 100, False, 0.8, 180),
        ("tied_below", t4, [0.8, 0.8], 10, 100, False, 0.79, 180),
        ("empty_bounds", t4, [], 10, 100, False, 0.99, 180),
        ("none_bounds", t4, None, 10, 100, False, 0.99, 180),
        ("narrow_below", t4[..., :2], [0.8, 0.8], 10, 100, False, 0.79, 180),
        ("narrow_oob", t4[..., :2], [0.8, 0.8], 10, 100, False, 0.8, 180),
        ("queried_code_3", np.full((58, 181, 2, 3), 3, dtype=np.int8), [0.7, 0.8], 10, 100, False, 0.75, 180),
        ("queried_neg1", np.full((58, 181, 2, 3), -1, dtype=np.int8), [0.7, 0.8], 10, 100, False, 0.75, 180),
    ]
    for name, tbl, bnds, s, d, sv, p, sl in edge_calls:
        try:
            raw_res = ("val", mdp.lookup_action(tbl, bnds, s, d, sv, p, sl))
        except Exception as exc:
            raw_res = ("exc", type(exc).__name__)
        if side == "head":
            try:
                g_act, g_meta = mdp.lookup_action_guarded(tbl, bnds, s, d, sv, p, sl)
                g_res = ("val", g_act, g_meta)
            except Exception as exc:
                g_res = ("exc", type(exc).__name__, None)
            if raw_res[0] == "exc":
                assert g_res[:2] == raw_res, (name, raw_res, g_res)
            elif name == "tied_at_edge":
                assert g_res == ("val", "double", {"rule": "monotone_skip_v1", "bin": 2, "from_bin": 1, "table_action": "skip"})
            else:
                assert g_res == ("val", raw_res[1], None), (name, raw_res, g_res)
        out["edge_tables"].append((name, raw_res))

    # 5. Non-bool savers
    savers = [0, 1, 2, -1, -2, -3, "0", "1", "false", None, np.bool_(False), np.bool_(True), np.int64(0), np.int64(1)]
    for idx, sv in enumerate(savers):
        for s, d, p in ((50, 180, 0.96), (50, 180, 0.83), (10, 100, 0.72)):
            try:
                r = ("val", mdp.lookup_action(art["policy_table"], art["boundaries"], s, d, sv, p))
            except Exception as exc:
                r = ("exc", type(exc).__name__)
            if side == "head":
                try:
                    ga, gm = mdp.lookup_action_guarded(art["policy_table"], art["boundaries"], s, d, sv, p)
                    gr = ("val", ga, gm)
                except Exception as exc:
                    gr = ("exc", type(exc).__name__, None)
                if r[0] == "exc":
                    assert gr[:2] == r, (idx, sv, r, gr)
                elif r[1] == "skip" and p == 0.96 and s == 50 and d == 180 and mdp.lookup_action(art["policy_table"], art["boundaries"], s, d, sv, 0.83) == "single":
                    assert gr == ("val", "single", GUARD_BIN4), (idx, sv, gr)
                else:
                    assert gr == ("val", r[1], None), (idx, sv, r, gr)
            out["savers"].append(((idx, s, d, p), r))

    return out


def _verify_item5() -> dict:
    # Load CSV affected set
    import csv
    with (ROOT / "docs/reviews/2026-11-18-whole-range/contract-g/affected-states.csv").open() as f:
        csv_map = {
            (int(r["streak"]), int(r["days_left"]), int(r["saver"]), int(r["bin"])): (
                r["guarded_action"],
                {"rule": "monotone_skip_v1", "bin": int(r["bin"]), "from_bin": int(r["from_bin"]), "table_action": "skip"},
            )
            for r in csv.DictReader(f)
        }

    base_p = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "item5", "base"],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=45,
        env=os.environ | {"PYTHONPATH": f"{BASE_SRC}:{ROOT}"},
    )
    head_p = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "item5", "head"],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=45,
        env=os.environ | {"PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"},
    )
    b_data = json.loads(base_p.stdout.strip().splitlines()[-1])
    h_data = json.loads(head_p.stdout.strip().splitlines()[-1])

    art = strategy._load_mdp()
    bounds = art["boundaries"]

    def _bin_of(p: float) -> int:
        q = 0
        for i, b in enumerate(bounds):
            if p >= b:
                q = i + 1
        return min(q, len(bounds))

    tail_overrides = 0
    for (key_b, rb), (key_h, rh) in zip(b_data["tail_boundary"], h_data["tail_boundary"]):
        assert key_b == key_h
        s, d, saver, _, _, p = key_b
        guard = rh.pop("guard")
        if rb["objective"] == "emax_season_best":
            assert rb == rh and guard is None
        else:
            q = _bin_of(p)
            hit = csv_map.get((s, d, int(saver), q))
            if hit:
                exp_act, exp_g = hit
                assert rb["action"] == "skip" and rh["action"] == exp_act and guard == exp_g
                tail_overrides += 1
            else:
                assert rb == rh and guard is None

    assert b_data["degraded_and_heuristic"] == h_data["degraded_and_heuristic"]
    assert b_data["early_returns"] == h_data["early_returns"]
    assert b_data["edge_tables"] == h_data["edge_tables"]
    assert b_data["savers"] == h_data["savers"]

    # Additional donor code edge tests: negative donor codes (-1, -2) must NOT be treated as single/double donors!
    neg_donor_tbl = np.zeros((11, 101, 2, 3), dtype=np.int8)
    neg_donor_tbl[10, 100, 0] = [-1, -2, 0]
    assert mdp.lookup_action_guarded(neg_donor_tbl, [0.7, 0.8], 10, 100, False, 0.85) == ("skip", None)

    # 4-bin repeated queries in both orders + synthetic tail donor hole
    four_tbl = np.zeros((11, 106, 2, 4), dtype=np.int8)
    four_tbl[10, 100, 0] = [2, 1, 0, 0]
    for q, p in ((3, 0.95), (2, 0.85), (3, 0.95), (1, 0.75), (0, 0.65)):
        act, g = mdp.lookup_action_guarded(four_tbl, [0.7, 0.8, 0.9], 10, 105, False, p, 100)
        if q >= 2:
            assert (act, g) == ("single", {"rule": "monotone_skip_v1", "bin": q, "from_bin": 1, "table_action": "skip"})
        elif q == 1:
            assert (act, g) == ("single", None)
        else:
            assert (act, g) == ("double", None)

    return {
        "tail_boundary_checked": len(b_data["tail_boundary"]),
        "reach57_csv_overrides_near_boundary": tail_overrides,
        "degraded_and_heuristic_checked": len(b_data["degraded_and_heuristic"]),
        "early_returns_checked": len(b_data["early_returns"]),
        "negative_donor_codes_rejected": True,
    }


def _verify_item6_standing_rows() -> dict:
    day = "2027-03-24"
    now = walk.REAL_DATETIME(2027, 3, 24, 16, 0, tzinfo=walk.ET)

    with TemporaryDirectory() as d:
        root = Path(d)
        # 1. Interrupt after lookup before first save, and immediately before os.replace
        with walk.offline(now):
            frame = walk._predictions(date.fromisoformat(day), 0.96)
            statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}

            orig_lookup = mdp.lookup_action_guarded

            def interrupt_after_lookup(*a, **k):
                res = orig_lookup(*a, **k)
                assert res[1] == GUARD_BIN4
                raise ProbeInterrupt("after_lookup")

            empty_dir = root / "pre_save"
            empty_dir.mkdir()
            with patch.object(mdp, "lookup_action_guarded", interrupt_after_lookup):
                try:
                    strategy.select_pick(
                        frame, day, empty_dir, streak=50, saver_available=False,
                        game_statuses_detailed=statuses, require_detailed_statuses=True,
                    )
                except ProbeInterrupt:
                    pass
                else:
                    raise AssertionError("Expected ProbeInterrupt")
            assert picks.load_pick(day, empty_dir) is None
            assert daily_decision.load_decision(day, empty_dir) is None

            # Interrupt immediately before os.replace with an existing prior pick file
            rep_dir = root / "atomic_rep"
            rep_dir.mkdir()
            sel_prior = strategy.select_pick(
                walk._predictions(date.fromisoformat(day), 0.83), day, rep_dir,
                streak=50, saver_available=False, game_statuses_detailed=statuses, require_detailed_statuses=True,
            )
            prior_path = picks.save_pick(sel_prior.pick_result.daily, rep_dir)
            prior_bytes = prior_path.read_bytes()

            sel_new = strategy.select_pick(
                frame, day, rep_dir, streak=50, saver_available=False,
                game_statuses_detailed=statuses, require_detailed_statuses=True,
            )
            with patch("bts.util.os.replace", side_effect=ProbeInterrupt("before_replace")):
                try:
                    picks.save_pick(sel_new.pick_result.daily, rep_dir)
                except ProbeInterrupt:
                    pass
                else:
                    raise AssertionError("Expected ProbeInterrupt")
            assert prior_path.read_bytes() == prior_bytes
            assert list(rep_dir.glob("*.tmp*")) == []
            assert daily_decision.load_decision(day, rep_dir) is None

        # 2. Malformed, oversized, directory, and FIFO base artifacts
        art_dir = root / "artifacts"
        art_dir.mkdir()
        corrupt = art_dir / "corrupt.npz"
        corrupt.write_bytes(b"not-a-zip")
        oversized = art_dir / "oversized.npz"
        oversized.write_bytes(b"0" * (tail_policy.MAX_ARTIFACT_BYTES + 10))
        directory = art_dir / "dir_policy"
        directory.mkdir()
        fifo = art_dir / "fifo_policy"
        os.mkfifo(fifo)

        orig_default = mdp.DEFAULT_POLICY_PATH
        try:
            for bad_path, err_sub in (
                (corrupt, "BadZipFile"),
                (oversized, "too large"),
                (directory, "IsADirectoryError"),
            ):
                mdp.DEFAULT_POLICY_PATH = bad_path
                strategy._mdp_cache = None
                loaded = strategy._load_mdp()
                assert loaded["policy_table"] is None and loaded["base_error"]
                assert loaded["tail"] is None and "unverifiable" in loaded["tail_error"]
                ctx = strategy.DecisionContext(0.96, 0.90, True, "2027-09-20", True, loaded, 10, "trusted")
                dec = strategy.resolve_policy_decision(ctx, 0, False)
                assert dec.objective == "emax_season_best" and dec.action == "single"
                assert dec.degraded_reason and dec.guard is None
        finally:
            mdp.DEFAULT_POLICY_PATH = orig_default
            strategy._mdp_cache = None

        # Bounded subprocess timeout probe for unwritten FIFO on BOTH base and head
        fifo_blocked = {}
        for side_name, py_src in (("base", BASE_SRC), ("head", ROOT / "src")):
            try:
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "fifo", str(fifo)],
                    cwd=ROOT, capture_output=True, text=True, timeout=0.35,
                    env=os.environ | {"PYTHONPATH": f"{py_src}:{ROOT}"},
                )
                fifo_blocked[side_name] = False
            except subprocess.TimeoutExpired:
                fifo_blocked[side_name] = True
        assert fifo_blocked == {"base": True, "head": True}, fifo_blocked

        # 3. Midnight ET & year boundary
        art = strategy._load_mdp()
        assert strategy._days_remaining("2026-12-31") == 0
        assert strategy._days_remaining("2027-01-01") == 269
        dec_dec31 = strategy.resolve_policy_decision(
            strategy.DecisionContext(0.96, 0.90, True, "2026-12-31", True, art, 50, "trusted"), 50, False
        )
        dec_jan01 = strategy.resolve_policy_decision(
            strategy.DecisionContext(0.96, 0.90, True, "2027-01-01", True, art, 50, "trusted"), 50, False
        )
        assert (dec_dec31.days_effective, dec_dec31.action, dec_dec31.guard) == (0, "skip", None)
        assert (dec_jan01.days_effective, dec_jan01.action, dec_jan01.guard) == (180, "single", GUARD_BIN4)

        for d_str, hr, mn in (("2026-12-31", 23, 59), ("2027-01-01", 0, 1)):
            at = walk.REAL_DATETIME.fromisoformat(f"{d_str}T00:00:00").replace(hour=hr, minute=mn, tzinfo=walk.ET)
            with walk.offline(at):
                empty_sel = strategy.select_pick(pd.DataFrame(), d_str, root / "yr", streak=50, saver_available=False)
                assert empty_sel.pick_result is None
                if d_str == "2027-01-01":
                    f_jan = walk._predictions(date(2027, 1, 1), 0.96)
                    st_jan = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in f_jan.game_pk}
                    jan_sel = strategy.select_pick(
                        f_jan, d_str, root / "yr_play", streak=50, saver_available=False,
                        game_statuses_detailed=st_jan, require_detailed_statuses=True,
                    )
                    assert jan_sel.action == "single" and jan_sel.decision.guard == GUARD_BIN4

        bc = walk.boundary_checks()
        assert bc["switch_2027_at_2027_12_31_2359"] == "accepted"
        assert bc["switch_2027_at_2028_01_01_0001"].startswith("refused")

        # 4. Failed/unconfirmed send, empty DM ID, private interrupt, and post-decision pre-flag crash
        for sub in ("caught_dm_and_public", "attempt_direct", "attempt_classify", "empty_dm", "private_interrupt", "post_decision_crash"):
            sub_dir = root / sub
            sub_dir.mkdir()
            with ExitStack() as stack:
                stack.enter_context(walk.offline(now))
                for helper_name in ("_trigger_live_forward_capture_on_lock", "_alert_delivery_outcome_unknown"):
                    stack.enter_context(patch.object(scheduler, helper_name, lambda *a, **k: None))
                stack.enter_context(patch.object(scheduler, "_void_committed_games", lambda *_: {}))
                stack.enter_context(patch.object(scheduler, "_now_et", lambda: now))
                frame = walk._predictions(date.fromisoformat(day), 0.96)
                statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
                sel = strategy.select_pick(
                    frame, day, sub_dir, streak=50, saver_available=False,
                    game_statuses_detailed=statuses, require_detailed_statuses=True,
                )
                daily = sel.pick_result.daily
                picks.save_pick(daily, sub_dir)
                cfg = {"orchestrator": {"picks_dir": str(sub_dir)}, "bluesky": {"dm_recipient": "op"},
                       "scheduler": {"pick_delivery": "dm"}}

                if sub == "caught_dm_and_public":
                    with patch.object(dm, "send_dm", side_effect=RuntimeError("net down")):
                        assert scheduler._deliver_and_lock_pick(daily, cfg, sub_dir, _make_state(), day, "p", selection=sel) is False
                    assert not picks.load_pick(day, sub_dir).delivery_attempted
                    assert daily_decision.load_decision(day, sub_dir) is None
                    cfg["scheduler"]["pick_delivery"] = "public"
                    with patch.object(posting, "post_to_bluesky", side_effect=RuntimeError("bsky down")):
                        assert scheduler._deliver_and_lock_pick(picks.load_pick(day, sub_dir), cfg, sub_dir, _make_state(), day, "p", selection=sel) is False
                    assert not picks.load_pick(day, sub_dir).delivery_attempted
                    assert daily_decision.load_decision(day, sub_dir) is None
                elif sub in ("attempt_direct", "attempt_classify"):
                    with patch.object(dm, "send_dm", side_effect=ProbeInterrupt("mid_send")):
                        try:
                            scheduler._deliver_and_lock_pick(daily, cfg, sub_dir, _make_state(), day, "p", selection=sel)
                        except ProbeInterrupt:
                            pass
                    persisted = picks.load_pick(day, sub_dir)
                    assert persisted.delivery_attempted is True
                    assert daily_decision.load_decision(day, sub_dir) is None
                    spy = Mock()
                    with patch.object(dm, "send_dm", spy):
                        if sub == "attempt_direct":
                            assert scheduler._deliver_and_lock_pick(persisted, cfg, sub_dir, _make_state(), day, "p") is False
                        else:
                            scheduler._write_classification_decision(
                                sub_dir, day, action="single", delivered=False, attempted=True,
                                primary=scheduler._row_from_daily(persisted.pick), double_down=None,
                                state=_make_state(), daily=persisted,
                            )
                    spy.assert_not_called()
                    rec = daily_decision.load_decision(day, sub_dir)
                    assert rec["delivery_status"] == "locked_unconfirmed" and rec["policy_guard"] == GUARD_BIN4
                elif sub == "empty_dm":
                    with patch.object(dm, "send_dm", return_value=""):
                        assert scheduler._deliver_and_lock_pick(daily, cfg, sub_dir, _make_state(), day, "p", selection=sel) is True
                    assert not picks.pick_was_delivered(picks.load_pick(day, sub_dir))
                    rec = daily_decision.load_decision(day, sub_dir)
                    assert rec["delivery_status"] == "delivered" and rec["policy_guard"] == GUARD_BIN4
                elif sub == "private_interrupt":
                    cfg["scheduler"]["pick_delivery"] = "private"
                    with patch.object(scheduler, "_trigger_live_forward_capture_on_lock", side_effect=ProbeInterrupt("priv")):
                        try:
                            scheduler._deliver_and_lock_pick(daily, cfg, sub_dir, _make_state(), day, "p", selection=sel)
                        except ProbeInterrupt:
                            pass
                    loaded_p = picks.load_pick(day, sub_dir)
                    assert loaded_p.policy_decision["guard"] == GUARD_BIN4
                    assert daily_decision.load_decision(day, sub_dir) is None
                    scheduler._write_classification_decision(
                        sub_dir, day, action="single", delivered=False, attempted=False,
                        primary=scheduler._row_from_daily(loaded_p.pick), double_down=None,
                        state=_make_state(), daily=loaded_p,
                    )
                    assert daily_decision.load_decision(day, sub_dir) is None
                    # Subsequent commit preserves original guard
                    assert scheduler._deliver_and_lock_pick(loaded_p, cfg, sub_dir, _make_state(), day, "p") is True
                    assert daily_decision.load_decision(day, sub_dir)["policy_guard"] == GUARD_BIN4
                elif sub == "post_decision_crash":
                    scheduler.save_state(_make_state(), sub_dir)
                    with patch.object(scheduler, "save_state", side_effect=ProbeInterrupt("after_dec")):
                        try:
                            scheduler._commit_decision_for_pick(
                                sub_dir, day, daily, selection=sel, delivery_status="delivered", state=_make_state()
                            )
                        except ProbeInterrupt:
                            pass
                    dec_p = daily_decision.decision_path(day, sub_dir)
                    exact_bytes = dec_p.read_bytes()
                    assert not scheduler.load_state(day, sub_dir).committed_pick_written
                    scheduler._write_endofday_skip(sub_dir, day, _make_state(final_skip_candidate={"source": "mdp", "primary": sel.primary_candidate}))
                    assert dec_p.read_bytes() == exact_bytes
                    scheduler._write_classification_decision(
                        sub_dir, day, action="single", delivered=True,
                        primary=scheduler._row_from_daily(daily.pick), double_down=None,
                        state=_make_state(), daily=daily,
                    )
                    assert dec_p.read_bytes() == exact_bytes

    return {"fifo_blocked": fifo_blocked, "standing_rows_all_verified": True}


def _verify_item7_journal() -> dict:
    art = strategy._load_mdp()
    b0, b1 = art["boundaries"][0], art["boundaries"][1]
    p_bin1 = (b0 + b1) / 2.0

    cases = [
        # (name, day, streak, saver, p, allow_double, diff_game, injected_mdp, exp_guard_line)
        ("unguarded_single", "2027-03-24", 50, False, 0.83, True, True, art, None),
        ("all_skip", "2027-03-24", 56, False, 0.99, True, True, art, None),
        ("tail_play", "2027-09-20", 0, False, 0.96, True, True, art, None),
        ("heuristic_play", "2027-03-24", 5, False, 0.90, True, True, None, None),
        (
            "capped_march24_bin4_single",
            "2027-03-24",
            50,
            False,
            0.96,
            True,
            True,
            art,
            "  POLICY GUARD — At streak 50 with 180 policy days left and saver input unavailable, "
            "the table skips bin 4. Bin 3 supplies single. Selected action is single.",
        ),
        (
            "saver_true_bin4_single",
            (END_2027 - timedelta(days=38)).isoformat(),
            42,
            True,
            0.96,
            True,
            True,
            art,
            "  POLICY GUARD — At streak 42 with 38 policy days left and saver input available, "
            "the table skips bin 4. Bin 3 supplies single. Selected action is single.",
        ),
        (
            "exec_guarded_double",
            "2027-08-29",
            1,
            False,
            p_bin1,
            True,
            True,
            art,
            "  POLICY GUARD — At streak 1 with 29 policy days left and saver input unavailable, "
            "the table skips bin 1. Bin 0 supplies double. Selected action is double.",
        ),
        (
            "clamped_allow_false",
            "2027-08-29",
            1,
            False,
            p_bin1,
            False,
            True,
            art,
            "  POLICY GUARD — At streak 1 with 29 policy days left and saver input unavailable, "
            "the table skips bin 1. Bin 0 supplies double. Selected action is single.",
        ),
        (
            "clamped_same_game",
            "2027-08-29",
            1,
            False,
            p_bin1,
            True,
            False,
            art,
            "  POLICY GUARD — At streak 1 with 29 policy days left and saver input unavailable, "
            "the table skips bin 1. Bin 0 supplies double. Selected action is single.",
        ),
        (
            "nonbool_str_zero",
            "2027-03-24",
            50,
            "0",
            0.96,
            True,
            True,
            art,
            "  POLICY GUARD — At streak 50 with 180 policy days left and saver input unavailable, "
            "the table skips bin 4. Bin 3 supplies single. Selected action is single.",
        ),
        (
            "nonbool_str_one",
            (END_2027 - timedelta(days=38)).isoformat(),
            42,
            "1",
            0.96,
            True,
            True,
            art,
            "  POLICY GUARD — At streak 42 with 38 policy days left and saver input available, "
            "the table skips bin 4. Bin 3 supplies single. Selected action is single.",
        ),
        (
            "nonbool_np_bool_true",
            (END_2027 - timedelta(days=38)).isoformat(),
            42,
            np.bool_(True),
            0.96,
            True,
            True,
            art,
            "  POLICY GUARD — At streak 42 with 38 policy days left and saver input available, "
            "the table skips bin 4. Bin 3 supplies single. Selected action is single.",
        ),
    ]

    rendered = {}
    for name, day, s, sv, p, allow, diff_g, inj_mdp, exp_line in cases:
        ctx = strategy.DecisionContext(p, 0.90 if diff_g else None, diff_g, day, allow, inj_mdp, s, "trusted")
        dec = strategy.resolve_policy_decision(ctx, s, sv)
        buf = io.StringIO()
        with redirect_stderr(buf):
            scheduler._log_policy_decision(SimpleNamespace(decision=dec, saver_available=sv))
        lines = [ln for ln in buf.getvalue().splitlines() if "POLICY GUARD" in ln]
        if exp_line is None:
            assert lines == [], (name, lines)
        else:
            assert lines == [exp_line], (name, lines, exp_line)
        rendered[name] = lines[0] if lines else None

    return {"cases_verified": len(cases), "rendered": rendered}


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "item5":
        print(json.dumps(_item5_side_snapshot(sys.argv[2])))
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "fifo":
        _fifo_worker(sys.argv[2])
        return

    r5 = _verify_item5()
    r6 = _verify_item6_standing_rows()
    r7 = _verify_item7_journal()

    out = {"item5": r5, "item6": r6, "item7": r7}
    (HERE / "probe_05_06_07_results.json").write_text(json.dumps(out, indent=2))
    print("PROBE_05_06_07_PASS", json.dumps({"item5": r5, "item6_fifo": r6["fifo_blocked"], "item7_cases": r7["cases_verified"]}))


if __name__ == "__main__":
    main()
