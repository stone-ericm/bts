"""Adversarial probe for Item 3 (A6): parity with archived base 0a2bec10 below streak 57 and at 0 raw days."""
from __future__ import annotations

import csv
from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta
from functools import cache
from hashlib import sha256
from itertools import product
import json
import logging
from math import nextafter
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def worker(src_dir: str) -> None:
    sys.path.insert(0, src_dir)
    sys.path.insert(1, str(REPO))
    import numpy as np
    import pandas as pd
    import bts
    from bts import daily_decision, picks, scheduler, strategy
    from bts.simulate import mdp, tail_policy as tp

    real_file = str(Path(bts.__file__).resolve())
    assert real_file == str((Path(src_dir) / "bts" / "__init__.py").resolve()), real_file
    print(f"IMPORTED={real_file}", flush=True)

    logging.disable(logging.CRITICAL)
    strategy._days_remaining = cache(strategy._days_remaining)
    shipped = strategy._load_mdp()
    tail = shipped["tail"]

    # Build a valid 2-bin tail and a 240-day extended base table + malformed paired base tables
    freq2, hit2, both2 = [0.5, 0.5], [0.75, 0.75], [0.5, 0.5]
    pol2 = tp.solve_emax_season_best(freq2, hit2, both2).policy
    man2 = tp.tail_manifest(n_bins=2, hits=3, both=2, late_seed_days=4, bin_counts=[2, 2])
    two_bin = replace(tail, boundaries=[0.82], policy_table=pol2, bin_freq=freq2, bin_p_hit=hit2, bin_p_both=both2, manifest=man2)
    tp.validate_tail_policy(two_bin, expected_base_sha=shipped["base_sha256"])

    ext_table = np.concatenate([shipped["policy_table"], np.repeat(shipped["policy_table"][:, -1:, :, :], 60, axis=1)], axis=1)
    ext_240 = shipped | {"policy_table": ext_table, "season_length": 240}
    short_25 = shipped | {"policy_table": shipped["policy_table"][:, :26, :, :], "season_length": 25}
    scalar_paired = shipped | {"policy_table": np.array(0, dtype=np.int8)}
    bad_cell_paired = shipped | {"policy_table": shipped["policy_table"].copy()}
    bad_cell_paired["policy_table"][57, 0, 0, 0] = 9

    modes = {
        "shipped": shipped,
        "tail_missing": shipped | {"tail": None, "tail_error": "tail missing"},
        "base_missing": shipped | {"policy_table": None, "base_sha256": None, "base_error": "missing"},
        "sha_mismatch": shipped | {"tail": replace(tail, base_policy_sha256="f" * 64)},
        "none": None,
        "two_bin": shipped | {"tail": two_bin},
        "short_25": short_25,
        "ext_240": ext_240,
        "horizon_zero": shipped | {"season_length": 0},
        "horizon_neg": shipped | {"season_length": -5},
    }

    digest = sha256()
    cases = 0
    raw_seen = set()
    eff_seen = set()

    def record(tag: str, streak: int, day: str, saver: bool, p: float, p2: float | None,
               has_dg: bool, allow: bool, mode_name: str, mode_val, best, status) -> dict:
        nonlocal cases
        raw = strategy._days_remaining(day)
        ctx = strategy.DecisionContext(p, p2, has_dg, day, allow, mode_val, best, status)
        dec = strategy.resolve_policy_decision(ctx, streak, saver)
        bar = strategy.effective_pick_bar(streak, day, saver, mdp=mode_val, best_streak=best, best_status=status)
        d_dict = asdict(dec)
        digest.update(
            json.dumps([tag, streak, day, saver, raw, p, p2, has_dg, allow, mode_name, best, status, d_dict, bar],
                       sort_keys=True).encode()
        )
        cases += 1
        raw_seen.add(raw)
        eff_seen.add(dec.days_effective)
        return d_dict

    # A. Full streak (0..56) x raw_days (0..271) x saver (False, True) x probability edges
    dates_2028 = [(date(2028, 9, 28) - timedelta(days=r)).isoformat() for r in range(272)]
    bounds = shipped["boundaries"]
    probs = [
        0.0, 0.01,
        *(x for b in (*bounds, 0.80) for x in (nextafter(b, 0.0), b, nextafter(b, 1.0))),
        0.99, 1.0,
    ]
    bests = [(None, None), *product((0, 12, 28, 55, 56, 57), ("trusted", "untrusted"))]
    for raw in range(272):
        day = dates_2028[raw]
        for s in range(57):
            for sv in (False, True):
                # Test 3 rotating probability points on every (raw, s, sv), and ALL 19 probs on key days
                p_list = probs if raw in (0, 1, 2, 3, 4, 15, 28, 29, 74, 116, 180, 181, 271) else (
                    probs[(raw + s + sv) % len(probs)],
                    probs[(raw * 3 + s * 5 + sv) % len(probs)],
                )
                for idx, p in enumerate(p_list):
                    b_val, b_st = bests[(raw + s + idx) % len(bests)]
                    has_dg = bool((raw + s + idx) % 3)
                    p2 = 0.88 if has_dg else None
                    allow = bool((raw + idx) % 2)
                    record("grid", s, day, sv, p, p2, has_dg, allow, "shipped", shipped, b_val, b_st)

    # B. All 1,511 G guard cells explicitly across all 5 upper bins & clamps
    guard_count = 0
    guard_csv = REPO / "docs/reviews/2026-11-18-whole-range/contract-g/affected-states.csv"
    with guard_csv.open() as fh:
        for row in csv.DictReader(fh):
            s, r_d, sv, q = int(row["streak"]), int(row["days_left"]), int(row["saver"]), int(row["bin"])
            day = (date(2027, 9, 27) - timedelta(days=r_d)).isoformat()
            p_edge = bounds[q - 1]
            d_dict = record("guard", s, day, bool(sv), p_edge, 0.91, True, True, "shipped", shipped, s, "trusted")
            assert d_dict["guard"] is not None and d_dict["action"] == row["guarded_action"]
            guard_count += 1
    assert guard_count == 1511

    # C. Heuristic double thresholds & partner clamps (exact nextafter on both sides of p1 * p2)
    for s, cut in ((0, 0.55), (9, 0.55), (10, 0.60), (15, 0.60), (16, 0.65), (45, 0.65), (46, None), (56, None)):
        p1 = 0.85
        p2_vals = [None, 0.50, 0.99]
        if cut is not None:
            exact_p2 = cut / p1
            p2_vals.extend([nextafter(exact_p2, 0.0), exact_p2, nextafter(exact_p2, 1.0)])
        for p2 in p2_vals:
            for has_dg, allow, mname in product((False, True), (False, True), ("none", "base_missing")):
                for p_top in (nextafter(0.80, 0.0), 0.80, p1):
                    record("heur", s, "2027-07-15", False, p_top, p2, has_dg, allow, mname, modes[mname], None, None)

    # D. Tail regime exhaustive best/trust/artifact modes + prove _validate_longest_base not reachable in tail!
    for mname, mval in list(modes.items()) + [("scalar_paired_in_tail", scalar_paired), ("bad_cell_in_tail", bad_cell_paired)]:
        for r_d, s, sv in product((1, 2, 3, 10, 28), (0, 10, 25, 40, 54), (False, True)):
            if s + 2 * r_d < 57:
                day = dates_2028[r_d]
                for b_val, b_st in ((None, None), (0, "trusted"), (s, "trusted"), (s + 2 * r_d, "trusted"), (57, "trusted"), (57, "untrusted")):
                    record("tail", s, day, sv, 0.84, 0.83, True, True, mname, mval, b_val, b_st)

    # E. Alternate horizons (25, 240, 0, -5), other years (2025, 2026, 2027, 2028, 2030), and zero days at >= 57
    other_dates = [
        "2025-07-15", "2025-09-28", "2026-07-15", "2026-09-28", "2026-12-31",
        "2027-01-01", "2027-09-26", "2027-09-27", "2027-12-31",
        "2028-01-01", "2028-09-27", "2028-09-28", "2030-06-01", "2030-09-28",
    ]
    for mname, mval in modes.items():
        for day in other_dates:
            raw_d = strategy._days_remaining(day)
            streaks = (0, 15, 40, 56, 57, 58, 100) if raw_d == 0 else (0, 15, 40, 56)
            for s, sv, p in product(streaks, (False, True), (0.70, 0.80, 0.96)):
                record("cal", s, day, sv, p, 0.90, True, True, mname, mval, 56, "trusted")

    # Also check scalar_paired at streak >= 57 with 0 raw days: lookup_indices returns None before indexing table!
    for s in (57, 58, 100):
        record("zero_ge57_bad_cell", s, "2027-09-27", False, 0.96, 0.95, True, True, "bad_cell", bad_cell_paired, 57, "trusted")

    # F. Real select_pick + save_pick + _commit_decision_for_pick below boundary and at 0-day 57
    with tempfile.TemporaryDirectory(dir=HERE, prefix="parity-sel-") as tdir:
        pdir = Path(tdir)
        frame = pd.DataFrame([
            {"batter_name": "B0", "batter_id": 10, "team": "T0", "game_pk": 1, "p_game_hit": 0.86,
             "lineup": 1, "pitcher_name": "P0", "pitcher_id": 90, "flags": "", "game_time": "2027-07-15T23:05:00Z", "pitcher_team": "U0"},
            {"batter_name": "B1", "batter_id": 11, "team": "T1", "game_pk": 2, "p_game_hit": 0.85,
             "lineup": 2, "pitcher_name": "P1", "pitcher_id": 91, "flags": "", "game_time": "2027-07-15T23:05:00Z", "pitcher_team": "U1"},
        ])
        statuses = {1: {"abstract": "P", "detailed": "Scheduled"}, 2: {"abstract": "P", "detailed": "Scheduled"}}
        for idx, (s, day) in enumerate([(0, "2027-07-15"), (12, "2027-09-20"), (56, "2027-07-15"), (56, "2027-09-26"), (57, "2027-09-27")]):
            sub = pdir / f"c{idx}"
            sub.mkdir()
            sel = strategy.select_pick(frame, day, sub, streak=s, saver_available=True, allow_double=True,
                                       game_statuses_detailed=statuses, best_streak=s, best_status="trusted")
            rec_out = None
            if sel.pick_result is not None:
                picks.save_pick(sel.pick_result.daily, sub)
                loaded = picks.load_pick(day, sub)
                st = scheduler.SchedulerState(day, None, [], [], [], False, None, None, None)
                scheduler._commit_decision_for_pick(sub, day, loaded, selection=sel, delivery_status="delivered", state=st)
                rec_out = daily_decision.load_decision(day, sub)
                rec_out.pop("finalized_at", None)
            digest.update(json.dumps([idx, sel.action, sel.source, asdict(sel.decision), rec_out], sort_keys=True).encode())

    # G. Check effective_pick_bar ordinary exception behavior separately
    def broken_loader():
        raise RuntimeError("loader down")

    orig_loader = strategy._load_mdp
    strategy._load_mdp = broken_loader
    try:
        try:
            val = strategy.effective_pick_bar(56, "2027-07-15", False)
            bar_err_behavior = f"returned:{val!r}"
        except RuntimeError:
            bar_err_behavior = "raised:RuntimeError"
    finally:
        strategy._load_mdp = orig_loader

    print(json.dumps({
        "cases": cases,
        "guards": guard_count,
        "raw_min": min(raw_seen),
        "raw_max": max(raw_seen),
        "raw_len": len(raw_seen),
        "eff_min": min(eff_seen),
        "eff_max": max(eff_seen),
        "eff_len": len(eff_seen),
        "digest": digest.hexdigest(),
        "bar_err_behavior": bar_err_behavior,
    }))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
        return

    # 1. Verify mdp.py is untouched byte-for-byte
    base_blob = subprocess.run(["git", "rev-parse", "0a2bec10:src/bts/simulate/mdp.py"],
                               cwd=REPO, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    head_blob = subprocess.run(["git", "rev-parse", "84805852:src/bts/simulate/mdp.py"],
                               cwd=REPO, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    wt_sha = sha256((REPO / "src/bts/simulate/mdp.py").read_bytes()).hexdigest()
    assert base_blob == head_blob, f"mdp.py changed: {base_blob} != {head_blob}"
    print(f"OK: mdp.py untouched (git blob {base_blob}, sha256 {wt_sha[:16]})")

    with tempfile.TemporaryDirectory(dir=HERE, prefix="scratch-base-0a2bec10-") as tmp:
        archive = subprocess.run(["git", "archive", "0a2bec10", "src"],
                                 cwd=REPO, capture_output=True, check=True, timeout=15).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True, timeout=15)

        outputs = []
        for label, src_path in (("base_0a2bec10", Path(tmp) / "src"), ("head_84805852", REPO / "src")):
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(src_path)],
                cwd=REPO, env=env, capture_output=True, text=True, check=True, timeout=90,
            )
            lines = proc.stdout.strip().splitlines()
            print(f"[{label}] {lines[0]}")
            data = json.loads(lines[1])
            print(f"[{label}] {data}")
            outputs.append(data)

    base_out, head_out = outputs
    assert base_out["bar_err_behavior"] == "raised:RuntimeError"
    assert head_out["bar_err_behavior"] == "returned:None"
    assert {k: v for k, v in base_out.items() if k != "bar_err_behavior"} == {
        k: v for k, v in head_out.items() if k != "bar_err_behavior"
    }
    print(
        f"OK: Exact parity across {head_out['cases']} cases + 5 live select/commit flows "
        f"(raw days {head_out['raw_min']}..{head_out['raw_max']} [{head_out['raw_len']} distinct], "
        f"effective days {head_out['eff_min']}..{head_out['eff_max']} [{head_out['eff_len']} distinct], "
        f"guards={head_out['guards']}, sha256={head_out['digest']})"
    )


if __name__ == "__main__":
    main()
