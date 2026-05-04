"""v2.6 Route A smoke — direct corrected_audit_pipeline call.

Validates the new n_block_bootstrap path on real data with tiny reps.
Bypasses CE-IS rare-event MC, diagnostic heatmap, and the rest of run_harness
for speed. Two passes:

1. Baseline: n_block_bootstrap=0 (uses 5-fold percentile CI)
2. Block-bootstrap: n_block_bootstrap=20 (uses pooled-block-bootstrap CI)

Acceptance:
- point_baseline == point_bb (deterministic given seed; only CI changes)
- ci_bb is finite (not None, not NaN)

Usage:
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/v2_6_smoke.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.mdp import solve_mdp
from bts.validate.ope import corrected_audit_pipeline


def main():
    # ---- Subset data: 2 seasons × 2 seeds ----
    profiles_paths = sorted(glob.glob("data/simulation/profiles_seed*_season*.parquet"))
    pa_paths = sorted(glob.glob("data/simulation/pa_predictions_seed*_season*.parquet"))
    if not profiles_paths or not pa_paths:
        raise RuntimeError(
            "No profile/PA parquets found. Run from the worktree root and verify "
            "data/simulation/*.parquet exist."
        )

    profiles_full = pd.concat(pd.read_parquet(p) for p in profiles_paths)
    pa_full = pd.concat(pd.read_parquet(p) for p in pa_paths)

    # Subset: keep only seasons {2024, 2025} and seeds in {0, 42}
    keep_seasons = [2024, 2025]
    keep_seeds = [0, 42]
    profiles = profiles_full[
        profiles_full["season"].isin(keep_seasons)
        & profiles_full["seed"].isin(keep_seeds)
    ].copy()
    pa_df = pa_full[pa_full["season"].isin(keep_seasons)].copy()

    print(f"Smoke data: profiles={len(profiles):,} rows, pa_df={len(pa_df):,} rows")
    print(f"  seasons: {sorted(profiles['season'].unique().tolist())}")
    print(f"  seeds: {sorted(profiles['seed'].unique().tolist())}")

    def _solve(corrected_bins):
        return solve_mdp(corrected_bins, season_length=153, late_phase_days=30)

    # ---- Pass 1: baseline (n_block_bootstrap=0) ----
    print("\n=== Pass 1: baseline (n_block_bootstrap=0) ===")
    result_baseline = corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=keep_seasons,
        mdp_solve_fn=_solve,
        n_bootstrap=20,
        rho_pair_n_permutations=10,
        pa_n_bootstrap=10,
        seed=42,
        n_block_bootstrap=0,
    )
    print(f"  point_estimate: {result_baseline.point_estimate:.6f}")
    print(f"  ci_lower: {result_baseline.ci_lower}")
    print(f"  ci_upper: {result_baseline.ci_upper}")
    print(f"  n_folds: {len(result_baseline.fold_metadata)}")

    # ---- Pass 2: block-bootstrap (n_block_bootstrap=20) ----
    print("\n=== Pass 2: block-bootstrap (n_block_bootstrap=20) ===")
    result_bb = corrected_audit_pipeline(
        profiles, pa_df,
        fold_seasons=keep_seasons,
        mdp_solve_fn=_solve,
        n_bootstrap=20,
        rho_pair_n_permutations=10,
        pa_n_bootstrap=10,
        seed=42,
        n_block_bootstrap=20,
        expected_block_length=7,
    )
    print(f"  point_estimate: {result_bb.point_estimate:.6f}")
    print(f"  ci_lower: {result_bb.ci_lower}")
    print(f"  ci_upper: {result_bb.ci_upper}")
    print(f"  n_folds: {len(result_bb.fold_metadata)}")

    # ---- Acceptance checks ----
    print("\n=== Acceptance checks ===")
    point_match = result_baseline.point_estimate == result_bb.point_estimate
    print(f"  point_baseline == point_bb: {point_match}")
    if not point_match:
        print(
            f"  FAIL: points differ — baseline={result_baseline.point_estimate}, "
            f"bb={result_bb.point_estimate}"
        )
        raise SystemExit(1)

    bb_ci_finite = (
        result_bb.ci_lower is not None
        and result_bb.ci_upper is not None
        and np.isfinite(result_bb.ci_lower)
        and np.isfinite(result_bb.ci_upper)
    )
    print(f"  bb ci finite: {bb_ci_finite}")
    if not bb_ci_finite:
        print(f"  FAIL: bb CI is None or NaN — lo={result_bb.ci_lower}, hi={result_bb.ci_upper}")
        raise SystemExit(1)

    # ---- Write smoke JSON for memo/inspection ----
    smoke_out = {
        "data_subset": {
            "seasons": keep_seasons,
            "seeds": keep_seeds,
            "profiles_rows": int(len(profiles)),
            "pa_rows": int(len(pa_df)),
        },
        "baseline": {
            "n_block_bootstrap": 0,
            "point": float(result_baseline.point_estimate),
            "ci_lower": result_baseline.ci_lower,
            "ci_upper": result_baseline.ci_upper,
            "n_folds": len(result_baseline.fold_metadata),
        },
        "block_bootstrap": {
            "n_block_bootstrap": 20,
            "expected_block_length": 7,
            "point": float(result_bb.point_estimate),
            "ci_lower": float(result_bb.ci_lower),
            "ci_upper": float(result_bb.ci_upper),
            "n_folds": len(result_bb.fold_metadata),
        },
        "acceptance": {
            "point_unchanged": bool(point_match),
            "bb_ci_finite": bool(bb_ci_finite),
        },
    }
    out_path = Path("/tmp/v2_6_smoke.json")
    out_path.write_text(json.dumps(smoke_out, indent=2))
    print(f"\nWrote smoke result: {out_path}")
    print("\nSMOKE PASSED.")


if __name__ == "__main__":
    main()
