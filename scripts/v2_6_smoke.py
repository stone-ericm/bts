"""v2.6 Route A smoke — direct corrected_audit_pipeline call.

Validates the new n_block_bootstrap path on real data with tiny reps.
Bypasses CE-IS rare-event MC, diagnostic heatmap, and the rest of run_harness
for speed. Two passes:

1. Baseline: n_block_bootstrap=0 (uses 5-fold percentile CI)
2. Block-bootstrap: n_block_bootstrap=<n> (uses pooled-block-bootstrap CI)

Acceptance:
- point_baseline == point_bb (deterministic given seed; only CI changes)
- ci_bb is finite (not None, not NaN)
- When --all-seeds + all 5 seasons: point estimate within 0.01 of 0.0333
  (4/120 trajectories = the cell 111 / v2 verdict)

Usage:
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/v2_6_smoke.py
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/v2_6_smoke.py --seeds 0 --seasons 2024,2025 --n-block-bootstrap 5
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/v2_6_smoke.py --all-seeds --seasons 2021,2022,2023,2024,2025 --n-block-bootstrap 20 --output /tmp/v2_6_smoke_111_nonzero.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.mdp import solve_mdp
from bts.validate.ope import corrected_audit_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="v2.6 smoke — corrected_audit_pipeline block-bootstrap validation"
    )
    parser.add_argument(
        "--seasons",
        default="2021,2022,2023,2024,2025",
        help="Comma-separated seasons to include (default: 2021,2022,2023,2024,2025)",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seeds",
        default="0,42",
        help="Comma-separated data seeds to include (default: 0,42). Mutually exclusive with --all-seeds.",
    )
    seed_group.add_argument(
        "--all-seeds",
        action="store_true",
        default=False,
        help="Load all seeds present in data. Mutually exclusive with --seeds.",
    )
    parser.add_argument(
        "--n-block-bootstrap",
        type=int,
        default=20,
        help="Number of block-bootstrap iterations (default: 20)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=20,
        help="Number of inner bootstrap iterations (default: 20)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10,
        help="Number of rho-pair permutations (default: 10)",
    )
    parser.add_argument(
        "--pa-n-bootstrap",
        type=int,
        default=10,
        help="Number of PA bootstrap iterations (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="/tmp/v2_6_smoke.json",
        help="Output JSON path (default: /tmp/v2_6_smoke.json)",
    )
    return parser.parse_args()


def load_parquets(
    data_dir: Path,
    seasons: list[int],
    seeds: list[int] | None,  # None means all-seeds
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load profiles and PA parquets, filtering by season and seed via glob.

    When seeds is None, all parquets matching the season filter are loaded.
    When seeds is a list, only parquets for those specific seeds are loaded.
    """
    profiles_frames = []
    pa_frames = []

    for season in seasons:
        if seeds is None:
            # --all-seeds: glob for all seeds for this season
            p_paths = sorted(
                data_dir.glob(f"profiles_seed*_season{season}.parquet")
            )
            pa_paths = sorted(
                data_dir.glob(f"pa_predictions_seed*_season{season}.parquet")
            )
        else:
            p_paths = []
            pa_paths = []
            for seed in seeds:
                p = data_dir / f"profiles_seed{seed}_season{season}.parquet"
                pa = data_dir / f"pa_predictions_seed{seed}_season{season}.parquet"
                if p.exists():
                    p_paths.append(p)
                if pa.exists():
                    pa_paths.append(pa)

        profiles_frames.extend(pd.read_parquet(p) for p in p_paths)
        pa_frames.extend(pd.read_parquet(p) for p in pa_paths)

    if not profiles_frames:
        raise RuntimeError(
            "No profile parquets found. Run from the worktree root and verify "
            "data/simulation/profiles_seed*_season*.parquet exist."
        )
    if not pa_frames:
        raise RuntimeError(
            "No PA parquets found. Run from the worktree root and verify "
            "data/simulation/pa_predictions_seed*_season*.parquet exist."
        )

    return pd.concat(profiles_frames, ignore_index=True), pd.concat(
        pa_frames, ignore_index=True
    )


def main() -> None:
    args = parse_args()

    seasons = [int(s.strip()) for s in args.seasons.split(",")]
    if args.all_seeds:
        seeds: list[int] | None = None
        seeds_label = "all-seeds"
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        seeds_label = seeds

    data_dir = Path("data/simulation")

    print(f"Loading parquets from {data_dir} ...")
    print(f"  seasons: {seasons}")
    print(f"  seeds: {seeds_label}")

    profiles, pa_df = load_parquets(data_dir, seasons, seeds)

    # Report actual data present
    profile_seeds_actual = sorted(profiles["seed"].unique().tolist())
    seasons_actual = sorted(profiles["season"].unique().tolist())
    pa_seeds_actual = (
        sorted(pa_df["seed"].unique().tolist())
        if "seed" in pa_df.columns
        else "no seed column"
    )

    print(f"Smoke data: profiles={len(profiles):,} rows, pa_df={len(pa_df):,} rows")
    print(f"  seasons actual: {seasons_actual}")
    print(f"  profile seeds actual: {profile_seeds_actual}")
    print(f"  pa seeds actual: {pa_seeds_actual}")

    def _solve(corrected_bins):
        return solve_mdp(corrected_bins, season_length=153, late_phase_days=30)

    # ---- Pass 1: baseline (n_block_bootstrap=0) ----
    print("\n=== Pass 1: baseline (n_block_bootstrap=0) ===")
    result_baseline = corrected_audit_pipeline(
        profiles,
        pa_df,
        fold_seasons=seasons,
        mdp_solve_fn=_solve,
        n_bootstrap=args.n_bootstrap,
        rho_pair_n_permutations=args.n_permutations,
        pa_n_bootstrap=args.pa_n_bootstrap,
        seed=args.seed,
        n_block_bootstrap=0,
    )
    print(f"  point_estimate: {result_baseline.point_estimate:.6f}")
    print(f"  ci_lower: {result_baseline.ci_lower}")
    print(f"  ci_upper: {result_baseline.ci_upper}")
    print(f"  n_folds: {len(result_baseline.fold_metadata)}")

    # ---- Pass 2: block-bootstrap ----
    n_bb = args.n_block_bootstrap
    print(f"\n=== Pass 2: block-bootstrap (n_block_bootstrap={n_bb}) ===")
    result_bb = corrected_audit_pipeline(
        profiles,
        pa_df,
        fold_seasons=seasons,
        mdp_solve_fn=_solve,
        n_bootstrap=args.n_bootstrap,
        rho_pair_n_permutations=args.n_permutations,
        pa_n_bootstrap=args.pa_n_bootstrap,
        seed=args.seed,
        n_block_bootstrap=n_bb,
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
        sys.exit(1)

    bb_ci_finite = (
        result_bb.ci_lower is not None
        and result_bb.ci_upper is not None
        and np.isfinite(result_bb.ci_lower)
        and np.isfinite(result_bb.ci_upper)
    )
    print(f"  bb ci finite: {bb_ci_finite}")
    if not bb_ci_finite:
        print(
            f"  FAIL: bb CI is None or NaN — lo={result_bb.ci_lower}, hi={result_bb.ci_upper}"
        )
        sys.exit(1)

    # Full-data value assertion: only when --all-seeds + all 5 seasons
    full_data_check: dict | None = None
    if args.all_seeds and set(seasons) == {2021, 2022, 2023, 2024, 2025}:
        expected = 0.0333  # 4/120
        tolerance = 0.01  # 1 metric tick at n=120
        delta = abs(result_bb.point_estimate - expected)
        within_tolerance = delta <= tolerance
        print(
            f"  full-data point estimate check: {result_bb.point_estimate:.4f} "
            f"(expected ~{expected}, tolerance {tolerance}) — "
            f"{'PASS' if within_tolerance else 'FAIL'} (delta={delta:.4f})"
        )
        if not within_tolerance:
            print(
                f"  FAIL: point estimate {result_bb.point_estimate:.6f} outside "
                f"tolerance of {expected} ± {tolerance}"
            )
            sys.exit(1)
        full_data_check = {
            "expected": expected,
            "tolerance": tolerance,
            "delta": float(delta),
            "within_tolerance": bool(within_tolerance),
        }

    # ---- Write smoke JSON ----
    smoke_out = {
        "data_subset": {
            "seasons_requested": seasons,
            "seasons_actual": seasons_actual,
            "seeds_requested": seeds_label,
            "profile_seeds_actual": profile_seeds_actual,
            "pa_seeds_actual": pa_seeds_actual,
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
            "n_block_bootstrap": n_bb,
            "expected_block_length": 7,
            "point": float(result_bb.point_estimate),
            "ci_lower": float(result_bb.ci_lower),
            "ci_upper": float(result_bb.ci_upper),
            "n_folds": len(result_bb.fold_metadata),
        },
        "acceptance": {
            "point_unchanged": bool(point_match),
            "bb_ci_finite": bool(bb_ci_finite),
            **({"full_data_value_check": full_data_check} if full_data_check else {}),
        },
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(smoke_out, indent=2))
    print(f"\nWrote smoke result: {out_path}")
    print("\nSMOKE PASSED.")


if __name__ == "__main__":
    main()
