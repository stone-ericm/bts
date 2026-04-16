#!/usr/bin/env python3
"""A/B validation for the Option 7 pooled policy.

Compares the shipped production policy (data/models/mdp_policy.npz) against
the new pooled policy (data/models/mdp_policy_pooled_v1.npz) on each of
the 16 audit seeds' individual bin distributions.

Two passes:
    1. WITHIN-POOL: evaluate both policies on each seed's own bins. The
       pooled policy was trained on all 16 seeds INCLUDING each seed
       being evaluated, so this is not a pure holdout — but it shows
       the distribution of per-seed performance.
    2. LEAVE-ONE-OUT: for each seed, rebuild the pooled policy from the
       OTHER 15 seeds, then evaluate it on the held-out seed's bins.
       This IS a proper holdout — each evaluation is against a seed
       the policy didn't see.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/pooled_policy_ab.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.mdp import MDPSolution, solve_mdp
from bts.simulate.pooled_policy import (
    build_pooled_policy,
    compute_pooled_bins,
    evaluate_mdp_policy,
    load_pooled_profiles,
    parse_seed_from_path,
    split_by_phase_pooled,
)


DEFAULT_PROD_POLICY_PATH = Path("data/models/mdp_policy.npz")
DEFAULT_POOLED_POLICY_PATH = Path("data/models/mdp_policy_pooled_v1.npz")
DEFAULT_PROFILES_ROOT = Path("data/hetzner_results/pooled_bins_run")

SEASON_LENGTH = 180
LATE_PHASE_DAYS = 30
N_BINS = 5


def load_mdp_solution_from_npz(path: Path, bins_for_shape: "object" = None) -> np.ndarray:
    """Return just the policy_table from a saved .npz.

    The shipped production policy doesn't preserve the QualityBins object,
    only the policy_table + boundaries + season_length, so we reconstruct
    a minimal wrapper later if needed.
    """
    data = np.load(path)
    return data["policy_table"], data["boundaries"].tolist(), int(data["season_length"])


def seed_bins_for(profiles: pd.DataFrame, late_phase_days: int, n_bins: int):
    early_df, late_df = split_by_phase_pooled(profiles, late_phase_days=late_phase_days)
    if len(late_df) == 0:
        eb = compute_pooled_bins(profiles, n_bins=n_bins)
        return eb, None
    eb = compute_pooled_bins(early_df, n_bins=n_bins)
    lb = compute_pooled_bins(late_df, n_bins=n_bins)
    return eb, lb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles-root", type=Path, default=DEFAULT_PROFILES_ROOT,
                    help="Directory containing per-box subdirs with simulation_seed* children")
    ap.add_argument("--prod-policy", type=Path, default=DEFAULT_PROD_POLICY_PATH,
                    help="Production policy .npz to compare against")
    ap.add_argument("--pooled-policy", type=Path, default=DEFAULT_POOLED_POLICY_PATH,
                    help="Candidate pooled policy .npz to evaluate")
    ap.add_argument("--out", type=Path, default=Path("data/validation/pooled_policy_ab.json"),
                    help="Output JSON path")
    args = ap.parse_args()

    all_seed_dirs = sorted(args.profiles_root.glob("*/simulation_seed*"))
    print(f"Found {len(all_seed_dirs)} seed dirs under {args.profiles_root}")

    # Load all profiles once, tagged with seed
    all_profiles = load_pooled_profiles(all_seed_dirs)
    all_profiles["season"] = all_profiles["season"].astype(int)
    seeds = sorted(int(s) for s in all_profiles["seed"].unique())
    print(f"Seeds: {seeds}")

    # Load production + pooled policies
    prod_table, prod_boundaries, prod_season_length = load_mdp_solution_from_npz(args.prod_policy)
    pool_table, pool_boundaries, pool_season_length = load_mdp_solution_from_npz(args.pooled_policy)
    print(f"\nProduction policy: season_length={prod_season_length}, "
          f"boundaries={[round(b, 4) for b in prod_boundaries]}, shape={prod_table.shape}")
    print(f"Pooled policy:     season_length={pool_season_length}, "
          f"boundaries={[round(b, 4) for b in pool_boundaries]}, shape={pool_table.shape}")

    # ---- Pass 1: WITHIN-POOL ----
    print("\n" + "=" * 60)
    print("PASS 1: within-pool (each seed evaluated against its own bins)")
    print("=" * 60)

    print(f"\n{'seed':>10}  {'prod P(57)':>12}  {'pooled P(57)':>13}  {'gap':>9}  {'n_early':>8}  {'n_late':>7}")
    within_results = []
    for seed in seeds:
        seed_df = all_profiles[all_profiles["seed"] == seed].copy()
        early_bins, late_bins = seed_bins_for(seed_df, LATE_PHASE_DAYS, N_BINS)

        v_prod = evaluate_mdp_policy(
            policy_table=prod_table, early_bins=early_bins,
            season_length=SEASON_LENGTH,
            late_bins=late_bins, late_phase_days=LATE_PHASE_DAYS,
        )
        v_pool = evaluate_mdp_policy(
            policy_table=pool_table, early_bins=early_bins,
            season_length=SEASON_LENGTH,
            late_bins=late_bins, late_phase_days=LATE_PHASE_DAYS,
        )
        n_early = sum(len(seed_df[seed_df["date"].isin(sorted(seed_df[seed_df["season"] == s]["date"].unique())[:-LATE_PHASE_DAYS])]) for s in seed_df["season"].unique() if len(sorted(seed_df[seed_df["season"] == s]["date"].unique())) > LATE_PHASE_DAYS) // 10
        n_late_days = 0
        for s in seed_df["season"].unique():
            dates = sorted(seed_df[seed_df["season"] == s]["date"].unique())
            n_late_days += min(LATE_PHASE_DAYS, len(dates))
        print(f"{seed:>10}  {v_prod:>11.3%}  {v_pool:>12.3%}  {v_pool - v_prod:>+8.3%}  "
              f"{'':>8}  {n_late_days:>7}")
        within_results.append({"seed": seed, "v_prod": v_prod, "v_pool": v_pool})

    within_prod = np.array([r["v_prod"] for r in within_results])
    within_pool = np.array([r["v_pool"] for r in within_results])
    gap_within = within_pool - within_prod
    print(f"\n  mean(prod)   = {within_prod.mean():.3%}  ± {within_prod.std(ddof=1):.3%}")
    print(f"  mean(pooled) = {within_pool.mean():.3%}  ± {within_pool.std(ddof=1):.3%}")
    print(f"  mean(gap)    = {gap_within.mean():+.3%}  ± {gap_within.std(ddof=1):.3%}")
    print(f"  pooled wins in {int((gap_within > 0).sum())}/{len(gap_within)} seeds")

    # ---- Pass 2: LEAVE-ONE-OUT ----
    print("\n" + "=" * 60)
    print("PASS 2: leave-one-out (each seed held out, policy built from other 15)")
    print("=" * 60)

    print(f"\n{'seed':>10}  {'prod P(57)':>12}  {'LOO P(57)':>11}  {'gap':>9}")
    loo_results = []
    for held_out in seeds:
        train_df = all_profiles[all_profiles["seed"] != held_out].copy()
        eval_df = all_profiles[all_profiles["seed"] == held_out].copy()

        loo_sol = build_pooled_policy(
            train_df,
            season_length=SEASON_LENGTH,
            late_phase_days=LATE_PHASE_DAYS,
            n_bins=N_BINS,
        )
        eval_early, eval_late = seed_bins_for(eval_df, LATE_PHASE_DAYS, N_BINS)

        v_prod = evaluate_mdp_policy(
            policy_table=prod_table, early_bins=eval_early,
            season_length=SEASON_LENGTH,
            late_bins=eval_late, late_phase_days=LATE_PHASE_DAYS,
        )
        v_loo = evaluate_mdp_policy(
            policy_table=loo_sol.policy_table, early_bins=eval_early,
            season_length=SEASON_LENGTH,
            late_bins=eval_late, late_phase_days=LATE_PHASE_DAYS,
        )
        print(f"{held_out:>10}  {v_prod:>11.3%}  {v_loo:>10.3%}  {v_loo - v_prod:>+8.3%}")
        loo_results.append({"seed": held_out, "v_prod": v_prod, "v_loo": v_loo,
                            "loo_optimal_p57": loo_sol.optimal_p57})

    loo_prod = np.array([r["v_prod"] for r in loo_results])
    loo_pool = np.array([r["v_loo"] for r in loo_results])
    gap_loo = loo_pool - loo_prod
    print(f"\n  mean(prod on holdout)    = {loo_prod.mean():.3%}  ± {loo_prod.std(ddof=1):.3%}")
    print(f"  mean(LOO pooled holdout) = {loo_pool.mean():.3%}  ± {loo_pool.std(ddof=1):.3%}")
    print(f"  mean(gap)                = {gap_loo.mean():+.3%}  ± {gap_loo.std(ddof=1):.3%}")
    print(f"  pooled wins in {int((gap_loo > 0).sum())}/{len(gap_loo)} seeds")

    # ---- Ship recommendation ----
    print("\n" + "=" * 60)
    print("SHIP DECISION")
    print("=" * 60)
    if gap_loo.mean() > 0 and (gap_loo > 0).sum() >= len(gap_loo) * 0.6:
        print(f"  ✓ SHIP: pooled policy beats production on {int((gap_loo > 0).sum())}/{len(gap_loo)} "
              f"held-out seeds with mean gap +{gap_loo.mean() * 100:.2f}pp")
    elif abs(gap_loo.mean()) < 0.005:
        print(f"  ~ TIE: mean gap is {gap_loo.mean() * 100:+.2f}pp (< 0.5pp); pooled policy is "
              f"equivalent to production but LESS NOISY. Shipping is defensible for robustness.")
    else:
        print(f"  ✗ HOLD: pooled policy underperforms by {gap_loo.mean() * 100:.2f}pp "
              f"on held-out seeds. Don't ship yet.")

    # Save results
    out_json = args.out
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "within_pool": within_results,
        "leave_one_out": loo_results,
        "within_pool_summary": {
            "mean_prod": float(within_prod.mean()),
            "mean_pool": float(within_pool.mean()),
            "mean_gap": float(gap_within.mean()),
            "std_gap": float(gap_within.std(ddof=1)),
            "pool_wins": int((gap_within > 0).sum()),
            "n_seeds": len(seeds),
        },
        "leave_one_out_summary": {
            "mean_prod": float(loo_prod.mean()),
            "mean_pool": float(loo_pool.mean()),
            "mean_gap": float(gap_loo.mean()),
            "std_gap": float(gap_loo.std(ddof=1)),
            "pool_wins": int((gap_loo > 0).sum()),
            "n_seeds": len(seeds),
        },
    }, indent=2))
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
