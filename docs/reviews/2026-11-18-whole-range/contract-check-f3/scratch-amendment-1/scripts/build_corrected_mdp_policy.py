"""Build a corrected MDP policy from Task 13's dependence-correction findings.

Acts on the falsification harness's HEADLINE_BROKEN verdict by producing
a corrected policy artifact that the production scheduler could (after
review) load instead of the current `data/models/mdp_policy.npz`.

DOES NOT replace the existing production policy — only produces a new file
`data/models/mdp_policy_corrected_<DATE>.npz` for inspection / future deploy.

Workflow:
  1. Load all profiles + PA data from data/simulation/profiles_seed*_season*
     (already pivoted to harness format by scripts/task13_merge.py).
  2. Compute headline bins (uncorrected baseline for comparison).
  3. Run dependence diagnostics (rho_PA, tau, rho_pair).
  4. Build corrected QualityBins via build_corrected_transition_table.
  5. Solve MDP on corrected bins.
  6. Save policy to data/models/mdp_policy_corrected_<DATE>.npz.
  7. Print summary comparing headline vs corrected.

Usage:
  UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/build_corrected_mdp_policy.py
"""
from __future__ import annotations

import glob
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.mdp import solve_mdp
from bts.validate.dependence import (
    build_corrected_transition_table,
    fit_logistic_normal_random_intercept,
    pa_residual_correlation,
    pair_residual_correlation,
)
from bts.validate.ope import _compute_bins_from_direct_profiles


def main() -> None:
    print("=== Building corrected MDP policy from Task 13 data ===\n")

    profiles_paths = sorted(glob.glob("data/simulation/profiles_seed*_season*.parquet"))
    pa_paths = sorted(glob.glob("data/simulation/pa_predictions_seed*_season*.parquet"))
    if not profiles_paths or not pa_paths:
        raise RuntimeError(
            "No profile / PA parquets found. Run scripts/task13_merge.py first "
            "(which converts data/simulation_seedN/ outputs into the pivoted "
            "harness-ready format)."
        )
    print(f"Loading {len(profiles_paths)} profile parquets + {len(pa_paths)} PA parquets...")
    profiles = pd.concat(pd.read_parquet(p) for p in profiles_paths)
    pa_df = pd.concat(pd.read_parquet(p) for p in pa_paths)
    print(f"  profiles: {len(profiles):,} rows × {len(profiles.columns)} cols")
    print(f"  pa_df:    {len(pa_df):,} rows × {len(pa_df.columns)} cols\n")

    # Step 1: headline bins (uncorrected baseline).
    print("Computing headline bins (uncorrected)...")
    headline_bins = _compute_bins_from_direct_profiles(profiles)
    headline_solution = solve_mdp(headline_bins, season_length=153, late_phase_days=30)
    print(f"  headline P(57) in-sample: {headline_solution.optimal_p57:.4f}\n")

    # Step 2: dependence diagnostics.
    print("Running dependence diagnostics (n_bootstrap=300, fast post-perf-fix)...")
    rho_PA, rho_PA_lo, rho_PA_hi, _ = pa_residual_correlation(pa_df, n_bootstrap=300)
    print(f"  rho_PA_within_game: {rho_PA:.4f} [{rho_PA_lo:.4f}, {rho_PA_hi:.4f}]")

    pa_for_lnri = pa_df.rename(columns={
        "batter_game_id": "group_id",
        "p_pa": "p_pred",
        "actual_hit": "y",
    })
    tau_hat, _, _ = fit_logistic_normal_random_intercept(pa_for_lnri)
    print(f"  tau_hat: {tau_hat:.4f}  (tau_squared: {tau_hat**2:.6f})")

    # Pair correlation: use canonical seed 42 (or 0).
    canonical_seed = next(
        (s for s in [42, 0] if s in profiles["seed"].unique()),
        int(profiles["seed"].iloc[0]),
    )
    canonical_profiles = profiles[profiles["seed"] == canonical_seed].copy()
    pair_df = canonical_profiles[
        ["date", "top1_p", "top1_hit", "top2_p", "top2_hit"]
    ].rename(columns={
        "top1_p": "p_rank1", "top1_hit": "y_rank1",
        "top2_p": "p_rank2", "top2_hit": "y_rank2",
    })
    rho_pair, rho_pair_lo, rho_pair_hi, _ = pair_residual_correlation(pair_df, n_permutations=300)
    print(f"  rho_pair_cross_game: {rho_pair:.4f} [{rho_pair_lo:.4f}, {rho_pair_hi:.4f}]\n")

    # Step 3: corrected bins + MDP.
    print("Building corrected transition table + re-solving MDP...")
    corrected_bins = build_corrected_transition_table(
        headline_bins,
        rho_PA_within_game=rho_PA,
        tau_squared=tau_hat ** 2,
        rho_pair_cross_game=rho_pair,
        n_pa_per_game=5,
    )
    corrected_solution = solve_mdp(corrected_bins, season_length=153, late_phase_days=30)
    print(f"  corrected P(57) in-sample: {corrected_solution.optimal_p57:.4f}")
    print(f"  delta vs headline:         {corrected_solution.optimal_p57 - headline_solution.optimal_p57:+.4f}\n")

    # Step 4: bin-by-bin comparison.
    print("Bin-by-bin comparison (p_hit, p_both):")
    print(f"  {'bin':>4} {'p_hit (orig→corr)':>22} {'p_both (orig→corr)':>22} {'freq':>8}")
    for h_bin, c_bin in zip(headline_bins.bins, corrected_bins.bins):
        print(
            f"  Q{h_bin.index+1:>2d}  "
            f"{h_bin.p_hit:.4f} → {c_bin.p_hit:.4f}  "
            f"({c_bin.p_hit - h_bin.p_hit:+.4f})       "
            f"{h_bin.p_both:.4f} → {c_bin.p_both:.4f}  "
            f"({c_bin.p_both - h_bin.p_both:+.4f})       "
            f"{h_bin.frequency:.4f}"
        )
    print()

    # Step 5: save corrected policy.
    out_path = Path(f"data/models/mdp_policy_corrected_{date.today().isoformat()}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corrected_solution.save(out_path)
    print(f"Wrote corrected policy: {out_path}")
    print(f"  size: {out_path.stat().st_size:,} bytes\n")

    # Step 6: deploy advisory.
    print("=== Deploy advisory ===")
    print("This script does NOT replace data/models/mdp_policy.npz.")
    print("Production deploy is a manual decision — the harness verdict was")
    print("HEADLINE_BROKEN (corrected_pipeline_p57 = 0.83% [0, 3.75%] vs the")
    print(f"headline 8.17%). The corrected policy here would deflate the")
    print(f"projected in-sample P(57) by {corrected_solution.optimal_p57 - headline_solution.optimal_p57:+.4f}.")
    print("")
    print("To deploy on bts-hetzner (after review):")
    print(f"  scp {out_path} root@bts-hetzner:/root/projects/bts/data/models/mdp_policy.npz")
    print(f"  ssh root@bts-hetzner 'systemctl restart bts-scheduler'  # bts user unit")
    print()
    print("Or stage as parallel policy via shadow_policy config (no production swap).")


if __name__ == "__main__":
    main()
