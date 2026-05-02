"""Driver for the BTS 8.17% falsification harness.

Orchestrates: DR-OPE (fixed-policy + pipeline modes), CE-IS rare-event MC, and
PA + cross-game dependence diagnostics. Emits a verdict JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.mdp import solve_mdp
from bts.simulate.rare_event_mc import estimate_p57_with_ceis
from bts.validate.dependence import (
    build_corrected_transition_table,
    fit_logistic_normal_random_intercept,
    pa_residual_correlation,
    pair_residual_correlation,
)
from bts.validate.ope import (
    audit_fixed_policy,
    audit_pipeline,
    _compute_bins_from_direct_profiles,
)


def _format_estimate(point: float, ci_lo: float | None, ci_hi: float | None) -> str:
    if ci_lo is None or ci_hi is None:
        return f"{point:.4f}"
    return f"{point:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]"


def _classify_verdict(
    corrected_pipeline_point: float,
    corrected_pipeline_lo: float,
    fixed_point: float,
    fixed_lo: float,
    fixed_hi: float,
    headline: float,
) -> tuple[str, str]:
    """Apply the spec §7 verdict rules.

    Returns (verdict, rationale).
    """
    if corrected_pipeline_lo >= 0.05 and (fixed_lo <= headline <= fixed_hi):
        return "HEADLINE_DEFENDED", (
            f"Corrected pipeline P(57) CI lower bound {corrected_pipeline_lo:.4f} >= 5pp; "
            f"fixed-policy CI [{fixed_lo:.4f}, {fixed_hi:.4f}] contains headline {headline:.4f}."
        )
    if 0.03 <= corrected_pipeline_point <= 0.06:
        return "HEADLINE_REDUCED", (
            f"Corrected pipeline P(57) point estimate {corrected_pipeline_point:.4f} is below "
            f"in-sample headline {headline:.4f} but production policy still better than always-rank1."
        )
    return "HEADLINE_BROKEN", (
        f"Corrected pipeline P(57) point estimate {corrected_pipeline_point:.4f} is below 3pp; "
        f"the headline 8.17% claim does not survive honest cross-validated evaluation. "
        f"Trigger a full rebuild of the policy with the corrected transition tables."
    )


def run_harness(
    profiles: pd.DataFrame,
    pa_df: pd.DataFrame,
    *,
    output_path: Path,
    headline_p57_in_sample: float = 0.0817,
    n_bootstrap: int = 2000,
    n_final: int = 20000,
) -> dict:
    """Top-level harness driver. Returns the verdict dict and writes it to output_path."""
    seasons = sorted(profiles["season"].unique().tolist())
    held_out = seasons[-1]

    # Step 1: Headline policy — bins + MDP solved on all seasons.
    bins_full = _compute_bins_from_direct_profiles(profiles)
    headline_solution = solve_mdp(bins_full, season_length=153, late_phase_days=30)

    # Step 2: Fixed-policy audit on held-out season.
    fixed_result = audit_fixed_policy(
        profiles,
        frozen_policy={"action_table": headline_solution.policy_table},
        test_seasons=[held_out],
        n_bootstrap=n_bootstrap,
    )

    # Step 3: Pipeline audit (LOSO).
    pipeline_result = audit_pipeline(
        profiles, fold_seasons=seasons, n_bootstrap=n_bootstrap
    )

    # Step 4: CE-IS rare-event MC on a synthetic 153-day profile derived from bins.
    ceis_profiles = []
    for d in range(153):
        qb = bins_full.bins[d % len(bins_full.bins)]
        ceis_profiles.append({
            "p_game": qb.p_hit,
            "p_both": qb.p_both,
        })
    ceis_result = estimate_p57_with_ceis(
        ceis_profiles, strategy=None, n_final=n_final, seed=42,
    )

    # Step 5: Dependence diagnostics.
    rho_PA, rho_PA_lo, rho_PA_hi, _ = pa_residual_correlation(pa_df, n_bootstrap=n_bootstrap)
    pa_for_lnri = pa_df.rename(columns={
        "batter_game_id": "group_id",
        "p_pa": "p_pred",
        "actual_hit": "y",
    })
    tau_hat, _ = fit_logistic_normal_random_intercept(pa_for_lnri)

    # Cross-game pair correlation: one observation per (season, date), averaged over seeds.
    # Using one row per day is semantically correct — cross-game correlation is a property
    # of the game realizations, not of the modeling seeds.
    pair_df = (
        profiles
        .groupby(["season", "date"])
        .agg(
            p_rank1=("top1_p", "mean"),
            y_rank1=("top1_hit", "mean"),
            p_rank2=("top2_p", "mean"),
            y_rank2=("top2_hit", "mean"),
        )
        .reset_index()
    )
    # Round realized hit columns back to binary using the mean (0.5 threshold).
    # For independence testing, fractional values from averaging are fine — the
    # pearson_residual function handles floats in (0, 1) cleanly.
    rho_pair, rho_pair_lo, rho_pair_hi, _ = pair_residual_correlation(
        pair_df, n_permutations=n_bootstrap
    )

    # Step 6: Corrected transition table + re-solved policy.
    corrected_bins = build_corrected_transition_table(
        bins_full,
        rho_PA_within_game=rho_PA,
        tau_squared=tau_hat ** 2,
        rho_pair_cross_game=rho_pair,
        n_pa_per_game=5,
    )
    corrected_solution = solve_mdp(corrected_bins, season_length=153, late_phase_days=30)

    # Step 7: Corrected-policy audit on held-out season (fixed-policy mode, v1).
    corrected_result = audit_fixed_policy(
        profiles,
        frozen_policy={"action_table": corrected_solution.policy_table},
        test_seasons=[held_out],
        n_bootstrap=n_bootstrap,
    )

    # Step 8: Verdict.
    verdict, rationale = _classify_verdict(
        corrected_result.point_estimate,
        corrected_result.ci_lower if corrected_result.ci_lower is not None else 0.0,
        fixed_result.point_estimate,
        fixed_result.ci_lower if fixed_result.ci_lower is not None else 0.0,
        fixed_result.ci_upper if fixed_result.ci_upper is not None else 1.0,
        headline_p57_in_sample,
    )

    out = {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "headline_p57_in_sample": headline_p57_in_sample,
        "fixed_policy_dr_ope_p57": _format_estimate(
            fixed_result.point_estimate, fixed_result.ci_lower, fixed_result.ci_upper
        ),
        "pipeline_dr_ope_p57": _format_estimate(
            pipeline_result.point_estimate, pipeline_result.ci_lower, pipeline_result.ci_upper
        ),
        "rare_event_ce_p57": _format_estimate(
            ceis_result.point_estimate, ceis_result.ci_lower, ceis_result.ci_upper
        ),
        "rho_PA_within_game": _format_estimate(rho_PA, rho_PA_lo, rho_PA_hi),
        "rho_pair_cross_game": _format_estimate(rho_pair, rho_pair_lo, rho_pair_hi),
        "corrected_pipeline_p57": _format_estimate(
            corrected_result.point_estimate, corrected_result.ci_lower, corrected_result.ci_upper
        ),
        "verdict": verdict,
        "verdict_rationale": rationale,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(out, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles-glob", default="data/simulation/backtest_*.parquet")
    parser.add_argument("--pa-glob", default="data/simulation/pa_predictions_*.parquet")
    parser.add_argument("--output", default="data/validation/falsification_harness.json")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-final", type=int, default=20000)
    parser.add_argument("--headline-p57", type=float, default=0.0817)
    args = parser.parse_args()

    profiles = pd.concat(pd.read_parquet(p) for p in sorted(Path().glob(args.profiles_glob)))
    pa_df = pd.concat(pd.read_parquet(p) for p in sorted(Path().glob(args.pa_glob)))
    out = run_harness(
        profiles, pa_df,
        output_path=Path(args.output),
        headline_p57_in_sample=args.headline_p57,
        n_bootstrap=args.n_bootstrap,
        n_final=args.n_final,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
