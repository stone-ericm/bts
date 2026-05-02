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
    corrected_pipeline_hi: float,
    fixed_point: float,
    fixed_lo: float,
    fixed_hi: float,
    headline: float,
) -> tuple[str, str]:
    """Apply hypothesis-test verdict gates calibrated for the trajectory count.

    Recalibrated 2026-05-02 after Codex review: the previous "CI lower ≥ 0.05"
    DEFENDED gate could not be achieved at n≈24 trajectories even for a true
    headline. The new gates use containment (DEFENDED) and 50% threshold (BROKEN)
    with an explicit INCONCLUSIVE category for cases where the data can't decide.

    Returns (verdict, rationale).
    """
    half_headline = 0.5 * headline

    # DEFENDED: corrected pipeline CI contains headline AND lower bound is at
    # or above half-headline. The lower-bound precision check rules out the
    # case where a CI is so wide it spans both the headline AND values in the
    # broken-territory range — which would be ambiguous, not defended. Without
    # this check, DEFENDED can win at low statistical power simply because the
    # CI is too wide to reject anything (false-defend).
    if (corrected_pipeline_lo <= headline <= corrected_pipeline_hi
        and corrected_pipeline_lo >= half_headline):
        return "HEADLINE_DEFENDED", (
            f"Corrected pipeline P(57) CI [{corrected_pipeline_lo:.4f}, "
            f"{corrected_pipeline_hi:.4f}] contains the headline {headline:.4f} "
            f"and lower bound {corrected_pipeline_lo:.4f} >= half-headline "
            f"{half_headline:.4f}; data is consistent with the claim under "
            f"correlation correction at adequate precision."
        )

    # BROKEN: corrected pipeline CI upper bound < 0.5 × headline (clear rejection).
    if corrected_pipeline_hi < half_headline:
        return "HEADLINE_BROKEN", (
            f"Corrected pipeline P(57) CI upper bound {corrected_pipeline_hi:.4f} "
            f"is below half the headline ({half_headline:.4f}); the 8.17% claim does "
            f"not survive honest cross-validated evaluation. Trigger a full rebuild "
            f"of the policy with the corrected transition tables."
        )

    # REDUCED: point estimate is clearly below half-headline AND CI doesn't contain.
    if corrected_pipeline_point < half_headline:
        return "HEADLINE_REDUCED", (
            f"Corrected pipeline P(57) point estimate {corrected_pipeline_point:.4f} "
            f"is below half the headline ({half_headline:.4f}); production policy may "
            f"still beat always-rank1 but the headline is partly artifact."
        )

    # INCONCLUSIVE: the data can't decide between defended and broken at this
    # statistical power level.
    return "HEADLINE_INCONCLUSIVE", (
        f"Corrected pipeline P(57) point estimate {corrected_pipeline_point:.4f} "
        f"with CI [{corrected_pipeline_lo:.4f}, {corrected_pipeline_hi:.4f}] does not "
        f"clearly defend or reject the headline {headline:.4f}. The trajectory count "
        f"may be insufficient at this statistical power; consider increasing the "
        f"effective n via season-month aggregation or running multi-season pipeline mode."
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
    # Build a 153-day season profile that respects each bin's empirical
    # frequency. Cycling through bins uniformly (bins[d % n_bins]) gave 1/5
    # days per bin regardless of the actual bin distribution; using qb.frequency
    # produces the right composition (e.g., 38 Q1 days, 30 Q2 days, etc.).
    total_days = 153
    days_per_bin = [int(round(qb.frequency * total_days)) for qb in bins_full.bins]
    # Adjust to ensure the sum is exactly total_days (rounding can drift by ±1-2).
    drift = sum(days_per_bin) - total_days
    if drift != 0:
        # Adjust the largest-frequency bin so it absorbs the rounding error.
        largest_idx = max(range(len(days_per_bin)), key=lambda i: bins_full.bins[i].frequency)
        days_per_bin[largest_idx] -= drift

    ceis_profiles = []
    for qb, n_days in zip(bins_full.bins, days_per_bin):
        for _ in range(n_days):
            ceis_profiles.append({"p_game": qb.p_hit, "p_both": qb.p_both})
    # Shuffle the day order. Without this, the profile is block-ordered by bin
    # (all Q1 days first, then all Q2, ...) which artificially suppresses streak
    # probability since high-prob days are concentrated at the end. A shuffled
    # mix matches a real season's day-to-day variation. Seeded for reproducibility.
    np.random.default_rng(42).shuffle(ceis_profiles)
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

    # Cross-game pair correlation: use the canonical (production) seed only.
    # Different seeds pick different rank-1/rank-2 batters, so averaging across
    # seeds produces fractional y values that break the permutation null. The
    # canonical seed gives one row per (season, date) with binary y values —
    # which is what pair_residual_correlation actually expects.
    canonical_seed_candidates = [42, 0]
    canonical_seed = next(
        (s for s in canonical_seed_candidates if s in profiles["seed"].unique()),
        int(profiles["seed"].iloc[0]),
    )
    canonical_profiles = profiles[profiles["seed"] == canonical_seed].copy()
    pair_df = canonical_profiles[
        ["date", "top1_p", "top1_hit", "top2_p", "top2_hit"]
    ].rename(columns={
        "top1_p": "p_rank1",
        "top1_hit": "y_rank1",
        "top2_p": "p_rank2",
        "top2_hit": "y_rank2",
    })
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
        corrected_result.ci_upper if corrected_result.ci_upper is not None else 1.0,
        fixed_result.point_estimate,
        fixed_result.ci_lower if fixed_result.ci_lower is not None else 0.0,
        fixed_result.ci_upper if fixed_result.ci_upper is not None else 1.0,
        headline_p57_in_sample,
    )

    out = {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "headline_p57_in_sample": headline_p57_in_sample,
        "fixed_policy_terminal_r_mc_p57": _format_estimate(
            fixed_result.point_estimate, fixed_result.ci_lower, fixed_result.ci_upper
        ),
        "pipeline_terminal_r_mc_p57": _format_estimate(
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
