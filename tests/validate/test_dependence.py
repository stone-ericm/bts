"""Tests for PA + cross-game dependence diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bts.validate.dependence import (
    pearson_residual,
    pa_residual_correlation,
    pair_residual_correlation,
)


class TestPearsonResidual:
    def test_residual_for_hit_with_predicted_p(self):
        # binary y=1, p=0.7
        # e = (1 - 0.7) / sqrt(0.7 * 0.3) = 0.3 / sqrt(0.21) ≈ 0.6547
        e = pearson_residual(1, 0.7)
        assert abs(e - 0.6547) < 1e-3

    def test_residual_for_miss_with_predicted_p(self):
        # binary y=0, p=0.7
        # e = (0 - 0.7) / sqrt(0.21) ≈ -1.5275
        e = pearson_residual(0, 0.7)
        assert abs(e - (-1.5275)) < 1e-3

    def test_residual_clips_extreme_p(self):
        # p very close to 0 or 1 shouldn't crash
        e = pearson_residual(1, 0.0)
        assert np.isfinite(e)
        e = pearson_residual(0, 1.0)
        assert np.isfinite(e)


class TestPAResidualCorrelation:
    def test_recovers_known_within_batter_correlation(self):
        """Synthetic PA data with known intra-batter-game correlation.

        Logistic-normal latent factor model: u ~ N(0, sigma_u^2) shifts each
        batter-game's log-odds. sigma_u=1.55 yields binary ICC ~0.30 (verified
        empirically; the logistic-normal mapping shrinks the observed binary ICC
        substantially relative to the latent parameter, so we use sigma_u directly
        rather than a rho_true/(1-rho_true) formula that targets the latent scale).
        """
        rng = np.random.default_rng(0)
        n_batter_games = 1000
        n_pa_per = 5
        sigma_u = 1.55  # calibrated to produce binary ICC ~0.30
        rho_target = 0.30
        rows = []
        for bg in range(n_batter_games):
            # Latent factor u induces within-batter-game correlation.
            u = rng.normal() * sigma_u
            for pa in range(n_pa_per):
                p_pred = 0.25
                logit_p = np.log(p_pred / (1 - p_pred)) + u
                p_realized = 1.0 / (1.0 + np.exp(-logit_p))
                y = int(rng.random() < p_realized)
                rows.append({"batter_game_id": bg, "pa_index": pa, "p_pa": p_pred, "actual_hit": y})
        df = pd.DataFrame(rows)
        rho_hat, ci_lo, ci_hi, p_value = pa_residual_correlation(df)
        # Recovered correlation should be within 0.10 of target rho (slack for small-n PA pairs).
        assert abs(rho_hat - rho_target) < 0.10, f"rho_hat={rho_hat:.3f} vs target={rho_target:.3f}"
        # Should be statistically significant from zero.
        assert p_value < 0.05

    def test_independence_yields_near_zero_correlation(self):
        """Independent PAs should produce rho_hat near zero."""
        rng = np.random.default_rng(1)
        n_batter_games = 500
        n_pa_per = 5
        rows = []
        for bg in range(n_batter_games):
            for pa in range(n_pa_per):
                p_pred = 0.25
                y = int(rng.random() < p_pred)
                rows.append({"batter_game_id": bg, "pa_index": pa, "p_pa": p_pred, "actual_hit": y})
        df = pd.DataFrame(rows)
        rho_hat, ci_lo, ci_hi, p_value = pa_residual_correlation(df)
        # Under independence, rho_hat should be near 0; CI should contain 0.
        assert abs(rho_hat) < 0.10, f"rho_hat={rho_hat:.3f} too far from 0 under independence"
        assert ci_lo <= 0 <= ci_hi, f"CI [{ci_lo:.3f}, {ci_hi:.3f}] should contain 0"


class TestLogisticNormalRandomIntercept:
    def test_recovers_known_tau(self):
        """Fit on synthetic data with known tau (latent Gaussian SD); recovered tau in same scale."""
        from bts.validate.dependence import fit_logistic_normal_random_intercept
        rng = np.random.default_rng(0)
        tau_true = 0.5
        n_groups = 500
        n_per = 5
        rows = []
        for g in range(n_groups):
            u = rng.normal(0, tau_true)
            for k in range(n_per):
                p_pred = 0.25
                logit_p_pred = np.log(p_pred / (1 - p_pred))
                p_realized = 1.0 / (1.0 + np.exp(-(logit_p_pred + u)))
                y = int(rng.random() < p_realized)
                rows.append({"group_id": g, "p_pred": p_pred, "y": y})
        df = pd.DataFrame(rows)
        tau_hat, integrate_fn = fit_logistic_normal_random_intercept(df)
        # Wide tolerance because the latent-to-observed transformation is noisy
        # at finite n. We mainly want to confirm the estimator is in the right
        # ballpark (recovers >0.2 tau when truth is 0.5) and doesn't blow up.
        assert 0.20 < tau_hat < 1.20, f"tau_hat={tau_hat:.3f}; expected 0.2-1.2 for tau_true=0.5"

    def test_integrate_fn_lowers_p_at_least_one(self):
        """integrate_fn(p_list) should return P(>=1 hit) lower than independence when tau>0."""
        from bts.validate.dependence import fit_logistic_normal_random_intercept
        rng = np.random.default_rng(0)
        tau_true = 0.8  # large enough that integrate_fn yields meaningfully different values
        n_groups = 500
        n_per = 5
        rows = []
        for g in range(n_groups):
            u = rng.normal(0, tau_true)
            for k in range(n_per):
                p_pred = 0.25
                logit_p_pred = np.log(p_pred / (1 - p_pred))
                p_realized = 1.0 / (1.0 + np.exp(-(logit_p_pred + u)))
                y = int(rng.random() < p_realized)
                rows.append({"group_id": g, "p_pred": p_pred, "y": y})
        df = pd.DataFrame(rows)
        tau_hat, integrate_fn = fit_logistic_normal_random_intercept(df)
        p_list = [0.25, 0.25, 0.25, 0.25, 0.25]
        p_at_least_one_corrected = integrate_fn(p_list)
        p_at_least_one_indep = 1.0 - 0.75**5  # ≈ 0.7626
        # Positive tau should reduce P(>=1 hit) below independence baseline.
        assert p_at_least_one_corrected < p_at_least_one_indep, (
            f"corrected={p_at_least_one_corrected:.4f} not below indep={p_at_least_one_indep:.4f}"
        )

    def test_zero_tau_returns_independence_aggregation(self):
        """When the data is iid (tau truly 0), integrate_fn should match independence."""
        from bts.validate.dependence import fit_logistic_normal_random_intercept
        rng = np.random.default_rng(1)
        n_groups = 500
        n_per = 5
        rows = []
        for g in range(n_groups):
            for k in range(n_per):
                p_pred = 0.25
                y = int(rng.random() < p_pred)
                rows.append({"group_id": g, "p_pred": p_pred, "y": y})
        df = pd.DataFrame(rows)
        tau_hat, integrate_fn = fit_logistic_normal_random_intercept(df)
        # tau_hat should be very small (truth is 0).
        assert tau_hat < 0.30, f"tau_hat={tau_hat:.3f} too large for tau_true=0 case"
        # integrate_fn at very small tau should return ≈ 1 - prod(1-p).
        p_list = [0.25, 0.25, 0.25, 0.25, 0.25]
        p_at_least_one = integrate_fn(p_list)
        p_indep = 1.0 - 0.75**5
        assert abs(p_at_least_one - p_indep) < 0.05


class TestBuildCorrectedTransitionTable:
    def test_zero_dependence_preserves_p_hit_and_returns_independence_p_both(self):
        """When rho_PA = tau = rho_pair = 0, p_hit unchanged AND p_both = p1*p2.

        Note: with the Codex-review fix to use Pearson reconstruction (rather than
        empirical-additive), the corrected p_both at zero correlation equals the
        independence product p1*p2, not the original empirical b.p_both.
        """
        from bts.validate.dependence import build_corrected_transition_table
        from bts.simulate.quality_bins import QualityBins, QualityBin
        bins = QualityBins(
            bins=[
                QualityBin(index=0, p_range=(0.7, 0.8), p_hit=0.75, p_both=0.55, frequency=1.0),
            ],
            boundaries=[],
        )
        corrected = build_corrected_transition_table(
            bins,
            rho_PA_within_game=0.0,
            tau_squared=0.0,
            rho_pair_cross_game=0.0,
            n_pa_per_game=5,
        )
        assert abs(corrected.bins[0].p_hit - 0.75) < 1e-9
        # p_both = p1*p2 at independence (synthetic from Pearson reconstruction).
        assert abs(corrected.bins[0].p_both - 0.5625) < 1e-9
        assert corrected.bins[0].index == 0
        assert corrected.bins[0].frequency == 1.0

    def test_positive_pa_dependence_lowers_p_hit(self):
        """Within-game positive correlation lowers P(at least one hit per game).

        This effect holds when p_hit is large enough that the kernel
        g(u) = (1 - sigmoid(logit(p_pa) + u))^n is concave and the Jensen
        correction pushes E[1-g(U)] below 1-g(0). At p_hit=0.70 with n=5 and
        tau_squared=0.5 the direction is reliably downward (delta ≈ -0.020).
        """
        from bts.validate.dependence import build_corrected_transition_table
        from bts.simulate.quality_bins import QualityBins, QualityBin
        bins = QualityBins(
            bins=[
                QualityBin(index=0, p_range=(0.65, 0.75), p_hit=0.70, p_both=0.45, frequency=1.0),
            ],
            boundaries=[],
        )
        corrected = build_corrected_transition_table(
            bins,
            rho_PA_within_game=0.20,  # included in result for caller record-keeping
            tau_squared=0.5,
            rho_pair_cross_game=0.0,
            n_pa_per_game=5,
        )
        assert corrected.bins[0].p_hit < 0.70, (
            f"PA positive dependence should lower p_hit; got {corrected.bins[0].p_hit:.4f}"
        )
        # Should still be a valid probability.
        assert 0.0 < corrected.bins[0].p_hit < 1.0

    def test_positive_pair_dependence_raises_p_both(self):
        """Cross-game positive correlation raises P(both hit)."""
        from bts.validate.dependence import build_corrected_transition_table
        from bts.simulate.quality_bins import QualityBins, QualityBin
        bins = QualityBins(
            bins=[
                QualityBin(index=0, p_range=(0.7, 0.8), p_hit=0.75, p_both=0.55, frequency=1.0),
            ],
            boundaries=[],
        )
        corrected = build_corrected_transition_table(
            bins,
            rho_PA_within_game=0.0,
            tau_squared=0.0,
            rho_pair_cross_game=0.15,
            n_pa_per_game=5,
        )
        assert corrected.bins[0].p_both > 0.55, (
            f"pair positive dependence should raise p_both; got {corrected.bins[0].p_both:.4f}"
        )
        # Frechet upper bound: min(p1, p2) = 0.75. Should be <= that.
        assert corrected.bins[0].p_both <= 0.75

    def test_build_corrected_transition_table_accepts_per_bin_rho_vector(self):
        """Passing a length-K rho_pair vector applies per-bin correction."""
        from bts.validate.dependence import build_corrected_transition_table
        from bts.simulate.quality_bins import QualityBins, QualityBin
        bins = QualityBins(
            bins=[
                QualityBin(index=0, p_range=(0.0, 0.5), p_hit=0.3, p_both=0.10, frequency=0.2),
                QualityBin(index=1, p_range=(0.5, 0.6), p_hit=0.55, p_both=0.30, frequency=0.2),
                QualityBin(index=2, p_range=(0.6, 0.7), p_hit=0.65, p_both=0.42, frequency=0.2),
                QualityBin(index=3, p_range=(0.7, 0.8), p_hit=0.75, p_both=0.55, frequency=0.2),
                QualityBin(index=4, p_range=(0.8, 1.0), p_hit=0.85, p_both=0.72, frequency=0.2),
            ],
            boundaries=[0.0, 0.5, 0.6, 0.7, 0.8, 1.0],
        )
        rho_per_bin = np.array([0.0, 0.0, 0.0, -0.10, +0.05])  # Q4 negative, Q5 positive
        corrected = build_corrected_transition_table(
            bins,
            rho_PA_within_game=0.0,
            tau_squared=0.0,
            rho_pair_cross_game=rho_per_bin,
            n_pa_per_game=5,
        )
        # Q1-Q3 use rho=0 → p_both should equal p1*p2 (within FH bounds).
        for i in [0, 1, 2]:
            b_orig = bins.bins[i]
            b_new = corrected.bins[i]
            expected_pboth = b_orig.p_hit ** 2
            assert abs(b_new.p_both - expected_pboth) < 1e-9, (
                f"Q{i+1}: rho=0 should give p_both = p_hit^2"
            )
        # Q4 uses rho=-0.10 → p_both should be BELOW p1*p2.
        b4_orig = bins.bins[3]
        b4_new = corrected.bins[3]
        p1 = p2 = b4_orig.p_hit
        sigma = np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
        expected_q4 = p1 * p2 + (-0.10) * sigma
        expected_q4 = max(0.0, p1 + p2 - 1.0) if expected_q4 < max(0.0, p1 + p2 - 1.0) else expected_q4
        expected_q4 = min(p1, p2) if expected_q4 > min(p1, p2) else expected_q4
        assert abs(b4_new.p_both - expected_q4) < 1e-9
        # Q5 uses rho=+0.05 → p_both should be ABOVE p1*p2.
        b5_orig = bins.bins[4]
        b5_new = corrected.bins[4]
        p1 = p2 = b5_orig.p_hit
        sigma = np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
        expected_q5 = p1 * p2 + 0.05 * sigma
        # Apply Frechet-Hoeffding clipping (small positive rho probably won't trigger,
        # but parallels the Q4 logic and locks down the formula).
        lower_fh = max(0.0, p1 + p2 - 1.0)
        upper_fh = min(p1, p2)
        expected_q5 = float(min(max(expected_q5, lower_fh), upper_fh))
        assert abs(b5_new.p_both - expected_q5) < 1e-9

    def test_build_corrected_transition_table_rejects_non_contiguous_bin_indices(self):
        """When bins have non-contiguous indices (e.g., index=0,1,3,4 from
        compute_bins skipping an empty group), per-bin rho input must raise
        ValueError instead of silently mis-assigning."""
        from bts.validate.dependence import build_corrected_transition_table
        from bts.simulate.quality_bins import QualityBins, QualityBin
        sparse_bins = QualityBins(
            bins=[
                QualityBin(index=0, p_range=(0.0, 0.5), p_hit=0.3, p_both=0.10, frequency=0.25),
                QualityBin(index=1, p_range=(0.5, 0.7), p_hit=0.6, p_both=0.36, frequency=0.25),
                QualityBin(index=3, p_range=(0.7, 0.85), p_hit=0.78, p_both=0.61, frequency=0.25),
                QualityBin(index=4, p_range=(0.85, 1.0), p_hit=0.9, p_both=0.81, frequency=0.25),
            ],
            boundaries=[0.0, 0.5, 0.7, 0.85, 1.0],
        )
        rho_per_bin = np.array([0.0, 0.0, 0.0, 0.0])  # length 4, matches len(bins.bins)
        with pytest.raises(ValueError, match="contiguous 0-based"):
            build_corrected_transition_table(
                sparse_bins,
                rho_PA_within_game=0.0,
                tau_squared=0.0,
                rho_pair_cross_game=rho_per_bin,
                n_pa_per_game=5,
            )
        # Scalar input on the same sparse bins should still work (no per-bin lookup).
        result = build_corrected_transition_table(
            sparse_bins,
            rho_PA_within_game=0.0,
            tau_squared=0.0,
            rho_pair_cross_game=0.05,
            n_pa_per_game=5,
        )
        assert len(result.bins) == 4

    def test_build_corrected_transition_table_scalar_input_still_works(self):
        """Backward compat: scalar rho_pair_cross_game applies to all bins."""
        from bts.validate.dependence import build_corrected_transition_table
        from bts.simulate.quality_bins import QualityBins, QualityBin
        bins = QualityBins(
            bins=[QualityBin(index=i, p_range=(i*0.2, (i+1)*0.2), p_hit=0.5, p_both=0.25, frequency=0.2) for i in range(5)],
            boundaries=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        corrected = build_corrected_transition_table(
            bins,
            rho_PA_within_game=0.0,
            tau_squared=0.0,
            rho_pair_cross_game=0.05,  # scalar
            n_pa_per_game=5,
        )
        for b in corrected.bins:
            assert abs(b.p_both - (0.5*0.5 + 0.05*0.25)) < 1e-9

    def test_build_corrected_transition_table_per_bin_rho_uses_post_tau_marginal(self):
        """REGRESSION GUARD: when both tau>0 AND per-bin rho!=0, the rho correction
        must use the tau-corrected p_hit (new_p_hit), NOT the original b.p_hit.

        v1 had this bug; v2 fixes it. This test would catch a revert.
        """
        from bts.validate.dependence import build_corrected_transition_table
        from bts.simulate.quality_bins import QualityBins, QualityBin
        bins = QualityBins(
            bins=[
                QualityBin(index=i, p_range=(i*0.2, (i+1)*0.2), p_hit=0.6, p_both=0.36, frequency=0.2)
                for i in range(5)
            ],
            boundaries=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        # Tau > 0 will pull new_p_hit BELOW 0.6 (within-game dependence inflates
        # game-level hit when integrated). Per-bin rho lookup of 0.10 in every bin.
        rho_per_bin = np.array([0.10] * 5)
        corrected = build_corrected_transition_table(
            bins,
            rho_PA_within_game=0.0,
            tau_squared=0.5,  # nonzero — exercises the tau path
            rho_pair_cross_game=rho_per_bin,
            n_pa_per_game=5,
        )
        # For each bin, the corrected p_both must equal new_p_hit^2 + rho * sigma(new_p_hit),
        # NOT b.p_hit^2 + rho * sigma(b.p_hit).
        for i, b_new in enumerate(corrected.bins):
            new_p_hit = b_new.p_hit  # the tau-corrected marginal
            # Verify tau actually moved the marginal (sanity check the test setup).
            assert abs(new_p_hit - 0.6) > 0.001, (
                f"Q{i+1}: tau=0.5 should have shifted p_hit from 0.6 (got {new_p_hit:.6f})"
            )
            # Now verify p_both uses new_p_hit, not 0.6.
            sigma_post_tau = np.sqrt(new_p_hit * (1 - new_p_hit) * new_p_hit * (1 - new_p_hit))
            expected_post_tau = new_p_hit * new_p_hit + 0.10 * sigma_post_tau
            # Apply FH clipping
            lower_fh = max(0.0, 2 * new_p_hit - 1.0)
            upper_fh = min(new_p_hit, new_p_hit)  # = new_p_hit
            expected_post_tau = float(min(max(expected_post_tau, lower_fh), upper_fh))

            # The bug-version would compute against b.p_hit=0.6
            sigma_pre_tau = np.sqrt(0.6 * 0.4 * 0.6 * 0.4)
            bug_value = 0.6 * 0.6 + 0.10 * sigma_pre_tau
            # Both expected values should differ — otherwise the test isn't sensitive.
            assert abs(expected_post_tau - bug_value) > 1e-6, (
                f"Q{i+1}: test setup degenerate — pre-tau and post-tau values "
                f"are too close ({expected_post_tau:.6f} vs {bug_value:.6f})"
            )
            assert abs(b_new.p_both - expected_post_tau) < 1e-9, (
                f"Q{i+1}: corrected p_both ({b_new.p_both:.6f}) should match "
                f"post-tau formula ({expected_post_tau:.6f}), NOT pre-tau ({bug_value:.6f})"
            )

    def test_build_corrected_transition_table_per_bin_rho_extreme_values_clip_to_FH_bounds(self):
        """rho=-1 and rho=+1 push p_both outside [max(0, p1+p2-1), min(p1,p2)];
        output must clip to those bounds for valid joint probabilities.
        """
        from bts.validate.dependence import build_corrected_transition_table
        from bts.simulate.quality_bins import QualityBins, QualityBin
        # Use p_hit=0.5 (max sigma=0.25, so rho=-1 sends p_both to 0.0).
        bins = QualityBins(
            bins=[
                QualityBin(index=i, p_range=(i*0.2, (i+1)*0.2), p_hit=0.5, p_both=0.25, frequency=0.2)
                for i in range(5)
            ],
            boundaries=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )
        # Q1 rho=-1 (max negative), Q5 rho=+1 (max positive), middle bins zero.
        rho_per_bin = np.array([-1.0, 0.0, 0.0, 0.0, +1.0])
        corrected = build_corrected_transition_table(
            bins,
            rho_PA_within_game=0.0,
            tau_squared=0.0,
            rho_pair_cross_game=rho_per_bin,
            n_pa_per_game=5,
        )
        # Q1 (rho=-1): formula gives 0.25 + (-1)(0.25) = 0.0. FH lower = max(0, 0.5+0.5-1) = 0.0.
        # Result: 0.0 (right at lower bound, no clipping needed but still valid).
        assert abs(corrected.bins[0].p_both - 0.0) < 1e-9
        # Q5 (rho=+1): formula gives 0.25 + 1*0.25 = 0.5. FH upper = min(0.5, 0.5) = 0.5.
        # Result: 0.5 (at upper bound).
        assert abs(corrected.bins[4].p_both - 0.5) < 1e-9
        # Validity: every p_both must be in [0, 1] AND in FH bounds.
        for b in corrected.bins:
            assert 0.0 <= b.p_both <= 1.0
            p1 = p2 = b.p_hit
            assert b.p_both >= max(0.0, p1 + p2 - 1.0) - 1e-9
            assert b.p_both <= min(p1, p2) + 1e-9


class TestPairResidualCorrelation:
    def test_independent_pairs_yield_nonsignificant_correlation(self):
        """If rank1 and rank2 are independent, the test should not reject H0 most of the time."""
        rng = np.random.default_rng(0)
        n_pairs = 100
        rows = []
        for t in range(n_pairs):
            p1, p2 = 0.75, 0.70
            y1 = int(rng.random() < p1)
            y2 = int(rng.random() < p2)
            rows.append({
                "date": t, "p_rank1": p1, "p_rank2": p2,
                "y_rank1": y1, "y_rank2": y2,
            })
        df = pd.DataFrame(rows)
        rho_hat, ci_lo, ci_hi, p_value = pair_residual_correlation(df, n_permutations=500)
        assert abs(rho_hat) < 0.20, f"rho_hat={rho_hat:.3f} too far from 0 under independence"

    def test_correlated_pairs_detected(self):
        """If rank1 and rank2 share a latent slate factor, test should detect."""
        rng = np.random.default_rng(0)
        n_pairs = 200
        rows = []
        for t in range(n_pairs):
            u = rng.normal()  # slate factor shared by both picks on day t
            p1, p2 = 0.75, 0.70
            # sigma=1.2 is a strong-but-plausible slate effect; sigma=0.8 lands
            # rho_hat right at the detection boundary for this seed and is not
            # reliably distinguishable from zero at n=200.
            logit_p1 = np.log(p1 / (1 - p1)) + 1.2 * u
            logit_p2 = np.log(p2 / (1 - p2)) + 1.2 * u
            y1 = int(rng.random() < 1.0 / (1.0 + np.exp(-logit_p1)))
            y2 = int(rng.random() < 1.0 / (1.0 + np.exp(-logit_p2)))
            rows.append({
                "date": t, "p_rank1": p1, "p_rank2": p2,
                "y_rank1": y1, "y_rank2": y2,
            })
        df = pd.DataFrame(rows)
        rho_hat, ci_lo, ci_hi, p_value = pair_residual_correlation(df, n_permutations=500)
        # Latent factor with sigma=1.2 and n=200 reliably produces rho_hat in 0.10-0.20.
        assert rho_hat > 0.04, f"rho_hat={rho_hat:.3f} not detecting positive correlation"
        # The permutation test should show p_value < 0.10 at this signal strength.
        assert p_value < 0.10, f"p_value={p_value:.3f} above significance under positive correlation"

    def test_pair_residual_correlation_per_bin_returns_dict_with_correct_shapes(self):
        """Per-bin rho returns shape-(K,) arrays for K unique bins."""
        rng = np.random.default_rng(42)
        n = 500
        df = pd.DataFrame({
            "p_rank1": rng.uniform(0.5, 0.95, n),
            "y_rank1": rng.binomial(1, 0.7, n),
            "p_rank2": rng.uniform(0.4, 0.85, n),
            "y_rank2": rng.binomial(1, 0.6, n),
        })
        bin_assignment = pd.Series(rng.integers(0, 5, n))  # 5 bins

        result = pair_residual_correlation(
            df, n_permutations=100, bin_assignment=bin_assignment,
        )

        assert isinstance(result, dict)
        assert result["rho_per_bin"].shape == (5,)
        assert result["ci_lo_per_bin"].shape == (5,)
        assert result["ci_hi_per_bin"].shape == (5,)
        assert result["p_value_per_bin"].shape == (5,)
        assert result["n_per_bin"].shape == (5,)
        assert isinstance(result["global_rho"], float)
        # Per-bin n_per_bin should sum to total
        assert int(result["n_per_bin"].sum()) == n
        np.testing.assert_array_equal(np.sort(result["bin_indices"]), np.arange(5))

    def test_pair_residual_correlation_scalar_path_preserved_for_back_compat(self):
        """When bin_assignment is None, return tuple (rho, ci_lo, ci_hi, p) as before."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "p_rank1": rng.uniform(0.5, 0.95, 200),
            "y_rank1": rng.binomial(1, 0.7, 200),
            "p_rank2": rng.uniform(0.4, 0.85, 200),
            "y_rank2": rng.binomial(1, 0.6, 200),
        })
        result = pair_residual_correlation(df, n_permutations=100)
        assert isinstance(result, tuple)
        assert len(result) == 4
        rho, ci_lo, ci_hi, p = result
        assert isinstance(rho, float)

    def test_pair_residual_correlation_missing_bin_returns_zero_at_correct_index(self):
        """CRITICAL: when bin 2 is missing from data but expected_bin_indices=[0..4],
        output[2] is the empty-bin slot — NOT the next observed bin's value.

        This protects against the silent indexing-shift bug where bin labels in
        output don't match bin labels the consumer indexes by.
        """
        rng = np.random.default_rng(42)
        n = 400
        # Build df where rank-1 bin is in {0,1,3,4} (no bin 2).
        df = pd.DataFrame({
            "p_rank1": rng.uniform(0.5, 0.95, n),
            "y_rank1": rng.binomial(1, 0.7, n),
            "p_rank2": rng.uniform(0.4, 0.85, n),
            "y_rank2": rng.binomial(1, 0.6, n),
        })
        bin_assignment = pd.Series(rng.choice([0, 1, 3, 4], n))

        result = pair_residual_correlation(
            df, n_permutations=50,
            bin_assignment=bin_assignment,
            expected_bin_indices=np.arange(5),  # [0,1,2,3,4]
        )
        # Output is shape-(5,) with bin 2 empty.
        assert result["rho_per_bin"].shape == (5,)
        assert result["n_per_bin"][2] == 0  # bin 2 absent
        assert result["rho_per_bin"][2] == 0.0  # empty bin → rho=0
        assert result["p_value_per_bin"][2] == 1.0  # empty bin → p=1
        # Bin 2 absent — verify ALL empty-bin defaults, not just rho/n.
        assert result["ci_lo_per_bin"][2] == 0.0
        assert result["ci_hi_per_bin"][2] == 0.0
        # Bins 0, 1, 3, 4 should have non-zero counts.
        for k in [0, 1, 3, 4]:
            assert result["n_per_bin"][k] > 0
        # bin_indices should match what was passed in.
        np.testing.assert_array_equal(result["bin_indices"], np.arange(5))

    def test_pair_residual_correlation_strict_label_validation_raises(self):
        """When bin_assignment contains labels outside expected_bin_indices,
        strict_bin_labels=True (default) raises ValueError; False allows drop."""
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "p_rank1": rng.uniform(0.5, 0.95, n),
            "y_rank1": rng.binomial(1, 0.7, n),
            "p_rank2": rng.uniform(0.4, 0.85, n),
            "y_rank2": rng.binomial(1, 0.6, n),
        })
        # Data has bin label 7, but expected only includes [0..4].
        bin_assignment = pd.Series([7] * n)

        # Default strict mode raises.
        with pytest.raises(ValueError, match="not in expected_bin_indices"):
            pair_residual_correlation(
                df, n_permutations=10,
                bin_assignment=bin_assignment,
                expected_bin_indices=np.arange(5),
            )

        # Opt-out allows silent drop.
        result = pair_residual_correlation(
            df, n_permutations=10,
            bin_assignment=bin_assignment,
            expected_bin_indices=np.arange(5),
            strict_bin_labels=False,
        )
        # All output bins are empty (label 7 was dropped).
        assert result["n_per_bin"].sum() == 0
