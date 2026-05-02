"""Tests for PA + cross-game dependence diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bts.validate.dependence import (
    pearson_residual,
    pa_residual_correlation,
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
        from bts.validate.dependence import pair_residual_correlation
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
        from bts.validate.dependence import pair_residual_correlation
        rho_hat, ci_lo, ci_hi, p_value = pair_residual_correlation(df, n_permutations=500)
        # Latent factor with sigma=1.2 and n=200 reliably produces rho_hat in 0.10-0.20.
        assert rho_hat > 0.04, f"rho_hat={rho_hat:.3f} not detecting positive correlation"
        # The permutation test should show p_value < 0.10 at this signal strength.
        assert p_value < 0.10, f"p_value={p_value:.3f} above significance under positive correlation"
