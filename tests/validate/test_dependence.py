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
