"""Tests for cross-entropy importance-sampling rare-event MC."""
from __future__ import annotations

import numpy as np
import pytest

from bts.simulate.rare_event_mc import LatentFactorSimulator


class TestLatentFactorSimulator:
    def test_zero_lambda_recovers_independent_simulator(self):
        """At lambda_d=lambda_g=0, the simulator behaves like independent Bernoulli draws."""
        rng = np.random.default_rng(42)
        # Profiles: 100 days, single game per day, p=0.7.
        profiles = [{"date": d, "p_game": 0.7} for d in range(100)]
        sim = LatentFactorSimulator(profiles, lambda_d=0.0, lambda_g=0.0)
        n_seasons = 5000
        outcomes = [sim.sample_season(rng) for _ in range(n_seasons)]
        empirical_hit_rate = np.array([sum(s) for s in outcomes]).mean() / 100
        # Mean should match 0.7 to within 1pp at n=5000.
        assert abs(empirical_hit_rate - 0.7) < 0.01

    def test_nonzero_lambda_introduces_correlation(self):
        """At lambda_d>0, day-level outcomes should differ in variance from the independent case."""
        rng = np.random.default_rng(0)
        profiles = [{"date": d, "p_game": 0.5} for d in range(200)]
        sim_indep = LatentFactorSimulator(profiles, lambda_d=0.0, lambda_g=0.0)
        sim_corr = LatentFactorSimulator(profiles, lambda_d=1.5, lambda_g=0.0)
        n = 5000
        # Compare distributional properties of the first day's outcome.
        # With lambda_d=1.5, the marginal P(hit) is still ~0.5 but the day-to-day
        # variation across resampled lambda noise is amplified — pointwise variance
        # of `outcomes[0]` (a single Bernoulli) is bounded at 0.25 either way, so
        # we instead compare the variance of the per-season hit-rate (sum/n_days).
        hit_rates_indep = np.array([np.mean(sim_indep.sample_season(rng)) for _ in range(n)])
        hit_rates_corr = np.array([np.mean(sim_corr.sample_season(rng)) for _ in range(n)])
        var_indep = float(hit_rates_indep.var())
        var_corr = float(hit_rates_corr.var())
        # Independent: variance of mean of N=200 Bernoulli p=0.5 = 0.25/200 = 0.00125.
        # Correlated: latent factor inflates per-day correlation, so per-season hit-rate
        # variance increases by a factor related to (1 + (n-1)*rho_eff). For lambda_d=1.5
        # we expect at least a 2-3x increase in variance.
        assert var_corr > var_indep * 1.5, (
            f"lambda_d=1.5 produced var_corr={var_corr:.5f} only marginally above "
            f"var_indep={var_indep:.5f}"
        )
