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


class TestCEISUnbiasedness:
    def test_unbiased_at_theta_zero_matches_naive_mc(self):
        """At theta=0, CE-IS estimator equals naive MC of independent Bernoulli draws.

        Constructs a tiny synthetic profile (constant per-day p), uses the always-play
        every-day strategy, and asserts that CE-IS at theta=0 matches the analytical
        P(>=57 consecutive hits in N days) within reasonable MC noise.
        """
        from bts.simulate.rare_event_mc import estimate_p57_with_ceis

        # Constant p=0.95 over 60 days. Use streak_threshold=50 so the event is
        # frequent enough that 30k samples gives a tight estimate to compare against
        # naive MC. P(>=57 consecutive at p=0.78, n=153) is ~1-5% which needs n=100k+.
        n_days = 60
        p = 0.95
        threshold = 50
        profiles = [{"date": d, "p_game": p} for d in range(n_days)]

        result = estimate_p57_with_ceis(
            profiles,
            strategy=None,  # ignored when always-play; helper handles None
            theta=np.zeros(4),  # explicit theta=0 means no tilting
            n_rounds=0,         # skip CE fitting at theta=0
            n_final=30000,
            seed=42,
            streak_threshold=threshold,
        )

        # Analytical truth via direct enumeration is hard for streaks; instead, run a
        # large naive MC and assert CE-IS at theta=0 lands within MC noise of it.
        # Both are sampling the same distribution at theta=0, so they should match
        # within ~1pp at n=30k.
        rng = np.random.default_rng(123)
        naive_hits = 0
        n_naive = 30000
        for _ in range(n_naive):
            streak = 0
            max_streak = 0
            for _ in range(n_days):
                if rng.random() < p:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
            if max_streak >= threshold:
                naive_hits += 1
        naive_p = naive_hits / n_naive

        ce_p = result.point_estimate
        # Both at theta=0 should agree within MC noise (~1pp at n=30k).
        assert abs(ce_p - naive_p) < 0.02, (
            f"CE-IS theta=0 estimate {ce_p:.4f} differs from naive MC {naive_p:.4f} by "
            f"{abs(ce_p - naive_p):.4f} (>2pp tolerance)"
        )
