"""Tests for reachability MDP solver."""

import numpy as np
import pytest
from bts.simulate.quality_bins import QualityBin, QualityBins
from bts.simulate.mdp import solve_mdp, MDPSolution


def _simple_bins():
    """One bin: p_hit=0.9, p_both=0.8, frequency=1.0."""
    return QualityBins(
        bins=[QualityBin(index=0, p_range=(0.8, 0.9), p_hit=0.9, p_both=0.8, frequency=1.0)],
        boundaries=[],
    )


def _two_bins():
    """Two bins: bad (50% hit, freq=0.3) and good (90% hit, freq=0.7)."""
    return QualityBins(
        bins=[
            QualityBin(index=0, p_range=(0.7, 0.8), p_hit=0.5, p_both=0.3, frequency=0.3),
            QualityBin(index=1, p_range=(0.8, 0.9), p_hit=0.9, p_both=0.8, frequency=0.7),
        ],
        boundaries=[0.8],
    )


class TestSolveMDP:
    def test_returns_mdp_solution(self):
        bins = _simple_bins()
        sol = solve_mdp(bins, season_length=50)
        assert isinstance(sol, MDPSolution)
        assert 0 <= sol.optimal_p57 <= 1

    def test_terminal_state_value_is_1(self):
        bins = _simple_bins()
        sol = solve_mdp(bins, season_length=50)
        # V(57, any_d, any_saver, any_q) should be 1.0
        for d in range(50):
            for saver in [0, 1]:
                for q in range(len(bins.bins)):
                    assert sol.value_table[57, d, saver, q] == 1.0

    def test_zero_days_value_is_0(self):
        bins = _simple_bins()
        sol = solve_mdp(bins, season_length=50)
        # V(s<57, 0, any, any) should be 0.0
        for s in range(57):
            for saver in [0, 1]:
                for q in range(len(bins.bins)):
                    assert sol.value_table[s, 0, saver, q] == 0.0

    def test_optimal_p57_positive_with_good_bins(self):
        bins = _simple_bins()  # p_hit = 0.9
        sol = solve_mdp(bins, season_length=200)
        assert sol.optimal_p57 > 0.01

    def test_optimal_beats_or_matches_always_single(self):
        """MDP optimal should be >= any fixed strategy."""
        from bts.simulate.exact import exact_p57
        from bts.simulate.strategies import Strategy

        bins = _two_bins()
        sol = solve_mdp(bins, season_length=100)
        p_single = exact_p57(Strategy(name="single"), bins, season_length=100)
        assert sol.optimal_p57 >= p_single - 1e-10

    def test_policy_returns_valid_action(self):
        bins = _simple_bins()
        sol = solve_mdp(bins, season_length=50)
        action = sol.policy(streak=10, days_remaining=30, saver=True, quality_bin=0)
        assert action in ("skip", "single", "double")

    def test_more_days_increases_value(self):
        bins = _simple_bins()
        sol = solve_mdp(bins, season_length=200)
        v_10 = sol.value_table[0, 10, 1, 0]
        v_100 = sol.value_table[0, 100, 1, 0]
        assert v_100 >= v_10

    def test_skip_optimal_for_bad_bin(self):
        """With a terrible bin (20% hit rate), the MDP should prefer skip."""
        bins = QualityBins(
            bins=[
                QualityBin(index=0, p_range=(0.5, 0.6), p_hit=0.2, p_both=0.05, frequency=0.3),
                QualityBin(index=1, p_range=(0.8, 0.9), p_hit=0.9, p_both=0.8, frequency=0.7),
            ],
            boundaries=[0.7],
        )
        sol = solve_mdp(bins, season_length=100)
        action = sol.policy(streak=20, days_remaining=80, saver=False, quality_bin=0)
        assert action == "skip"
