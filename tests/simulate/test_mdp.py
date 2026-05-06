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


class TestMDPObjectiveSpec:
    """Tests-as-documentation for the reachability semantics of solve_mdp.

    See docs/sota_audit/2026-05-06-mdp-objective-audit.md for the formal
    specification these tests harden. The contract is that solve_mdp returns
    a value table V where V[s, d, saver, q] is the optimal probability of
    reaching streak 57 within d remaining days when the agent acts optimally.
    """

    def test_value_equals_closed_form_reachability_on_tiny_mdp(self):
        """Hand-computed reachability values on a 1-bin MDP, no saver zone.

        With p_hit=0.9, p_both=0.8, freq=1.0 and saver=0 outside [10,15]:
          V[57, *, *, *] = 1                    (terminal)
          V[s, 0, *, *]  = 0  for s < 57         (no days left)
          V[56, 1, 0, 0] = max(skip=0,
                               single=0.9*1 + 0.1*0 = 0.9,
                               double=0.8*1 + 0.2*0 = 0.8)
                         = 0.9                   (single)
          V[55, 1, 0, 0] = max(skip=0,
                               single=0.9*V[56,0]+0.1*V[0,0] = 0,
                               double=0.8*1 + 0.2*0 = 0.8)
                         = 0.8                   (double)
          V[55, 2, 0, 0] = max(skip=V[55,1,0,0]=0.8,
                               single=0.9*V[56,1,0,0]+0.1*V[0,1,0,0]
                                     =0.9*0.9 + 0.1*0 = 0.81,
                               double=0.8*V[57,1,0,0]+0.2*V[0,1,0,0]
                                     =0.8*1.0 + 0.2*0 = 0.8)
                         = 0.81                  (single)
          V[0, 1, 0, 0]  = 0                     (cannot reach 57 in 1 day from 0)
        """
        bins = _simple_bins()
        sol = solve_mdp(bins, season_length=10)
        V = sol.value_table

        assert V[57, 1, 0, 0] == pytest.approx(1.0)
        assert V[56, 0, 0, 0] == pytest.approx(0.0)
        assert V[56, 1, 0, 0] == pytest.approx(0.9, abs=1e-12)
        assert V[55, 1, 0, 0] == pytest.approx(0.8, abs=1e-12)
        assert V[55, 2, 0, 0] == pytest.approx(0.81, abs=1e-12)
        assert V[0, 1, 0, 0] == pytest.approx(0.0)

        # And the policy at the diagnostic states matches the closed-form argmax.
        assert sol.policy(streak=56, days_remaining=1, saver=False, quality_bin=0) == "single"
        assert sol.policy(streak=55, days_remaining=1, saver=False, quality_bin=0) == "double"
        assert sol.policy(streak=55, days_remaining=2, saver=False, quality_bin=0) == "single"

    def test_optimal_p57_matches_initial_state_expectation(self):
        """sol.optimal_p57 equals freq[q]-weighted V[0, season_length, 1, q].

        This locks in the contract of mdp.py:205 — the reported headline metric
        is the bin-frequency expectation of the initial-state value, computed
        with the early-phase frequencies (or late-phase if late_bins is set;
        here we use single-phase only).
        """
        bins = _two_bins()
        season_length = 100
        sol = solve_mdp(bins, season_length=season_length)
        freq = np.array([b.frequency for b in bins.bins])
        V_initial = sol.value_table[0, season_length, 1, :]
        expected = float(np.dot(freq, V_initial))
        assert sol.optimal_p57 == pytest.approx(expected, abs=1e-12)

    def test_value_function_is_probability_in_unit_interval(self):
        """Every V entry lies in [0, 1] — necessary for any reachability semantic.

        Catches a future objective change that allows V outside [0, 1] (e.g.,
        signed rewards, expected-streak-length, etc.) by having the existing
        test suite fail loudly rather than silently inheriting the new contract.
        """
        bins = _two_bins()
        sol = solve_mdp(bins, season_length=80)
        V = sol.value_table
        assert V.min() >= 0.0
        assert V.max() <= 1.0 + 1e-12
