"""Tests for exact P(57) computation via absorbing Markov chain."""

import numpy as np
import pytest
from bts.simulate.quality_bins import QualityBin, QualityBins
from bts.simulate.exact import exact_p57, build_transition_matrix
from bts.simulate.strategies import Strategy


def _simple_bins():
    """One-bin QualityBins where every day is identical (p_hit=0.9, p_both=0.8)."""
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


class TestBuildTransitionMatrix:
    def test_rows_sum_to_1(self):
        bins = _simple_bins()
        strategy = Strategy(name="always-single")
        T = build_transition_matrix(strategy, bins)
        # All rows except absorbing state (57) should sum to 1
        for s in range(57):
            assert abs(T[s].sum() - 1.0) < 1e-10, f"Row {s} sums to {T[s].sum()}"
        # Absorbing state stays at 57
        assert T[57, 57] == 1.0

    def test_shape(self):
        bins = _simple_bins()
        strategy = Strategy(name="test")
        T = build_transition_matrix(strategy, bins)
        assert T.shape == (58, 58)


class TestExactP57:
    def test_perfect_hit_rate_high_p57(self):
        """With 90% hit rate and 200 plays, P(57) should be substantial."""
        bins = _simple_bins()  # p_hit = 0.9
        strategy = Strategy(name="always-single")
        p = exact_p57(strategy, bins, season_length=200)
        assert p > 0.01  # should be meaningfully positive

    def test_zero_hit_rate_zero_p57(self):
        """With 0% hit rate, P(57) = 0."""
        bins = QualityBins(
            bins=[QualityBin(index=0, p_range=(0.5, 0.6), p_hit=0.0, p_both=0.0, frequency=1.0)],
            boundaries=[],
        )
        strategy = Strategy(name="test")
        p = exact_p57(strategy, bins, season_length=200)
        assert p == 0.0

    def test_more_plays_increases_p57(self):
        """More plays (longer season) should increase P(57)."""
        bins = _simple_bins()
        strategy = Strategy(name="test")
        p_short = exact_p57(strategy, bins, season_length=100)
        p_long = exact_p57(strategy, bins, season_length=300)
        assert p_long >= p_short

    def test_skip_strategy_reduces_plays(self):
        """A strategy that skips bad days should differ from always-play."""
        bins = _two_bins()  # bad=0.3freq, good=0.7freq
        always_play = Strategy(name="always", skip_threshold=None)
        skip_bad = Strategy(name="skip", skip_threshold=0.8)  # skips bin 0
        p_always = exact_p57(always_play, bins, season_length=180)
        p_skip = exact_p57(skip_bad, bins, season_length=180)
        # With 50% hit on bad days dragging down the average, skipping should help
        assert p_skip > p_always

    def test_doubling_changes_p57(self):
        """Doubling should produce different P(57) than singles-only."""
        bins = _simple_bins()
        singles = Strategy(name="singles")
        doubles = Strategy(name="doubles", double_threshold=0.50)
        p_singles = exact_p57(singles, bins, season_length=180)
        p_doubles = exact_p57(doubles, bins, season_length=180)
        assert p_singles != p_doubles

    def test_saver_increases_p57(self):
        """Streak saver should increase P(57)."""
        bins = _simple_bins()
        no_saver = Strategy(name="no-saver", streak_saver=False)
        with_saver = Strategy(name="saver", streak_saver=True)
        p_no = exact_p57(no_saver, bins, season_length=180)
        p_yes = exact_p57(with_saver, bins, season_length=180)
        assert p_yes >= p_no
