"""Tests for Monte Carlo streak simulation."""

import numpy as np
import pytest
from bts.simulate.strategies import Strategy, ALL_STRATEGIES
from bts.simulate.monte_carlo import DailyProfile, simulate_season, SeasonResult


def _profile(top1_p: float, top1_hit: int, top2_p: float = 0.70, top2_hit: int = 1) -> DailyProfile:
    """Create a daily profile for testing."""
    return DailyProfile(top1_p=top1_p, top1_hit=top1_hit, top2_p=top2_p, top2_hit=top2_hit)


class TestSimulateSeason:
    def test_all_hits_produces_full_streak(self):
        """10 days, all hits, no skipping → streak of 10."""
        profiles = [_profile(0.85, 1)] * 10
        strategy = ALL_STRATEGIES["baseline"]
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 10
        assert result.play_days == 10

    def test_miss_resets_streak(self):
        """Hit, hit, miss, hit → max streak 2."""
        profiles = [
            _profile(0.85, 1),
            _profile(0.85, 1),
            _profile(0.85, 0),
            _profile(0.85, 1),
        ]
        strategy = ALL_STRATEGIES["baseline"]
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 2

    def test_skip_preserves_streak(self):
        """With skip threshold 0.80: high-conf hit, low-conf skip, high-conf hit → streak 2."""
        profiles = [
            _profile(0.85, 1),
            _profile(0.75, 0),  # below threshold AND would miss — but we skip
            _profile(0.85, 1),
        ]
        strategy = Strategy(name="test-skip", skip_threshold=0.80)
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 2
        assert result.play_days == 2

    def test_double_down_advances_by_two(self):
        """Both hit on a double → streak advances by 2."""
        profiles = [_profile(0.85, 1, 0.82, 1)] * 5
        strategy = Strategy(name="test-double", double_threshold=0.50)
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 10  # 5 days × 2
        assert result.play_days == 5

    def test_double_down_miss_resets(self):
        """One miss in a double → reset."""
        profiles = [
            _profile(0.85, 1, 0.82, 1),
            _profile(0.85, 1, 0.82, 0),  # second pick misses
            _profile(0.85, 1, 0.82, 1),
        ]
        strategy = Strategy(name="test-double", double_threshold=0.50)
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 2  # first day

    def test_double_threshold_prevents_double(self):
        """P(both) below threshold → single pick only."""
        profiles = [_profile(0.75, 1, 0.70, 1)] * 3
        # P(both) = 0.75 * 0.70 = 0.525, below 0.65 threshold
        strategy = Strategy(name="test-high-thresh", double_threshold=0.65)
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 3  # singles only

    def test_streak_saver_saves_at_10(self):
        """10 hits then a miss → saver preserves streak at 10."""
        profiles = [_profile(0.85, 1)] * 10 + [_profile(0.85, 0)] + [_profile(0.85, 1)] * 3
        strategy = Strategy(name="test-saver", streak_saver=True)
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 13  # 10 + saved + 3 more
        assert result.streak_saver_used is True

    def test_streak_saver_does_not_save_above_15(self):
        """16 hits then a miss → no save, reset."""
        profiles = [_profile(0.85, 1)] * 16 + [_profile(0.85, 0)]
        strategy = Strategy(name="test-saver", streak_saver=True)
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 16
        assert result.streak_saver_used is False  # wasn't eligible

    def test_streak_saver_only_fires_once(self):
        """Save at 10, rebuild to 12, miss again → reset."""
        profiles = (
            [_profile(0.85, 1)] * 10  # streak = 10
            + [_profile(0.85, 0)]      # saved at 10
            + [_profile(0.85, 1)] * 2  # streak = 12
            + [_profile(0.85, 0)]      # no save, reset
            + [_profile(0.85, 1)] * 5  # new streak = 5
        )
        strategy = Strategy(name="test-saver", streak_saver=True)
        result = simulate_season(profiles, strategy)
        assert result.max_streak == 12

    def test_empty_profiles(self):
        result = simulate_season([], ALL_STRATEGIES["baseline"])
        assert result.max_streak == 0
        assert result.play_days == 0
