"""Tests for strategy profile definitions and threshold resolution."""

import pytest
from bts.simulate.strategies import Strategy, get_thresholds, ALL_STRATEGIES


class TestStrategy:
    def test_baseline_has_no_skip_no_double(self):
        s = ALL_STRATEGIES["baseline"]
        assert s.skip_threshold is None
        assert s.double_threshold is None
        assert s.streak_saver is True

    def test_current_has_double_at_065(self):
        s = ALL_STRATEGIES["current"]
        assert s.skip_threshold is None
        assert s.double_threshold == 0.65

    def test_sprint_has_aggressive_double(self):
        s = ALL_STRATEGIES["sprint"]
        assert s.double_threshold == 0.50


class TestGetThresholds:
    def test_flat_strategy_ignores_streak(self):
        s = ALL_STRATEGIES["current"]
        skip, double = get_thresholds(s, streak=0)
        assert skip is None
        assert double == 0.65
        skip2, double2 = get_thresholds(s, streak=40)
        assert skip2 is None
        assert double2 == 0.65

    def test_streak_aware_early_phase(self):
        s = ALL_STRATEGIES["streak-aware"]
        skip, double = get_thresholds(s, streak=5)
        assert skip is None
        assert double == 0.55

    def test_streak_aware_saver_phase(self):
        s = ALL_STRATEGIES["streak-aware"]
        skip, double = get_thresholds(s, streak=12)
        assert skip is None
        assert double == 0.60

    def test_streak_aware_protect_phase(self):
        s = ALL_STRATEGIES["streak-aware"]
        skip, double = get_thresholds(s, streak=25)
        assert skip == 0.78
        assert double == 0.65

    def test_streak_aware_lockdown_phase(self):
        s = ALL_STRATEGIES["streak-aware"]
        skip, double = get_thresholds(s, streak=35)
        assert skip == 0.80
        assert double is None

    def test_streak_aware_sprint_finish(self):
        s = ALL_STRATEGIES["streak-aware"]
        skip, double = get_thresholds(s, streak=50)
        assert skip == 0.78
        assert double == 0.60

    def test_all_strategies_registered(self):
        expected = {
            "baseline", "current", "skip-conservative", "skip-aggressive",
            "sprint", "streak-aware", "combined",
        }
        assert set(ALL_STRATEGIES.keys()) == expected
