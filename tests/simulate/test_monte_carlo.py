"""Tests for Monte Carlo streak simulation."""

import numpy as np
import pytest
from bts.simulate.strategies import Strategy, ALL_STRATEGIES
from bts.simulate.monte_carlo import DailyProfile, simulate_season, SeasonResult
from bts.simulate.monte_carlo import (
    load_profiles, run_monte_carlo, MonteCarloResult,
    load_all_profiles, load_season_profiles, run_replay,
)

import pandas as pd
from click.testing import CliRunner
from bts.simulate.cli import simulate


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


def _make_profile_df(n_days: int = 30, hit_rate: float = 0.85) -> pd.DataFrame:
    """Create a synthetic backtest profile DataFrame."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_days):
        date = f"2024-{(i // 28) + 4:02d}-{(i % 28) + 1:02d}"
        for rank in range(1, 11):
            p = max(0.5, 0.90 - rank * 0.02 + rng.normal(0, 0.02))
            hit = 1 if rng.random() < hit_rate else 0
            rows.append({"date": date, "rank": rank, "batter_id": rank * 1000,
                          "p_game_hit": p, "actual_hit": hit, "n_pas": 4})
    return pd.DataFrame(rows)


class TestLoadProfiles:
    def test_extracts_top2_per_day(self):
        df = _make_profile_df(n_days=5)
        profiles = load_profiles(df)
        assert len(profiles) == 5
        assert all(isinstance(p, DailyProfile) for p in profiles)

    def test_profiles_use_rank_1_and_2(self):
        df = _make_profile_df(n_days=1)
        profiles = load_profiles(df)
        r1 = df[df["rank"] == 1].iloc[0]
        r2 = df[df["rank"] == 2].iloc[0]
        assert profiles[0].top1_p == r1["p_game_hit"]
        assert profiles[0].top1_hit == r1["actual_hit"]
        assert profiles[0].top2_p == r2["p_game_hit"]
        assert profiles[0].top2_hit == r2["actual_hit"]


class TestRunMonteCarlo:
    def test_returns_correct_shape(self):
        df = _make_profile_df(n_days=60)
        profiles = load_profiles(df)
        result = run_monte_carlo(profiles, ALL_STRATEGIES["baseline"], n_trials=100, season_length=30)
        assert isinstance(result, MonteCarloResult)
        assert result.n_trials == 100
        assert len(result.max_streaks) == 100
        assert 0 <= result.p_57 <= 1
        assert result.median_streak >= 0
        assert result.p95_streak >= result.median_streak

    def test_perfect_hit_rate_reaches_57(self):
        """If every profile is a hit, P(57) should be 1.0 with enough days."""
        profiles = [_profile(0.90, 1)] * 60
        result = run_monte_carlo(profiles, ALL_STRATEGIES["baseline"], n_trials=50, season_length=60)
        assert result.p_57 == 1.0

    def test_zero_hit_rate_never_reaches_57(self):
        profiles = [_profile(0.50, 0)] * 60
        result = run_monte_carlo(profiles, ALL_STRATEGIES["baseline"], n_trials=50, season_length=60)
        assert result.p_57 == 0.0


class TestRunReplay:
    def test_replays_each_season(self):
        season_profiles = {
            2024: [_profile(0.85, 1)] * 10 + [_profile(0.85, 0)] + [_profile(0.85, 1)] * 5,
            2025: [_profile(0.85, 1)] * 20,
        }
        results = run_replay(season_profiles, ALL_STRATEGIES["baseline"])
        assert len(results) == 2
        assert 2024 in results
        assert 2025 in results
        assert results[2025].max_streak == 20


class TestCLI:
    def test_simulate_run_with_synthetic_profiles(self, tmp_path):
        """CLI runs Monte Carlo on saved profile parquets."""
        df = _make_profile_df(n_days=30, hit_rate=0.85)
        df.to_parquet(tmp_path / "backtest_2024.parquet", index=False)

        runner = CliRunner()
        result = runner.invoke(simulate, [
            "run", "--profiles-dir", str(tmp_path), "--trials", "100",
        ])
        assert result.exit_code == 0
        assert "baseline" in result.output
        assert "P(57)" in result.output

    def test_simulate_run_replay_only(self, tmp_path):
        """CLI replay mode."""
        df = _make_profile_df(n_days=30, hit_rate=0.85)
        df.to_parquet(tmp_path / "backtest_2024.parquet", index=False)

        runner = CliRunner()
        result = runner.invoke(simulate, [
            "run", "--profiles-dir", str(tmp_path), "--replay-only",
        ])
        assert result.exit_code == 0
        assert "Replay" in result.output

    def test_simulate_run_single_strategy(self, tmp_path):
        """CLI with --strategy flag runs only that strategy."""
        df = _make_profile_df(n_days=30, hit_rate=0.85)
        df.to_parquet(tmp_path / "backtest_2024.parquet", index=False)

        runner = CliRunner()
        result = runner.invoke(simulate, [
            "run", "--profiles-dir", str(tmp_path),
            "--strategy", "sprint", "--trials", "50",
        ])
        assert result.exit_code == 0
        assert "sprint" in result.output
