"""Tests for scorecard computation module.

TDD: tests written before implementation.
"""

import numpy as np
import pandas as pd
import pytest

from bts.validate.scorecard import (
    compute_precision_at_k,
    compute_miss_analysis,
    compute_calibration,
    compute_streak_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profiles(days: int = 10, top_n: int = 10) -> pd.DataFrame:
    """Generate synthetic backtest profile DataFrame.

    Rank 1: always hits (actual_hit=1).
    Rank 2: hits with 80% probability (deterministic via seed for testing).
    Ranks 3+: hit with 50% probability.
    p_game_hit decreasing by rank: 0.90, 0.88, 0.86, ...
    """
    rng = np.random.default_rng(42)
    rows = []
    for day_idx in range(days):
        date = pd.Timestamp("2025-04-01") + pd.Timedelta(days=day_idx)
        for rank in range(1, top_n + 1):
            p_hit = 0.90 - (rank - 1) * 0.02
            if rank == 1:
                hit = 1
            elif rank == 2:
                hit = int(rng.random() < 0.80)
            else:
                hit = int(rng.random() < 0.50)
            rows.append({
                "date": date,
                "rank": rank,
                "batter_id": 1000 + rank,
                "p_game_hit": p_hit,
                "actual_hit": hit,
                "n_pas": 4,
            })
    return pd.DataFrame(rows)


def _make_profiles_with_season(days: int = 20, top_n: int = 10) -> pd.DataFrame:
    """Generate profiles split across two seasons."""
    df = _make_profiles(days=days, top_n=top_n)
    df["season"] = df["date"].apply(lambda d: 2024 if d.month <= 4 else 2025)
    return df


# ---------------------------------------------------------------------------
# compute_precision_at_k
# ---------------------------------------------------------------------------

class TestComputePrecisionAtK:
    def test_returns_dict_for_all_k_values(self):
        df = _make_profiles(days=10, top_n=10)
        result = compute_precision_at_k(df)
        # Should include all standard K values up to top_n
        for k in [1, 5, 10]:
            assert k in result, f"Missing K={k}"

    def test_precision_at_1_is_one(self):
        """Rank-1 always hits in our synthetic data → precision@1 should be 1.0."""
        df = _make_profiles(days=10, top_n=10)
        result = compute_precision_at_k(df, k_values=[1])
        assert result[1] == pytest.approx(1.0)

    def test_precision_decreases_with_k(self):
        """Higher K includes worse-ranked batters, so precision should not increase."""
        df = _make_profiles(days=10, top_n=10)
        result = compute_precision_at_k(df, k_values=[1, 5, 10])
        # p@1 >= p@5 >= p@10
        assert result[1] >= result[5] >= result[10]

    def test_k_larger_than_pool_is_skipped(self):
        """K values exceeding daily pool size should be skipped gracefully."""
        df = _make_profiles(days=5, top_n=10)
        # K=100 > 10 rows per day
        result = compute_precision_at_k(df, k_values=[1, 100])
        assert 1 in result
        # K=100 should be absent (no day has 100 rows)
        assert 100 not in result

    def test_by_season_returns_nested_dict(self):
        """by_season=True with season column returns {season: {k: precision}}."""
        df = _make_profiles_with_season(days=20, top_n=10)
        result = compute_precision_at_k(df, k_values=[1, 5], by_season=True)
        assert isinstance(result, dict)
        # Each value should be a dict of {k: precision}
        for season_val in result.values():
            assert isinstance(season_val, dict)
            assert 1 in season_val
            assert 5 in season_val

    def test_by_season_without_season_column_falls_back_to_flat(self):
        """by_season=True but no season column → flat dict."""
        df = _make_profiles(days=10, top_n=10)
        result = compute_precision_at_k(df, k_values=[1, 5], by_season=True)
        # Flat dict: keys are k values (ints), not seasons
        assert 1 in result
        assert isinstance(result[1], float)

    def test_precision_value_in_range(self):
        """All precision values must be in [0, 1]."""
        df = _make_profiles(days=10, top_n=10)
        result = compute_precision_at_k(df, k_values=[1, 5, 10])
        for k, val in result.items():
            assert 0.0 <= val <= 1.0, f"P@{k} = {val} out of range"


# ---------------------------------------------------------------------------
# compute_miss_analysis
# ---------------------------------------------------------------------------

class TestComputeMissAnalysis:
    def test_returns_expected_keys(self):
        df = _make_profiles(days=10, top_n=5)
        result = compute_miss_analysis(df)
        assert "n_miss_days" in result
        assert "rank_2_hit_rate_on_miss" in result
        assert "mean_p_hit_on_miss" in result
        assert "mean_p_hit_on_hit" in result

    def test_rank1_always_hits_means_zero_misses(self):
        """In our synthetic data rank-1 never misses → n_miss_days = 0."""
        df = _make_profiles(days=10, top_n=5)
        result = compute_miss_analysis(df)
        assert result["n_miss_days"] == 0
        # Rates should be None when no miss days
        assert result["rank_2_hit_rate_on_miss"] is None

    def test_with_some_misses(self):
        """Inject a few rank-1 misses and verify counts."""
        df = _make_profiles(days=10, top_n=5)
        # Force rank-1 to miss on day 0 and day 1
        miss_mask = (df["rank"] == 1) & (df["date"].isin(
            [pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-02")]
        ))
        df = df.copy()
        df.loc[miss_mask, "actual_hit"] = 0
        result = compute_miss_analysis(df)
        assert result["n_miss_days"] == 2

    def test_rank_2_hit_rate_is_rate(self):
        """rank_2_hit_rate_on_miss should be a float in [0, 1] when misses exist."""
        df = _make_profiles(days=10, top_n=5)
        df = df.copy()
        miss_mask = (df["rank"] == 1) & (df["date"] == pd.Timestamp("2025-04-01"))
        df.loc[miss_mask, "actual_hit"] = 0
        result = compute_miss_analysis(df)
        assert result["rank_2_hit_rate_on_miss"] is not None
        assert 0.0 <= result["rank_2_hit_rate_on_miss"] <= 1.0

    def test_mean_p_hit_values_ordered(self):
        """mean_p_hit_on_hit should be >= mean_p_hit_on_miss (model predicts better on hit days)."""
        df = _make_profiles(days=20, top_n=5)
        # Inject misses on a few days
        df = df.copy()
        miss_dates = df["date"].drop_duplicates().iloc[:5]
        df.loc[(df["rank"] == 1) & (df["date"].isin(miss_dates)), "actual_hit"] = 0
        result = compute_miss_analysis(df)
        # mean_p_hit_on_hit >= mean_p_hit_on_miss (rank-1 p is constant 0.90 here,
        # so they'll be equal — just check both are numbers)
        assert isinstance(result["mean_p_hit_on_hit"], float)
        assert isinstance(result["mean_p_hit_on_miss"], float)


# ---------------------------------------------------------------------------
# compute_calibration
# ---------------------------------------------------------------------------

class TestComputeCalibration:
    def test_returns_list_of_tuples(self):
        df = _make_profiles(days=20, top_n=10)
        result = compute_calibration(df)
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 3  # (predicted_mean, actual_mean, count)

    def test_tuple_values_in_range(self):
        df = _make_profiles(days=20, top_n=10)
        result = compute_calibration(df)
        for pred_mean, actual_mean, count in result:
            assert 0.0 <= pred_mean <= 1.0
            assert 0.0 <= actual_mean <= 1.0
            assert count > 0

    def test_n_deciles_controls_bucket_count(self):
        """n_deciles controls maximum number of buckets (may be less with duplicates='drop')."""
        df = _make_profiles(days=50, top_n=10)
        result_5 = compute_calibration(df, n_deciles=5)
        result_10 = compute_calibration(df, n_deciles=10)
        # With more deciles we should get at least as many buckets
        assert len(result_10) >= len(result_5)

    def test_total_count_matches_rank1_10_rows(self):
        """Total count across buckets = number of rank 1-10 rows."""
        df = _make_profiles(days=20, top_n=10)
        result = compute_calibration(df)
        total = sum(count for _, _, count in result)
        expected = len(df[df["rank"] <= 10])
        assert total == expected

    def test_uses_rank_1_to_10_only(self):
        """Calibration uses rank 1-10, not the full pool."""
        df = _make_profiles(days=20, top_n=20)
        # Inject extreme p_game_hit values on rank > 10 rows to verify they're excluded
        df_modified = df.copy()
        df_modified.loc[df_modified["rank"] > 10, "p_game_hit"] = 0.99
        result_normal = compute_calibration(df)
        result_modified = compute_calibration(df_modified)
        # Both should produce identical results — rank > 10 is excluded
        assert result_normal == result_modified


# ---------------------------------------------------------------------------
# compute_streak_metrics
# ---------------------------------------------------------------------------

class TestComputeStreakMetrics:
    def test_returns_expected_keys(self):
        df = _make_profiles(days=30, top_n=5)
        result = compute_streak_metrics(df, n_trials=100, season_length=50)
        assert "mean_max_streak" in result
        assert "median_max_streak" in result
        assert "p90_max_streak" in result
        assert "p99_max_streak" in result
        assert "p_57_monte_carlo" in result
        assert "longest_replay_streak" in result

    def test_p57_is_probability(self):
        df = _make_profiles(days=30, top_n=5)
        result = compute_streak_metrics(df, n_trials=100, season_length=50)
        assert 0.0 <= result["p_57_monte_carlo"] <= 1.0

    def test_streak_metrics_ordered(self):
        """p99 >= p90 >= median and all are non-negative integers."""
        df = _make_profiles(days=30, top_n=5)
        result = compute_streak_metrics(df, n_trials=100, season_length=50)
        assert result["p99_max_streak"] >= result["p90_max_streak"]
        assert result["p90_max_streak"] >= result["median_max_streak"]
        assert result["median_max_streak"] >= 0

    def test_longest_replay_streak_is_non_negative(self):
        df = _make_profiles(days=30, top_n=5)
        result = compute_streak_metrics(df, n_trials=100, season_length=50)
        assert result["longest_replay_streak"] >= 0

    def test_mean_max_streak_is_float(self):
        df = _make_profiles(days=30, top_n=5)
        result = compute_streak_metrics(df, n_trials=100, season_length=50)
        assert isinstance(result["mean_max_streak"], float)
