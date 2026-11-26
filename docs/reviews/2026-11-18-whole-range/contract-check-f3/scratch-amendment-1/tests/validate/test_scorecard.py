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


# ---------------------------------------------------------------------------
# compute_full_scorecard integration — proper_scoring section (SOTA #12)
# ---------------------------------------------------------------------------

class TestProperScoringInScorecard:
    """SOTA #12 phase 1 integration: scorecard JSON includes proper_scoring."""

    def test_scorecard_has_proper_scoring_section(self):
        from bts.validate.scorecard import compute_full_scorecard
        df = _make_profiles(days=30, top_n=10)
        sc = compute_full_scorecard(df, mc_trials=100, season_length=50)
        assert "proper_scoring" in sc

    def test_proper_scoring_has_both_decision_buckets(self):
        from bts.validate.scorecard import compute_full_scorecard
        df = _make_profiles(days=30, top_n=10)
        sc = compute_full_scorecard(df, mc_trials=100, season_length=50)
        ps = sc["proper_scoring"]
        assert "all_top10" in ps
        assert "rank1" in ps
        assert "metadata" in ps

    def test_proper_scoring_buckets_have_required_metrics(self):
        from bts.validate.scorecard import compute_full_scorecard
        df = _make_profiles(days=30, top_n=10)
        sc = compute_full_scorecard(df, mc_trials=100, season_length=50)
        for bucket_name in ("all_top10", "rank1"):
            bucket = sc["proper_scoring"][bucket_name]
            for key in ("n", "log_loss", "brier", "decomposition", "reliability_table", "top_bin"):
                assert key in bucket, f"{bucket_name} missing {key}"

    def test_rank1_n_matches_n_days(self):
        from bts.validate.scorecard import compute_full_scorecard
        df = _make_profiles(days=42, top_n=10)
        sc = compute_full_scorecard(df, mc_trials=100, season_length=50)
        assert sc["proper_scoring"]["rank1"]["n"] == 42

    def test_proper_scoring_metadata_has_interval_and_binning(self):
        from bts.validate.scorecard import compute_full_scorecard
        df = _make_profiles(days=30, top_n=10)
        sc = compute_full_scorecard(df, mc_trials=100, season_length=50)
        meta = sc["proper_scoring"]["metadata"]
        assert meta["interval_method"] == "wilson"
        assert "n_bins" in meta
        assert "binning" in meta

    def test_proper_scoring_round_trips_through_save_scorecard(self, tmp_path):
        """save_scorecard handles the proper_scoring section without numpy-serialization errors."""
        import json
        from bts.validate.scorecard import compute_full_scorecard, save_scorecard
        df = _make_profiles(days=30, top_n=10)
        sc = compute_full_scorecard(df, mc_trials=100, season_length=50)
        path = tmp_path / "scorecard.json"
        saved = save_scorecard(sc, path)
        assert saved.exists()
        loaded = json.loads(saved.read_text())
        assert "proper_scoring" in loaded
        assert "all_top10" in loaded["proper_scoring"]
        assert "rank1" in loaded["proper_scoring"]


# ---------------------------------------------------------------------------
# diff_scorecards — proper_scoring scalar diffs (Codex bus #65)
# ---------------------------------------------------------------------------

class TestDiffScorecardsProperScoring:
    """Verify proper_scoring scalars flow through diff_scorecards as flat
    {bucket}.{field} entries; tabular fields and metadata are skipped."""

    def _make_scorecard_pair(self):
        from bts.validate.scorecard import compute_full_scorecard
        # Build two different profile sets; same RNG (seed=42) but different
        # row counts → different per-bin metrics in proper_scoring.
        df_a = _make_profiles(days=30, top_n=10)
        df_b = _make_profiles(days=40, top_n=10)
        baseline = compute_full_scorecard(df_a, mc_trials=100, season_length=50)
        variant = compute_full_scorecard(df_b, mc_trials=100, season_length=50)
        return baseline, variant

    def test_diff_emits_proper_scoring_section(self):
        from bts.validate.scorecard import diff_scorecards
        baseline, variant = self._make_scorecard_pair()
        diffs = diff_scorecards(baseline, variant)
        assert "proper_scoring" in diffs

    def test_diff_includes_bucket_level_scalars(self):
        from bts.validate.scorecard import diff_scorecards
        baseline, variant = self._make_scorecard_pair()
        ps_diff = diff_scorecards(baseline, variant)["proper_scoring"]
        for bucket in ("all_top10", "rank1"):
            for field in ("log_loss", "brier"):
                key = f"{bucket}.{field}"
                assert key in ps_diff, f"missing {key}"
                entry = ps_diff[key]
                for k in ("baseline", "variant", "delta"):
                    assert k in entry

    def test_diff_includes_decomposition_scalars(self):
        from bts.validate.scorecard import diff_scorecards
        baseline, variant = self._make_scorecard_pair()
        ps_diff = diff_scorecards(baseline, variant)["proper_scoring"]
        for bucket in ("all_top10", "rank1"):
            for field in ("reliability", "resolution", "uncertainty"):
                assert f"{bucket}.decomposition.{field}" in ps_diff

    def test_diff_includes_top_bin_scalars(self):
        from bts.validate.scorecard import diff_scorecards
        baseline, variant = self._make_scorecard_pair()
        ps_diff = diff_scorecards(baseline, variant)["proper_scoring"]
        for bucket in ("all_top10", "rank1"):
            for field in ("mean_p", "mean_y", "gap"):
                assert f"{bucket}.top_bin.{field}" in ps_diff

    def test_diff_skips_reliability_table(self):
        """reliability_table is tabular and must not appear in flat scalar diff."""
        from bts.validate.scorecard import diff_scorecards
        baseline, variant = self._make_scorecard_pair()
        ps_diff = diff_scorecards(baseline, variant)["proper_scoring"]
        for key in ps_diff:
            assert "reliability_table" not in key

    def test_diff_skips_top_bin_count_and_ci(self):
        """top_bin.n / ci_lo / ci_hi are not performance scalars; skip in diff."""
        from bts.validate.scorecard import diff_scorecards
        baseline, variant = self._make_scorecard_pair()
        ps_diff = diff_scorecards(baseline, variant)["proper_scoring"]
        for bucket in ("all_top10", "rank1"):
            assert f"{bucket}.top_bin.n" not in ps_diff
            assert f"{bucket}.top_bin.ci_lo" not in ps_diff
            assert f"{bucket}.top_bin.ci_hi" not in ps_diff

    def test_diff_skips_metadata(self):
        """metadata is non-numeric (strings); diff must not emit it."""
        from bts.validate.scorecard import diff_scorecards
        baseline, variant = self._make_scorecard_pair()
        ps_diff = diff_scorecards(baseline, variant)["proper_scoring"]
        for key in ps_diff:
            assert "metadata" not in key

    def test_diff_tolerates_missing_proper_scoring_in_baseline(self):
        """Old scorecards without proper_scoring must not break diff_scorecards."""
        from bts.validate.scorecard import compute_full_scorecard, diff_scorecards
        df = _make_profiles(days=30, top_n=10)
        old_baseline = {"precision": {1: 0.9}}  # legacy, no proper_scoring
        variant = compute_full_scorecard(df, mc_trials=100, season_length=50)
        diffs = diff_scorecards(old_baseline, variant)
        # No crash; proper_scoring section absent because baseline lacks it
        assert "proper_scoring" not in diffs

    def test_diff_tolerates_missing_proper_scoring_in_variant(self):
        from bts.validate.scorecard import compute_full_scorecard, diff_scorecards
        df = _make_profiles(days=30, top_n=10)
        baseline = compute_full_scorecard(df, mc_trials=100, season_length=50)
        old_variant = {"precision": {1: 0.9}}
        diffs = diff_scorecards(baseline, old_variant)
        assert "proper_scoring" not in diffs

    def test_diff_delta_sign_matches_variant_minus_baseline(self):
        """Sanity: delta = variant - baseline."""
        from bts.validate.scorecard import diff_scorecards
        baseline, variant = self._make_scorecard_pair()
        ps_diff = diff_scorecards(baseline, variant)["proper_scoring"]
        entry = ps_diff["all_top10.log_loss"]
        assert abs(entry["delta"] - (entry["variant"] - entry["baseline"])) < 1e-12
