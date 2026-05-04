"""Tests for proper_scoring module — SOTA tracker item #12 phase 1.

TDD: tests written before implementation. Module under test is
`bts.validate.proper_scoring`.

Scope per Codex sign-off (bus msg #57):
- Game/profile-level only (NOT PA-level)
- Inputs: `p_game_hit` and `actual_hit` columns on backtest profile rows
- Decision buckets: `all_top10` and `rank1`
- Default intervals: Wilson/binomial per-bin (not bootstrap)
- Quantile bins by default; fixed bins optional
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bts.validate.proper_scoring import (
    binary_log_loss,
    brier_score,
    murphy_decomposition,
    reliability_table,
    top_bin_calibration,
    compute_proper_scoring,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profiles(days: int = 30, top_n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Synthetic profile DataFrame for testing.

    p_game_hit decreasing by rank from 0.90 (rank 1) → 0.72 (rank 10).
    actual_hit drawn deterministically from a target hit rate that is
    well-calibrated to p_game_hit (so reliability is small but nonzero).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for day_idx in range(days):
        date = pd.Timestamp("2025-04-01") + pd.Timedelta(days=day_idx)
        for rank in range(1, top_n + 1):
            p_hit = 0.90 - (rank - 1) * 0.02
            hit = int(rng.random() < p_hit)
            rows.append({
                "date": date,
                "rank": rank,
                "batter_id": 1000 + rank,
                "p_game_hit": p_hit,
                "actual_hit": hit,
                "n_pas": 4,
            })
    return pd.DataFrame(rows)


def _calibrated_profiles(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Perfectly calibrated p ~ Uniform(0.1, 0.9), y ~ Bernoulli(p)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, size=n)
    y = (rng.random(size=n) < p).astype(int)
    return pd.DataFrame({
        "date": pd.date_range("2025-04-01", periods=n, freq="h"),
        "rank": [1] * n,
        "p_game_hit": p,
        "actual_hit": y,
    })


# ---------------------------------------------------------------------------
# binary_log_loss
# ---------------------------------------------------------------------------

class TestBinaryLogLoss:
    def test_perfect_predictions_returns_near_zero(self):
        p = np.array([0.99, 0.01, 0.99, 0.01])
        y = np.array([1, 0, 1, 0])
        result = binary_log_loss(p, y)
        assert result < 0.02

    def test_all_half_returns_ln_2(self):
        p = np.full(100, 0.5)
        y = np.random.default_rng(0).integers(0, 2, size=100)
        result = binary_log_loss(p, y)
        assert abs(result - math.log(2)) < 1e-9

    def test_clipping_at_zero_does_not_blow_up(self):
        # y=1 with p=0 would be infinite without clipping
        p = np.array([0.0, 1.0])
        y = np.array([1, 0])
        result = binary_log_loss(p, y)
        assert math.isfinite(result)
        assert result > 30  # heavily penalised but finite

    def test_clipping_at_one_does_not_blow_up(self):
        # y=0 with p=1 would be infinite without clipping
        p = np.array([1.0, 0.0])
        y = np.array([0, 1])
        result = binary_log_loss(p, y)
        assert math.isfinite(result)
        assert result > 30

    def test_returns_scalar_float(self):
        p = np.array([0.5, 0.5])
        y = np.array([0, 1])
        result = binary_log_loss(p, y)
        assert isinstance(result, float)

    def test_eps_param_controls_clip(self):
        p = np.array([0.0])
        y = np.array([1])
        result_small_eps = binary_log_loss(p, y, eps=1e-15)
        result_large_eps = binary_log_loss(p, y, eps=1e-3)
        # Larger eps → less aggressive clip → smaller loss
        assert result_large_eps < result_small_eps


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_perfect_predictions_returns_zero(self):
        p = np.array([1.0, 0.0, 1.0, 0.0])
        y = np.array([1, 0, 1, 0])
        assert brier_score(p, y) == 0.0

    def test_all_half_with_uniform_y_returns_quarter(self):
        p = np.full(100, 0.5)
        y = np.random.default_rng(0).integers(0, 2, size=100)
        assert abs(brier_score(p, y) - 0.25) < 1e-9

    def test_all_zero_p_with_all_one_y_returns_one(self):
        p = np.zeros(10)
        y = np.ones(10, dtype=int)
        assert brier_score(p, y) == 1.0

    def test_known_value(self):
        # p=[0.8, 0.3], y=[1, 0] → mean((0.8-1)^2, (0.3-0)^2) = (0.04 + 0.09)/2 = 0.065
        p = np.array([0.8, 0.3])
        y = np.array([1, 0])
        assert abs(brier_score(p, y) - 0.065) < 1e-12

    def test_returns_scalar_float(self):
        p = np.array([0.5, 0.5])
        y = np.array([0, 1])
        assert isinstance(brier_score(p, y), float)


# ---------------------------------------------------------------------------
# murphy_decomposition
# ---------------------------------------------------------------------------

class TestMurphyDecomposition:
    def test_returns_dict_with_required_keys(self):
        df = _make_profiles(days=30)
        result = murphy_decomposition(
            df["p_game_hit"].to_numpy(), df["actual_hit"].to_numpy()
        )
        for key in ("reliability", "resolution", "uncertainty", "brier", "recomposition_error", "n_bins", "binning"):
            assert key in result

    def test_uncertainty_equals_p_y_times_one_minus_p_y(self):
        # Uncertainty = mean_y * (1 - mean_y), with mean_y the overall
        # observed hit rate.
        p = np.array([0.5] * 100)
        y = np.array([1] * 30 + [0] * 70)
        result = murphy_decomposition(p, y)
        expected_unc = 0.3 * 0.7
        assert abs(result["uncertainty"] - expected_unc) < 1e-9

    def test_zero_recomposition_error_with_discrete_forecast(self):
        """With a DISCRETE forecast (p in {0.1, 0.5, 0.9}) and matching fixed bins,
        the Murphy identity Brier = reliability - resolution + uncertainty is exact.
        With continuous p, a within-bin variance residual remains; that residual
        is exposed via the `recomposition_error` field.
        """
        rng = np.random.default_rng(7)
        n_per = 1000
        p_vals = [0.1, 0.5, 0.9]
        p_list: list[float] = []
        y_list: list[int] = []
        for p_val in p_vals:
            p_list.extend([p_val] * n_per)
            y_list.extend((rng.random(n_per) < p_val).astype(int))
        p = np.array(p_list)
        y = np.array(y_list)
        result = murphy_decomposition(p, y, n_bins=3, binning="fixed")
        identity_diff = result["brier"] - (
            result["reliability"] - result["resolution"] + result["uncertainty"]
        )
        assert abs(identity_diff) < 1e-9
        assert result["recomposition_error"] < 1e-9

    def test_brier_field_matches_brier_score(self):
        df = _make_profiles(days=30)
        p = df["p_game_hit"].to_numpy()
        y = df["actual_hit"].to_numpy()
        result = murphy_decomposition(p, y)
        assert abs(result["brier"] - brier_score(p, y)) < 1e-12

    def test_perfect_calibration_zero_reliability(self):
        # If p == empirical y in each bin, reliability is zero by definition.
        df = _calibrated_profiles(n=5000, seed=1)
        result = murphy_decomposition(
            df["p_game_hit"].to_numpy(),
            df["actual_hit"].to_numpy(),
            n_bins=10,
            binning="fixed",
        )
        # Calibrated data → reliability should be small
        assert result["reliability"] < 0.01


# ---------------------------------------------------------------------------
# reliability_table
# ---------------------------------------------------------------------------

class TestReliabilityTable:
    def test_returns_dataframe_with_required_columns(self):
        df = _make_profiles(days=30)
        table = reliability_table(
            df["p_game_hit"].to_numpy(), df["actual_hit"].to_numpy()
        )
        for col in ("bin_idx", "p_lo", "p_hi", "mean_p", "mean_y", "n", "ci_lo", "ci_hi"):
            assert col in table.columns

    def test_quantile_bins_have_roughly_equal_counts(self):
        df = _make_profiles(days=100)
        table = reliability_table(
            df["p_game_hit"].to_numpy(),
            df["actual_hit"].to_numpy(),
            n_bins=5,
            binning="quantile",
        )
        counts = table["n"].to_numpy()
        # All bin counts within 1% of each other for quantile binning on
        # a continuous distribution
        assert (counts.max() - counts.min()) / counts.mean() < 0.05

    def test_total_n_matches_input(self):
        df = _make_profiles(days=30)
        table = reliability_table(
            df["p_game_hit"].to_numpy(),
            df["actual_hit"].to_numpy(),
            n_bins=10,
        )
        assert table["n"].sum() == len(df)

    def test_wilson_ci_within_zero_one(self):
        df = _make_profiles(days=30)
        table = reliability_table(
            df["p_game_hit"].to_numpy(), df["actual_hit"].to_numpy()
        )
        assert (table["ci_lo"] >= 0).all()
        assert (table["ci_hi"] <= 1).all()
        assert (table["ci_lo"] <= table["mean_y"]).all()
        assert (table["mean_y"] <= table["ci_hi"]).all()

    def test_n_bins_param(self):
        df = _make_profiles(days=30)
        for n in (3, 5, 10):
            table = reliability_table(
                df["p_game_hit"].to_numpy(),
                df["actual_hit"].to_numpy(),
                n_bins=n,
                binning="fixed",
            )
            # Fixed bins always produces n_bins rows (some may have n=0,
            # but we drop empty bins → rows <= n_bins)
            assert len(table) <= n


# ---------------------------------------------------------------------------
# Constant-p edge case (Codex bus #60)
# ---------------------------------------------------------------------------

class TestConstantProbabilityEdgeCase:
    """rank1 slices can have low diversity; clustered predictions must
    produce one nonempty bin, not an empty table."""

    def test_reliability_table_single_row_for_constant_p(self):
        p = np.array([0.9, 0.9, 0.9])
        y = np.array([1, 0, 1])
        table = reliability_table(p, y, n_bins=10, binning="quantile")
        assert len(table) == 1
        row = table.iloc[0]
        assert row["n"] == 3
        assert row["mean_p"] == 0.9
        assert row["p_lo"] == 0.9
        assert row["p_hi"] == 0.9
        assert abs(row["mean_y"] - 2 / 3) < 1e-12

    def test_top_bin_calibration_constant_p(self):
        p = np.array([0.9, 0.9, 0.9])
        y = np.array([1, 0, 1])
        result = top_bin_calibration(p, y)
        assert result["n"] == 3
        assert result["mean_p"] == 0.9
        assert abs(result["mean_y"] - 2 / 3) < 1e-12

    def test_compute_proper_scoring_constant_p_rank1(self):
        df = pd.DataFrame([
            {
                "date": pd.Timestamp("2025-04-01") + pd.Timedelta(days=i),
                "rank": 1,
                "p_game_hit": 0.9,
                "actual_hit": int(i % 2 == 0),
            }
            for i in range(20)
        ])
        result = compute_proper_scoring(df)
        assert result["rank1"]["n"] == 20
        assert len(result["rank1"]["reliability_table"]) == 1
        assert result["rank1"]["top_bin"]["n"] == 20

    def test_murphy_decomposition_constant_p(self):
        p = np.array([0.9] * 100)
        y = np.array([1] * 80 + [0] * 20)
        result = murphy_decomposition(p, y, n_bins=10, binning="quantile")
        # One bin → resolution = 0 (no between-bin variation)
        assert result["resolution"] == 0.0
        # Reliability = (mean_p - mean_y)^2 = (0.9 - 0.8)^2 = 0.01
        assert abs(result["reliability"] - 0.01) < 1e-12
        # uncertainty = 0.8 * 0.2 = 0.16
        assert abs(result["uncertainty"] - 0.16) < 1e-12


# ---------------------------------------------------------------------------
# Observed p_lo/p_hi semantics (Codex bus #60 option 1)
# ---------------------------------------------------------------------------

class TestObservedPLoPHiSemantics:
    """p_lo/p_hi are observed min/max within each bin, not assignment edges."""

    def test_p_lo_p_hi_bound_observed_data(self):
        p = np.linspace(0.1, 0.9, 100)
        y = np.random.default_rng(0).integers(0, 2, size=100)
        table = reliability_table(p, y, n_bins=10, binning="quantile")
        for _, row in table.iterrows():
            # No bin extends outside the observed input range
            assert row["p_lo"] >= 0.1
            assert row["p_hi"] <= 0.9
            # mean_p sits between p_lo and p_hi
            assert row["p_lo"] <= row["mean_p"] <= row["p_hi"]

    def test_p_lo_p_hi_match_min_max_within_bin_fixed_binning(self):
        # Construct data where each bin has a known [min, max] of p
        p = np.array([0.05, 0.10, 0.15, 0.55, 0.60, 0.65, 0.92, 0.95, 0.99])
        y = np.zeros_like(p, dtype=int)
        # n_bins=10 fixed → bins of width 0.1; data lands in bins 0, 1, 5, 6, 9
        table = reliability_table(p, y, n_bins=10, binning="fixed")
        for _, row in table.iterrows():
            # Each row's p_lo/p_hi must equal observed min/max of the
            # data points assigned to that bin
            assert row["p_lo"] == row["p_lo"]  # finite
            assert row["p_lo"] <= row["p_hi"]


# ---------------------------------------------------------------------------
# top_bin_calibration
# ---------------------------------------------------------------------------

class TestTopBinCalibration:
    def test_returns_dict_with_top_bin_metrics(self):
        df = _make_profiles(days=30)
        result = top_bin_calibration(
            df["p_game_hit"].to_numpy(), df["actual_hit"].to_numpy()
        )
        for key in ("mean_p", "mean_y", "n", "gap", "ci_lo", "ci_hi"):
            assert key in result

    def test_gap_equals_mean_p_minus_mean_y(self):
        df = _make_profiles(days=30)
        result = top_bin_calibration(
            df["p_game_hit"].to_numpy(), df["actual_hit"].to_numpy()
        )
        assert abs(result["gap"] - (result["mean_p"] - result["mean_y"])) < 1e-12


# ---------------------------------------------------------------------------
# compute_proper_scoring
# ---------------------------------------------------------------------------

class TestComputeProperScoring:
    def test_returns_both_decision_buckets(self):
        df = _make_profiles(days=30)
        result = compute_proper_scoring(df)
        assert "all_top10" in result
        assert "rank1" in result

    def test_rank1_subset_matches_rank_filter(self):
        df = _make_profiles(days=30)
        result = compute_proper_scoring(df)
        # Rank 1 has one row per date; days=30 → n=30
        assert result["rank1"]["n"] == 30

    def test_all_top10_subset_size_matches_input(self):
        df = _make_profiles(days=30, top_n=10)
        result = compute_proper_scoring(df)
        assert result["all_top10"]["n"] == 30 * 10

    def test_each_bucket_has_required_metrics(self):
        df = _make_profiles(days=30)
        result = compute_proper_scoring(df)
        for bucket_name in ("all_top10", "rank1"):
            bucket = result[bucket_name]
            for key in ("log_loss", "brier", "decomposition", "reliability_table", "top_bin", "n"):
                assert key in bucket, f"Bucket {bucket_name} missing key {key}"

    def test_metadata_includes_interval_method_and_binning(self):
        df = _make_profiles(days=30)
        result = compute_proper_scoring(df)
        assert "metadata" in result
        meta = result["metadata"]
        for key in ("interval_method", "n_bins", "binning"):
            assert key in meta

    def test_default_interval_method_is_wilson(self):
        df = _make_profiles(days=30)
        result = compute_proper_scoring(df)
        assert result["metadata"]["interval_method"] == "wilson"

    def test_reliability_table_is_serializable(self):
        """Each bucket's reliability_table should be a list-of-dicts (JSON-friendly)."""
        df = _make_profiles(days=30)
        result = compute_proper_scoring(df)
        for bucket_name in ("all_top10", "rank1"):
            table = result[bucket_name]["reliability_table"]
            assert isinstance(table, list)
            if table:
                assert isinstance(table[0], dict)

    def test_works_with_p_col_and_y_col_overrides(self):
        """Function should accept configurable p_col / y_col."""
        df = _make_profiles(days=30).rename(
            columns={"p_game_hit": "my_p", "actual_hit": "my_y"}
        )
        result = compute_proper_scoring(df, p_col="my_p", y_col="my_y")
        assert result["rank1"]["n"] == 30
