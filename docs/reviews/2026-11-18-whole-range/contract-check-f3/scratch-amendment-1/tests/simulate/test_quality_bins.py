"""Tests for empirical quality bin computation."""

import numpy as np
import pandas as pd
import pytest
from bts.simulate.quality_bins import (
    QualityBin,
    QualityBins,
    compute_bins,
    compute_bins_with_boundaries,
)


def _make_profiles(n_days=100):
    """Create synthetic profile DataFrame with known distribution."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_days):
        date = f"2024-{(i // 28) + 4:02d}-{(i % 28) + 1:02d}"
        # Rank 1: confidence varies, hit rate correlates with confidence
        p1 = 0.75 + rng.random() * 0.15  # 0.75-0.90
        hit1 = 1 if rng.random() < (0.5 + p1 * 0.5) else 0
        # Rank 2: slightly lower
        p2 = p1 - 0.03 + rng.normal(0, 0.01)
        hit2 = 1 if rng.random() < (0.5 + p2 * 0.5) else 0
        for rank, p, hit in [(1, p1, hit1), (2, p2, hit2)]:
            rows.append({"date": date, "rank": rank, "batter_id": rank * 1000,
                          "p_game_hit": p, "actual_hit": hit, "n_pas": 4})
    return pd.DataFrame(rows)


class TestComputeBins:
    def test_returns_5_bins(self):
        df = _make_profiles()
        bins = compute_bins(df)
        assert isinstance(bins, QualityBins)
        assert len(bins.bins) == 5

    def test_frequencies_sum_to_1(self):
        df = _make_profiles()
        bins = compute_bins(df)
        total = sum(b.frequency for b in bins.bins)
        assert abs(total - 1.0) < 0.01

    def test_bins_ordered_by_confidence(self):
        df = _make_profiles()
        bins = compute_bins(df)
        for i in range(len(bins.bins) - 1):
            assert bins.bins[i].p_range[1] <= bins.bins[i + 1].p_range[1]

    def test_p_hit_between_0_and_1(self):
        df = _make_profiles()
        bins = compute_bins(df)
        for b in bins.bins:
            assert 0 <= b.p_hit <= 1
            assert 0 <= b.p_both <= 1

    def test_classify_returns_bin_index(self):
        df = _make_profiles()
        bins = compute_bins(df)
        idx = bins.classify(0.82)
        assert 0 <= idx <= 4

    def test_classify_extreme_values(self):
        df = _make_profiles()
        bins = compute_bins(df)
        assert bins.classify(0.50) == 0  # below all boundaries → lowest bin
        assert bins.classify(0.99) == 4  # above all boundaries → highest bin

    def test_fixed_boundaries_are_preserved(self):
        df = _make_profiles()
        boundaries = [0.78, 0.81, 0.84, 0.87]

        bins = compute_bins_with_boundaries(df, boundaries)

        assert bins.boundaries == boundaries
        assert len(bins.bins) == 5
        assert sum(b.frequency for b in bins.bins) == pytest.approx(1.0)

    def test_fixed_boundaries_keep_empty_bins_for_policy_shape(self):
        df = _make_profiles()
        boundaries = [0.1, 0.2, 0.3, 0.4]

        bins = compute_bins_with_boundaries(df, boundaries)

        assert len(bins.bins) == 5
        assert [b.frequency for b in bins.bins[:4]] == [0.0, 0.0, 0.0, 0.0]
        assert bins.bins[4].frequency == 1.0
