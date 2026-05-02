"""Tests for the conformal lower bounds module."""
from __future__ import annotations

import math

import pytest

from bts.model.conformal import wilson_lower_one_sided
from bts.model.conformal import (
    BucketWilsonCalibrator,
    fit_bucket_wilson_calibrator,
    apply_bucket_wilson,
)


class TestWilsonLowerOneSided:
    def test_known_textbook_value(self):
        # 80 hits in 100 trials. One-sided 95% lower bound (alpha=0.05).
        # Wilson formula: phat = 0.8, n = 100, z = 1.6449 (one-sided)
        # Expected ~0.728 (validated against scipy.stats.binom_test inversions
        # and standard textbook references).
        result = wilson_lower_one_sided(hits=80, n=100, alpha=0.05)
        assert 0.71 < result < 0.74, f"got {result}"

    def test_perfect_score_below_one(self):
        # 100/100. Lower bound is < 1 because finite n.
        result = wilson_lower_one_sided(hits=100, n=100, alpha=0.05)
        assert 0.95 < result < 1.0

    def test_zero_hits_returns_zero(self):
        result = wilson_lower_one_sided(hits=0, n=100, alpha=0.05)
        assert result == 0.0

    def test_zero_n_returns_zero(self):
        # Edge case: empty bucket — fall back to 0
        result = wilson_lower_one_sided(hits=0, n=0, alpha=0.05)
        assert result == 0.0

    def test_alpha_increases_lower_bound(self):
        # Higher alpha = wider credible interval = LOWER lower bound
        # (more conservative when alpha is smaller / coverage is higher)
        # Wait: alpha=0.05 means 95% coverage which should give the LOWEST
        # lower bound (most conservative); alpha=0.20 means 80% coverage,
        # which should give a HIGHER lower bound (less conservative).
        l_95 = wilson_lower_one_sided(hits=80, n=100, alpha=0.05)
        l_90 = wilson_lower_one_sided(hits=80, n=100, alpha=0.10)
        l_80 = wilson_lower_one_sided(hits=80, n=100, alpha=0.20)
        assert l_95 < l_90 < l_80


class TestBucketWilsonCalibrator:
    def _pairs(self, n_per_bucket: dict[float, tuple[int, int]]):
        """Build calibration pairs from {bucket_low: (n, hits)} spec."""
        out = []
        for low, (n, h) in n_per_bucket.items():
            for i in range(n):
                p = low + 0.0125  # mid of 0.025-wide bucket
                hit = 1 if i < h else 0
                out.append((p, hit))
        return out

    def test_fit_basic_three_buckets(self):
        pairs = self._pairs({
            0.65: (40, 26),  # 65% realized
            0.70: (50, 38),  # 76% realized
            0.75: (60, 50),  # 83% realized
        })
        cal = fit_bucket_wilson_calibrator(pairs, alphas=[0.05, 0.10, 0.20])
        assert isinstance(cal, BucketWilsonCalibrator)
        assert cal.alphas == [0.05, 0.10, 0.20]
        assert cal.bucket_n[0.65] == 40
        assert cal.bucket_hit_rate[0.65] == 26 / 40
        # Three lower bounds per bucket (one per alpha)
        assert len(cal.bucket_lower[0.65]) == 3

    def test_fit_filters_sparse_buckets(self):
        # Buckets with n < 30 (default) should be excluded
        pairs = self._pairs({
            0.50: (10, 5),    # too sparse, dropped
            0.75: (50, 38),   # kept
        })
        cal = fit_bucket_wilson_calibrator(pairs, alphas=[0.10], min_bucket_n=30)
        assert 0.50 not in cal.bucket_n
        assert 0.75 in cal.bucket_n

    def test_apply_returns_bucket_lower(self):
        pairs = self._pairs({0.75: (50, 40)})  # 80% realized
        cal = fit_bucket_wilson_calibrator(pairs, alphas=[0.10])
        # Predicted_p in the bucket [0.75, 0.775) → look up bucket 0.75
        result = apply_bucket_wilson(cal, predicted_p=0.76, alpha_index=0)
        # Wilson lower for 40/50 at alpha=0.10 ≈ 0.71
        assert 0.65 < result < 0.78

    def test_apply_sparse_bucket_returns_none(self):
        # Bucket below min_bucket_n threshold → no lookup → return None
        pairs = self._pairs({0.75: (5, 4)})
        cal = fit_bucket_wilson_calibrator(pairs, alphas=[0.10], min_bucket_n=30)
        result = apply_bucket_wilson(cal, predicted_p=0.76, alpha_index=0)
        assert result is None

    def test_apply_for_p_outside_any_bucket_returns_none(self):
        pairs = self._pairs({0.75: (50, 40)})
        cal = fit_bucket_wilson_calibrator(pairs, alphas=[0.10])
        # p=0.30 falls in a bucket that doesn't exist in calibration
        assert apply_bucket_wilson(cal, predicted_p=0.30, alpha_index=0) is None
