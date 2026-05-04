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


import numpy as np
import pandas as pd
from bts.model.conformal import fit_lr_classifier, compute_lr_weights


class TestFitLRClassifier:
    def test_returns_classifier_with_predict_proba(self):
        # Mock calibration data: 50 rows from "year=2025", 50 from "year=2026"
        # Use a simple feature that differs between groups.
        rng = np.random.default_rng(42)
        cal_features = pd.DataFrame({
            "feat_a": np.concatenate([
                rng.normal(0.5, 0.1, 50),  # year 2025
                rng.normal(0.7, 0.1, 50),  # year 2026 (shifted higher)
            ]),
        })
        years = np.array([2025] * 50 + [2026] * 50)
        clf = fit_lr_classifier(cal_features, years, target_year=2026)
        # Classifier should predict P(year=2026) higher for higher feat_a values
        proba_low = clf.predict_proba([[0.5]])[0, 1]
        proba_high = clf.predict_proba([[0.7]])[0, 1]
        assert proba_high > proba_low

    def test_compute_lr_weights_shape(self):
        rng = np.random.default_rng(42)
        cal_features = pd.DataFrame({
            "feat_a": np.concatenate([
                rng.normal(0.5, 0.1, 50),
                rng.normal(0.7, 0.1, 50),
            ]),
        })
        years = np.array([2025] * 50 + [2026] * 50)
        clf = fit_lr_classifier(cal_features, years, target_year=2026)
        weights = compute_lr_weights(clf, cal_features)
        assert weights.shape == (100,)
        assert np.all(weights > 0)
        # Weights should be higher for rows that "look more like" target
        # (higher feat_a, since year=2026 has higher mean)
        assert weights[-1] > weights[0]  # last (year=2026) row weights > first

    def test_weights_clipped_for_stability(self):
        # Extreme features could give 0/inf weights; helper should clip
        rng = np.random.default_rng(42)
        n = 200
        cal_features = pd.DataFrame({
            "feat_a": np.concatenate([
                rng.normal(0.0, 0.01, n // 2),    # very tight, year 2025
                rng.normal(1.0, 0.01, n // 2),    # very tight, year 2026
            ]),
        })
        years = np.array([2025] * (n // 2) + [2026] * (n // 2))
        clf = fit_lr_classifier(cal_features, years, target_year=2026)
        weights = compute_lr_weights(clf, cal_features)
        # No weight should be exactly 0 or inf
        assert np.all(np.isfinite(weights))
        assert np.all(weights > 0)


from bts.model.conformal import weighted_quantile


class TestWeightedQuantile:
    def test_uniform_weights_match_unweighted(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        # 0.9 quantile of [1..5] uniform = 5 (with finite-sample correction
        # ⌈(5+1)·0.9⌉ / (5+1) = 6/6 = 1.0 → highest score)
        result = weighted_quantile(scores, weights, alpha=0.10, n_for_correction=5)
        assert result == 5.0

    def test_higher_weights_skew_quantile(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        weights = [10.0, 1.0, 1.0, 1.0, 1.0]  # skew toward 1.0
        # Cumulative weight up to score=1 is 10/14 ≈ 0.71; not enough for 0.9
        # Cumulative up to score=2: 11/14 ≈ 0.79
        # Cumulative up to score=3: 12/14 ≈ 0.86
        # Cumulative up to score=4: 13/14 ≈ 0.93 → first to exceed
        # ⌈(5+1)·0.9⌉ / (5+1) = 6/6 = 1.0
        # So we need cumulative >= 1.0 → only score=5 qualifies
        result = weighted_quantile(scores, weights, alpha=0.10, n_for_correction=5)
        assert result == 5.0

    def test_handles_unsorted_input(self):
        scores = [3.0, 1.0, 5.0, 2.0, 4.0]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = weighted_quantile(scores, weights, alpha=0.10, n_for_correction=5)
        assert result == 5.0


from bts.model.conformal import (
    WeightedMondrianConformalCalibrator,
    fit_weighted_mondrian_conformal_calibrator,
    apply_weighted_mondrian_conformal,
)


class TestWeightedMondrianConformal:
    def _build_calibration_data(self, n=200):
        """Build calibration: predicted ~ Beta-ish, hits ~ Bernoulli(predicted * .9)."""
        rng = np.random.default_rng(42)
        predicted = rng.uniform(0.6, 0.85, n)
        hits = (rng.uniform(0, 1, n) < (predicted * 0.95)).astype(int)
        weights = np.ones(n)
        # 16-feature placeholder DataFrame
        features = pd.DataFrame({
            "feat_0": rng.uniform(0, 1, n),
            "feat_1": rng.uniform(0, 1, n),
        })
        return predicted, hits, features, weights

    def test_fit_basic_returns_calibrator(self):
        predicted, hits, features, weights = self._build_calibration_data(n=400)
        cal = fit_weighted_mondrian_conformal_calibrator(
            predicted_p=predicted,
            actual_hit=hits,
            weights=weights,
            alphas=[0.05, 0.10, 0.20],
        )
        assert isinstance(cal, WeightedMondrianConformalCalibrator)
        assert cal.alphas == [0.05, 0.10, 0.20]
        # Marginal quantile populated
        assert len(cal.marginal_quantiles) == 3

    def test_apply_in_populated_bucket(self):
        predicted, hits, features, weights = self._build_calibration_data(n=400)
        cal = fit_weighted_mondrian_conformal_calibrator(
            predicted_p=predicted, actual_hit=hits, weights=weights, alphas=[0.10],
        )
        # Pick a predicted_p in a populated bucket (most fall in [0.6, 0.85))
        result = apply_weighted_mondrian_conformal(cal, predicted_p=0.72, alpha_index=0)
        assert result is not None
        assert 0.0 <= result <= 0.72

    def test_apply_falls_back_to_marginal_for_sparse_bucket(self):
        predicted, hits, features, weights = self._build_calibration_data(n=400)
        cal = fit_weighted_mondrian_conformal_calibrator(
            predicted_p=predicted, actual_hit=hits, weights=weights, alphas=[0.10],
            min_bucket_eff_n=999999,  # force all buckets sparse
        )
        # No bucket meets threshold → marginal fallback
        result = apply_weighted_mondrian_conformal(cal, predicted_p=0.72, alpha_index=0)
        assert result is not None  # marginal still computed

    def test_apply_clamps_to_zero_below(self):
        # When q is large (very over-confident model), L = p - q can go negative.
        # Result must clamp to 0.
        predicted = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])  # all in 0.575-0.625
        hits = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # all miss
        weights = np.ones(8)
        cal = fit_weighted_mondrian_conformal_calibrator(
            predicted_p=predicted, actual_hit=hits, weights=weights,
            alphas=[0.20], min_bucket_eff_n=5,
        )
        result = apply_weighted_mondrian_conformal(cal, predicted_p=0.6, alpha_index=0)
        # score s = 0.6 - 0 = 0.6 for every row; q ≈ 0.6; L = 0.6 - 0.6 = 0
        assert result == 0.0

    def test_apply_clamps_to_predicted_p_above(self):
        # When q is negative (very under-confident), L = p - q > p. Clamp at p.
        predicted = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        hits = np.array([1, 1, 1, 1, 1, 1, 1, 1])  # all hit
        weights = np.ones(8)
        cal = fit_weighted_mondrian_conformal_calibrator(
            predicted_p=predicted, actual_hit=hits, weights=weights,
            alphas=[0.20], min_bucket_eff_n=5,
        )
        result = apply_weighted_mondrian_conformal(cal, predicted_p=0.6, alpha_index=0)
        # score s = 0.6 - 1 = -0.4; q ≈ -0.4; L = 0.6 - (-0.4) = 1.0 → clamp to 0.6
        assert result == 0.6
