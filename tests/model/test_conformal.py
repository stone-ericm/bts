"""Tests for the conformal lower bounds module."""
from __future__ import annotations

import math

import pytest

from bts.model.conformal import wilson_lower_one_sided


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
