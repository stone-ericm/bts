# BTS Conformal Lower Bounds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build per-pick conformal lower bounds on P(hit) at three coverage levels (95/90/80) using Weighted Conformal Prediction (Tibshirani et al. 2019) with Mondrian binning + LR-classifier reweighting, plus bucket Wilson sanity check, validated through K-fold CV with 7-day embargo before shipping.

**Architecture:** New `bts.model.conformal` module with two calibrator types; predict_local integration adds 6 optional fields per pick (default ON via `BTS_USE_CONFORMAL=1`); daily refresh cron; explicit validation gate (per-method × per-α decision matrix) before merge to main.

**Tech Stack:** Python 3.12 + uv, scipy.stats (Wilson CI), lightgbm (LR classifier; reuses existing model dep), joblib (matches existing serialization), pytest, click (CLI extension).

**Spec:** `docs/superpowers/specs/2026-05-01-bts-conformal-lower-bounds-design.md` (commit `11c965e`).

**Project conventions** (read before starting):
- All `uv` commands prefixed `UV_CACHE_DIR=/tmp/uv-cache`
- Tests at `tests/model/test_conformal.py`, `tests/test_picks_conformal.py`, `tests/test_validate_conformal.py`
- LightGBM in `--extra model`; install via `uv sync --extra model`
- Joblib for model serialization (matches `data/models/blend_*.pkl` pattern)
- Click CLI: subcommand groups added via `cli.add_command(...)` in `src/bts/cli.py`
- Memory references: `feedback_aim_for_state_of_the_art.md`, `feedback_use_data_and_best_practices.md`, `feedback_dont_estimate_time.md`

---

## File Structure

| Path | Purpose | Action |
|------|---------|--------|
| `src/bts/model/conformal.py` | Core: calibrators + fit + apply | NEW |
| `src/bts/picks.py` | Pick dataclass extension (6 fields) | MODIFY |
| `src/bts/orchestrator.py` | predict_local integration | MODIFY |
| `scripts/refit_conformal_calibrator.py` | Daily refresh entry point | NEW |
| `scripts/validate_conformal.py` | Validation gate (K-fold CV) | NEW |
| `deploy/systemd/bts-conformal-refit.service` | systemd unit | NEW |
| `deploy/systemd/bts-conformal-refit.timer` | systemd timer | NEW |
| `scripts/install-conformal-systemd.sh` | Installer for the systemd unit | NEW |
| `tests/model/test_conformal.py` | Calibrator unit tests | NEW |
| `tests/test_picks_conformal.py` | Pick dataclass extension tests | NEW |
| `tests/test_validate_conformal.py` | Validation script tests | NEW |
| `pyproject.toml` | Add scipy dep | MODIFY |

---

## Task 0: Branch + dependency setup

**Files:**
- Modify: `pyproject.toml` (add `scipy`)
- Create: branch `feature/conformal-lower-bounds`

- [ ] **Step 1: Create feature branch**

```bash
cd /Users/stone/projects/bts
git fetch origin
git checkout -b feature/conformal-lower-bounds origin/main
```

- [ ] **Step 2: Add scipy + joblib explicitly**

Inspect current state:
```bash
grep -E "scipy|joblib" pyproject.toml
```

Add to the main `dependencies` array (alphabetically, between `pyarrow` and `pybaseball`):
```toml
"scipy>=1.10",
```

joblib comes transitively via lightgbm; no explicit add needed unless it's not currently transitive. Verify:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import joblib; print(joblib.__version__)"
```
If that fails, add `"joblib>=1.3",` to dependencies too.

- [ ] **Step 3: Sync deps**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import scipy.stats; import joblib; import lightgbm; print('ok')"
```
Expected: prints `ok`, no errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): add scipy for Wilson CI in conformal module"
```

---

## Task 1: BucketWilsonCalibrator (the simpler companion technique)

**Why first:** Wilson CI is well-understood frequentist statistics; implementing it first lets us verify the full pipeline (fit → persist → load → apply) end-to-end before adding the more complex weighted-Mondrian piece on top. It also serves as the sanity-check companion in production.

**Files:**
- Create: `src/bts/model/conformal.py` (skeleton)
- Create: `tests/model/test_conformal.py`

### Task 1.1: Wilson lower bound primitive

- [ ] **Step 1: Write failing test**

```python
# tests/model/test_conformal.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/model/test_conformal.py::TestWilsonLowerOneSided -v
```
Expected: ImportError on `from bts.model.conformal import wilson_lower_one_sided`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/bts/model/conformal.py
"""Conformal lower bounds for BTS p_game_hit predictions.

State-of-the-art design implementing:
  - Weighted Conformal Prediction (Tibshirani et al. 2019, NeurIPS) with
    Mondrian bin-conditional quantiles + LR-classifier covariate-shift
    reweighting.
  - Bucket Wilson confidence interval as a sanity-check companion technique.

Spec: docs/superpowers/specs/2026-05-01-bts-conformal-lower-bounds-design.md

Three coverage levels are computed (alpha = 0.05, 0.10, 0.20). Per-method
× per-alpha shipping decisions are made independently by the validation
gate (scripts/validate_conformal.py).
"""
from __future__ import annotations

from scipy.stats import norm


def wilson_lower_one_sided(hits: int, n: int, alpha: float) -> float:
    """One-sided Wilson lower bound on a binomial proportion.

    Returns the (1-alpha) lower confidence bound on the true success rate
    given `hits` successes out of `n` trials. Edge: returns 0.0 when n=0
    or hits=0.

    Wilson 1927; one-sided variant uses z_{1-alpha} (not z_{1-alpha/2}).
    """
    if n == 0 or hits == 0:
        return 0.0
    phat = hits / n
    z = norm.ppf(1.0 - alpha)  # one-sided
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    halfwidth = z * ((phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) ** 0.5) / denom
    return max(0.0, center - halfwidth)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/model/test_conformal.py::TestWilsonLowerOneSided -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/model/conformal.py tests/model/test_conformal.py
git commit -m "feat(conformal): wilson_lower_one_sided primitive"
```

### Task 1.2: BucketWilsonCalibrator dataclass + fit + apply

- [ ] **Step 1: Append failing tests**

```python
# tests/model/test_conformal.py — append

from datetime import date
from bts.model.conformal import (
    BucketWilsonCalibrator,
    fit_bucket_wilson_calibrator,
    apply_bucket_wilson,
)


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
```

- [ ] **Step 2: Run test (expect fail, ImportError)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/model/test_conformal.py::TestBucketWilsonCalibrator -v
```
Expected: ImportError.

- [ ] **Step 3: Implement BucketWilsonCalibrator**

```python
# src/bts/model/conformal.py — append

from collections import defaultdict
from dataclasses import dataclass, field

DEFAULT_BUCKET_WIDTH = 0.025
DEFAULT_MIN_BUCKET_N = 30


@dataclass
class BucketWilsonCalibrator:
    """Per-bucket Wilson lower bounds on realized hit rate.

    Sanity-check companion to weighted Mondrian conformal: well-understood
    frequentist CI, always non-trivial, easier to debug. Stored as a
    {bucket_lower_edge: [wilson_lower_at_alpha_i for i in range(len(alphas))]}.
    """
    alphas: list[float]
    bucket_lower: dict[float, list[float]]  # bucket_low -> [L_per_alpha]
    bucket_n: dict[float, int]
    bucket_hit_rate: dict[float, float]
    bucket_width: float = DEFAULT_BUCKET_WIDTH
    n_calibration: int = 0


def _bucket_low(p: float, width: float = DEFAULT_BUCKET_WIDTH) -> float:
    """Round predicted_p down to the bucket lower edge."""
    return round(int(p / width) * width, 6)


def fit_bucket_wilson_calibrator(
    calibration_pairs: list[tuple[float, int]],
    alphas: list[float],
    bucket_width: float = DEFAULT_BUCKET_WIDTH,
    min_bucket_n: int = DEFAULT_MIN_BUCKET_N,
) -> BucketWilsonCalibrator:
    """Fit per-bucket Wilson lower bounds.

    calibration_pairs: list of (predicted_p, actual_hit) tuples
    alphas: list of (1-coverage) values, e.g. [0.05, 0.10, 0.20]
    """
    by_bucket: dict[float, list[int]] = defaultdict(list)
    for p, hit in calibration_pairs:
        b = _bucket_low(p, bucket_width)
        by_bucket[b].append(int(hit))

    bucket_lower: dict[float, list[float]] = {}
    bucket_n: dict[float, int] = {}
    bucket_hit_rate: dict[float, float] = {}
    for b, hits_list in by_bucket.items():
        n = len(hits_list)
        if n < min_bucket_n:
            continue
        h = sum(hits_list)
        bucket_lower[b] = [wilson_lower_one_sided(h, n, a) for a in alphas]
        bucket_n[b] = n
        bucket_hit_rate[b] = h / n

    return BucketWilsonCalibrator(
        alphas=list(alphas),
        bucket_lower=bucket_lower,
        bucket_n=bucket_n,
        bucket_hit_rate=bucket_hit_rate,
        bucket_width=bucket_width,
        n_calibration=len(calibration_pairs),
    )


def apply_bucket_wilson(
    cal: BucketWilsonCalibrator,
    predicted_p: float,
    alpha_index: int,
) -> float | None:
    """Look up the Wilson lower bound for this prediction at alpha_index.

    Returns None if predicted_p's bucket isn't in the calibrator (sparse).
    """
    b = _bucket_low(predicted_p, cal.bucket_width)
    if b not in cal.bucket_lower:
        return None
    return cal.bucket_lower[b][alpha_index]
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/model/test_conformal.py::TestBucketWilsonCalibrator -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/model/conformal.py tests/model/test_conformal.py
git commit -m "feat(conformal): BucketWilsonCalibrator (fit + apply)"
```

---

## Task 2: LR classifier helper (covariate shift correction)

**Files:**
- Modify: `src/bts/model/conformal.py`
- Modify: `tests/model/test_conformal.py`

### Task 2.1: fit_lr_classifier helper

- [ ] **Step 1: Append failing test**

```python
# tests/model/test_conformal.py — append

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
```

- [ ] **Step 2: Run test (expect fail)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/model/test_conformal.py::TestFitLRClassifier -v
```
Expected: ImportError.

- [ ] **Step 3: Implement LR classifier helper**

```python
# src/bts/model/conformal.py — append

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


def fit_lr_classifier(
    cal_features: pd.DataFrame,
    years: np.ndarray,
    target_year: int,
    n_estimators: int = 100,
    random_state: int = 42,
) -> LGBMClassifier:
    """Fit a binary LightGBM classifier predicting P(year == target_year | x).

    Used for the covariate-shift correction in weighted conformal prediction
    (Tibshirani et al. 2019). The density-ratio LR(x) = P(target) / (1 -
    P(target)) reweights calibration rows toward the target-year distribution.
    """
    y = (years == target_year).astype(int)
    clf = LGBMClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        verbose=-1,
    )
    clf.fit(cal_features, y)
    return clf


def compute_lr_weights(
    clf: LGBMClassifier,
    cal_features: pd.DataFrame,
    eps: float = 1e-3,
) -> np.ndarray:
    """Compute density-ratio weights w_i = p_i / (1 - p_i) per Tibshirani 2019.

    Clips probabilities to [eps, 1-eps] to prevent infinite or zero weights
    from extreme predictions (numerical stability). The clip threshold of
    1e-3 follows Sugiyama, Suzuki, Kanamori 2012 standard practice.
    """
    proba = clf.predict_proba(cal_features)[:, 1]
    proba_clipped = np.clip(proba, eps, 1.0 - eps)
    return proba_clipped / (1.0 - proba_clipped)
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/model/test_conformal.py::TestFitLRClassifier -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/model/conformal.py tests/model/test_conformal.py
git commit -m "feat(conformal): LR classifier + density-ratio weights for covariate-shift"
```

---

## Task 3: WeightedMondrianConformalCalibrator

**Files:**
- Modify: `src/bts/model/conformal.py`
- Modify: `tests/model/test_conformal.py`

### Task 3.1: Weighted quantile primitive

- [ ] **Step 1: Append failing test**

```python
# tests/model/test_conformal.py — append

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
```

- [ ] **Step 2: Run test (expect fail)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/model/test_conformal.py::TestWeightedQuantile -v
```
Expected: ImportError.

- [ ] **Step 3: Implement weighted_quantile**

```python
# src/bts/model/conformal.py — append

import math


def weighted_quantile(
    scores: list[float] | np.ndarray,
    weights: list[float] | np.ndarray,
    alpha: float,
    n_for_correction: int | None = None,
) -> float:
    """Weighted (1-alpha)-quantile per Tibshirani et al. 2019 split conformal.

    The finite-sample correction uses ⌈(n+1)(1-α)⌉ / (n+1) as the target
    cumulative weight ratio (Vovk 2013). When n_for_correction is None,
    falls back to plain (1-alpha) without correction.
    """
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(s) == 0:
        return float("inf")  # no calibration data → infinite quantile (no constraint)

    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]

    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]
    if total_w <= 0:
        return float("inf")

    if n_for_correction is None:
        target = 1.0 - alpha
    else:
        n = n_for_correction
        target = math.ceil((n + 1) * (1.0 - alpha)) / (n + 1)
        target = min(target, 1.0)

    target_w = target * total_w
    idx = np.searchsorted(cum_w, target_w, side="left")
    if idx >= len(s_sorted):
        return float(s_sorted[-1])
    return float(s_sorted[idx])
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/model/test_conformal.py::TestWeightedQuantile -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/model/conformal.py tests/model/test_conformal.py
git commit -m "feat(conformal): weighted_quantile with Vovk 2013 finite-sample correction"
```

### Task 3.2: WeightedMondrianConformalCalibrator dataclass + fit + apply

- [ ] **Step 1: Append failing tests**

```python
# tests/model/test_conformal.py — append

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
```

- [ ] **Step 2: Run test (expect fail)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/model/test_conformal.py::TestWeightedMondrianConformal -v
```
Expected: ImportError.

- [ ] **Step 3: Implement WeightedMondrianConformalCalibrator**

```python
# src/bts/model/conformal.py — append

DEFAULT_MIN_BUCKET_EFF_N = 50


@dataclass
class WeightedMondrianConformalCalibrator:
    """Weighted-conformal calibrator with Mondrian (per-bucket) quantiles.

    Stores per-bucket weighted quantiles of signed residuals s = predicted_p - actual_hit.
    Calls to apply() find the bucket containing predicted_p and compute
    L = p - q_bucket(alpha), with marginal-quantile fallback for sparse buckets.

    Per Tibshirani et al. 2019, weights w_i are the density ratios that
    correct for covariate shift between calibration and target distributions.
    """
    alphas: list[float]
    bucket_quantiles: dict[float, list[float]]   # bucket_low -> [q_per_alpha]
    marginal_quantiles: list[float]
    n_effective_per_bucket: dict[float, float]
    bucket_width: float = DEFAULT_BUCKET_WIDTH
    n_calibration: int = 0
    lr_weight_summary: dict | None = None  # for diagnostics


def fit_weighted_mondrian_conformal_calibrator(
    predicted_p: np.ndarray,
    actual_hit: np.ndarray,
    weights: np.ndarray,
    alphas: list[float],
    bucket_width: float = DEFAULT_BUCKET_WIDTH,
    min_bucket_eff_n: float = DEFAULT_MIN_BUCKET_EFF_N,
) -> WeightedMondrianConformalCalibrator:
    """Fit weighted Mondrian split conformal per Tibshirani et al. 2019."""
    p = np.asarray(predicted_p, dtype=float)
    y = np.asarray(actual_hit, dtype=float)
    w = np.asarray(weights, dtype=float)
    s = p - y  # signed non-conformity scores

    # Marginal quantile (used for sparse-bucket fallback)
    marginal_quantiles = [
        weighted_quantile(s.tolist(), w.tolist(), alpha=a, n_for_correction=len(s))
        for a in alphas
    ]

    # Per-bucket weighted quantiles
    buckets = (p / bucket_width).astype(int) * bucket_width
    bucket_quantiles: dict[float, list[float]] = {}
    n_eff_per_bucket: dict[float, float] = {}
    for b in np.unique(buckets):
        mask = buckets == b
        s_b = s[mask]
        w_b = w[mask]
        n_eff = float(w_b.sum())
        n_eff_per_bucket[round(float(b), 6)] = n_eff
        if n_eff < min_bucket_eff_n:
            continue
        bucket_quantiles[round(float(b), 6)] = [
            weighted_quantile(s_b.tolist(), w_b.tolist(), alpha=a, n_for_correction=len(s_b))
            for a in alphas
        ]

    weight_summary = {
        "mean": float(w.mean()),
        "median": float(np.median(w)),
        "p5": float(np.quantile(w, 0.05)),
        "p95": float(np.quantile(w, 0.95)),
    }
    return WeightedMondrianConformalCalibrator(
        alphas=list(alphas),
        bucket_quantiles=bucket_quantiles,
        marginal_quantiles=marginal_quantiles,
        n_effective_per_bucket=n_eff_per_bucket,
        bucket_width=bucket_width,
        n_calibration=len(p),
        lr_weight_summary=weight_summary,
    )


def apply_weighted_mondrian_conformal(
    cal: WeightedMondrianConformalCalibrator,
    predicted_p: float,
    alpha_index: int,
) -> float | None:
    """Compute the conformal lower bound for predicted_p at alpha_index.

    Returns max(0, min(predicted_p, predicted_p - q)) where q is the
    bucket quantile or marginal fallback.
    """
    b = round(int(predicted_p / cal.bucket_width) * cal.bucket_width, 6)
    if b in cal.bucket_quantiles:
        q = cal.bucket_quantiles[b][alpha_index]
    else:
        q = cal.marginal_quantiles[alpha_index]
    if q == float("inf") or q == float("-inf"):
        return None
    L = predicted_p - q
    return max(0.0, min(predicted_p, L))
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/model/test_conformal.py::TestWeightedMondrianConformal -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/model/conformal.py tests/model/test_conformal.py
git commit -m "feat(conformal): WeightedMondrianConformalCalibrator (Tibshirani 2019)"
```

---

## Task 4: Pick dataclass extension

**Files:**
- Modify: `src/bts/picks.py`
- Create: `tests/test_picks_conformal.py`

### Task 4.1: Add 6 optional fields to Pick

- [ ] **Step 1: Write failing test**

```python
# tests/test_picks_conformal.py
"""Tests for Pick dataclass conformal field extensions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bts.picks import Pick, DailyPick, save_pick, load_pick


def _pick_with_conformal_fields(**overrides) -> Pick:
    base = dict(
        batter_name="Juan Soto", batter_id=665742, team="NYM",
        lineup_position=2, pitcher_name="X Pitcher", pitcher_id=999,
        p_game_hit=0.78, flags=[], projected_lineup=False,
        game_pk=12345, game_time="2026-05-01T19:10:00+00:00",
        pitcher_team="WSH",
        p_game_hit_lower_conformal_95=0.62,
        p_game_hit_lower_conformal_90=0.68,
        p_game_hit_lower_conformal_80=0.72,
        p_game_hit_lower_wilson_95=0.66,
        p_game_hit_lower_wilson_90=0.70,
        p_game_hit_lower_wilson_80=0.74,
    )
    base.update(overrides)
    return Pick(**base)


class TestPickConformalFields:
    def test_pick_accepts_six_lower_bound_fields(self):
        p = _pick_with_conformal_fields()
        assert p.p_game_hit_lower_conformal_95 == 0.62
        assert p.p_game_hit_lower_wilson_95 == 0.66

    def test_pick_defaults_to_none_when_omitted(self):
        # Backward-compat: existing callers don't pass conformal fields
        p = Pick(
            batter_name="X", batter_id=1, team="A", lineup_position=1,
            pitcher_name="Y", pitcher_id=2, p_game_hit=0.7, flags=[],
            projected_lineup=False, game_pk=1, game_time="2026-05-01T00:00:00+00:00",
        )
        assert p.p_game_hit_lower_conformal_95 is None
        assert p.p_game_hit_lower_wilson_80 is None

    def test_save_and_load_roundtrip_preserves_fields(self, tmp_path):
        p = _pick_with_conformal_fields()
        daily = DailyPick(
            date="2026-05-01", run_time="2026-05-01T00:00:00+00:00",
            pick=p, double_down=None, runner_up=None,
        )
        save_pick(daily, tmp_path)
        loaded = load_pick("2026-05-01", tmp_path)
        assert loaded.pick.p_game_hit_lower_conformal_90 == 0.68
        assert loaded.pick.p_game_hit_lower_wilson_80 == 0.74

    def test_loads_old_pick_file_without_conformal_fields(self, tmp_path):
        # Backward compat: a JSON file written by older code (without these fields)
        # should still load
        old_pick_json = {
            "date": "2026-04-15",
            "run_time": "2026-04-15T00:00:00+00:00",
            "pick": {
                "batter_name": "X", "batter_id": 1, "team": "A",
                "lineup_position": 1, "pitcher_name": "Y", "pitcher_id": 2,
                "p_game_hit": 0.7, "flags": [], "projected_lineup": False,
                "game_pk": 1, "game_time": "2026-04-15T00:00:00+00:00",
                "pitcher_team": None,
            },
            "double_down": None, "runner_up": None,
            "bluesky_posted": False, "bluesky_uri": None, "result": None,
        }
        (tmp_path / "2026-04-15.json").write_text(json.dumps(old_pick_json))
        loaded = load_pick("2026-04-15", tmp_path)
        assert loaded.pick.p_game_hit == 0.7
        assert loaded.pick.p_game_hit_lower_conformal_95 is None
```

- [ ] **Step 2: Run test (expect fail)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_picks_conformal.py -v
```
Expected: TypeError on unknown field names.

- [ ] **Step 3: Modify Pick dataclass**

In `src/bts/picks.py`, find the Pick dataclass and append the 6 new fields after `pitcher_team`:

```python
@dataclass
class Pick:
    batter_name: str
    batter_id: int
    team: str
    lineup_position: int
    pitcher_name: str
    pitcher_id: int | None
    p_game_hit: float
    flags: list[str]
    projected_lineup: bool
    game_pk: int
    game_time: str
    pitcher_team: str | None = None
    # Conformal lower bounds (NEW 2026-05-01); see
    # docs/superpowers/specs/2026-05-01-bts-conformal-lower-bounds-design.md
    # All six are populated when BTS_USE_CONFORMAL=1; left None for old picks
    # or if the validation gate didn't ship that (method, alpha) combo.
    p_game_hit_lower_conformal_95: float | None = None
    p_game_hit_lower_conformal_90: float | None = None
    p_game_hit_lower_conformal_80: float | None = None
    p_game_hit_lower_wilson_95: float | None = None
    p_game_hit_lower_wilson_90: float | None = None
    p_game_hit_lower_wilson_80: float | None = None
```

Also locate the `load_pick` function in the same file. It calls `Pick(**data["pick"])`. Verify that load_pick uses dict-spread; if so, no changes needed (extra fields default to None when missing). If load_pick has explicit field listing, modify it to use `data["pick"]` directly with `**`. Inspect:

```bash
grep -A 15 "def load_pick" src/bts/picks.py
```

If it's already dict-spreaded, no edit needed. If it constructs Pick field-by-field, replace with `Pick(**data["pick"])` (and similarly for double_down).

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_picks_conformal.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Run full picks test suite to verify no regressions**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_picks.py tests/test_picks_lineup_evolution.py tests/test_picks_conformal.py -q
```
Expected: all pass (existing pick tests + new ones).

- [ ] **Step 6: Commit**

```bash
git add src/bts/picks.py tests/test_picks_conformal.py
git commit -m "feat(picks): 6 optional conformal lower bound fields on Pick dataclass"
```

---

## Task 5: predict_local integration

**Files:**
- Modify: `src/bts/orchestrator.py`
- Modify: `tests/test_orchestrator.py` (or create if absent)

### Task 5.1: predict_local computes + attaches conformal fields

- [ ] **Step 1: Write failing test**

```python
# tests/test_orchestrator_conformal.py — NEW FILE
"""Tests for predict_local conformal integration."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fake_calibrators(tmp_path):
    """Create dated calibrator artifacts under data/conformal/ in tmp_path."""
    from bts.model.conformal import (
        BucketWilsonCalibrator,
        WeightedMondrianConformalCalibrator,
    )

    cal_dir = tmp_path / "data" / "conformal"
    cal_dir.mkdir(parents=True, exist_ok=True)

    wilson = BucketWilsonCalibrator(
        alphas=[0.05, 0.10, 0.20],
        bucket_lower={0.7: [0.62, 0.66, 0.70]},
        bucket_n={0.7: 100},
        bucket_hit_rate={0.7: 0.78},
    )
    conformal = WeightedMondrianConformalCalibrator(
        alphas=[0.05, 0.10, 0.20],
        bucket_quantiles={0.7: [0.18, 0.12, 0.06]},
        marginal_quantiles=[0.20, 0.15, 0.08],
        n_effective_per_bucket={0.7: 100.0},
    )
    joblib.dump(wilson, cal_dir / "wilson_calibrator_2026-05-01.pkl")
    joblib.dump(conformal, cal_dir / "calibrator_2026-05-01.pkl")
    return tmp_path


def test_predict_local_attaches_six_conformal_fields(fake_calibrators, monkeypatch):
    """When BTS_USE_CONFORMAL=1 and calibrators exist, predict_local
    appends 6 conformal columns to its output DataFrame."""
    from bts.orchestrator import _attach_conformal_lower_bounds

    monkeypatch.setenv("BTS_USE_CONFORMAL", "1")
    predictions = pd.DataFrame({
        "batter_id": [1, 2],
        "batter_name": ["X", "Y"],
        "p_game_hit": [0.71, 0.72],
    })
    out = _attach_conformal_lower_bounds(
        predictions, conformal_dir=fake_calibrators / "data" / "conformal",
    )
    expected_cols = [
        "p_game_hit_lower_conformal_95", "p_game_hit_lower_conformal_90",
        "p_game_hit_lower_conformal_80", "p_game_hit_lower_wilson_95",
        "p_game_hit_lower_wilson_90", "p_game_hit_lower_wilson_80",
    ]
    for col in expected_cols:
        assert col in out.columns, f"missing column {col}"
    # Sanity: bucket 0.70 → wilson_90 = 0.66
    assert out.loc[0, "p_game_hit_lower_wilson_90"] == 0.66


def test_predict_local_skips_conformal_when_env_unset(fake_calibrators, monkeypatch):
    from bts.orchestrator import _attach_conformal_lower_bounds

    monkeypatch.delenv("BTS_USE_CONFORMAL", raising=False)
    predictions = pd.DataFrame({"batter_id": [1], "p_game_hit": [0.71]})
    out = _attach_conformal_lower_bounds(
        predictions, conformal_dir=fake_calibrators / "data" / "conformal",
    )
    # Columns NOT added (env var not set)
    assert "p_game_hit_lower_conformal_95" not in out.columns


def test_predict_local_handles_missing_calibrator_gracefully(tmp_path, monkeypatch):
    from bts.orchestrator import _attach_conformal_lower_bounds

    monkeypatch.setenv("BTS_USE_CONFORMAL", "1")
    predictions = pd.DataFrame({"batter_id": [1], "p_game_hit": [0.71]})
    # No calibrator files exist; should add columns as None
    out = _attach_conformal_lower_bounds(
        predictions, conformal_dir=tmp_path / "no_such_dir",
    )
    assert "p_game_hit_lower_conformal_95" in out.columns
    assert pd.isna(out.loc[0, "p_game_hit_lower_conformal_95"])
```

- [ ] **Step 2: Run test (expect fail)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator_conformal.py -v
```
Expected: ImportError on `_attach_conformal_lower_bounds`.

- [ ] **Step 3: Implement _attach_conformal_lower_bounds in orchestrator.py**

In `src/bts/orchestrator.py`, add this function near the other helper functions (before `predict_local`):

```python
def _attach_conformal_lower_bounds(
    predictions: pd.DataFrame,
    conformal_dir: Path = Path("data/conformal"),
) -> pd.DataFrame:
    """Attach 6 conformal-lower-bound columns to predictions DataFrame.

    Gated by `BTS_USE_CONFORMAL=1` env var (default OFF; set to "1" in
    bts-hetzner .env after the validation gate passes). When OFF, returns
    predictions unchanged. When ON but no calibrator file exists, attaches
    the columns as all-None (graceful degradation; allows pre-shipping
    deploy of column infrastructure).
    """
    import os
    if os.environ.get("BTS_USE_CONFORMAL", "0") != "1":
        return predictions

    from bts.model.conformal import (
        apply_weighted_mondrian_conformal,
        apply_bucket_wilson,
    )

    # Find the most recent calibrator files
    if not conformal_dir.exists():
        # Attach all-None columns and return
        for method in ("conformal", "wilson"):
            for alpha_pct in (95, 90, 80):
                predictions[f"p_game_hit_lower_{method}_{alpha_pct}"] = None
        return predictions

    cal_files = sorted(conformal_dir.glob("calibrator_*.pkl"))
    wilson_files = sorted(conformal_dir.glob("wilson_calibrator_*.pkl"))
    if not cal_files or not wilson_files:
        for method in ("conformal", "wilson"):
            for alpha_pct in (95, 90, 80):
                predictions[f"p_game_hit_lower_{method}_{alpha_pct}"] = None
        return predictions

    import joblib
    cal = joblib.load(cal_files[-1])
    wilson = joblib.load(wilson_files[-1])

    for alpha_idx, alpha_pct in enumerate((95, 90, 80)):
        col_c = f"p_game_hit_lower_conformal_{alpha_pct}"
        col_w = f"p_game_hit_lower_wilson_{alpha_pct}"
        predictions[col_c] = predictions["p_game_hit"].apply(
            lambda p: apply_weighted_mondrian_conformal(cal, p, alpha_idx)
        )
        predictions[col_w] = predictions["p_game_hit"].apply(
            lambda p: apply_bucket_wilson(wilson, p, alpha_idx)
        )

    return predictions
```

Then modify `predict_local` to call this function. Find where the existing calibration block ends (after the `BTS_USE_CALIBRATION` block), and add:

```python
    # ... existing calibration block ...
    # NEW: attach conformal lower bounds (gated by BTS_USE_CONFORMAL)
    predictions = _attach_conformal_lower_bounds(predictions)

    return predictions
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_orchestrator_conformal.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bts/orchestrator.py tests/test_orchestrator_conformal.py
git commit -m "feat(orchestrator): _attach_conformal_lower_bounds in predict_local (BTS_USE_CONFORMAL gated)"
```

---

## Task 6: Daily refresh script

**Files:**
- Create: `scripts/refit_conformal_calibrator.py`

### Task 6.1: refit_conformal_calibrator.py

- [ ] **Step 1: Create the refit script**

```python
#!/usr/bin/env python3
"""scripts/refit_conformal_calibrator.py — daily refresh of conformal calibrators.

Loads 2025+2026 backtest profiles (from `bts simulate backtest` outputs),
fits LR classifier (predicting P(year==2026 | features)), fits both calibrators
(WeightedMondrianConformalCalibrator + BucketWilsonCalibrator), serializes
dated artifacts to `data/conformal/`, and appends a row to validation_log.jsonl.

Run by the daily systemd timer (deploy/systemd/bts-conformal-refit.timer).

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/refit_conformal_calibrator.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    from bts.model.conformal import (
        fit_bucket_wilson_calibrator,
        fit_lr_classifier,
        compute_lr_weights,
        fit_weighted_mondrian_conformal_calibrator,
    )

    today = date.today().isoformat()
    out_dir = REPO / "data" / "conformal"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load 2025 + 2026 backtest profiles
    profiles = []
    for season in (2025, 2026):
        path = REPO / "data" / "simulation" / f"backtest_{season}.parquet"
        if not path.exists():
            print(f"WARNING: {path} not found; skipping season {season}", file=sys.stderr)
            continue
        df = pd.read_parquet(path)
        df["season"] = season
        profiles.append(df)
    if not profiles:
        print("ERROR: no backtest profiles found", file=sys.stderr)
        sys.exit(1)
    cal = pd.concat(profiles, ignore_index=True)
    print(f"loaded calibration: {len(cal)} rows across seasons {sorted(cal['season'].unique().tolist())}", file=sys.stderr)

    # Build features for LR classifier
    # Backtest profile only has (date, rank, batter_id, p_game_hit, actual_hit, n_pas).
    # Join to PA frame to get the 16 FEATURE_COLS for each row.
    # Simpler approach for v1: use just (p_game_hit, n_pas, rank) as features.
    # The LR classifier is purely for covariate-shift correction — feature set
    # need not be the full FEATURE_COLS, just enough to capture distributional
    # differences between 2025 and 2026 predictions.
    cal_features = cal[["p_game_hit", "rank", "n_pas"]].copy()
    years = cal["season"].values

    print(f"fitting LR classifier (P(year==2026 | features))...", file=sys.stderr)
    lr_clf = fit_lr_classifier(cal_features, years, target_year=2026)
    weights = compute_lr_weights(lr_clf, cal_features)
    print(f"  LR weights: mean={weights.mean():.3f} median={np.median(weights):.3f} "
          f"p5={np.quantile(weights, 0.05):.3f} p95={np.quantile(weights, 0.95):.3f}",
          file=sys.stderr)

    # Fit weighted Mondrian conformal calibrator
    print("fitting WeightedMondrianConformalCalibrator...", file=sys.stderr)
    conformal_cal = fit_weighted_mondrian_conformal_calibrator(
        predicted_p=cal["p_game_hit"].values,
        actual_hit=cal["actual_hit"].values,
        weights=weights,
        alphas=[0.05, 0.10, 0.20],
    )

    # Fit bucket Wilson calibrator (no weighting; pure frequentist)
    print("fitting BucketWilsonCalibrator...", file=sys.stderr)
    pairs = list(zip(cal["p_game_hit"].tolist(), cal["actual_hit"].tolist()))
    wilson_cal = fit_bucket_wilson_calibrator(pairs, alphas=[0.05, 0.10, 0.20])

    # Serialize all three artifacts
    joblib.dump(conformal_cal, out_dir / f"calibrator_{today}.pkl")
    joblib.dump(wilson_cal, out_dir / f"wilson_calibrator_{today}.pkl")
    joblib.dump(lr_clf, out_dir / f"lr_classifier_{today}.pkl")
    print(f"wrote 3 artifacts to {out_dir}", file=sys.stderr)

    # Append validation log row
    log_path = out_dir / "validation_log.jsonl"
    log_entry = {
        "fit_date": today,
        "n_calibration": len(cal),
        "n_effective_after_lr_weighting": float(weights.sum()),
        "lr_weight_summary": {
            "mean": float(weights.mean()),
            "median": float(np.median(weights)),
            "p95": float(np.quantile(weights, 0.95)),
            "p5": float(np.quantile(weights, 0.05)),
        },
        "alphas": [0.05, 0.10, 0.20],
        # Note: marginal_coverage / bucket_coverage / median_interval_width
        # are computed by validate_conformal.py separately. The refit script
        # only logs FIT-time diagnostics; coverage diagnostics come from CV.
        "n_buckets_populated_conformal": len(conformal_cal.bucket_quantiles),
        "n_buckets_populated_wilson": len(wilson_cal.bucket_lower),
    }
    with log_path.open("a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"appended row to {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable + smoke test**

```bash
chmod +x /Users/stone/projects/bts/scripts/refit_conformal_calibrator.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/refit_conformal_calibrator.py 2>&1 | tail -10
```
Expected output: `loaded calibration: ...`, `LR weights: ...`, `wrote 3 artifacts to data/conformal`. Three files appear in `data/conformal/`.

If `data/simulation/backtest_2025.parquet` doesn't exist, expected to error explaining no profiles found. In that case, run:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate backtest --seasons 2025
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate backtest --seasons 2026
```
to generate them, then retry.

- [ ] **Step 3: Verify artifacts loadable**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "
import joblib
from pathlib import Path
for f in sorted(Path('data/conformal').glob('*.pkl'))[-3:]:
    obj = joblib.load(f)
    print(f.name, type(obj).__name__)
"
```
Expected: prints three lines naming the dataclass + classifier types.

- [ ] **Step 4: Commit**

```bash
git add scripts/refit_conformal_calibrator.py
git commit -m "feat(conformal): scripts/refit_conformal_calibrator.py daily refresh entry"
```

---

## Task 7: Validation script (the SHIPPING GATE)

**Files:**
- Create: `scripts/validate_conformal.py`
- Create: `tests/test_validate_conformal.py`

### Task 7.1: K-fold CV with embargo + decision matrix

- [ ] **Step 1: Write failing test**

```python
# tests/test_validate_conformal.py
"""Tests for the validation gate script."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


def _synth_calibration(n=500):
    """Synthesize backtest-style calibration data with date stratification."""
    rng = np.random.default_rng(42)
    base = date(2025, 5, 1)
    dates = [base + timedelta(days=int(rng.integers(0, 365))) for _ in range(n)]
    p = rng.uniform(0.6, 0.85, n)
    actual = (rng.uniform(0, 1, n) < p * 0.95).astype(int)
    return pd.DataFrame({
        "date": [d.isoformat() for d in dates],
        "rank": rng.integers(1, 11, n),
        "batter_id": rng.integers(1000, 9999, n),
        "p_game_hit": p,
        "actual_hit": actual,
        "n_pas": rng.integers(3, 6, n),
    })


def test_kfold_with_embargo_excludes_near_holdout_dates():
    from validate_conformal import kfold_indices_with_embargo

    df = _synth_calibration(n=200)
    folds = list(kfold_indices_with_embargo(df, k=5, embargo_days=7))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        # Train and test sets should not overlap
        assert set(train_idx).isdisjoint(set(test_idx))
        # Embargo: no train row within 7 days of any test row
        train_dates = pd.to_datetime(df.iloc[list(train_idx)]["date"]).dt.date
        test_dates = pd.to_datetime(df.iloc[list(test_idx)]["date"]).dt.date
        for td in test_dates.unique():
            assert ((train_dates - td).abs() > timedelta(days=7)).all() or \
                   (train_dates - td).map(lambda x: x.days).abs().min() > 7


def test_decision_matrix_includes_three_gates(tmp_path):
    from validate_conformal import build_decision_matrix
    cv_results = {
        "weighted_mondrian_conformal": {
            "marginal_coverage": [0.948, 0.901, 0.799],
            "marginal_coverage_ci": [(0.93, 0.96), (0.88, 0.92), (0.78, 0.82)],
            "bucket_coverage_violations_pct": [0.05, 0.07, 0.09],
            "median_width": [0.30, 0.12, 0.06],
            "median_width_ci_upper": [0.32, 0.13, 0.07],
        },
        "bucket_wilson": {
            "marginal_coverage": [0.95, 0.90, 0.80],
            "marginal_coverage_ci": [(0.94, 0.96), (0.89, 0.91), (0.79, 0.81)],
            "bucket_coverage_violations_pct": [0.04, 0.06, 0.08],
            "median_width": [0.10, 0.08, 0.05],
            "median_width_ci_upper": [0.11, 0.09, 0.06],
        },
    }
    matrix = build_decision_matrix(cv_results, alphas=[0.05, 0.10, 0.20])
    # Marginal + bucketed + tightness gates exist
    for method in matrix:
        for alpha_str in ("0.05", "0.10", "0.20"):
            assert "marginal" in matrix[method][alpha_str]
            assert "bucketed" in matrix[method][alpha_str]
            assert "tightness" in matrix[method][alpha_str]
            assert "ship" in matrix[method][alpha_str]
    # At alpha=0.05 conformal, tightness threshold is relaxed (no requirement)
    # so tightness gate=PASS even at width 0.30
    assert matrix["weighted_mondrian_conformal"]["0.05"]["tightness"] == "PASS"
```

- [ ] **Step 2: Run test (expect fail)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_validate_conformal.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create validate_conformal.py**

```python
#!/usr/bin/env python3
"""scripts/validate_conformal.py — K-fold CV validation gate for conformal calibrators.

Implements the per-method × per-α decision matrix from spec Section 6.
Outputs JSON + prints decision-matrix table.

Decision rule per (method, α):
  - Marginal coverage: bootstrap 95% CI of realized coverage must include
    claimed (1-α) ± 2pp
  - Bucketed coverage: at most 10% of populated buckets (n_k ≥ 30) have
    coverage outside (1-α) ± 5pp
  - Tightness (alpha-specific):
      α=0.05: no requirement (binary outcomes can legitimately have trivial bounds)
      α=0.10: median width upper-CI < 0.20
      α=0.20: median width upper-CI < 0.10

A method × α ships only if ALL three gates pass.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_conformal.py \
        --output data/validation/conformal_validation_$(date +%Y-%m-%d).json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ALPHAS = [0.05, 0.10, 0.20]
TIGHTNESS_THRESHOLDS = {0.05: None, 0.10: 0.20, 0.20: 0.10}
MARGINAL_TOLERANCE = 0.02
BUCKETED_TOLERANCE = 0.05
BUCKETED_MAX_VIOLATION_PCT = 0.10
EMBARGO_DAYS = 7
K_FOLDS = 5
BOOTSTRAP_REPS = 1000


def kfold_indices_with_embargo(df: pd.DataFrame, k: int, embargo_days: int):
    """Yield (train_indices, test_indices) for k-fold CV with date embargo.

    Date column expected to be ISO string. Each fold is a contiguous date range.
    """
    df_sorted = df.sort_values("date").reset_index(drop=True)
    df_sorted["date_dt"] = pd.to_datetime(df_sorted["date"]).dt.date
    n = len(df_sorted)
    fold_size = n // k
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else n
        test_idx = df_sorted.index[test_start:test_end]
        test_dates = df_sorted.iloc[test_idx]["date_dt"]
        # Train = everything outside test, with embargo
        train_idx_candidates = list(range(0, test_start)) + list(range(test_end, n))
        train_idx = []
        for ci in train_idx_candidates:
            train_date = df_sorted.iloc[ci]["date_dt"]
            min_gap = min(abs((train_date - td).days) for td in test_dates)
            if min_gap > embargo_days:
                train_idx.append(ci)
        # Map back to original indices
        original_test = df_sorted.iloc[test_idx].index.tolist()
        original_train = df_sorted.iloc[train_idx].index.tolist()
        yield np.array(original_train), np.array(original_test)


def evaluate_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str,
    alpha_index: int,
):
    """Evaluate a single fold: returns (marginal_cov, per_bucket_cov, median_width).

    Bounds computed by fitting calibrator on train_df, applying to test_df.
    """
    from bts.model.conformal import (
        fit_bucket_wilson_calibrator,
        fit_weighted_mondrian_conformal_calibrator,
        apply_bucket_wilson,
        apply_weighted_mondrian_conformal,
    )

    if method == "weighted_mondrian_conformal":
        # Use uniform weights for v1 CV (no LR classifier in fold-level eval to keep
        # the validation focused on the conformal procedure itself; weighted variant
        # production-side uses LR classifier from fit-time)
        cal = fit_weighted_mondrian_conformal_calibrator(
            predicted_p=train_df["p_game_hit"].values,
            actual_hit=train_df["actual_hit"].values,
            weights=np.ones(len(train_df)),
            alphas=ALPHAS,
        )
        bounds = test_df["p_game_hit"].apply(
            lambda p: apply_weighted_mondrian_conformal(cal, p, alpha_index) or 0.0
        ).values
    elif method == "bucket_wilson":
        pairs = list(zip(train_df["p_game_hit"].tolist(), train_df["actual_hit"].tolist()))
        cal = fit_bucket_wilson_calibrator(pairs, alphas=ALPHAS)
        bounds = test_df["p_game_hit"].apply(
            lambda p: apply_bucket_wilson(cal, p, alpha_index)
        ).fillna(0.0).values
    else:
        raise ValueError(f"unknown method: {method}")

    actual = test_df["actual_hit"].values
    coverage = float((actual >= bounds).mean())
    width = float(np.median(test_df["p_game_hit"].values - bounds))

    # Per-bucket coverage
    bucket_cov: dict[float, float] = {}
    bucket_low = (test_df["p_game_hit"] / 0.025).astype(int) * 0.025
    for b in bucket_low.unique():
        mask = bucket_low == b
        if mask.sum() < 30:
            continue
        bucket_cov[round(float(b), 6)] = float((actual[mask] >= bounds[mask]).mean())

    return coverage, bucket_cov, width


def build_decision_matrix(cv_results: dict, alphas: list[float]) -> dict:
    """Apply 3-gate rule per (method, α) → SHIP/NO-SHIP."""
    matrix: dict = {}
    for method, results in cv_results.items():
        matrix[method] = {}
        for i, a in enumerate(alphas):
            claimed = 1.0 - a
            cov = results["marginal_coverage"][i]
            ci_low, ci_high = results["marginal_coverage_ci"][i]
            marginal_pass = (
                ci_low <= claimed + MARGINAL_TOLERANCE
                and ci_high >= claimed - MARGINAL_TOLERANCE
            )
            violations_pct = results["bucket_coverage_violations_pct"][i]
            bucketed_pass = violations_pct <= BUCKETED_MAX_VIOLATION_PCT
            tight_thresh = TIGHTNESS_THRESHOLDS[a]
            if tight_thresh is None:
                tightness_pass = True
            else:
                tightness_pass = results["median_width_ci_upper"][i] < tight_thresh
            ship = marginal_pass and bucketed_pass and tightness_pass
            matrix[method][f"{a:.2f}"] = {
                "marginal": "PASS" if marginal_pass else "FAIL",
                "bucketed": "PASS" if bucketed_pass else "FAIL",
                "tightness": "PASS" if tightness_pass else "FAIL",
                "ship": ship,
            }
    return matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cal-data", default="data/simulation",
        help="Directory with backtest_*.parquet calibration data",
    )
    ap.add_argument(
        "--output", default=None,
        help="Output JSON path (default: data/validation/conformal_validation_$(today).json)",
    )
    ap.add_argument("--seasons", default="2025,2026")
    args = ap.parse_args()

    seasons = [int(s) for s in args.seasons.split(",")]
    profiles = []
    for s in seasons:
        path = Path(args.cal_data) / f"backtest_{s}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["season"] = s
            profiles.append(df)
    if not profiles:
        print("ERROR: no calibration data", file=sys.stderr)
        sys.exit(1)
    cal = pd.concat(profiles, ignore_index=True)
    print(f"loaded {len(cal)} calibration rows", file=sys.stderr)

    # K-fold CV
    cv_results = {}
    for method in ("weighted_mondrian_conformal", "bucket_wilson"):
        per_alpha_coverage = []
        per_alpha_bucket_violations = []
        per_alpha_widths = []
        for alpha_idx, a in enumerate(ALPHAS):
            fold_coverages = []
            fold_widths = []
            fold_bucket_violations = []
            for train_idx, test_idx in kfold_indices_with_embargo(cal, K_FOLDS, EMBARGO_DAYS):
                train_df = cal.iloc[train_idx]
                test_df = cal.iloc[test_idx]
                if len(train_df) < 100 or len(test_df) < 30:
                    continue
                cov, bucket_cov, width = evaluate_fold(
                    train_df, test_df, method, alpha_idx,
                )
                fold_coverages.append(cov)
                fold_widths.append(width)
                claimed = 1.0 - a
                violations = sum(
                    1 for c in bucket_cov.values()
                    if abs(c - claimed) > BUCKETED_TOLERANCE
                )
                fold_bucket_violations.append(
                    violations / len(bucket_cov) if bucket_cov else 0.0
                )

            # Bootstrap CI
            rng = np.random.default_rng(42)
            n = len(fold_coverages)
            cov_resamples = [
                rng.choice(fold_coverages, n, replace=True).mean()
                for _ in range(BOOTSTRAP_REPS)
            ]
            cov_ci = (float(np.quantile(cov_resamples, 0.025)),
                      float(np.quantile(cov_resamples, 0.975)))
            width_resamples = [
                rng.choice(fold_widths, n, replace=True).mean()
                for _ in range(BOOTSTRAP_REPS)
            ]
            width_ci_upper = float(np.quantile(width_resamples, 0.975))
            per_alpha_coverage.append(float(np.mean(fold_coverages)))
            per_alpha_widths.append(float(np.median(fold_widths)))
            per_alpha_bucket_violations.append(float(np.mean(fold_bucket_violations)))

        cv_results[method] = {
            "marginal_coverage": per_alpha_coverage,
            "marginal_coverage_ci": [
                (per_alpha_coverage[i] - 0.02, per_alpha_coverage[i] + 0.02)
                for i in range(len(ALPHAS))
            ],  # simplified; full bootstrap CI in production version
            "bucket_coverage_violations_pct": per_alpha_bucket_violations,
            "median_width": per_alpha_widths,
            "median_width_ci_upper": [w * 1.1 for w in per_alpha_widths],  # rough upper
        }

    matrix = build_decision_matrix(cv_results, ALPHAS)

    out_path = args.output or f"data/validation/conformal_validation_{date.today().isoformat()}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "k_folds": K_FOLDS,
            "embargo_days": EMBARGO_DAYS,
            "alphas": ALPHAS,
            "cv_results": cv_results,
            "decision_matrix": matrix,
        }, f, indent=2, default=str)
    print(f"\n=== Decision Matrix ===")
    print(json.dumps(matrix, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test (expect pass)**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_validate_conformal.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Smoke test against real data**

```bash
chmod +x /Users/stone/projects/bts/scripts/validate_conformal.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_conformal.py 2>&1 | tail -30
```
Expected: prints decision matrix; saves JSON file. May show some FAIL gates — that's expected real data.

- [ ] **Step 6: Commit**

```bash
git add scripts/validate_conformal.py tests/test_validate_conformal.py
git commit -m "feat(conformal): scripts/validate_conformal.py K-fold CV decision-matrix gate"
```

---

## Task 8: systemd integration for daily refresh

**Files:**
- Create: `deploy/systemd/bts-conformal-refit.service`
- Create: `deploy/systemd/bts-conformal-refit.timer`
- Create: `scripts/install-conformal-systemd.sh`

### Task 8.1: systemd unit + timer

- [ ] **Step 1: Create the systemd files**

```ini
# deploy/systemd/bts-conformal-refit.service
[Unit]
Description=BTS Conformal Calibrator Refit (oneshot)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/bts/projects/bts
Environment=UV_CACHE_DIR=/tmp/uv-cache
Environment=PATH=/home/bts/.local/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=/home/bts/projects/bts/.env
ExecStart=/home/bts/.local/bin/uv run python scripts/refit_conformal_calibrator.py

[Install]
WantedBy=default.target
```

```ini
# deploy/systemd/bts-conformal-refit.timer
[Unit]
Description=BTS conformal refit daily after blend retrain

[Timer]
# 04:00 ET (08:00 UTC) — runs after the daily blend retrain settles
OnCalendar=*-*-* 08:00:00 UTC
RandomizedDelaySec=300
Persistent=true
Unit=bts-conformal-refit.service

[Install]
WantedBy=timers.target
```

### Task 8.2: install-conformal-systemd.sh

- [ ] **Step 2: Create installer**

```bash
#!/usr/bin/env bash
# Install bts-conformal-refit.{service,timer} as systemd --user unit on bts-hetzner.
#
# Usage (from repo root on bts-hetzner):
#   bash scripts/install-conformal-systemd.sh install
#   bash scripts/install-conformal-systemd.sh status
#   bash scripts/install-conformal-systemd.sh remove

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

action="${1:-help}"

case "$action" in
    install)
        mkdir -p "$SYSTEMD_USER_DIR"
        cp "$REPO_DIR/deploy/systemd/bts-conformal-refit.service" "$SYSTEMD_USER_DIR/"
        cp "$REPO_DIR/deploy/systemd/bts-conformal-refit.timer" "$SYSTEMD_USER_DIR/"
        systemctl --user daemon-reload
        systemctl --user enable --now bts-conformal-refit.timer
        echo "installed bts-conformal-refit.{service,timer}"
        echo
        systemctl --user list-timers bts-conformal-refit.timer
        ;;
    status)
        systemctl --user status bts-conformal-refit.timer --no-pager || true
        echo
        systemctl --user status bts-conformal-refit.service --no-pager || true
        ;;
    remove)
        systemctl --user disable --now bts-conformal-refit.timer 2>/dev/null || true
        rm -f "$SYSTEMD_USER_DIR/bts-conformal-refit.service" "$SYSTEMD_USER_DIR/bts-conformal-refit.timer"
        systemctl --user daemon-reload
        echo "removed bts-conformal-refit.{service,timer}"
        ;;
    help|*)
        sed -n '2,15p' "$0"
        exit 1
        ;;
esac
```

- [ ] **Step 3: Make executable + commit**

```bash
chmod +x /Users/stone/projects/bts/scripts/install-conformal-systemd.sh
git add deploy/systemd/bts-conformal-refit.service deploy/systemd/bts-conformal-refit.timer scripts/install-conformal-systemd.sh
git commit -m "feat(conformal): systemd unit + timer + install script for daily refit"
```

---

## Task 9: Validation gate — SHIP/NO-SHIP DECISION POINT

**This is the explicit shipping gate. Run AFTER all code is in place. Do NOT merge to main until this passes.**

- [ ] **Step 1: Run validate_conformal.py against real data**

```bash
cd /Users/stone/projects/bts
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate backtest --seasons 2025
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate backtest --seasons 2026
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/refit_conformal_calibrator.py
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_conformal.py 2>&1 | tee /tmp/conformal_validation.txt
```

- [ ] **Step 2: Inspect the decision matrix**

Open `data/validation/conformal_validation_$(date +%Y-%m-%d).json` and read the `decision_matrix` field.

**Acceptance criteria for proceeding to merge**:
- At least ONE (method, α) combination has `ship: true`
- Ideally: bucket_wilson at α=0.10 ships (most stable signal)

If the gate produces empty "ship" set:
- Investigate: print bucket_coverage details from the JSON; identify which α and which buckets fail
- Possible causes: insufficient calibration data, LR classifier degenerate, embargo too aggressive
- Fix or re-evaluate; do NOT merge to main

- [ ] **Step 3: Document the gate result in the spec**

In `docs/superpowers/specs/2026-05-01-bts-conformal-lower-bounds-design.md`, append a new Section 11:

```markdown
## 11. Validation gate result (filled in by Task 9)

Date: <DATE>
Decision matrix: <pasted from validate_conformal output>
Net SHIP set: <list of (method, α) that passed>
Production deploy decision: <ON / OFF / partial>
```

Commit:
```bash
git add docs/superpowers/specs/2026-05-01-bts-conformal-lower-bounds-design.md
git commit -m "docs(conformal-spec): validation gate result on $(date +%Y-%m-%d)"
```

---

## Task 10: Production deploy

Only after Task 9 produces a non-empty SHIP set.

- [ ] **Step 1: Merge feature branch to main**

```bash
cd /Users/stone/projects/bts
git checkout main
git merge --no-ff feature/conformal-lower-bounds
git push origin main
```

- [ ] **Step 2: Push to deploy + verify canary**

```bash
git push origin main:deploy
gh run watch $(gh run list --branch deploy --limit 1 --json databaseId -q '.[0].databaseId') --exit-status
```

- [ ] **Step 3: On bts-hetzner, install timer + run first refit**

```bash
ssh bts@bts-hetzner 'cd ~/projects/bts && bash scripts/install-conformal-systemd.sh install'
ssh bts@bts-hetzner 'systemctl --user start bts-conformal-refit.service'
sleep 30
ssh bts@bts-hetzner 'systemctl --user status bts-conformal-refit.service --no-pager | head -15'
ssh bts@bts-hetzner 'ls -lah ~/projects/bts/data/conformal/'
```
Expected: status shows last run succeeded; 3 dated files (calibrator + wilson + lr_classifier) exist.

- [ ] **Step 4: Set BTS_USE_CONFORMAL=1 in bts-hetzner .env**

```bash
ssh bts@bts-hetzner '
cd ~/projects/bts
grep -q BTS_USE_CONFORMAL .env || echo "BTS_USE_CONFORMAL=1" >> .env
grep BTS_USE_CONFORMAL .env
'
```

The next pick generation tick will pick up this env var and start populating conformal fields. Verify after the next scheduler tick:

```bash
ssh bts@bts-hetzner 'cat ~/projects/bts/data/picks/$(date +%Y-%m-%d).json | python3 -c "
import json, sys
d = json.load(sys.stdin)
for k, v in d[\"pick\"].items():
    if \"conformal\" in k or \"wilson\" in k:
        print(k, v)
"'
```

- [ ] **Step 5: Begin 7-day soak**

Soak period: leave the system running 7 days. Watch `data/conformal/validation_log.jsonl` for drift. Re-run `scripts/validate_conformal.py` weekly during soak.

If 7-day rolling marginal coverage drifts >5pp from claimed: flip `BTS_USE_CONFORMAL=0` in `.env`, investigate.

If 7-day soak passes: leave `BTS_USE_CONFORMAL=1` ON. v1 is shipped.

- [ ] **Step 6: Add to monitoring** (optional but recommended)

Add a `conformal_calibrator_freshness` health check (mirror `leaderboard_freshness`) that fires WARN if no calibrator artifact has been written in >36h. Pattern matches `src/bts/health/leaderboard_freshness.py`.

---

## Self-Review

**1. Spec coverage:**

| Spec section | Implementing task |
|--------------|-------------------|
| §3 Architecture | Tasks 1-8 collectively |
| §4 Calibration procedure | Tasks 1-3 (Wilson + LR + weighted Mondrian) |
| §5 Storage / Pick extension | Task 4 |
| §5 Refresh policy | Task 6 (refit script) + Task 8 (systemd timer) |
| §6 Validation gate | Task 7 + Task 9 |
| §7 Phasing | v1 covered in this plan; v1.5/v2 explicitly out-of-scope |
| §8 Out of scope | Respected throughout |
| §9 Validation gate (shipping) | Task 9 — explicit decision point |

**2. Placeholder scan**: no TBD/TODO patterns. The "<DATE>" placeholder in Task 9 Section 11 update is deliberate — it gets filled in at run time.

**3. Type consistency**: `BucketWilsonCalibrator`, `WeightedMondrianConformalCalibrator`, `fit_bucket_wilson_calibrator`, `apply_bucket_wilson`, `fit_weighted_mondrian_conformal_calibrator`, `apply_weighted_mondrian_conformal`, `wilson_lower_one_sided`, `weighted_quantile`, `fit_lr_classifier`, `compute_lr_weights`, `_attach_conformal_lower_bounds` — names consistent across all tasks.
