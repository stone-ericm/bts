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

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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
