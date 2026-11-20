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
from statistics import NormalDist


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
    z = NormalDist().inv_cdf(1.0 - alpha)  # one-sided
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
