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
