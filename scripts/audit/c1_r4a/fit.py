"""C1 rank 4a, T4: the one calibration map (registration `docs/sota_audit/2026-10-04-prereg-c1-calibration.md` §§1, 3).

g_a(p) = logistic(logit(p) + a), with a Normal(0, 0.5²) prior on a. The MAP maximizes

    Σ_d (1/n_d) Σ_i [y log g_a(p) + (1 − y) log(1 − g_a(p))] − a² / (2 · 0.5²)

over the eligible fit dates. Date means are summed: a date with many rows counts once. This is a fixed weighted
likelihood with a regularizing prior, not a calibrated posterior. p is clipped to [1e−15, 1 − 1e−15] for fitting;
invalid, non-numeric, boolean, non-finite or out-of-[0, 1] probabilities are refused (the population step excludes
and counts them first). The objective is strictly concave, so Newton's method converges. A fit needs at least 25 fit
dates with known eligible rows; otherwise the study is inconclusive (insufficient support).
"""
from __future__ import annotations

import math

import numpy as np

PRIOR_SD = 0.5
CLIP = 1e-15
MIN_FIT_DATES = 25


def _valid(p) -> np.ndarray:
    if any(isinstance(x, (bool, np.bool_)) for x in p):
        raise ValueError("boolean probability")
    a = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(a)) or np.any((a < 0) | (a > 1)):
        raise ValueError("a probability is non-finite or outside [0, 1]")
    return a


def logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, CLIP, 1 - CLIP)
    return np.log(q / (1 - q))


def apply_map(p, a: float) -> np.ndarray:
    """g_a(p); g_a(0) = 0 and g_a(1) = 1 by continuity."""
    p = _valid(p)
    out = 1 / (1 + np.exp(-(logit(p) + a)))
    return np.where(p == 0, 0.0, np.where(p == 1, 1.0, out))


def fit_intercept(dates, *, tol: float = 1e-13, max_iter: int = 100) -> float:
    """dates: [(probabilities, outcomes 0/1), ...], one entry per eligible fit date."""
    prepared = []
    for p, y in dates:
        z = logit(_valid(p))
        yy = np.asarray(y, dtype=float)
        if z.size == 0 or yy.shape != z.shape:
            raise ValueError("each date needs matching non-empty probabilities and outcomes")
        prepared.append((z, yy))
    a = 0.0
    for _ in range(max_iter):
        grad, hess = -a / PRIOR_SD ** 2, -1 / PRIOR_SD ** 2
        for z, yy in prepared:
            g = 1 / (1 + np.exp(-(z + a)))
            grad += float(np.mean(yy - g))
            hess -= float(np.mean(g * (1 - g)))
        step = grad / hess
        a -= step
        if abs(step) < tol:
            return a
    raise RuntimeError("the MAP did not converge")


def fit_or_inconclusive(dates) -> dict:
    if len(dates) < MIN_FIT_DATES:
        return {"status": "inconclusive", "reason": "insufficient support", "fit_dates": len(dates)}
    a = fit_intercept(dates)
    if not math.isfinite(a):
        raise RuntimeError("non-finite MAP")
    return {"status": "fitted", "a": a, "fit_dates": len(dates)}
