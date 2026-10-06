"""C1 rank 4a, T4: the one calibration map (registration `docs/sota_audit/2026-10-04-prereg-c1-calibration.md` §§1, 3).

g_a(p) = logistic(logit(p) + a), with a Normal(0, 0.5²) prior on a. The MAP maximizes

    Σ_d (1/n_d) Σ_i [y log g_a(p) + (1 − y) log(1 − g_a(p))] − a² / (2 · 0.5²)

over the eligible fit dates. Date means are summed: a date with many rows counts once. This is a fixed weighted
likelihood with a regularizing prior, not a calibrated posterior. p is clipped to [1e−15, 1 − 1e−15] for fitting;
invalid, non-numeric, boolean, non-finite or out-of-[0, 1] probabilities are refused (the population step excludes
and counts them first). The objective is strictly concave, so Newton's method converges. A fit needs at least 25 fit
dates with known eligible rows; otherwise the study is inconclusive (insufficient support).

**The fit-window interval (descriptive, registration §§1, 4):** the central 95% interval of the normalized weighted
likelihood times the prior, exp(objective(a)), integrated numerically on a fixed grid of ±12 Laplace standard
deviations around the MAP (24,001 points, trapezoid rule). It is labelled descriptive: the weighted likelihood is a
fixed weighting, not a calibrated posterior, and the interval never selects a map or changes a disposition.
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


INTERVAL_LEVEL = 0.95
INTERVAL_METHOD = ("central interval of exp(objective(a)), normalized numerically on a grid of MAP +/- 12 Laplace SDs "
                   "(24,001 points, trapezoid rule)")
INTERVAL_LABEL = "descriptive: a normalized weighted-likelihood posterior, not calibrated posterior uncertainty"


def _objective_on(grid: np.ndarray, prepared) -> np.ndarray:
    out = -grid ** 2 / (2 * PRIOR_SD ** 2)
    for z, yy in prepared:
        s = z[None, :] + grid[:, None]
        out = out + np.mean(-yy * np.logaddexp(0, -s) - (1 - yy) * np.logaddexp(0, s), axis=1)
    return out


def posterior_interval(dates, a_map: float, *, level: float = INTERVAL_LEVEL, points: int = 24_001) -> dict:
    """The descriptive fit-window interval for the MAP `a_map` of `dates` (the same input as fit_intercept)."""
    prepared = [(logit(_valid(p)), np.asarray(y, dtype=float)) for p, y in dates]
    curvature = 1 / PRIOR_SD ** 2 + sum(float(np.mean((g := 1 / (1 + np.exp(-(z + a_map)))) * (1 - g)))
                                        for z, _ in prepared)
    sd = 1 / math.sqrt(curvature)
    grid = np.linspace(a_map - 12 * sd, a_map + 12 * sd, points)
    obj = _objective_on(grid, prepared)
    dens = np.exp(obj - obj.max())
    cdf = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(grid))])
    cdf /= cdf[-1]
    tail = (1 - level) / 2
    lo, hi = np.interp([tail, 1 - tail], cdf, grid)
    return {"level": level, "bounds": [float(lo), float(hi)], "laplace_sd": sd, "method": INTERVAL_METHOD,
            "label": INTERVAL_LABEL}
