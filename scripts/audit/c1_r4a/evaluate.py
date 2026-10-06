"""C1 rank 4a, T5: the registered evaluation (registration §§3–4).

- **Primary:** the mean over test dates of each date's mean log-loss difference (map minus identity), on that
  date's known eligible rows. A 10,000-draw bootstrap of the date-difference vector (with replacement; each sampled
  occurrence keeps its multiplicity; seed 20270101) gives the 2.5th/97.5th percentile interval. It conditions on the
  once-fitted map.
- **Guardrail:** the 95th percentile of a 10,000-draw bootstrap of the original rank-1 date differences (seed
  20270103) must be at most 0.005 nats. A point estimate alone cannot pass it.
- **Dispositions:**
  - **positive:** a difference ≤ −0.001 with an upper bound < 0, and the guardrail holds;
  - **negative:** a difference ≥ 0;
  - **inconclusive:** otherwise.

  At least 50 scoreable test dates and 50 known rank-1 dates are required; with fewer the result is inconclusive
  (insufficient support).
"""
from __future__ import annotations

import numpy as np

from scripts.audit.c1_r4a.fit import CLIP, apply_map

DRAWS = 10_000
SEED_PRIMARY = 20270101
SEED_GUARD = 20270103
BAR = -0.001
GUARD_LIMIT = 0.005
MIN_DATES = 50


def _log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    q = np.clip(p, CLIP, 1 - CLIP)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def date_differences(dates, *, a: float) -> list[float]:
    out = []
    for p, y in dates:
        p, y = np.asarray(p, float), np.asarray(y, float)
        out.append(float(np.mean(_log_loss(apply_map(p, a), y) - _log_loss(p, y))))
    return out


def bootstrap(values, *, seed: int, draws: int = DRAWS) -> np.ndarray:
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    return v[rng.integers(0, v.size, size=(draws, v.size))].mean(axis=1)


def bootstrap_interval(values, *, seed: int = SEED_PRIMARY) -> tuple[float, float]:
    m = bootstrap(values, seed=seed)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def guardrail_p95(rank1_values, *, seed: int = SEED_GUARD) -> float:
    return float(np.percentile(bootstrap(rank1_values, seed=seed), 95))


def disposition(*, diff: float, upper: float, guard_p95: float, n_dates: int, n_rank1: int) -> dict:
    if n_dates < MIN_DATES or n_rank1 < MIN_DATES:
        return {"disposition": "inconclusive", "reason": "insufficient support", "n_dates": n_dates, "n_rank1": n_rank1}
    if diff >= 0:
        return {"disposition": "negative"}
    if diff <= BAR and upper < 0 and guard_p95 <= GUARD_LIMIT:
        return {"disposition": "positive"}
    return {"disposition": "inconclusive",
            "reason": "the practical bar, the interval or the guardrail was not met"}
