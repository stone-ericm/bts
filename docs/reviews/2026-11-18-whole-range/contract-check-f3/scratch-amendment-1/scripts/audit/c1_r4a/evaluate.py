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
- **Secondary, descriptive** (`secondary`; never selects a map or changes a disposition): on the same known rows,
  each date's mean, with dates weighing equally (registration §2; review r1 R10):
  - the Brier score of each arm, and their paired difference (map minus identity);
  - stated minus realized, before and after the map, on all rows and on the known rank-1 rows;
  - a reliability table by decile of p (`RELIABILITY_RULE` states the bin edges and the weighting).
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


WEIGHTING = "each date's mean over its known eligible rows; dates weigh equally"
RELIABILITY_RULE = ("bins: the empirical deciles of p over all known rows (numpy.quantile, linear interpolation). Bin k "
                    "holds edges[k] <= p < edges[k+1], and the last bin also holds p == edges[10]. Within a bin each "
                    "row weighs 1/n_d, n_d being its date's known-row count, so dates weigh equally; n is the row "
                    "count")


def secondary(dates, *, a: float) -> dict:
    """dates: [(probabilities, outcomes 0/1, rank-1 position or None), ...] over each test date's known rows."""
    per = [(np.asarray(p, float), np.asarray(y, float), r1) for p, y, r1 in dates if len(p)]
    if not per:
        return {"dates": 0, "weighting": WEIGHTING}
    mapped = [apply_map(p, a) for p, _, _ in per]
    b_id = np.array([np.mean((p - y) ** 2) for p, y, _ in per])
    b_map = np.array([np.mean((m - y) ** 2) for m, (_, y, _) in zip(mapped, per)])
    r1 = [(p[r], m[r], y[r]) for m, (p, y, r) in zip(mapped, per) if r is not None]
    out = {"weighting": WEIGHTING, "dates": len(per),
           "brier": {"identity": float(b_id.mean()), "map": float(b_map.mean()),
                     "difference_map_minus_identity": float((b_map - b_id).mean())},
           "stated_minus_realized": {"identity": float(np.mean([p.mean() - y.mean() for p, y, _ in per])),
                                     "map": float(np.mean([m.mean() - y.mean() for m, (_, y, _) in zip(mapped, per)]))},
           "rank1_stated_minus_realized": None if not r1 else {
               "identity": float(np.mean([p - y for p, _, y in r1])), "map": float(np.mean([m - y for _, m, y in r1])),
               "dates": len(r1)}}
    p_all = np.concatenate([p for p, _, _ in per])
    y_all = np.concatenate([y for _, y, _ in per])
    m_all = np.concatenate(mapped)
    w_all = np.concatenate([np.full(p.size, 1 / p.size) for p, _, _ in per])
    edges = np.quantile(p_all, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, p_all, side="right") - 1, 0, 9)
    table = []
    for k in range(10):
        sel = bins == k
        if sel.any():
            w = w_all[sel]
            table.append({"bin": k, "n": int(sel.sum()), "weight": float(w.sum()),
                          "mean_p": float(np.average(p_all[sel], weights=w)),
                          "mean_map": float(np.average(m_all[sel], weights=w)),
                          "realized": float(np.average(y_all[sel], weights=w))})
    out["reliability"] = {"rule": RELIABILITY_RULE, "edges": [float(e) for e in edges], "bins": table}
    return out
