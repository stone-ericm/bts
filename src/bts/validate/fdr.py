"""Multiple-testing FDR baseline for realized-picks calibration.

p-value FDR baseline ONLY — NOT e-BH. Per Codex bus #225/#227:
- Wang & Ramdas (2022) e-BH requires e-values with E_H0[e] <= 1; 1/p has
  infinite expectation under Uniform(0,1) null and is NOT a valid universal
  p-to-e calibrator. e-BH deferred until valid e-values are constructed
  (likelihood-ratio under a prespecified alternative, or a documented
  calibrator family such as kappa * p^(kappa-1)).
- Cell p-values use Poisson-binomial under heterogeneous H0 (each row's
  p_game_hit), via scipy.stats.poisson_binom. Inclusive upper-tail via
  sf(x - 1) so P(X >= x) is computed correctly on the discrete distribution.
- BH (Benjamini & Hochberg 1995) for PRDS dependence baseline.
- BY (Benjamini & Yekutieli 2001) with c(m) = sum_{i=1..m} 1/i harmonic
  penalty for arbitrary-dependence conservatism.
- Tail direction: overconfidence iff p_lower < p_upper (observed hits LOW
  vs expected). Per Codex's correction, this is the direction in which
  the model was overconfident.

This module produces q-values and tail directions only. It does NOT set a
deploy-gate threshold — that decision is left to memo interpretation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# Cell-key fields identifying a Cut C cell; used by cell_pvalues_from_artifact.
CUT_C_KEY_FIELDS: tuple[str, ...] = (
    "regime",
    "slot",
    "is_park_driven",
    "batter_skill_quartile",
)


# ---- BH/BY adjustment ----


def _validate_pvalues(p: np.ndarray) -> None:
    if p.size == 0:
        return
    if np.any(np.isnan(p)):
        raise ValueError("p-values contain NaN; raise rather than silently drop")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p-values must be in [0, 1]")


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted q-values (1995, PRDS dependence).

    Computes q_(i) = min over k>=i of (m / k * p_(k)) with the standard
    cap at 1.0. Returns q-values aligned to original input order (input is
    NOT sorted in place). Empty input returns empty output. NaN p-values
    raise rather than silently drop.
    """
    p = np.asarray(pvalues, dtype=float)
    _validate_pvalues(p)
    if p.size == 0:
        return np.zeros(0, dtype=float)

    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]

    ranks = np.arange(1, m + 1, dtype=float)
    raw = (m / ranks) * p_sorted
    q_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)

    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q


def by_qvalues(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Yekutieli adjusted q-values (2001, arbitrary-dependence).

    BY = min(1, BH * c(m)), c(m) = sum_{i=1..m} 1/i. More conservative than
    BH; valid under arbitrary positive or negative dependence.
    """
    p = np.asarray(pvalues, dtype=float)
    _validate_pvalues(p)
    if p.size == 0:
        return np.zeros(0, dtype=float)

    m = len(p)
    c_m = float(np.sum(1.0 / np.arange(1, m + 1, dtype=float)))
    bh = bh_qvalues(p)
    return np.minimum(bh * c_m, 1.0)


# ---- Poisson-binomial cell p-values ----


def cell_pvalue(
    actual_hits: int,
    row_probabilities: np.ndarray,
) -> dict[str, Any]:
    """Compute Poisson-binomial p-values for a single cell under H0=calibrated.

    Each row contributes a Bernoulli(p_i) where p_i is the model's predicted
    probability for that pick; X = sum is Poisson-binomial. The H0 test asks
    "would the observed hits be unusual if every prediction were exactly the
    Bernoulli rate of its outcome?"

    Returns a dict with:
      p_lower (float): P(X <= actual_hits | H0). The OVERCONFIDENCE tail
        (observed below expected => model was overconfident).
      p_upper (float): P(X >= actual_hits | H0), inclusive via sf(x - 1).
        The UNDERCONFIDENCE tail.
      p_two_sided (float): min(1, 2 * min(p_lower, p_upper)). Standard
        discrete double-the-smaller convention; not an exact-optimal
        two-sided test.
      tail_direction (str): "overconfidence" iff p_lower < p_upper, else
        "underconfidence" iff p_upper < p_lower, else "balanced".

    Raises:
      ValueError on empty/NaN/out-of-range row_probabilities, or actual_hits
      outside [0, n].
    """
    probs = np.asarray(row_probabilities, dtype=float)
    if probs.size == 0:
        raise ValueError("row_probabilities cannot be empty")
    if np.any(np.isnan(probs)):
        raise ValueError("row_probabilities contain NaN")
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("row_probabilities must be in [0, 1]")
    n = len(probs)
    if not isinstance(actual_hits, (int, np.integer)):
        raise ValueError(
            f"actual_hits must be int, got {type(actual_hits).__name__}"
        )
    actual_hits_int = int(actual_hits)
    if actual_hits_int < 0 or actual_hits_int > n:
        raise ValueError(
            f"actual_hits={actual_hits_int} out of [0, {n}]"
        )

    pb = stats.poisson_binom(probs)
    p_lower = float(pb.cdf(actual_hits_int))
    if actual_hits_int == 0:
        p_upper = 1.0
    else:
        p_upper = float(pb.sf(actual_hits_int - 1))
    p_two_sided = min(1.0, 2.0 * min(p_lower, p_upper))

    if p_lower < p_upper:
        direction = "overconfidence"
    elif p_upper < p_lower:
        direction = "underconfidence"
    else:
        direction = "balanced"

    return {
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_two_sided": p_two_sided,
        "tail_direction": direction,
    }


# ---- Cut-C extraction ----


def cell_pvalues_from_artifact(df: pd.DataFrame) -> dict[str, Any]:
    """Extract Cut C cells from a canonical realized-picks artifact + compute
    per-cell Poisson-binomial p-values + tail direction.

    Cell key: (regime, slot, is_park_driven, batter_skill_quartile).

    Excluded:
      - Pending rows (result_status != "resolved").
      - Rows where any cell-key field is NA. (NA in skill_quartile is the
        common case — pre_pooled_mdp early-season picks below MIN_PRIOR_PA.)

    Returns:
      {
        "cells": [
          {regime, slot, is_park_driven, batter_skill_quartile, n, hits,
           mean_p, observed_rate, p_lower, p_upper, p_two_sided,
           tail_direction},
          ...
        ],
        "m": int,            # tested family size
        "excluded_pending_rows": int,
        "excluded_na_rows": int,
      }
    """
    resolved = df[df["result_status"] == "resolved"].copy()
    excluded_pending = len(df) - len(resolved)

    mask_complete = np.ones(len(resolved), dtype=bool)
    for f in CUT_C_KEY_FIELDS:
        mask_complete &= resolved[f].notna().to_numpy()
    excluded_na = int((~mask_complete).sum())
    eligible = resolved[mask_complete].copy()

    cells: list[dict[str, Any]] = []
    for keys, group in eligible.groupby(list(CUT_C_KEY_FIELDS), dropna=False):
        regime, slot, is_park_driven, quartile = keys
        n = len(group)
        if n == 0:
            continue
        probs = group["p_game_hit"].astype(float).to_numpy()
        actual = group["actual_hit"].astype(bool).to_numpy()
        hits = int(actual.sum())
        mean_p = float(probs.mean())
        observed_rate = hits / n
        result = cell_pvalue(actual_hits=hits, row_probabilities=probs)
        cells.append({
            "regime": str(regime),
            "slot": str(slot),
            "is_park_driven": bool(is_park_driven),
            "batter_skill_quartile": int(quartile),
            "n": n,
            "hits": hits,
            "mean_p": mean_p,
            "observed_rate": observed_rate,
            "p_lower": result["p_lower"],
            "p_upper": result["p_upper"],
            "p_two_sided": result["p_two_sided"],
            "tail_direction": result["tail_direction"],
        })

    return {
        "cells": cells,
        "m": len(cells),
        "excluded_pending_rows": excluded_pending,
        "excluded_na_rows": excluded_na,
    }
