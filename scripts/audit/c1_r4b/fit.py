"""C1 4b fold fitting (registration §3): the five-bin classifier and the availability-conditioned type environment.

- **Cutpoints:** the pooled rank-1 probabilities of the fitting seasons' known opportunity days, at quantiles
  0.2, 0.4, 0.6, 0.8 (numpy's default linear interpolation). A value equal to a cutpoint enters the upper bin.
- **Types:** within each phase (late = raw days left <= late_days by each season's own calendar), bin and partner
  availability, the frequency is the share of that phase's known opportunity days.
  - **With a partner:** p_hit and p_both both come from those same partner-eligible days.
  - **Partnerless:** p_hit comes from the partnerless days, and p_both is 0.
- **Primary absence** is never evidenced by the profiles, so no absent type is fitted.
- **An empty (phase, bin) cell stops** the fit (`solvers.validate`); nothing is imputed.
"""
from __future__ import annotations

import numpy as np

from scripts.audit.c1_r4b.solvers import DayType, Environment, validate

QUANTILES = (0.2, 0.4, 0.6, 0.8)


def cutpoints(p1: np.ndarray) -> np.ndarray:
    p1 = np.asarray(p1, float)
    if p1.size == 0 or not np.isfinite(p1).all():
        raise ValueError("cutpoints need finite primary probabilities")
    cuts = np.quantile(p1, QUANTILES)
    if not (np.diff(cuts) > 0).all():
        raise ValueError(f"degenerate cutpoints {cuts}: stop")
    return cuts


def classify(p: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Bin index; equality with a cutpoint enters the upper bin (as the production lookups do)."""
    return np.searchsorted(np.asarray(cuts, float), np.asarray(p, float), side="right")


def _known(d: dict) -> np.ndarray:
    return d["opp"] & d["known"]


def fit_environment(season_days: list[dict], cuts: np.ndarray, *, late_days: int = 30) -> tuple[Environment, dict]:
    n_bins = len(cuts) + 1
    phases, stats = {}, {}
    for name, is_late in (("early", False), ("late", True)):
        p1, hit1, partner, hit2 = [], [], [], []
        for d in season_days:
            k = _known(d) & ((d["d_raw"] <= late_days) if is_late else (d["d_raw"] > late_days))
            p1.append(d["p1"][k]); hit1.append(d["hit1"][k]); partner.append(d["partner"][k]); hit2.append(d["hit2"][k])
        p1, hit1, partner, hit2 = map(np.concatenate, (p1, hit1, partner, hit2))
        n = p1.size
        if n == 0:
            raise ValueError(f"{name}: no known opportunity days to fit")
        q = classify(p1, cuts)
        types, cells = [], {}
        for b in range(n_bins):
            for has_partner in (True, False):
                k = (q == b) & (partner == has_partner)
                c = int(k.sum())
                cells[(b, has_partner)] = {"n": c, "hit1": int(hit1[k].sum()),
                                           "both": int((hit1[k] & hit2[k]).sum()) if has_partner else 0}
                if c == 0:
                    continue
                ph = float(hit1[k].mean())
                pb = float((hit1[k] & hit2[k]).mean()) if has_partner else 0.0
                types.append(DayType(freq=c / n, q=b, partner=has_partner, p_hit=ph, p_both=pb))
        phases[name] = tuple(types)
        stats[name] = {"days": n, "cells": {f"{b}|{'P' if p else 'N'}": c for (b, p), c in cells.items()}}  # zeros kept
    env = Environment(n_bins=n_bins, early=phases["early"], late=phases["late"])
    validate(env)
    return env, stats


def r_bar(season_days: list[dict]) -> float:
    """The fitting seasons' partner-eligible leg hit rate (for the Δ thinning probability min(1, Δ / r_bar))."""
    legs = np.concatenate([d["hit2"][_known(d) & d["partner"]] for d in season_days])
    if legs.size == 0:
        raise ValueError("no partner-eligible days")
    return float(legs.mean())
