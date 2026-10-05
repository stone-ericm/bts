"""C1 4b fixed-policy projection of P(reach the target), registration §5 ("P(57)", "Comparable environment",
"Calibration grid", "Dependence grid"). Every number this module produces is a model projection, not a measured
jackpot frequency.

- **Forward propagation** of the state distribution over (s, m, saver, r) along a season calendar. Each day is
  (opportunity, raw days left).
  - **No-opportunity day:** a forced skip that resets r.
  - **Opportunity day:** a type is drawn from the phase's environment. A primary-absent type is a forced skip that
    resets r. Otherwise the arm's fixed policy (a lookup, never re-optimized) gives a raw action for the type's bin,
    and the legal clamp executes a partnerless double as a single.
- **Outcome categories per type:** primary miss, primary-only hit and joint hit.
- **The run counter r** (consecutive calendar-day primary hits, capped at r_cap) moves on the *primary* outcome
  whatever the arm does, so the dependence stress is exogenous to each arm's streak.
- **Transform order:**
  1. calibration c, subtracted from both rates with a zero floor and p_both <= p_hit enforced;
  2. the analytic leg stress Δ: p_both := max(0, p_both − Δ·p_hit);
  3. at r = r_cap, the dependence h: p_hit' = max(0, p_hit − h), p_both scaled by p_hit'/p_hit (0 if p_hit was 0).

  h = 0 reproduces iid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.audit.c1_r4b.solvers import DOUBLE, SINGLE, SKIP, DayType, Environment

R_CAP = 8


class Unavailable(ValueError):
    """The environment cannot support this projection (a phase without types, or frequencies not summing to 1);
    reported as unavailable, never as a zero probability."""


@dataclass(frozen=True)
class Stress:
    c: float = 0.0
    h: float = 0.0
    delta: float = 0.0


def categories(t: DayType, stress: Stress, *, at_cap: bool) -> tuple[float, float, float]:
    """(primary miss, primary-only hit, joint hit) probabilities for a type under the stress."""
    ph = max(0.0, t.p_hit - stress.c)
    pb = min(max(0.0, t.p_both - stress.c), ph)
    pb = max(0.0, pb - stress.delta * ph)
    if at_cap and stress.h > 0.0:
        new_ph = max(0.0, ph - stress.h)
        pb = pb * new_ph / ph if ph > 0.0 else 0.0
        ph = new_ph
    return 1.0 - ph, ph - pb, pb


def project(policy, env: Environment, days: list[tuple[bool, int]], *, target: int, saver_zone: tuple[int, int],
            late_days: int, stress: Stress, r_cap: int = R_CAP, init_saver: int = 1) -> dict:
    """policy(d_raw, q) -> (T+1, T+1, 2) raw actions (the arm applies its own caps/routing). Returns the projected
    P(reach target), the projected E[season best] and the total mass (1 up to rounding)."""
    T1, R1 = target + 1, r_cap + 1
    S, M, SV = np.meshgrid(np.arange(T1), np.arange(T1), np.arange(2), indexing="ij")
    lo, hi = saver_zone
    catch = (SV == 1) & (S >= lo) & (S <= hi)
    miss_s, miss_sv = np.where(catch, S, 0), np.where(catch, 0, SV)
    s1 = np.minimum(S + 1, target); m1 = np.maximum(M, s1)
    s2 = np.minimum(S + 2, target); m2 = np.maximum(M, s2)
    absorbed = S >= target
    dist = np.zeros((T1, T1, 2, R1))
    dist[0, 0, init_saver, 0] = 1.0

    def add(new, mask, s_idx, m_idx, sv_idx, r, vals):
        if mask.any():
            np.add.at(new, (s_idx[mask], m_idx[mask], sv_idx[mask], np.full(int(mask.sum()), r)), vals[mask])

    for opp, d in days:
        new = np.zeros_like(dist)
        if not opp:
            new[..., 0] = dist.sum(axis=3)
            dist = new
            continue
        new[absorbed] += dist[absorbed]
        live = dist.copy()
        live[absorbed] = 0.0
        types = env.types(d, late_days)
        if not types or abs(sum(t.freq for t in types) - 1.0) > 1e-9:
            raise Unavailable(f"day with {d} days left: phase types missing or frequencies do not sum to 1")
        for t in types:
            if t.q is None:
                new[..., 0] += t.freq * live.sum(axis=3)
                continue
            A = np.asarray(policy(d, t.q))
            skip = (A == SKIP) & ~absorbed
            single = ((A == SINGLE) | ((A == DOUBLE) & (not t.partner))) & ~absorbed
            dbl = (A == DOUBLE) & bool(t.partner) & ~absorbed
            for r in range(R1):
                X = t.freq * live[..., r]
                if not X.any():
                    continue
                pm, po, pj = categories(t, stress, at_cap=(r == r_cap))
                rh = min(r_cap, r + 1)
                new[..., rh] += np.where(skip, X * (po + pj), 0.0)
                new[..., 0] += np.where(skip, X * pm, 0.0)
                add(new, single, s1, m1, SV, rh, X * (po + pj))
                add(new, single, miss_s, M, miss_sv, 0, X * pm)
                add(new, dbl, s2, m2, SV, rh, X * pj)
                add(new, dbl, miss_s, M, miss_sv, rh, X * po)
                add(new, dbl, miss_s, M, miss_sv, 0, X * pm)
        dist = new
    by_best = dist.sum(axis=(0, 2, 3))
    return {"p_reach": float(dist[target].sum()), "e_best": float(by_best @ np.arange(T1)),
            "mass": float(dist.sum())}
