"""C1 4b corrected realized-sequence replay (registration §4) and the arm providers.

Corrections relative to the July `replay_vectorized`:
- **(a) Calendar clock:** days left is the raw calendar count, taken from the season's frozen calendar.
- **(b) Legal clamp:** a partnerless double executes as a single on the primary's outcome (counted as demoted).
- **(c) Season best in state:** every provider receives the arm's own running best m.
- **(d) A0 routing:** A0 is the deployed hybrid: the base table while s + 2d >= 57, else the tail with a trusted best.

No-opportunity and unknown-coverage days call no policy but still consume calendar time.

A provider maps (s, m, d_raw, saver, p1, partner) to raw actions, vectorized over thinning replicates. The registered
Δ stress thins only eligible partner hits, with probability min(1, Δ / r_bar). Its uniforms are drawn once per stable
(season, seed) identity, so masks are nested across Δ and shared by every arm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.audit.c1_r4b.solvers import DOUBLE, SINGLE, SKIP, TARGET, Solution

THIN_SEED = 20261004
SAVER_ZONE = (10, 15)


def const(action: int):
    return lambda s, m, d, sv, p1, partner: np.full(np.shape(s), action, dtype=np.int64)


def _bin_ge(p: float, bounds) -> int:
    """Production's classification: bin = number of boundaries <= p (equality enters the upper bin)."""
    return int(sum(1 for b in bounds if p >= b))


@dataclass
class A0:
    """The pinned deployed hybrid, replicating `strategy.resolve_policy_decision` with a trusted best."""
    base: np.ndarray
    base_bounds: list
    base_season_length: int
    tail: np.ndarray
    tail_bounds: list
    target: int = TARGET

    def __call__(self, s, m, d_raw, sv, p1, partner):
        s, m, sv = np.asarray(s), np.asarray(m), np.asarray(sv)
        d_eff = max(0, min(int(d_raw), int(self.base_season_length)))
        tail_mode = (d_eff > 0) & (s < self.target) & (s + 2 * d_eff < self.target)
        if d_raw <= 0:
            base_act = np.full(s.shape, SKIP)
        else:
            q = min(_bin_ge(p1, self.base_bounds), self.base.shape[3] - 1)
            base_act = self.base[np.minimum(s, self.target - 1), min(int(d_raw), self.base_season_length), sv, q]
            base_act = np.where(s >= self.target, SKIP, base_act)
        if tail_mode.any():
            m_eff = np.minimum(self.target, np.maximum(s, m))
            qt = _bin_ge(p1, self.tail_bounds)
            d_t = min(d_eff, self.tail.shape[2] - 1)
            tail_act = self.tail[np.minimum(s, self.target), m_eff, d_t, sv, qt]
        else:
            tail_act = base_act
        return np.where(tail_mode, tail_act, base_act).astype(np.int64)


@dataclass
class Table:
    """A1/A2-style fitted table: the fold classifier and the table's own horizon cap."""
    solution: Solution
    cuts: np.ndarray

    def __call__(self, s, m, d_raw, sv, p1, partner):
        d = max(0, min(int(d_raw), self.solution.horizon))
        q = int(np.searchsorted(self.cuts, p1, side="right"))
        return self.solution.policy[np.asarray(s), np.asarray(m), d, np.asarray(sv), q].astype(np.int64)


@dataclass
class HybridArm:
    """A1: the matched reach table while 57 is reachable, else the matched E[best] continuation."""
    reach: Solution
    emax: Solution
    cuts: np.ndarray
    target: int = TARGET

    def __call__(self, s, m, d_raw, sv, p1, partner):
        s, m, sv = np.asarray(s), np.asarray(m), np.asarray(sv)
        d = max(0, min(int(d_raw), self.emax.horizon))
        q = int(np.searchsorted(self.cuts, p1, side="right"))
        r = self.reach.policy[s, m, d, sv, q]
        e = self.emax.policy[s, m, d, sv, q]
        return np.where(s + 2 * d >= self.target, r, e).astype(np.int64)


def thin_masks(hit2: np.ndarray, *, delta: float, r_bar: float, reps: int, identity: tuple) -> np.ndarray:
    hit2 = np.asarray(hit2, bool)
    if delta == 0:
        return hit2[None, :].copy()
    if not r_bar > 0:
        raise ValueError("r_bar is zero: the stress is unavailable")
    q = min(1.0, delta / r_bar)
    u = np.random.default_rng([THIN_SEED, *map(int, identity)]).random((reps, hit2.size))
    return hit2[None, :] & ~(u < q)


def _executed(a: np.ndarray, partner: bool) -> tuple[np.ndarray, np.ndarray]:
    demoted = (a == DOUBLE) & (not partner)
    return np.where(demoted, SINGLE, a), demoted


def replay(days: dict, arms: dict, *, hit2_masks: np.ndarray, target: int = TARGET,
           zone: tuple[int, int] = SAVER_ZONE, compare: tuple[str, str] | None = None) -> dict:
    reps, n = hit2_masks.shape
    lo, hi = zone
    st = {k: {"s": np.zeros(reps, np.int64), "m": np.zeros(reps, np.int64), "sv": np.ones(reps, np.int64),
              "resets": np.zeros(reps, np.int64),
              "actions": {a: np.zeros(reps, np.int64) for a in ("skip", "single", "double", "demoted")}}
          for k in arms}
    cons = None
    if compare:
        x, y = compare
        cons = {f"{x}_states": {"visits": 0, "differ": 0, "visits_s10": 0, "differ_s10": 0},
                f"{y}_states": {"visits": 0, "differ": 0, "visits_s10": 0, "differ_s10": 0},
                "own_trajectory": {"days": 0, "differ": 0}}
    for i in range(n):
        if not (days["opp"][i] and days["known"][i]):
            continue
        d, p1, partner, h1 = int(days["d_raw"][i]), float(days["p1"][i]), bool(days["partner"][i]), bool(days["hit1"][i])
        h2 = hit2_masks[:, i]
        if cons is not None:
            x, y = compare
            for a_name, b_name in ((x, y), (y, x)):
                sa = st[a_name]
                live = sa["s"] < target
                ea, _ = _executed(arms[a_name](sa["s"], sa["m"], d, sa["sv"], p1, partner), partner)
                eb, _ = _executed(arms[b_name](sa["s"], sa["m"], d, sa["sv"], p1, partner), partner)
                rec = cons[f"{a_name}_states"]
                diff = (ea != eb) & live
                hi10 = live & (sa["s"] >= 10)
                rec["visits"] += int(live.sum()); rec["differ"] += int(diff.sum())
                rec["visits_s10"] += int(hi10.sum()); rec["differ_s10"] += int((diff & hi10).sum())
            sx, sy = st[x], st[y]
            ex, _ = _executed(arms[x](sx["s"], sx["m"], d, sx["sv"], p1, partner), partner)
            ey, _ = _executed(arms[y](sy["s"], sy["m"], d, sy["sv"], p1, partner), partner)
            both = (sx["s"] < target) & (sy["s"] < target)
            cons["own_trajectory"]["days"] += int(both.sum())
            cons["own_trajectory"]["differ"] += int(((ex != ey) & both).sum())
        for name, prov in arms.items():
            a_st = st[name]
            s, m, sv = a_st["s"], a_st["m"], a_st["sv"]
            live = s < target
            raw = np.where(live, prov(s, m, d, sv, p1, partner), SKIP)
            a, demoted = _executed(raw, partner)
            a_st["actions"]["skip"] += (live & (a == SKIP)).astype(np.int64)
            a_st["actions"]["single"] += (a == SINGLE).astype(np.int64)
            a_st["actions"]["double"] += (a == DOUBLE).astype(np.int64)
            a_st["actions"]["demoted"] += (demoted & live).astype(np.int64)
            played = a != SKIP
            success = played & h1 & ((a == SINGLE) | h2)
            miss = played & ~success
            catch = miss & (sv == 1) & (s >= lo) & (s <= hi)
            hard = miss & ~catch
            s = np.where(success, np.minimum(target, s + np.where(a == DOUBLE, 2, 1)), s)
            s = np.where(hard, 0, s)
            a_st["sv"] = np.where(catch, 0, sv)
            a_st["resets"] = a_st["resets"] + hard
            a_st["s"], a_st["m"] = s, np.maximum(m, s)
    out = {}
    for name, a_st in st.items():
        mx = a_st["m"]
        out[name] = {"max": mx, "resets": a_st["resets"], "actions": a_st["actions"],
                     "reach20": mx >= 20, "reach30": mx >= 30, "reach40": mx >= 40, "reach57": mx >= target}
    if cons is not None:
        out["_consequences"] = cons
    return out
