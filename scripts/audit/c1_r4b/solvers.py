"""C1 4b solvers: phase-aware whole-season backward induction with the legal availability kernel.

Registration `docs/sota_audit/2026-10-04-prereg-c1-longest-streak-policy.md` §§2–3, 5 (FROZEN).

- **State:** streak s, season best m (both capped at the target; reaching it is absorbing), days left d, saver.
  The raw policy is also indexed by today's quality bin q.
- **Day types (per phase):** each has a frequency, a bin q (None means no primary: forced skip), whether a
  different-game partner exists, p_hit and p_both. On a partnerless day a raw double executes as a single, so its
  +1 transition uses p_hit. The value of a raw action at bin q integrates that legal execution over the bin's types
  (one raw action per bin; availability only applies the clamp).
- **Phase:** a day with d <= late_days uses the late types.
- **Objectives:**
  - `emax` maximizes E[season best], with the explicit stop rule (skip iff min(T, s + 2d) <= m) and play-first
    exact ties (single, then double, then skip). This extends `tail_policy.solve_emax_season_best` to two phases
    and the legal kernel.
  - `reach` maximizes P(reach T), with skip-first exact ties as in `mdp.solve_mdp`.
- **Saver:** a miss at lo <= s <= hi with the saver available keeps the streak and spends it, as in
  `mdp.transition_outcomes`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SKIP, SINGLE, DOUBLE = 0, 1, 2
TARGET = 57
SAVER_ZONE = (10, 15)
LATE_DAYS = 30
OBJECTIVES = ("emax", "reach")


@dataclass(frozen=True)
class DayType:
    freq: float
    q: int | None
    partner: bool
    p_hit: float
    p_both: float


@dataclass(frozen=True)
class Environment:
    n_bins: int
    early: tuple[DayType, ...]
    late: tuple[DayType, ...]

    def types(self, d: int, late_days: int = LATE_DAYS) -> tuple[DayType, ...]:
        return self.late if d <= late_days else self.early


def validate(env: Environment) -> None:
    """Coherent rates (0 <= p_both <= p_hit <= 1; p_both = 0 without a partner; a primary-absent type has no rates),
    frequencies summing to 1 per phase, and every bin present with positive weight in both phases: an empty fitting
    (phase, bin) cell is a stop condition under the registration, never silently imputed."""
    for name, types in (("early", env.early), ("late", env.late)):
        freqs = [t.freq for t in types]
        if not freqs or any(not math.isfinite(f) or f < 0 for f in freqs) or abs(sum(freqs) - 1.0) > 1e-9:
            raise ValueError(f"{name}: type frequencies must be finite, >= 0 and sum to 1 (got {sum(freqs)!r})")
        for t in types:
            if t.q is None:
                if t.partner or t.p_hit != 0.0 or t.p_both != 0.0:
                    raise ValueError(f"{name}: a primary-absent type carries no partner or rates: {t}")
                continue
            if not (0 <= t.q < env.n_bins):
                raise ValueError(f"{name}: bin {t.q} outside 0..{env.n_bins - 1}")
            if not (0.0 <= t.p_both <= t.p_hit <= 1.0) or not math.isfinite(t.p_hit + t.p_both):
                raise ValueError(f"{name}: need 0 <= p_both <= p_hit <= 1: {t}")
            if not t.partner and t.p_both != 0.0:
                raise ValueError(f"{name}: a partnerless type must have p_both = 0: {t}")
        for q in range(env.n_bins):
            if sum(t.freq for t in types if t.q == q) <= 0.0:
                raise ValueError(f"{name}: bin {q} has no positive-weight type (empty fitting cell: stop)")


@dataclass
class Solution:
    objective: str
    target: int
    horizon: int
    late_days: int
    saver_zone: tuple[int, int]
    value: np.ndarray    # (T+1, T+1, D+1, 2): pre-observation value of (s, m, d, saver)
    policy: np.ndarray   # (T+1, T+1, D+1, 2, n_bins) int8 raw actions


def solve(env: Environment, *, horizon: int, objective: str, target: int = TARGET,
          saver_zone: tuple[int, int] = SAVER_ZONE, late_days: int = LATE_DAYS) -> Solution:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    validate(env)
    T, D, nq = int(target), int(horizon), env.n_bins
    S, M = np.meshgrid(np.arange(T + 1), np.arange(T + 1), indexing="ij")
    emax = objective == "emax"
    value = np.zeros((T + 1, T + 1, D + 1, 2))
    policy = np.zeros((T + 1, T + 1, D + 1, 2, nq), dtype=np.int8)
    terminal = M.astype(float) if emax else (S >= T).astype(float)
    value[:, :, 0, :] = terminal[:, :, None]
    s1 = np.minimum(S + 1, T); m1 = np.maximum(M, s1)
    s2 = np.minimum(S + 2, T); m2 = np.maximum(M, s2)
    zero = np.zeros_like(S)
    lo, hi = saver_zone
    absorbed = S >= T
    for d in range(1, D + 1):
        types = env.types(d, late_days)
        EV = value[:, :, d - 1, :]
        stop = (np.minimum(T, S + 2 * d) <= M) if emax else np.zeros_like(S, dtype=bool)
        for sv in (0, 1):
            catch = (sv == 1) & (S >= lo) & (S <= hi)
            ev_miss = np.where(catch, EV[S, M, 0], EV[zero, M, sv])
            v_skip = EV[S, M, sv]
            ev_one, ev_two = EV[s1, m1, sv], EV[s2, m2, sv]
            total = sum(t.freq for t in types if t.q is None) * v_skip
            for q in range(nq):
                tq = [t for t in types if t.q == q]
                w = sum(t.freq for t in tq)
                v_single = sum(t.freq * (t.p_hit * ev_one + (1 - t.p_hit) * ev_miss) for t in tq) / w
                v_double = sum(t.freq * ((t.p_both * ev_two + (1 - t.p_both) * ev_miss) if t.partner
                                         else (t.p_hit * ev_one + (1 - t.p_hit) * ev_miss)) for t in tq) / w
                best = np.maximum(np.maximum(v_skip, v_single), v_double)
                if emax:
                    act = np.where(v_single == best, SINGLE, np.where(v_double == best, DOUBLE, SKIP))
                    act = np.where(stop, SKIP, act)
                    # At stop cells every branch equals m by construction; take v_skip (exactly m).
                    chosen = np.where(act == SINGLE, v_single, np.where(act == DOUBLE, v_double, v_skip))
                else:
                    act = np.where(v_skip == best, SKIP, np.where(v_single == best, SINGLE, DOUBLE))
                    chosen = best
                act = np.where(absorbed, SKIP, act)
                policy[:, :, d, sv, q] = act.astype(np.int8)
                total = total + w * chosen
            value[:, :, d, sv] = np.where(absorbed, terminal, total)
    return Solution(objective=objective, target=T, horizon=D, late_days=late_days, saver_zone=tuple(saver_zone),
                    value=value, policy=policy)


@dataclass
class Hybrid:
    """A1's routing (and A0's shape): the reach table while 57 is still reachable (s + 2d >= T), else the E[best]
    continuation. Both tables come from the same fitted environment; d is the caller's effective days left."""
    reach: Solution
    emax: Solution
    target: int = TARGET

    def action(self, s: int, m: int, d: int, sv: int, q: int) -> int:
        table = self.reach if s + 2 * d >= self.target else self.emax
        return int(table.policy[s, m, d, sv, q])
