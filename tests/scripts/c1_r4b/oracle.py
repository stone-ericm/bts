"""Independent brute-force oracle for C1 4b (registration §5 "Validation"): explicit path enumeration with no state
merging, written separately from the vectorized solvers. Tiny targets and horizons only.

A day type is (freq, q, partner, p_hit, p_both); q is None for a primary-absent day (forced skip). A raw double on a
partnerless day executes as a single (+1 on the primary's outcome). A miss at streak lo..hi with the saver available
keeps the streak and spends the saver; otherwise the streak resets to 0. Streak and best cap at the target, and
reaching it is absorbing."""
from __future__ import annotations

SKIP, SINGLE, DOUBLE = 0, 1, 2


def _step(s, m, sv, a, t, target, zone):
    """All (probability, s', m', sv') branches of executing raw action a on day type t."""
    _, q, partner, ph, pb = t
    if q is None or a == SKIP or s >= target:
        return [(1.0, s, m, sv)]
    double = a == DOUBLE and partner
    p, up = (pb, 2) if double else (ph, 1)
    s_win = min(s + up, target)
    lo, hi = zone
    if sv == 1 and lo <= s <= hi:
        miss = (s, m, 0)
    else:
        miss = (0, m, sv)
    return [(p, s_win, max(m, s_win), sv), (1.0 - p, *miss)]


def phase_types(env, d, late_days):
    return env["late"] if d <= late_days else env["early"]


def terminal(objective, s, m, target):
    return float(m) if objective == "emax" else (1.0 if s >= target else 0.0)


def optimal(env, objective, s, m, d, sv, *, target, zone, late_days):
    """Expectimax over every path under the registered information set: the agent observes today's quality bin q
    (not partner availability) and picks one raw action per bin; availability then clamps its execution. So: E over
    bins, max over raw actions of E over that bin's types and outcomes, recurse."""
    if d == 0 or s >= target:
        return terminal(objective, s, m, target)
    types = phase_types(env, d, late_days)
    rec = lambda s2, m2, sv2: optimal(env, objective, s2, m2, d - 1, sv2, target=target, zone=zone,
                                      late_days=late_days)
    total = sum(t[0] * rec(s, m, sv) for t in types if t[1] is None)
    for q in sorted({t[1] for t in types if t[1] is not None}):
        tq = [t for t in types if t[1] == q]
        total += max(sum(t[0] * sum(p * rec(s2, m2, sv2) for p, s2, m2, sv2 in _step(s, m, sv, a, t, target, zone))
                         for t in tq)
                     for a in (SKIP, SINGLE, DOUBLE))
    return total


def evaluate(env, policy, objective, s, m, d, sv, *, target, zone, late_days):
    """Value of following policy(s, m, d, sv, q) -> raw action, enumerated over every path."""
    if d == 0 or s >= target:
        return terminal(objective, s, m, target)
    total = 0.0
    for t in phase_types(env, d, late_days):
        a = SKIP if t[1] is None else policy(s, m, d, sv, t[1])
        total += t[0] * sum(p * evaluate(env, policy, objective, s2, m2, d - 1, sv2, target=target, zone=zone,
                                         late_days=late_days)
                            for p, s2, m2, sv2 in _step(s, m, sv, a, t, target, zone))
    return total
