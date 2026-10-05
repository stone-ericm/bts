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


def independent_categories(p_hit, p_both, c=0.0, h=0.0, delta=0.0, at_cap=False):
    """Re-derived from registration §5 text, independent of project.categories."""
    ph = max(0.0, p_hit - c)
    pb = min(max(0.0, p_both - c), ph)
    pb = max(0.0, pb - delta * ph)
    if at_cap and h > 0:
        ph2 = max(0.0, ph - h)
        pb = pb * ph2 / ph if ph > 0 else 0.0
        ph = ph2
    return 1 - ph, ph - pb, pb


def projection(policy_fn, e, days, *, target, zone, late_days, stress, r_cap=8, s=0, m=0, sv=1, r=0):
    if s >= target:
        return 1.0
    if not days:
        return 0.0
    (opp, d), rest = days[0], days[1:]
    nxt = lambda s2, m2, sv2, r2: projection(policy_fn, e, rest, target=target, zone=zone, late_days=late_days,
                                         stress=stress, r_cap=r_cap, s=s2, m=m2, sv=sv2, r=r2)
    if not opp:
        return nxt(s, m, sv, 0)
    total = 0.0
    lo, hi = zone
    miss = (s, m, 0) if (sv == 1 and lo <= s <= hi) else (0, m, sv)
    for t in e.types(d, late_days):
        if t.q is None:
            total += t.freq * nxt(s, m, sv, 0)
            continue
        pm, po, pj = independent_categories(t.p_hit, t.p_both, stress.c, stress.h, stress.delta, r == r_cap)
        a = policy_fn(s, m, d, sv, t.q)
        rh = min(r_cap, r + 1)
        if a == SKIP:
            v = (po + pj) * nxt(s, m, sv, rh) + pm * nxt(s, m, sv, 0)
        elif a == SINGLE or not t.partner:
            s1 = min(target, s + 1)
            v = (po + pj) * nxt(s1, max(m, s1), sv, rh) + pm * nxt(*miss, 0)
        else:
            s2 = min(target, s + 2)
            v = pj * nxt(s2, max(m, s2), sv, rh) + po * nxt(*miss, rh) + pm * nxt(*miss, 0)
        total += t.freq * v
    return total


def scalar_replay(days, policy, *, hit2=None, target=57, zone=(10, 15)):
    """Independent scalar reference: one trajectory, explicit branches."""
    s = m = 0
    sv, resets = 1, 0
    acts = {"skip": 0, "single": 0, "double": 0, "demoted": 0}
    h2 = days["hit2"] if hit2 is None else hit2
    for i in range(len(days["opp"])):
        if not (days["opp"][i] and days["known"][i]) or s >= target:
            continue
        a = policy(s, m, int(days["d_raw"][i]), sv, float(days["p1"][i]), bool(days["partner"][i]))
        if a == SKIP:
            acts["skip"] += 1
            continue
        if a == DOUBLE and not days["partner"][i]:
            acts["demoted"] += 1
            a = SINGLE
        acts["double" if a == DOUBLE else "single"] += 1
        ok = bool(days["hit1"][i]) and (a == SINGLE or bool(h2[i]))
        if ok:
            s = min(target, s + (2 if a == DOUBLE else 1))
        elif sv == 1 and zone[0] <= s <= zone[1]:
            sv = 0
        else:
            s, resets = 0, resets + 1
        m = max(m, s)
    return {"max": m, "resets": resets, **acts}
