"""Check the arithmetic, dispatch and every decision field."""
from dataclasses import asdict
from itertools import product
from math import nextafter
from unittest.mock import patch

import bts
from bts import strategy
from bts.simulate import tail_policy as tp


def oracle(days, hit, both):
    extra = 0.0
    action = "skip"
    for _ in range(days):
        one = hit * (1.0 + extra)
        two = both * (2.0 + extra)
        if one >= extra and one >= two:
            action, extra = "single", one
        elif two >= extra:
            action, extra = "double", two
        else:
            action = "skip"
    return action


def main():
    artifact = strategy._load_mdp()
    tail = artifact["tail"]
    shipped = tail.bin_p_hit[0], tail.bin_p_both[0]
    rates = [0.0, 5e-324, 1e-300, 0.01, 0.25, 0.5, 0.6, 0.75, 0.99,
             nextafter(1.0, 0.0), 1.0]
    pairs = {shipped, *((hit, both) for hit in rates for both in rates if both <= hit)}
    for hit in rates:
        pairs.update((hit, v) for v in (hit / 2, nextafter(hit / 2, 0), nextafter(hit / 2, 1))
                     if 0 <= v <= hit)
    for days, (hit, both) in product(range(1, 401), sorted(pairs)):
        action = tp.longest_streak_action(days, hit, both)
        assert action in {"single", "double"} and action == oracle(days, hit, both), (days, hit, both)
    assert tp.longest_streak_action(0, *shipped) == "skip"
    assert tp.longest_streak_action(-1, *shipped) == "skip"
    boundary = tp.LONGEST_STREAK_FROM
    streaks = (55, 56, 57, 58, 59, 60, 100, 10**100, -1, False, True)
    positive = (1, 2, 3, 4, 28, 180, 181, 271)
    cases = 0
    for streak, raw, horizon in product(streaks, (0, -3, *positive), (0, -1, 20, 180, 272)):
        active = streak >= boundary and raw > 0
        legacy = min(max(0, raw), horizon)
        legacy_objective = "reach57" if legacy <= 0 or streak + 2 * legacy >= 57 else "emax_season_best"
        want = "longest_streak" if active else legacy_objective
        assert tp.mdp_objective(streak, raw, horizon) == want, (streak, raw, horizon, want)
        mdp = artifact | {"season_length": horizon}
        date = "2027-09-26"
        with patch.object(strategy, "_days_remaining", return_value=raw):
            effective = min(raw, 180) if active else tp.effective_days(raw, horizon or 180)
            want_regime = ("longest_streak" if active else tp.mdp_objective(streak, raw, horizon or 180),
                           effective)
            assert strategy._regime(mdp, streak, date) == want_regime
            if not active:
                continue
            for saver, p, partner, allow, (best, status) in product(
                    (False, True), (.01, .49, .99), (False, True), (False, True),
                    ((None, "trusted"), (9, "trusted"), (57, "untrusted"), (57, None))):
                ctx = strategy.DecisionContext(p, .93 if partner else None, partner,
                                               date, allow, mdp, best, status)
                actual = asdict(strategy.resolve_policy_decision(ctx, streak, saver))
                degraded = horizon <= 0
                policy = "single" if degraded else oracle(effective, *shipped)
                action = "single" if policy == "double" and not (allow and partner) else policy
                expected = dict(
                    action=action, policy_action=policy, source="mdp", objective=want,
                    streak=streak, days_effective=effective, best_supplied=best,
                    best_status="missing" if best is None else status or "untrusted",
                    effective_best=None, tail_sha256=None if degraded else tail.sha256,
                    degraded_reason=("longest-streak rule unavailable (base policy season length "
                                     "must be positive); forced single") if degraded else None,
                    reason=None, pick_bar=None, guard=None,
                )
                assert actual == expected, (actual, expected)
                cases += 1
            with patch.object(strategy, "_load_mdp", side_effect=AssertionError("display read disk")) as loader:
                assert strategy.effective_pick_bar(streak, date, False) == 0.0
                loader.assert_not_called()
    print(f"{bts.__file__}: boundary {boundary}, {400 * len(pairs)} arithmetic cases, "
          f"{cases} whole longest-streak decisions")
    for s in (boundary - 1, boundary, boundary + 1, 100, 10**100, -1, False, True):
        outcomes = []
        for date in ("2027-09-27", "2027-09-26", "2027-09-24", "2027-09-23",
                     "2028-04-01", "2028-03-31", "2028-01-01"):
            ctx = strategy.DecisionContext(.96, .95, True, date, True, artifact)
            d = strategy.resolve_policy_decision(ctx, s, True)
            outcomes.append((strategy._days_remaining(date), d.days_effective, d.objective, d.action))
        print(repr(s), outcomes)


if __name__ == "__main__":
    import logging

    logging.disable(logging.CRITICAL)
    main()
