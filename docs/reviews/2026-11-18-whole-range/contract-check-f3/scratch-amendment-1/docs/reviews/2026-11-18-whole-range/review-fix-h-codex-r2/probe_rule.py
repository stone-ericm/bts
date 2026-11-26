from dataclasses import asdict
from datetime import date, timedelta
from itertools import product
import math
import random
from unittest.mock import patch

from bts import strategy
from bts.simulate import tail_policy as tp


def check_arithmetic():
    rng = random.Random(57)
    hits = [0., math.ulp(0.), .125, .5, .75, 1., math.nextafter(1., 0.)]
    rates = [(hit, hit * fraction) for hit, fraction in product(hits, [0., .25, .5, .75, 1.])]
    rates += [(hit, rng.uniform(0., hit)) for hit in [rng.random() for _ in range(53)]]
    rates += [(.7336111111111111, .5511111111111111), (.75, math.nextafter(.375, 0.)),
              (.75, math.nextafter(.375, 1.))]
    for hit, both in rates:
        previous = 0.
        for days in range(1, 401):
            single = hit * (1. + previous)
            double = both * (2. + previous)
            if single >= double and single >= previous:
                expected, previous = 'single', single
            elif double >= previous:
                expected, previous = 'double', double
            else:
                expected = 'skip'
            assert tp.longest_streak_action(days, hit, both) == expected != 'skip', (days, hit, both)
    print('rule cases', 400 * len(rates))


def check_dispatch():
    print('boundary', tp.LONGEST_STREAK_FROM)
    boundary = tp.LONGEST_STREAK_FROM
    policy = strategy._load_mdp()
    count, inherited = 0, 0
    for raw, streak, allow, partner, saver, best in product(
            (-2, 0, 1, 2, 3, 4, 180, 181, 271),
            (55, 56, 57, 58, 59, 100, 10**85, -1, True, False),
            (False, True), (False, True), (False, True),
            ((None, None), (57, 'trusted'), (37, 'untrusted'))):
        days = min(180, max(0, raw))
        objective = ('longest_streak' if streak >= boundary and raw > 0 else
                     'reach57' if not days or streak >= 57 or streak + 2 * days >= 57 else
                     'emax_season_best')
        assert tp.mdp_objective(streak, raw) == objective
        context = strategy.DecisionContext(.01 if saver else .99, .9 if partner else None,
                                           partner, '2027-09-26', allow, policy, *best)
        with patch.object(strategy, '_days_remaining', return_value=raw):
            assert strategy._regime(policy, streak, context.date) == (objective, days)
            try:
                decision = asdict(strategy.resolve_policy_decision(context, streak, saver))
            except IndexError:
                assert type(streak) is bool and objective == 'reach57' and raw > 0
                inherited += 1
                continue
            assert decision['objective'] == objective
            if objective == 'longest_streak':
                pick = 'double' if days <= 3 else 'single'
                assert decision == dict(action=pick if allow and partner else 'single',
                    policy_action=pick, source='mdp', objective=objective, streak=streak,
                    days_effective=days, best_supplied=best[0], best_status=best[1] or 'missing',
                    effective_best=None, tail_sha256=policy['tail'].sha256, degraded_reason=None,
                    reason=None, pick_bar=None, guard=None)
                assert strategy.effective_pick_bar(streak, context.date, saver, policy) == 0.
            count += 1
    print('dispatch', count, 'legacy boolean exceptions', inherited)


def check_fresh_boundary():
    from scripts.audit.season_walk import walk

    from pathlib import Path
    from tempfile import TemporaryDirectory
    import datetime

    assert walk.DEFAULT_STATES[-1].streak == tp.LONGEST_STREAK_FROM
    day = date(2027, 9, 26)
    frame = walk._predictions(day, .87)
    statuses = {int(pk): {'abstract': 'P', 'detailed': 'Scheduled'} for pk in frame.game_pk}
    at = datetime.datetime.combine(day, datetime.time(12), walk.ET)
    with TemporaryDirectory() as tmp, walk.offline(at):
        for streak in (tp.LONGEST_STREAK_FROM - 1, tp.LONGEST_STREAK_FROM):
            selection = strategy.select_pick(frame, str(day), Path(tmp), streak=streak,
                saver_available=False, game_statuses_detailed=statuses)
            assert (selection.decision.objective == tp.OBJECTIVE_LONGEST) == (streak == tp.LONGEST_STREAK_FROM)
    print('fresh selection and walk state follow boundary')


if __name__ == '__main__':
    check_arithmetic()
    check_dispatch()
    check_fresh_boundary()
