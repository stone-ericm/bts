import argparse
import math
from collections import Counter
from datetime import timedelta
from hashlib import sha256

import bts
from bts import strategy
from support import DAY


def signature(ctx, active, saver):
    d = strategy.resolve_policy_decision(ctx, active, saver)
    return (d.action, d.policy_action, d.objective, d.days_effective, d.effective_best,
            d.reason, d.guard, d.pick_bar, d.tail_sha256, d.degraded_reason)


def compare(mdp, first, stop):
    edges = sorted(set(mdp['boundaries']) | set(mdp['tail'].boundaries))
    points = sorted({0., .96, 1., *(p for edge in edges for p in
                                    (math.nextafter(edge, 0.), edge, math.nextafter(edge, 1.)))})
    days = [str(DAY + timedelta(days=1 - d)) for d in range(270)]
    contexts = [(day, p, saver) for day in days for p in points for saver in (False, True)]
    count = Counter()
    guard = 0
    digest = sha256()
    for streak in range(first, stop):
        for day, p, saver in contexts:
            args = (p, .95, True, day, True, mdp)
            expected = signature(strategy.DecisionContext(*args, 57, 'trusted'), streak, saver)
            guard += expected[6] is not None
            for best in (58, 100, 400):
                actual = signature(strategy.DecisionContext(*args, best, 'trusted'), streak, saver)
                assert actual == expected, (streak, day, p, saver, best, expected, actual)
                count[actual[2]] += 1
            digest.update(repr(expected).encode())
    print('range', first, stop, 'raw day counts', len(days), 'bin probes', len(points),
          'actions', dict(count), 'guarded cells', guard, 'total', sum(count.values()), 'digest', digest.hexdigest())


def controls(mdp):
    for active in (57, 58, 100, 399, 400):
        for d in (0, 1, 2, 3, 4, 180, 269):
            day = str(DAY + timedelta(days=1 - d))
            for saver in (False, True):
                for allow, different in ((True, True), (True, False), (False, True)):
                    args = (.96, .95 if different else None, different, day, allow, mdp)
                    expected = signature(strategy.DecisionContext(*args, 57, 'trusted'), active, saver)
                    for best, status in ((58, 'trusted'), (100, 'trusted'), (400, 'trusted'), (None, 'missing')):
                        assert signature(strategy.DecisionContext(*args, best, status), active, saver) == expected
                    if d:
                        assert expected[2] == 'longest_streak'
                        assert expected[0] == ('double' if d <= 3 and allow and different else 'single')
                    else:
                        assert expected[2] != 'longest_streak'
    print('boundary: zero days, 57/58/100/399/400, both savers and double restrictions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('first', type=int)
    parser.add_argument('stop', type=int)
    args = parser.parse_args()
    print(bts.__file__)
    mdp = strategy._load_mdp()
    assert mdp is not None and mdp['tail'] is not None
    controls(mdp)
    compare(mdp, args.first, args.stop)
