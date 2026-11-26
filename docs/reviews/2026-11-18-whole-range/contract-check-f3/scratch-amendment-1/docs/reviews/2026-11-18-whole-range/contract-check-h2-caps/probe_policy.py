import argparse
import math
from dataclasses import replace
from datetime import date, timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from bts import contest_state as cs, strategy
from bts.simulate.tail_policy import OBJECTIVE_LONGEST, OBJECTIVE_REACH57, OBJECTIVE_TAIL
from probe_live import at, model_and_switch, fetch, state
from probe_validators import dated, call
from scripts.audit.season_walk import walk as w


def resolve(day, active, best, best_status, saver=False, quality=.96, mdp=None):
    ctx = strategy.DecisionContext(quality, .95, True, str(day), True, mdp, best, best_status)
    decision = strategy.resolve_policy_decision(ctx, active, saver)
    behavior = (decision.action, decision.policy_action, decision.objective, decision.days_effective,
                decision.effective_best, decision.reason, decision.pick_bar, decision.guard)
    return behavior, decision


def validate_cases(mdp):
    reach = date(2027, 7, 15)
    tail = date(2027, 9, 26)
    for day, active in ((reach, 12), (tail, 12), (reach, 58), (tail, 58), (tail, 100), (tail, 400),
                         (tail + timedelta(days=1), 100)):
        baseline = resolve(day, active, max(57, active), 'trusted', mdp=mdp)[0]
        for best in (57, 58, 100, 400):
            result, decision = resolve(day, active, best, 'trusted', mdp=mdp)
            print('policy', day, active, best, result, 'best fields', decision.best_supplied, decision.best_status)
            assert result == baseline
        if active < 57 and day == tail:
            for blob in (mdp, None):
                play, _ = resolve(day, active, 157, 'untrusted', mdp=blob)
                skip, _ = resolve(day, active, 157, 'trusted', mdp=blob)
                print('trust delta', blob is not None, play, skip)
                assert play[0] in {'single', 'double'} and skip[0] == 'skip'

    with TemporaryDirectory() as tmp, w.offline(at(reach)):
        path = Path(tmp)
        for counts in ((60, 75), (58, 58), (400, 400)):
            print('M3 dated', counts, call(lambda: cs.validate_dated_observation(dated(*counts), path / 'dated')))
        model_and_switch(path, reach, prior=12)
        print('M3 automatic', fetch(path, reach, 12, 60, recipient=False),
              state(path).streak, state(path).best_streak, state(path).best_status)
        print('coverage-missing', fetch(path, reach, 58, 58, recipient=False, coverage=False),
              state(path).streak, state(path).best_status, state(path).status)


def compare_grid(mdp, first, stop):
    edges = mdp['boundaries']
    points = sorted({.0, .96, 1., *(value for edge in edges for value in (math.nextafter(edge, 0.), edge, math.nextafter(edge, 1.)))})
    end = date(2027, 9, 27)
    count = 0
    objectives = {OBJECTIVE_REACH57: 0, OBJECTIVE_TAIL: 0, OBJECTIVE_LONGEST: 0}
    digest = sha256()
    for streak in range(first, stop):
        for days in range(1, 181):
            day = end - timedelta(days=days)
            for saver in (False, True):
                for p in points:
                    expected, _ = resolve(day, streak, 57, 'trusted', saver, p, mdp)
                    for best in (58, 100, 400):
                        actual, _ = resolve(day, streak, best, 'trusted', saver, p, mdp)
                        assert expected == actual, (streak, days, saver, p, best, expected, actual)
                        count += 1
                        objectives[actual[2]] += 1
                    digest.update(repr(expected).encode())
    print('grid', first, stop, count, objectives, digest.hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('first', type=int, nargs='?')
    parser.add_argument('stop', type=int, nargs='?')
    args = parser.parse_args()
    print(cs.__file__)
    artifacts = strategy._load_mdp()
    if args.first is None:
        validate_cases(artifacts)
    else:
        compare_grid(artifacts, args.first, args.stop)
