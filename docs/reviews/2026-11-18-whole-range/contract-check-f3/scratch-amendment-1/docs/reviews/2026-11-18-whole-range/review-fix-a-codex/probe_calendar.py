"""Compare the executable policy and health calendar against the archived source."""

from collections import defaultdict
from datetime import date, timedelta
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import subprocess
import sys


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]


def sample():
    from bts import strategy
    from bts.health import tail_policy as health
    from bts.simulate import mdp, tail_policy

    policy = strategy._load_mdp()
    assert policy['policy_table'] is not None and policy['tail'] is not None
    assert policy['base_sha256'] == policy['tail'].base_policy_sha256
    dates = {}
    digest = sha256()
    for year in (2026, 2027):
        for offset in range(365):
            day = date(year, 1, 1) + timedelta(days=offset)
            iso = day.isoformat()
            remaining = strategy._days_remaining(iso)
            assert health._days_remaining(day) == remaining
            if year == 2026:
                assert remaining == max(0, (date(year, 9, 28) - day).days)
                effective = tail_policy.effective_days(remaining, policy['season_length'])
                # Include both shipped policy day-slices in the date-by-date comparison.
                digest.update(policy['policy_table'][:, effective].tobytes())
                if effective <= policy['tail'].max_days:
                    digest.update(policy['tail'].policy_table[:, :, effective].tobytes())
            dates[iso] = remaining

    probabilities = [0.0, *map(float, policy['boundaries']), .96]
    for streak in range(57):
        for available in (False, True):
            for probability in probabilities:
                for offset in range(365):
                    iso = (date(2026, 1, 1) + timedelta(days=offset)).isoformat()
                    actual = strategy._mdp_action_from(policy, probability, streak, iso, available)
                    expected = mdp.lookup_action(policy['policy_table'], policy['boundaries'],
                                                 streak, dates[iso], available, probability,
                                                 policy['season_length'])
                    assert actual == expected
                    digest.update(actual.encode())

    def state(day, streak, best, saver, p):
        context = strategy.DecisionContext(p, p, True, day, True, policy, best, 'trusted')
        decision = strategy.resolve_policy_decision(context, streak, saver)
        assert decision.degraded_reason is None
        return decision.action, decision.objective, decision.days_effective

    late = {'|'.join(map(str, (day, streak, best, saver, p))): state(day, streak, best, saver, p)
            for day in (f'2027-09-{n}' for n in range(17, 27))
            for streak in range(40, 57) for best in range(streak, 58)
            for saver in (False, True) for p in (.96, .60)}
    alerts = health.check(base_path=mdp.DEFAULT_POLICY_PATH,
                          tail_path=tail_policy.DEFAULT_TAIL_POLICY_PATH, today=date(2028, 9, 26))
    assert len(alerts) == 1 and alerts[0].level == 'WARN' and 'not configured' in alerts[0].message
    assert strategy.season_end_date('2028-03-24') == '2028-09-28'
    assert strategy._days_remaining('2028-09-26') == 2
    points = [dates[day] for day in ('2027-03-24','2027-09-26','2027-09-27','2027-09-28')]
    return {'source': strategy.__file__, 'digest': digest.hexdigest(), 'points': points, 'late': late}


def compare():
    def collect(source):
        env = dict(os.environ, PYTHONPATH=str(source))
        return json.loads(subprocess.check_output([sys.executable, __file__, 'sample'], cwd=ROOT, env=env))

    base, head = collect(REVIEW / 'base/src'), collect(ROOT / 'src')
    assert 'base/src' in base['source'] and 'base/src' not in head['source']
    assert base['digest'] == head['digest']
    assert head['points'] == [187, 1, 0, 0]
    print('2026 all day slices and reach-57 lookups:', head['digest'])
    print('2027 opening/finale/next/after:', base['points'], '->', head['points'])
    changed = defaultdict(list)
    for key, old in base['late'].items():
        new = head['late'][key]
        if old[0] != new[0]:
            day, streak, best, saver, p = key.split('|')
            changed[p, day, saver, old[0], new[0]].append((int(streak), int(best)))
    print('Total input states:', len(head['late']))
    for key, pairs in sorted(changed.items()):
        streaks = defaultdict(list)
        for streak, best in pairs:
            streaks[streak].append(best)
        print(*key, len(pairs), dict(streaks))


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    if sys.argv[1:] == ['sample']:
        print(json.dumps(sample()))
    else:
        compare()
