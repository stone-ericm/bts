from dataclasses import astuple, replace
from datetime import date, timedelta
from functools import cache, lru_cache
from hashlib import sha256
from itertools import product
import logging
from math import nextafter
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def worker(src, work, base):
    import bts
    import numpy as np
    from bts import daily_decision as dd, strategy
    from bts.simulate import tail_policy as tp

    print('bts.__file__', bts.__file__)
    assert Path(bts.__file__).resolve() == src / 'bts' / '__init__.py'
    h = dd.load_decision('2027-07-15', work)
    assert h and dd.scoreable_commit_from(h, None)
    assert dd.decision_objective(h) == ('unknown' if base else 'longest_streak')
    can_write = dd.write_decision('2027-07-16', work, action='single', source='mdp',
                                  objective='longest_streak', delivery_status='delivered', scoreable=True)
    assert bool(can_write) != base
    print('H commit', dd.decision_objective(h), 'scoreable; H write', bool(can_write))
    logging.disable(logging.CRITICAL)
    strategy._days_remaining = cache(strategy._days_remaining)
    shipped = strategy._load_mdp()
    table, tail = shipped['policy_table'], shipped['tail']
    two = replace(tail, boundaries=[.82], bin_p_hit=[.75, .75], bin_p_both=[.5, .5], bin_freq=[.5, .5],
                  policy_table=np.repeat(tail.policy_table, 2, axis=-1),
                  manifest=tp.tail_manifest(n_bins=2, hits=3, both=2, late_seed_days=4, bin_counts=[2, 2]))
    tp.validate_tail_policy(two, expected_base_sha=shipped['base_sha256'])
    variants = {'shipped': shipped, 'no-artifacts': None,
                'missing-tail': shipped | {'tail': None, 'tail_error': 'missing'},
                'bad-tail': shipped | {'tail': None, 'tail_error': 'invalid'},
                'missing-base': shipped | {'policy_table': None, 'base_sha256': None},
                'mismatch': shipped | {'tail': replace(tail, base_policy_sha256='b' * 64)},
                'two-bins': shipped | {'tail': two},
                'bad-type': shipped | {'season_length': 'oops'},
                'negative': shipped | {'season_length': -1},
                'zero': shipped | {'season_length': 0},
                'short': shipped | {'season_length': 1, 'policy_table': table[:, :2]},
                'long': shipped | {'season_length': 240,
                                   'policy_table': np.pad(table, [(0, 0), (0, 60), (0, 0), (0, 0)], mode='edge')},
                'bad-cells': shipped | {'policy_table': table.astype(float)},
                'bad-boundaries': shipped | {'boundaries': ['oops'] * 4}}
    counts, raw_days, effective_days, digest = {}, set(), set(), sha256()

    @lru_cache(maxsize=None, typed=True)
    def bar(mode, streak, slate, saver, best):
        return strategy.effective_pick_bar(streak, slate, saver, variants[mode], *best)

    def compare(mode, streak, slate, saver, score, partner, allow=True, best=(None, None)):
        policy = variants[mode]
        ctx = strategy.DecisionContext(score, partner, partner is not None, slate, allow, policy, *best)
        raw_days.add(strategy._days_remaining(slate))
        try:
            outcome = strategy.resolve_policy_decision(ctx, streak, saver)
            row = astuple(outcome)
            effective_days.add(outcome.days_effective)
        except (IndexError, TypeError, ValueError) as exc:
            outcome, row = None, (type(exc).__name__, str(exc))
        digest.update(repr((mode, streak, slate, saver, score, partner, allow, best, row,
                            bar(mode, streak, slate, saver, best))).encode())
        counts[mode] = counts.get(mode, 0) + 1
        return outcome

    dates = [str(date(2028, 9, 28) - timedelta(days=raw)) for raw in range(272)]
    for streak, slate, saver in product(range(57), dates, (False, True)):
        compare('shipped', streak, slate, saver, .89, .85)
    edges = sorted({.80, .82, *shipped['boundaries']})
    edges = [score for edge in edges for score in (nextafter(edge, 0), edge, nextafter(edge, 1))]
    for p, streak, d, saver in product(edges, range(57), (0, 1, 3, 4, 28, 180, 271), (False, True)):
        compare('shipped', streak, dates[d], saver, p, None if d % 2 else .85, bool(streak % 2))
    for mode, date_, streak, saver, best in product(
            variants, [dates[n] for n in (0, 1, 4, 28, 180, 181, 271)] + ['2026-09-27', '2027-12-31'],
            (-1, False, True, 0, 9, 15, 45, 56, 57, 58, 100, 10**85), (False, True),
            ((None, None), (0, 'trusted'), (22, 'untrusted'), (57, 'trusted'))):
        if streak < 57 or strategy._days_remaining(date_) == 0:
            compare(mode, streak, date_, saver, .87, .83, bool(saver), best)
    for info, streak, d, mode in product(product(range(58), ('trusted', 'untrusted')), (0, 12, 50, 56),
                                          (1, 28, 180), ('shipped', 'short', 'no-artifacts', 'two-bins')):
        compare(mode, streak, dates[d], bool(streak % 2), .8, .83, True, info)
    for streak, cut in [(9, .55), (15, .6), (45, .65)]:
        for partner, allow in product((None, nextafter(cut, 0), cut, nextafter(cut, 1)), (False, True)):
            compare('no-artifacts', streak, dates[180], True, 1., partner, allow)
    guards = 0
    for streak, d, saver, q in np.argwhere((table == 0) & (np.cumsum(table != 0, axis=-1) != 0)):
        if streak >= 57 or d == 0:
            continue
        p = shipped['boundaries'][q - 1]
        outcome = compare('shipped', int(streak), dates[d], bool(saver), p, .99)
        assert outcome.guard['bin'] == q
        guards += 1
    assert guards == 1511
    print('parity', sum(counts.values()), 'guard cells', guards, 'raw', min(raw_days), max(raw_days),
          'effective', min(effective_days), max(effective_days), digest.hexdigest())
    with patch.object(strategy, '_load_mdp', side_effect=TypeError('broken loader')):
        try:
            fallback = strategy.effective_pick_bar(56, '2027-09-26', False)
        except TypeError:
            assert base
            print('legacy display loader exception propagated on base')
        else:
            assert fallback is None and not base
            print('legacy display loader exception converted to None under A2')


def main():
    from bts import daily_decision as dd

    with TemporaryDirectory() as tmp:
        work = Path(tmp).resolve()
        archive = subprocess.run(['git', 'archive', '0a2bec10', 'src'], cwd=ROOT, capture_output=True,
                                 check=True, timeout=6).stdout
        subprocess.run(['tar', '-xf', '-', '-C', str(work)], input=archive, check=True, timeout=6)
        assert dd.write_decision('2027-07-15', work, action='single', source='mdp', objective='longest_streak',
                                 delivery_status='delivered', scoreable=True)
        output = []
        for src, base in ((work / 'src', True), (ROOT / 'src', False)):
            env = os.environ | {'PYTHONPATH': os.pathsep.join((str(src), str(ROOT)))}
            result = subprocess.run([sys.executable, __file__, str(src), str(work), str(int(base))], cwd=ROOT,
                                    env=env, text=True, capture_output=True, timeout=18)
            print(result.stdout, end='')
            assert result.returncode == 0, result.stderr
            output.append(next(line for line in result.stdout.splitlines() if line.startswith('parity ')))
        assert output[0] == output[1], output


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        worker(Path(sys.argv[1]), Path(sys.argv[2]), bool(int(sys.argv[3])))
