"""Compare full archived packages in separate interpreters, against the CSV."""
from dataclasses import asdict
from datetime import date, timedelta
from functools import cache
from itertools import product
from pathlib import Path
import csv
import math
import os
import pickle
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTRACT = ROOT / 'docs/reviews/2026-11-18-whole-range/contract-g'
END = date(2027, 9, 27)


def capture(side):
    import bts
    from bts import strategy
    from bts.simulate import mdp

    source = ROOT / 'src' if side == 'head' else HERE / 'base/src'
    assert Path(bts.__file__).resolve().is_relative_to(source), bts.__file__
    print(side, bts.__file__, flush=True)
    artifact = strategy._load_mdp()
    assert artifact['base_sha256'].startswith('66d15471') and artifact['tail'] is not None
    table, bounds = artifact['policy_table'], artifact['boundaries']
    pairs = list(zip([0., *bounds], [*bounds, 1.]))
    points = [((lo + hi) / 2, lo, math.nextafter(hi, -math.inf)) for lo, hi in pairs]
    points[0] += (-0.2, float('nan'))
    points[-1] += (1., 1.2)
    ppm = [(q, p) for q, group in enumerate(points) for p in group]
    raw = bytearray()
    guarded = {}
    for s, d, saver in product(range(58), range(181), (False, True)):
        for q, p in ppm:
            args = (table, bounds, s, d, saver, p)
            action = mdp.lookup_action(*args)
            if side == 'head':
                action_g, guard = mdp.lookup_action_guarded(*args)
                if guard is not None:
                    assert type(guard) is dict
                    assert all(type(k) is str for k in guard)
                    assert all(type(guard[k]) is int for k in ('bin', 'from_bin'))
                    assert all(type(guard[k]) is str for k in ('rule', 'table_action'))
                    key = (s, d, int(saver), q)
                    guarded.setdefault(key, (action_g, guard))
                    assert guarded[key] == (action_g, guard) and action != action_g
                else:
                    assert action == action_g
            raw.append(mdp.ACTIONS.index(action))

    strategy._days_remaining = cache(strategy._days_remaining)
    shared, bars, overrides, coverage = [], [], 0, set()
    with (CONTRACT / 'affected-states.csv').open() as f:
        expected = {}
        for row in csv.DictReader(f):
            key = tuple(int(row[k]) for k in ('streak', 'days_left', 'saver', 'bin'))
            expected[key] = (row['guarded_action'], dict(rule='monotone_skip_v1', bin=key[3],
                             from_bin=int(row['from_bin']), table_action=row['table_action']))
    if side == 'head':
        assert guarded == expected and len(guarded) == 1511

    days = [(date(2027, 3, 24) + timedelta(days=i)).isoformat() for i in range(188)]
    for day, s, saver in product(days, range(58), (False, True)):
        bar = strategy.effective_pick_bar(s, day, saver, artifact, s, 'trusted')
        bars.append(bar)
        for q, p in ppm:
            ctx = strategy.DecisionContext(p, .97, True, day, True, artifact, s, 'trusted')
            actual = strategy.resolve_policy_decision(ctx, s, saver)
            payload = vars(actual).copy()
            guard = payload.pop('guard', None)
            key = (s, actual.days_effective, int(saver), q)
            hit = expected.get(key) if actual.objective == 'reach57' else None
            if side == 'head':
                if hit:
                    assert (actual.action, guard) == hit
                    assert actual.action == actual.policy_action and actual.pick_bar is None
                    assert all(type(guard[k]) is int for k in ('bin', 'from_bin'))
                    payload.update(action='skip', policy_action='skip', pick_bar=bar)
                    overrides += 1
                    coverage.add(key)
                else:
                    assert guard is None
                if actual.objective == 'reach57':
                    if bar is None:
                        assert actual.action == 'skip'
                    elif p >= bar:
                        assert actual.action != 'skip'
                    if p == bar:
                        assert guard is None
            elif hit:
                assert actual.action == actual.policy_action == 'skip' and actual.pick_bar == bar
            shared.append(tuple(payload.values()))
    if side == 'head':
        assert coverage == set(expected)
        print(f'{len(guarded)} distinct changed cells; {overrides} changed resolver calls')
    with (HERE / f'{side}-grid.pickle').open('wb') as f:
        pickle.dump((raw, shared, bars), f)
    print(f'{len(raw)} raw calls, {len(shared)} resolver calls, {len(bars)} direct bars; {len(ppm)} probabilities')


if __name__ == '__main__':
    if len(sys.argv) == 2:
        capture(sys.argv[1])
    else:
        for side, src in [('base', HERE / 'base/src'), ('head', ROOT / 'src')]:
            subprocess.run([sys.executable, __file__, side], cwd=ROOT, check=True, timeout=40,
                           env=os.environ | {'PYTHONPATH': str(src)})
        def load(side):
            with (HERE / f'{side}-grid.pickle').open('rb') as f:
                return pickle.load(f)
        assert load('base') == load('head')
        out = HERE / 'affected-states.csv'
        subprocess.run([sys.executable, str(CONTRACT / 'enumerate_states.py'), str(ROOT), str(out)],
                       check=True, timeout=10)
        assert out.read_bytes() == (CONTRACT / out.name).read_bytes()
        print('All unaffected fields and raw actions agree, and CSV regenerated byte for byte.')
