from collections import Counter
from datetime import date, timedelta
import csv
from functools import cache
import gzip
from itertools import product
import json
from pathlib import Path

import numpy as np

from bts import strategy
from bts.simulate import mdp
from scripts.audit.season_walk import walk


ROOT = Path('.codex-review/wr-check-g')
CONTRACT = Path('docs/reviews/2026-11-18-whole-range/contract-g/affected-states.csv')
MODELS = Path('data/models')


def csv_index():
    with CONTRACT.open() as stream:
        return {
            tuple(int(row[key]) for key in ('streak', 'days_left', 'saver', 'bin')): row
            for row in csv.DictReader(stream)
        }


def model_examples():
    for path in sorted(MODELS.glob('mdp_policy*.npz')):
        table, bounds, season_length = mdp.load_policy(path)
        prior_plays = np.maximum.accumulate(table != 0, axis=-1)
        hits = (table[..., 1:] == 0) & prior_plays[..., :-1]
        hits = hits[:57, 1:season_length + 1].sum(axis=-1)
        print(path.name, table.shape, season_length, bounds,
              'affected states', int((hits > 0).sum()), 'multi', int((hits > 1).sum()))
    table = np.zeros((58, 181, 2, 3), dtype=np.int8)
    table[10, 100, 0] = (0, 2, 0)
    for name, columns, bounds, probability in (
        ('tied, raw skipped bin has unreachable lower double', 3, [0.8, 0.8], 0.8),
        ('empty boundary vector', 3, [], 0.99),
        ('None boundaries', 3, None, 0.99),
        ('table narrower than bins, in range', 2, [0.8, 0.8], 0.79),
        ('table narrower than bins, out of range', 2, [0.8, 0.8], 0.8),
    ):
        try:
            result = mdp.lookup_action(table[..., :columns], bounds, 10, 100, False, probability)
        except (TypeError, IndexError) as exc:
            result = f'{type(exc).__name__}: {exc}'
        print(name, result)
    for streak, days in ((57, 100), (10, 0), (10, -1)):
        assert mdp.lookup_action(None, None, streak, days, False, 0.99) == 'skip'


def lookup_grid(table, bounds, groups, affected):
    counts = Counter()
    for streak, days, saver, quality in product(range(58), range(181), (False, True), range(5)):
        row = table[streak, days, saver * 1]
        expected = 'skip' if streak == 57 or days == 0 else mdp.ACTIONS[row[quality]]
        hit = affected.get((streak, days, saver, quality))
        counts['cells'] += 1
        counts['affected'] += bool(hit)
        playable = np.flatnonzero(row)
        if hit:
            assert expected == 'skip' and quality > playable[0]
            lower = playable[playable < quality]
            assert int(hit['from_bin']) == lower[-1]
            assert hit['guarded_action'] == mdp.ACTIONS[row[lower[-1]]]
        for probability in groups[quality]:
            assert mdp.lookup_action(table, bounds, streak, days, saver, probability) == expected
            counts['calls'] += 1
    print('lookup', counts)


def decision_grid(artifact, groups, affected, early):
    start = date(2027, 3, 24) if early else date(2027, 3, 31)
    end = date(2027, 3, 30) if early else date(2027, 9, 27)
    counts = Counter()
    seen = set()
    bar_for = cache(strategy.effective_pick_bar)
    current_day = start
    while current_day <= end:
        day = current_day.isoformat()
        for streak, saver, quality in product(range(58), (False, True), range(5)):
            for probability in groups[quality]:
                context = strategy.DecisionContext(probability, 0.95, True, day, True, artifact,
                                                   best_streak=streak, best_status='trusted')
                decision = strategy.resolve_policy_decision(context, streak, saver)
                counts['calls'] += 1
                counts[decision.objective] += 1
                key = streak, decision.days_effective, saver, quality
                hit = affected.get(key) if decision.objective == 'reach57' else None
                if hit:
                    assert decision.action == 'skip' and decision.pick_bar is not None
                    counts['affected'] += 1
                    seen.add(key)
                    if hit['guarded_action'] == 'double':
                        counts['affected_double'] += 1
                bar = bar_for(streak, day, saver)
                if decision.objective != 'reach57' or decision.action != 'skip':
                    continue
                assert decision.pick_bar == bar
                if bar is None:
                    counts['skip_without_bar'] += 1
                    if streak < 57:
                        counts['all_skip_below_streak_57'] += 1
                elif probability >= bar:
                    counts['base_skip_at_or_above_bar'] += 1
                    counts['projected_skip_at_or_above_bar'] += hit is None
        current_day += timedelta(days=1)
    print(start, end, counts, 'unique affected reachable', len(seen))
    return seen


def grids():
    affected = csv_index()
    artifact = strategy._load_mdp()
    table, bounds = artifact['policy_table'], artifact['boundaries']
    edges = [0.0, *bounds, 1.0]
    groups = [
        (low, (low + high) / 2, np.nextafter(high, low) if high < 1.0 else 0.999)
        for low, high in zip(edges, edges[1:])
    ]
    neighbor_singles = [row for row in affected.values()
                        if row['guarded_action'] == 'double' and row['row'][2] == 's']
    print('inherited doubles followed by a single', len(neighbor_singles),
          Counter(row['row'] for row in affected.values() if row['guarded_action'] == 'double'))
    print('CSV at d=180', sum(key[1] == 180 for key in affected))
    model_examples()
    lookup_grid(table, bounds, groups, affected)
    strategy._days_remaining = cache(strategy._days_remaining)
    seen = decision_grid(artifact, groups, affected, False)
    seen.update(decision_grid(artifact, groups, affected, True))
    assert seen == affected.keys()
    for day in ('2026-12-31', '2027-01-01', '2027-03-01', '2027-03-24', '2027-03-30',
                '2027-03-31', '2027-04-01', '2027-09-26', '2027-09-27'):
        example = strategy.DecisionContext(0.96, 0.95, True, day, True, artifact,
                                           best_streak=0, best_status='trusted')
        decision = strategy.resolve_policy_decision(example, 0, True)
        print(day, strategy._days_remaining(day), decision.days_effective,
              decision.objective, decision.action)
    print('finite-bar examples:', strategy.effective_pick_bar(50, '2027-03-24', False),
          strategy.effective_pick_bar(56, '2027-03-24', False))


def walk_comparison():
    affected = csv_index()
    with gzip.open('scripts/audit/season_walk/results/walk-bc143131.json.gz', 'rt') as stream:
        saved = json.load(stream)
    base = json.loads((ROOT / 'walk-base.json').read_text())
    assert base.keys() == saved.keys()
    print('walk changed top-level fields', [key for key in base if base[key] != saved[key]])
    assert base == saved
    artifact = strategy._load_mdp()
    table, bounds = artifact['policy_table'], artifact['boundaries']
    counts = Counter()
    for record in base['records']:
        _, streak, _, _, probability = record['state'].split('/')
        streak, probability = int(streak), float(probability)
        quality = sum(probability >= edge for edge in bounds)
        days, saver = record['days_effective'], record['saver_available']
        hit = affected.get((streak, days, saver, quality)) if record['objective'] == 'reach57' else None
        flagged = bool(walk.check([record]))
        if hit:
            assert record['action'] == 'skip' and hit['guarded_action'] == 'single'
            counts['all predicted skip-to-single'] += 1
            counts['cleared warnings' if flagged else 'unflagged changes'] += 1
            if not record['in_season']:
                counts['preseason changes'] += 1
        elif flagged:
            assert not table[streak, days, int(saver)].any()
            assert record['pick_bar'] is None and record['skip_bar'] is None
            counts['still flagged'] += 1
            counts[f'still flagged at streak {streak}'] += 1
    print('walk projection', counts)
    print('boundary', base['boundary'])


if __name__ == '__main__':
    import sys

    {'grid': grids, 'walk': walk_comparison}[sys.argv[1]]()
