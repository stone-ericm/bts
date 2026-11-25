from collections import Counter
from datetime import date, timedelta
from pathlib import Path
import gzip
import json
import sys


check = Path(__file__).resolve().parent
mode, selection = sys.argv[1:]
scratch = check / 'scratch'
sys.path[:0] = [str((check / 'base' if mode == 'base' else scratch) / 'src'), str(scratch)]
import bts
from scripts.audit.season_walk import walk

print('bts', bts.__file__, 'walk', walk.__file__, flush=True)
if selection == 'new':
    state = walk.State('dated', 57, 57, 'used', .96)
    records = walk.walk(states=(state,))
    problems = walk.check(records)
    print('new', len(records), Counter((r['action'], r['objective'], r['in_season'],
                                       r['chosen_on_kind'], bool(r['double'])) for r in records))
    print('double dates', [r['day'] for r in records if r['action'] == 'double'])
    print('problems', len(problems), problems[:2])
    final = next(r for r in records if r['day'] == '2027-09-26')
    print('wrong single:', walk.check([final | {'action': 'single', 'batter': 'Batter 0', 'double': None}]))
    print('named skip:', walk.check([final | {'action': 'skip', 'named': True, 'skip_dm': ''}]))
    print('boundary', walk.boundary_checks())
    with gzip.open(check / f'walk-{mode}-new.json.gz', 'wt') as stream:
        json.dump({'records': records, 'problems': problems}, stream)
else:
    lo, hi = map(int, selection.split('-'))
    start = walk.WALK_START + timedelta(days=lo)
    end = walk.WALK_START + timedelta(days=hi)
    with gzip.open(walk.REPO / 'scripts/audit/season_walk/results/walk-g.json.gz', 'rt') as stream:
        saved = json.load(stream)
    print('saved totals', saved['runs'], len(saved['problems']),
          'guard key count', sum('guard' in r for r in saved['records']),
          'policy_guard key count', sum('policy_guard' in r for r in saved['records']),
          'findings by state', Counter(row.split()[1].split('/')[1] for row in saved['problems']))
    baseline = [r for r in saved['records'] if start.isoformat() <= r['day'] <= end.isoformat()]
    saved_problems = [p for p in saved['problems'] if start.isoformat() <= p.split()[0] <= end.isoformat()]
    current = walk.walk(start=start, end=end)
    problems = walk.check(current)
    assert [(r['day'], r['state']) for r in baseline] == [(r['day'], r['state']) for r in current]
    print('existing', len(current), 'changed fields', Counter(
        key for a, b in zip(baseline, current) for key in a.keys() | b.keys() if a.get(key) != b.get(key)),
          'changed shapes', sum(a.keys() != b.keys() for a, b in zip(baseline, current)),
          'problem comparison', len(problems), problems == saved_problems)
    assert current == baseline and problems == saved_problems
