from collections import Counter
from datetime import date, timedelta
import gzip
import json
from pathlib import Path
import sys


CHECK = Path(__file__).resolve().parent
mode = sys.argv[1]
originals = sys.argv[2] if len(sys.argv) > 2 else None
sys.path.insert(0, str(CHECK / 'scratch'))
sys.path.insert(0, str(CHECK / ('base' if mode == 'base' else 'scratch') / 'src'))

import bts
from scripts.audit.season_walk import walk

print('bts', bts.__file__, 'walk', walk.__file__, flush=True)
if originals:
    saved = json.loads(gzip.decompress(next((walk.REPO / 'scripts/audit/season_walk/results').glob('*.json.gz')).read_bytes()))
    lo, hi = (int(value) for value in originals.split('-'))
    start = walk.WALK_START + timedelta(days=lo)
    end = walk.WALK_START + timedelta(days=hi)
    baseline = [r for r in saved['records'] if start.isoformat() <= r['day'] <= end.isoformat()]
    current = walk.walk(start=start, end=end)
    keys = ('action', 'pick_bar', 'skip_bar', 'days_left', 'days_effective', 'objective')
    assert [(r['day'], r['state']) for r in baseline] == [(r['day'], r['state']) for r in current]
    print('existing records', len(current), 'saved', saved['runs'],
          'diffs', Counter(k for a, b in zip(baseline, current) for k in keys if a.get(k) != b.get(k)),
          'checks', len(walk.check(current)))
else:
    state = walk.State('dated', 57, 57, 'used', 0.96)
    current = walk.walk(states=(state,))
    checks = walk.check(current)
    print('new records', len(current), Counter((r['in_season'], r['action'], r['objective']) for r in current),
          'checks', len(checks), checks[:2])
    for day in ('2027-07-15', '2027-09-24', '2027-09-25', '2027-09-26', '2027-09-27'):
        print(next(r for r in current if r['day'] == day))
    m = next(r for r in current if r['day'] == '2027-09-26')
    print('does check reject single on finale?', walk.check([m | {'action': 'single', 'batter': 'Batter 0', 'double': None}]))
    print('low-p skip on finale:', walk.check([walk.walk_one(date(2027, 9, 26),
                                                            walk.State('dated', 57, 57, 'used', .72))]))
    print('boundaries', walk.boundary_checks())
    with gzip.open(CHECK / f'walk-{mode}-57.json.gz', 'wt') as handle:
        json.dump({'records': current, 'problems': checks}, handle)
