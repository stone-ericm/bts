import argparse
import gzip
import json
from collections import Counter
from datetime import date

from bts import contest_state as cs
from scripts.audit.season_walk import walk as w


parser = argparse.ArgumentParser()
parser.add_argument('--existing', action='store_true')
parser.add_argument('--start', type=date.fromisoformat, default=w.WALK_START)
parser.add_argument('--end', type=date.fromisoformat, default=w.WALK_END)
args = parser.parse_args()
print(cs.__file__, 'dates', args.start, args.end)
new_state = w.State('dated', 58, 60, 'used', .96)
new_rows = w.walk(args.start, args.end, states=(new_state,))
findings = w.check(new_rows)
print('new rows', len(new_rows), Counter(r['action'] for r in new_rows), len(findings), findings[:3])
if hasattr(cs, 'MAX_SEASON_STREAK'):
    assert not findings
else:
    assert all(r['setup'] and r['error'] for r in new_rows)
if args.existing:
    path = w.REPO / 'scripts/audit/season_walk/results/walk-ea-g-h.json.gz'
    with gzip.open(path, 'rt') as f:
        saved = json.load(f)
    expected = [r for r in saved['records'] if str(args.start) <= r['day'] <= str(args.end)]
    rows = w.walk(args.start, args.end)
    differences = [dict(day=a['day'], state=a['state'], changed=sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k)))
                   for a, b in zip(rows, expected) if a != b]
    problems = w.check(rows)
    print('existing rows', len(rows), 'differences', len(differences), differences[:5], 'findings', problems[:3])
    assert len(rows) == len(expected) and not differences and not problems
