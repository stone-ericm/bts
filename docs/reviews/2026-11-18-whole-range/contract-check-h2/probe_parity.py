from collections import Counter
from dataclasses import asdict, replace
from datetime import date, timedelta
from functools import cache
from itertools import product
from pathlib import Path
import csv
import hashlib
import json
import logging
import os
import sys

import numpy as np


check = Path(__file__).resolve().parent
mode = sys.argv[1]
scratch = check / 'scratch'
sys.path.insert(0, str((check / 'base' if mode == 'base' else scratch) / 'src'))
os.chdir(scratch)
import bts
from bts import strategy
from bts.simulate import tail_policy


print('bts', bts.__file__, flush=True)
logging.disable(logging.CRITICAL)
strategy._days_remaining = cache(strategy._days_remaining)
artifact = strategy._load_mdp()
tail = artifact['tail']
freq, ph, pb = [.5, .5], [.75, .75], [.5, .5]
two = replace(tail, boundaries=[.82], bin_freq=freq, bin_p_hit=ph, bin_p_both=pb,
              policy_table=tail_policy.solve_emax_season_best(freq, ph, pb).policy,
              manifest=tail_policy.tail_manifest(n_bins=2, hits=3, both=2, late_seed_days=4, bin_counts=[2, 2]))
tail_policy.validate_tail_policy(two, expected_base_sha=artifact['base_sha256'])
modes = (artifact, artifact | {'tail': None, 'tail_error': 'tail missing'},
         artifact | {'policy_table': None, 'base_sha256': None},
         artifact | {'tail': replace(tail, base_policy_sha256='0' * 64)},
         None, artifact | {'tail': two}, artifact | {'season_length': 90},
         artifact | {'season_length': -1})
digest = hashlib.sha256()
guard_rows = 0
count = 0
g_diffs = Counter()


def compare(s, day, saver, p, p2, art, best, status, allow):
    global count, guard_rows
    ctx = strategy.DecisionContext(p, p2, p2 is not None, day, allow, art, best, status)
    decision = strategy.resolve_policy_decision(ctx, s, saver)
    bar = strategy.effective_pick_bar(s, day, saver, mdp=art, best_streak=best, best_status=status)
    digest.update(json.dumps([asdict(decision), bar], sort_keys=True).encode())
    count += 1
    guard_rows += decision.guard is not None
    return decision, bar


probs = [.01, .99, *(float(p) for edge in [*artifact['boundaries'], .8, .82]
                    for p in [np.nextafter(edge, 0.), edge, np.nextafter(edge, 1.)])]
best_inputs = [(None, None), *((n, status) for n, status in product(range(58), ('trusted', 'untrusted')))]
for art in modes:
    for d, s, saver in product(range(272), range(57), (False, True)):
        day = (date(2028, 9, 28) - timedelta(days=d)).isoformat()
        i = d + s * 3 + saver
        compare(s, day, saver, probs[i % len(probs)], .86 if i % 3 else None, art,
                *best_inputs[i % len(best_inputs)], bool(i % 2))
    for s, day, saver, p in product((57, 58, 60, 100), ('2027-09-27', '2027-12-31', '2028-09-28'),
                                    (True, False), (.01, .8, .99)):
        compare(s, day, saver, p, .87, art, 57, 'trusted', True)
for s, cut in ((9, .55), (15, .60), (45, .65)):
    for p2 in (np.nextafter(cut, 0.), cut, np.nextafter(cut, 1.), None):
        compare(s, '2027-07-15', True, 1., p2, None, s, 'trusted', True)


guard_file = 'docs/reviews/2026-11-18-whole-range/contract-g/affected-states.csv'
with open(guard_file) as stream:
    for row in csv.DictReader(stream):
        s, d, v, q = (int(row[field]) for field in ('streak', 'days_left', 'saver', 'bin'))
        day = (date(2027, 9, 27) - timedelta(days=d)).isoformat()
        decision, _ = compare(s, day, bool(v), artifact['boundaries'][q - 1], .9, artifact, s, 'trusted', True)
        assert decision.guard and decision.action == row['guarded_action'], row
print('parity', count, 'cases,', guard_rows, 'guards', digest.hexdigest())

edges = [0., *artifact['boundaries'], 1.]
reps = [p for lo, hi in zip(edges, edges[1:]) for p in (lo, (lo + hi) / 2, np.nextafter(hi, lo))]
for offset, v in product(range(187), (True, False)):
    day = (date(2027, 3, 24) + timedelta(days=offset)).isoformat()
    bar = strategy.effective_pick_bar(57, day, v, artifact, 57, 'trusted')
    g_diffs['bar == old_bar (line 131)'] += bar is not None
    for p in reps:
        decision = strategy.resolve_policy_decision(strategy.DecisionContext(
            p, .95, True, day, True, artifact, 57, 'trusted'), 57, v)
        g_diffs['decision fields equal (line 150)'] += (decision.action, decision.objective) != ('skip', 'reach57')
        g_diffs['old.action != skip at p == bar (line 157)'] += p == bar
print('G comparison clauses evaluated without stopping at the first failure', g_diffs)
