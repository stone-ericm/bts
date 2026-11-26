"""Check saved walk evidence against a fresh walk and the independently listed cells."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / 'scripts/audit/season_walk/results'
CSV = ROOT / 'docs/reviews/2026-11-18-whole-range/contract-g/affected-states.csv'


def keyed(records):
    return {(r['day'], r['state']): r for r in records}


def warned(records):
    return {tuple(line.split(':', 1)[0].split()): line for line in records}


with (HERE / 'walk.json').open() as file:
    live = json.load(file)
with gzip.open(RESULTS / 'walk-bc143131.json.gz', 'rt') as file:
    old = json.load(file)
with gzip.open(RESULTS / 'walk-g.json.gz', 'rt') as file:
    saved = file.read()
assert json.loads(saved) == live
assert saved == (HERE / 'walk.json').read_text()
assert (RESULTS / 'walk-g.txt').read_bytes() == (HERE / 'walk.txt').read_bytes()
assert old['boundary'] == live['boundary'] and not live['boundary']['problems']
with CSV.open() as file:
    affected = {tuple(int(row[k]) for k in ('streak', 'days_left', 'saver', 'bin'))
                for row in csv.DictReader(file)}
with np.load(ROOT / 'data/models/mdp_policy.npz') as npz:
    table, boundaries = npz['policy_table'], npz['boundaries']
base_warnings, current_warnings = warned(old['problems']), warned(live['problems'])
base, current = keyed(old['records']), keyed(live['records'])
assert len(base) == len(current) == 4599 and base.keys() == current.keys()
changes = Counter()
for key, row in current.items():
    before = base[key]
    kind, streak, best, saver_state, raw_p = row['state'].split('/')
    streak, days, saver = int(streak), row['days_effective'], int(row['saver_available'])
    q = sum(float(raw_p) >= b for b in boundaries)
    if row['objective'] == 'reach57' and (streak, days, saver, q) in affected:
        assert before['action'] == 'skip' and before['named'] and before['skip_dm']
        assert row == before | dict(action='single', named=False, batter='Batter 0',
                                    chosen_on_kind=kind, skip_dm=None, skip_bar=None,
                                    pick_bar=None, best_p=None)
        assert (key in base_warnings) == row['in_season'] and key not in current_warnings
        changes['flagged' if key in base_warnings else 'preseason'] += 1
    else:
        assert row == before, key
    if key in current_warnings:
        assert current_warnings[key] == base_warnings[key]
        assert row['in_season'] and row['action'] == 'skip' and row['objective'] == 'reach57'
        assert not np.any(table[streak, days, saver])
        changes['all-skip warning'] += 1
assert len(old['problems']) == 841 and len(live['problems']) == 612
assert changes == {'flagged': 229, 'preseason': 23, 'all-skip warning': 612}
print('4,599 records, 252 skip-to-single, 612 all-skip warnings, zero other diffs.', changes)
print('Fresh JSON and text match the saved walk byte for byte (JSON decompressed).')
