from collections import Counter
import gzip
import json
from pathlib import Path

from scripts.audit.season_walk import walk


work = Path(__file__).resolve().parent
saved = work.parent.parent / 'scripts/audit/season_walk/results'
with gzip.open(saved / 'walk-bc143131.json.gz', 'rt') as stream:
    baseline = json.load(stream)
with gzip.open(saved / 'walk-e.json.gz', 'rt') as stream:
    carried = json.load(stream)
current = json.loads((work / 'walk.json').read_text())
assert baseline['runs'] == current['runs'] == 4599
assert current['by_day'] == baseline['by_day'] and current['problems'] == []
omit = {'skip_dm', 'best_p'}
flagged = Counter()
for old, new in zip(baseline['records'], current['records'], strict=True):
    assert {k: v for k, v in old.items() if k not in omit} == {
        k: v for k, v in new.items() if k not in omit}, (old, new)
    if any('skip message' in problem for problem in walk.check([old])):
        if 'at or above' in new['skip_dm']:
            flagged['K3'] += 1
            example = new
        elif 'No probability comparison' in new['skip_dm']:
            flagged['K4'] += 1
        else:
            raise AssertionError(new)
assert flagged == {'K3': 229, 'K4': 612}
assert current == carried
planted = dict(example, skip_dm='Planted regression: below the pick bar.')
problems = walk.check([planted])
assert len(problems) == 1 and 'skip message' in problems[0]
print('4599 matching runs, all non-display fields except candidate probability compared individually')
print('Every field matches carried walk-e; old flags:', dict(flagged))
print(problems[0])
