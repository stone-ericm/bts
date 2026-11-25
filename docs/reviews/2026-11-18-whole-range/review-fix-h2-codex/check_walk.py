import gzip
import json
from collections import Counter
from pathlib import Path


RESULTS = Path('scripts/audit/season_walk/results')
HIGH = 'dated/58/60/used/0.96'


def read_gzip(name):
    with gzip.open(RESULTS / name, 'rt') as f:
        return json.load(f)


if __name__ == '__main__':
    baseline = read_gzip('walk-ea-g-h.json.gz')
    saved = read_gzip('walk-h2.json.gz')
    fresh = json.loads(Path('.codex-review/wr-h2c/walk.json').read_text())
    assert fresh == saved, 'fresh walk differs from the saved H2 walk'
    old = [r for r in fresh['records'] if r['state'] != HIGH]
    new = [r for r in fresh['records'] if r['state'] == HIGH]
    assert len(old) == 4818 and old == baseline['records'] and fresh['boundary'] == baseline['boundary']
    assert fresh['problems'] == [] and fresh['runs'] == 5037 and len(new) == 219
    assert len({r['day'] for r in new}) == len(new)
    counts = Counter(r['action'] for r in new)
    doubles = [r['day'] for r in new if r['action'] == 'double']
    skips = [r['day'] for r in new if r['action'] == 'skip']
    assert counts == {'single': 207, 'double': 3, 'skip': 9}
    assert doubles == ['2027-09-24', '2027-09-25', '2027-09-26']
    assert skips[0] == '2027-09-27' and skips[-1] == '2027-10-05'
    print(len(old), 'baseline rows and boundary identical, fresh output equals stored H2,', dict(counts), doubles, skips)
