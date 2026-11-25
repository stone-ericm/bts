from collections import Counter
from datetime import date
import gzip
from hashlib import sha256
import json
from pathlib import Path

from scripts.audit.season_walk import walk


REVIEW = Path(__file__).resolve().parent
RESULTS = REVIEW.parents[1] / 'scripts' / 'audit' / 'season_walk' / 'results'


def load(path):
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == '.gz' else raw


def verify_run():
    fresh_bytes = load(REVIEW / 'walk-run.json')
    current = json.loads(fresh_bytes)
    previous = json.loads(load(RESULTS / 'walk-g.json.gz'))
    boundary = walk.LONGEST_STREAK_FROM
    added = f'dated/{boundary}/{boundary}/used/0.96'
    new = [row for row in current['records'] if row['state'] == added]
    original = [row for row in current['records'] if row['state'] != added]
    assert original == previous['records'] and len(original) == 4599
    assert current['problems'] == previous['problems'] and len(current['problems']) == 612
    assert current['boundary'] == previous['boundary'] and current['states'] == previous['states'] + [added]
    assert len(new) == 219 and current['runs'] == len(new) + len(original)
    counts = Counter(row['action'] for row in new)
    assert counts == {'single': 207, 'double': 3, 'skip': 9}
    assert [row['day'] for row in new if row['action'] == 'double'] == [f'2027-09-{d}' for d in [24, 25, 26]]
    assert sum(not row['in_season'] for row in new if row['action'] == 'single') == 23
    for row in new:
        day = date.fromisoformat(row['day'])
        raw = max(0, (date(2027, 9, 27) - day).days)
        assert (row['days_left'], row['days_effective']) == (raw, min(180, raw))
        if raw:
            assert row['chosen_on_valid'] and row['chosen_on_kind'] == 'dated'
            assert row['objective'] == 'longest_streak' and row['source'] == 'mdp'
            assert (row['batter'], row['double']) == ('Batter 0', 'Batter 1' if raw <= 3 else None)
        else:
            assert row['action'] == 'skip' and row['objective'] == 'reach57'
    assert not walk.check(new)
    assert fresh_bytes == load(RESULTS / 'walk-h.json.gz')
    fresh_text = (REVIEW / 'walk.log').read_text().splitlines()
    saved_text = (RESULTS / 'walk-h.txt').read_text().splitlines()
    assert fresh_text == saved_text[:len(fresh_text)]
    print('original records', len(original), 'existing warnings', len(current['problems']),
          'warning streaks', Counter(msg.split('/')[1] for msg in current['problems']))
    print('added', counts, '23 preseason singles, 210 validated choices; H match', sha256(fresh_bytes).hexdigest())
    return {row['day']: row for row in new}


def plant_errors(rows):
    summer, finale = rows['2027-07-15'], rows['2027-09-26']
    for row, changes, expected in [
        (summer, {'action': 'skip', 'batter': None, 'named': True, 'objective': 'reach57'}, 'longest-streak action'),
        (finale, {'action': 'single', 'double': None, 'objective': 'reach57'}, 'longest-streak action'),
        (finale, {'days_effective': None}, 'effective days'),
        (finale, {'source': None}, 'decision source'),
        (finale, {'artifacts': None}, 'policy artifacts missing'),
        (finale, {'chosen_on_valid': False}, 'invalid chosen_on'),
        (finale, {'double': None}, 'selected slots'),
    ]:
        seen = walk.check([row | changes])
        assert any(expected in msg for msg in seen), seen
        print('planted', changes, seen)
    self_consistent_wrong = finale | {'expected_days_left': 4, 'days_left': 4, 'days_effective': 4,
                                      'action': 'single', 'double': None}
    print('checker trusts planted calendar reference', not walk.check([self_consistent_wrong]))


if __name__ == '__main__':
    plant_errors(verify_run())
