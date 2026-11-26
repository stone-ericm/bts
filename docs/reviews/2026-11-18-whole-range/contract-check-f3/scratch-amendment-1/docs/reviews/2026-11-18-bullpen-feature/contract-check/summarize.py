import csv
import json
from pathlib import Path


root = Path(__file__).resolve().parent
raw = Path('/Users/eric/projects/bts/data/raw/2025')


def rows(name):
    with (root / name).open() as source:
        return list(csv.DictReader(source))


gaps = rows('affected-game-sides.csv')
for gap in gaps:
    if gap['probable']:
        continue
    feed = json.loads((raw / f'{gap["game_pk"]}.json').read_text())['gameData']
    side = next(key for key in ('home', 'away') if feed['teams'][key]['id'] == int(gap['team']))
    print('missing cached probable', gap['game_pk'], side, 'raw probable', feed.get('probablePitchers', {}).get(side))

print('same-year following games with equal serve value', [
    (item['team'], item['previous_date'], item['date']) for item in rows('following-games-2025.csv')
    if float(item['delta_serving']) == 0
])
for name in ('parity-2025-only.csv', 'parity-2017-2025-frozen-lookup.csv'):
    values = rows(name)
    print(name)
    for first, second in [('old', 'serve_old'), ('asof', 'naive'), ('asof', 'serve_asof')]:
        changes = [abs(float(item[first]) - float(item[second])) for item in values if item[first] and item[second]]
        print(first, second, 'left missing only', sum(not item[first] and bool(item[second]) for item in values),
              'right missing only', sum(bool(item[first]) and not item[second] for item in values),
              'numeric differences', sum(delta != 0 for delta in changes), 'max', max(changes),
              'inclusive mean abs', sum(changes) / len(changes))
    lead = {(r['team'], r['date']) for r in gaps}
    added = [r for r in values if r['asof'] and not r['old'] and (r['team'], r['date']) not in lead]
    print('extra eligible gaps', [(r['team'], r['date'], r['n']) for r in added])
