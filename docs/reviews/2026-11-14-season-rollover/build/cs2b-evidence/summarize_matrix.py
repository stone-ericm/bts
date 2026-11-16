"""Verify and summarize the recorded base/branch matrix; no live package imports."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent
base = json.loads((D / 'matrix-base.json').read_text())
branch = json.loads((D / 'matrix-branch.json').read_text())
assert len(base) == len(branch) == 126
lines = ['# Recorded date matrix', '',
         'Base `d8eb50d0` versus the CS2B branch, 18 values × 7 real entry paths.',
         'Raw records include return/exception, output, observed reads, pick result, and exact streak bytes (hex).',
         'All clocks fixed to 2027-03-24 23:00 ET; external services mocked.', '',
         '`a`: season-scoped read/write; `b`: unknown dated write rejected; `D`: required dateless tag preservation (outside the requested a/b classes).', '',
         '| Value | Shadow | Void | Final peer | Early peer | CLI | Strategy | Update |',
         '|---|---|---|---|---|---|---|---|']
changes = []
counts = {}
for index in range(0, len(base), 7):
    cells = []
    for old, new in zip(base[index:index+7], branch[index:index+7]):
        assert (old['row'], old['path']) == (new['row'], new['path'])
        season = new['season']
        path = new['path']
        fields = [k for k in old if old[k] != new[k]]
        if path != 'update':
            assert old['before_hex'] == new['before_hex'] == old['after_hex'] == new['after_hex'], (path, new['row'])
            assert old['pick_result'] == new['pick_result']
            if season is None:
                assert old == new, (path, new['row'], fields)
            if 'exception' in old['outcome']:
                assert old['outcome'] == new['outcome'], (path, new['row'])
            if path == 'cli' and old['outcome'].get('value', {}).get('exception'):
                assert old['outcome'] == new['outcome']
            category = 'a' if fields else '='
        elif season is not None:
            assert old['outcome'] == {'value': 14}
            assert new['outcome'] == {'value': 1}
            state = json.loads(new['after'])
            assert (state['season'], state['streak'], state['saver_available']) == (2027, 1, True)
            assert len(state['applied_dates']) == 1
            category = 'a'
        elif new['row'] == 'none':
            assert old['outcome'] == new['outcome'] == {'value': 14}
            old_state, new_state = json.loads(old['after']), json.loads(new['after'])
            assert new_state.pop('season') == 2026
            assert old_state == new_state
            category = 'D'
        else:
            assert new['outcome']['exception'] == 'ValueError'
            assert new['before_hex'] == new['after_hex']
            assert new['reads'] == []
            category = 'b'
        cells.append(category)
        counts[category] = counts.get(category, 0) + 1
        if fields:
            changes.append(f"| {new['row']} | {path} | {category} | {', '.join(fields)} |")
    labels = {
        'padded': '"2027-03-24"', 'unpadded-month': '"2027-3-24"',
        'unpadded-day': '"2027-03-4"', 'unpadded-both': '"2027-3-4"',
        'compact': '"20270324"', 'week': '"2027-W12-3"',
        'leading-space': '" 2027-03-24"', 'trailing-space': '"2027-03-24 "',
        'time': '"2027-03-24T10:00"', 'slash': '"2027/03/24"',
        'dot': '"2027.03.24"', 'impossible': '"2027-02-30"',
        'junk': '"junk"', 'empty': '""', 'date-object': 'date(2027, 3, 24)',
        'none': 'None', 'bytes': 'b"2027-03-24"', 'integer': '20270324 (int)',
    }
    lines.append('| ' + base[index]['row'] + ': `' + labels[base[index]['row']] + '` | ' + ' | '.join(cells) + ' |')
lines += ['', f'Counts: {counts}.', '',
          'For a: scheduler streak 13 → 0, CLI void display 13 → 0 where its age guard permits entry; strategy saver False → True (later base date exceptions remain identical); scorer 14 → 1, saver False → True, season 2027, original marker retained.',
          'For b: base returns 14 except date/bytes objects raise TypeError; branch raises ValueError before reads and preserves all bytes.',
          'For D: date=None is dateless. Both return 14, but the base removes the tag and the branch preserves season 2026, explicitly required by rows B/D. No row-L implementation can remove this difference while preserving those rows.', '',
          '## Every differing pair', '', '| Value | Path | Class | Differing recorded fields |', '|---|---|---|---|'] + changes
(D / 'date-matrix-summary.md').write_text('\n'.join(lines) + '\n')
print(counts)
