import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

from review_support import DAY, HERE, ROOT, RIGHT, WRONG, World, assert_source, health, installed, load_line, row
from probe_support import clock_slots


def schedules(root, base):
    seed = ('0 9 * * * echo foreign\n0 9 * * * echo replace # BTS-HETZNER\n'
            '# */15 10-23 * * * bts check-pick-entered\n')
    result = {}
    for intent in ('enter', 'research'):
        paths = {v: tree / 'scripts/cron-setup-hetzner.sh' for v, tree in (('base', base), ('f5', ROOT))}
        lines = {v: installed(root, path, intent, seed) for v, path in paths.items()}
        assert lines['base'][0] == lines['f5'][0] == seed.splitlines()[0]
        watchdogs = {v: [line for line in variants if ' bts check-pick-entered ' in line]
                     for v, variants in lines.items()}
        for v, text in watchdogs.items():
            assert len(text) == (intent == 'enter'), (v, text)
        assert [ln for ln in lines['base'] if ln not in watchdogs['base']] == [
            ln for ln in lines['f5'] if ln not in watchdogs['f5']]
        if intent == 'enter':
            old, = watchdogs['base']
            new, = watchdogs['f5']
            assert new.replace('*/15 * * * * ', '*/15 10-23 * * * ', 1).replace(
                'check-pick-entered --cron ', 'check-pick-entered ', 1) == old
            result = lines
            print(old, new, sep='\n')
    (HERE / 'generated.json').write_text(json.dumps(result, indent=2) + '\n')
    for day, count in (('2027-03-14', 92), (DAY, 96), ('2027-11-07', 100)):
        labels = [slot.strftime('%H:%M') for slot in clock_slots(load_line('f5'), day)]
        assert len(labels) == count and (labels[0], labels[-1]) == ('00:00', '23:45')
        assert labels.count('01:45') == (2 if count == 100 else 1)
        print(day, count)


def collect(root, version):
    records = {}
    for manual in (True, False):
        early = '09:45' if manual else '10:00'
        for case in ('no-pick', 'preview', 'too-early', 'strict-cutoff', 'confirmed-marker',
                     'missing', 'wrong-game', 'correct-game', 'unverified', 'wrong-account',
                     'fetch-failed', 'dm-failed-retry-confirm', 'list-marker', 'bad-encoding'):
            name = f'{manual}-{case}'
            w = World(root / name, version)
            pitch = ('10:05' if manual else '10:20')
            if case != 'no-pick':
                w.commit_at({'too-early': '13:00', 'strict-cutoff': '09:50' if manual else '10:05'}.get(case, pitch),
                            at='09:30', delivery=None if case == 'preview' else 'delivered')
            if case in ('confirmed-marker', 'list-marker', 'bad-encoding'):
                w.set_marker({'date': DAY, 'status': 'confirmed'})
                if case == 'list-marker':
                    w.marker.write_bytes(b'[]')
                if case == 'bad-encoding':
                    w.marker.write_bytes(b'\xff')
            if case in ('wrong-game', 'correct-game', 'dm-failed-retry-confirm'):
                w.capture([WRONG if case == 'wrong-game' else RIGHT], f'{DAY}T030000Z')
            kwargs = {
                'pending': [row()] if case in ('wrong-game', 'correct-game', 'unverified') else [],
                'auth': {'wrong-account': 'wrong-user', 'fetch-failed': 'pending'}.get(case, 'ok'),
                'crosswalk': {} if case == 'unverified' else None,
                'dm': 'failed' if case == 'dm-failed-retry-confirm' else 'sent',
                'manual': manual,
            }

            def snapshot():
                return {str(path.relative_to(w.root)): path.read_bytes()
                        for path in w.root.rglob('*') if path.is_file()}

            def state(at, **kw):
                before = snapshot()
                try:
                    trace = w.invoke(at, **kw)
                except (ValueError, AttributeError):
                    raise AssertionError('fixture could not display a malformed live marker')
                after = snapshot()
                return trace, {name: data for name, data in after.items() if before.get(name) != data}, sorted(before.keys() - after.keys())

            if case in ('list-marker', 'bad-encoding'):
                from unittest.mock import patch
                with patch.object(w, 'read_marker', lambda: None):
                    first = state(early, **kwargs)
            else:
                first = state(early, **kwargs)
            trace = [first]
            if case == 'dm-failed-retry-confirm':
                trace.append(state('09:46' if manual else '10:01', manual=manual))
                trace.append(state('09:47' if manual else '10:02', manual=manual, pending=[row()]))

            def health_outcome():
                try:
                    return health(w)
                except (AttributeError, UnicodeDecodeError) as exc:
                    return type(exc).__name__

            records[name] = (trace, health_outcome(), w.morning())
    return records


def normalized(value, root):
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, sort_keys=True, default=lambda data: data.hex() if b'\xff' in data else data.decode())
    value = value.replace(str(root), '<fixture>')
    ids = dict.fromkeys(re.findall(r'(?<![a-f0-9])[a-f0-9]{32}(?![a-f0-9])', value))
    for index, ident in enumerate(ids):
        value = value.replace(ident, f'<id-{index}>')
    return value


if __name__ == '__main__':
    mode, scratch = sys.argv[1:]
    scratch = Path(scratch)
    if mode == 'compare':
        before, after = (json.loads((HERE / f'parity-{v}.json').read_text()) for v in ('base', 'f5'))
        diffs = [name for name in before if before[name] != after[name]]
        print(f'{len(before)} paired cases, differences: {diffs}')
        assert not diffs
    else:
        assert_source((scratch / 'base' if mode == 'base' else ROOT) / 'src')
        with TemporaryDirectory() as temp:
            root = Path(temp)
            if mode == 'schedule':
                schedules(root, scratch / 'base')
            else:
                data = {k: normalized(v, root) for k, v in collect(root, mode).items()}
                (HERE / f'parity-{mode}.json').write_text(json.dumps(data, indent=2) + '\n')
                print(len(data), 'cases')
