"""Exercise identical manual CLI sequences using the worktree or the archived base."""
from itertools import product
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import bts
from probe_support import Replay
from tests.test_wr_f1_wrong_game import RIGHT, WRONG, row


def normalize(value, root):
    uuids = {}
    for i, event in enumerate(value):
        receipt = event.get('receipt')
        if receipt is not None:
            uuids[receipt['attempt_id']] = f'attempt-{i}'
    data = json.dumps(value, sort_keys=True).replace(str(root), '<sandbox>')
    for identifier, replacement in uuids.items():
        data = data.replace(identifier, replacement)
    return json.loads(data)


def capture_matrix(scratch):
    lengths = {}
    traces = {}
    for sequence, (clock, data, state, account) in enumerate(product(
            ('05:15', '09:45', '10:00', '18:15'),
            ('absent', 'preview', 'committed', 'double'),
            ('empty', 'confirmed', 'alerted'),
            ('match', 'wrong_game', 'missing', 'fetch_error'))):
        root = scratch / str(sequence)
        world = Replay(root)
        pitch = {'05:15': '06:10', '09:45': '10:50', '10:00': '10:50', '18:15': '19:10'}[clock]
        if data != 'absent':
            world.commit_at(pitch, dd=(1020, 202, '19:30') if data == 'double' else None,
                            delivery=None if data == 'preview' else 'delivered')
        world.capture([WRONG if account == 'wrong_game' else RIGHT, {'id': 78, 'roundId': 1, 'feedId': 202}])
        if state != 'empty':
            world.set_marker({'date': world.day, 'status': state})
        pending = ([row(), row(78, 2, 2)] if data == 'double' else [row()]) if account in ('match', 'wrong_game') else []
        event = world.invoke(clock, manual=True, pending=pending, auth='profile' if account == 'fetch_error' else 'ok')
        assert event['receipt'] is not None, event
        label = '/'.join((clock, data, state, account))
        traces[label] = normalize([event], root)
        outcome = event['receipt']['outcome']
        lengths[outcome] = lengths.get(outcome, 0) + 1
    for label, setup in {
        'invalid-pick': lambda w: (w.picks / f'{w.day}.json').write_text('{broken'),
        'invalid-decision': lambda w: (w.picks / w.day / 'decision.json').write_text('{broken'),
        'marker-null': lambda w: w.set_marker(None),
        'invalid-pitch': lambda w: (w.picks / f'{w.day}.json').write_text(re.sub(
            r'"game_time": "[^"]+"', '"game_time": "bad-date"', (w.picks / f'{w.day}.json').read_text())),
        'marker-directory': lambda w: w.marker.mkdir(parents=True),
        'dm-retry': lambda w: None,
        'elapsed-cutoff': lambda w: None,
    }.items():
        root = scratch / label
        world = Replay(root)
        world.commit_at('10:50')
        setup(world)
        argv = {'manual': True, 'dm': 'failed' if label == 'dm-retry' else 'sent',
                'elapsed': 1200 if label == 'elapsed-cutoff' else 0}
        events = [world.invoke('10:30', **argv)]
        if label == 'dm-retry':
            events.extend([world.invoke('10:31', manual=True), world.invoke('10:32', manual=True, pending=[row()])])
        traces[label] = normalize(events, root)
    print('manual outcomes:', lengths)
    return traces


if __name__ == '__main__':
    module = Path(bts.__file__).resolve()
    print(f'bts.__file__ = {module}')
    assert ('/base/src/' in str(module)) == (sys.argv[1] == 'base')
    out = Path(__file__).parent / f'manual-{sys.argv[1]}.json'
    with TemporaryDirectory() as temp:
        traces = capture_matrix(Path(temp))
    out.write_text(json.dumps(traces, indent=2, sort_keys=True) + '\n')
    if sys.argv[1] == 'f5':
        base = json.loads(out.with_name('manual-base.json').read_text())
        assert base.keys() == traces.keys()
        failures = [(name, base[name], traces[name]) for name in traces if base[name] != traces[name]]
        assert not failures, failures[:5]
        print(f'{len(traces)} matched sequences, {sum(map(len, traces.values()))} invocations per tree')
