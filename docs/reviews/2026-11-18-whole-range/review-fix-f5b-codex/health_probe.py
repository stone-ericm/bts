from datetime import timedelta
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from bts import entry_receipt
from review_support import DAY, ROOT, RIGHT, World, assert_source, health, ReadTimedOut, row


def input_matrix(root):
    for field, contents in {
        'missing': None, 'directory': None, 'fifo': None, 'invalid-utf8': b'\xff',
        'invalid-json': b'{', 'list': b'[]', 'null': b'null', 'scalar': b'1',
        'older': b'{"date":"2027-03-23","status":"dm_failed"}',
        'newer': b'{"date":"2027-03-25","status":"confirmed"}',
        'same-quiet': b'{"date":"2027-03-24","status":"present_unverified"}',
        'same-unrecognized': b'{"date":"2027-03-24","status":[]}',
        'same-failed': b'{"date":"2027-03-24","status":"dm_failed","reason":"no_pick"}',
    }.items():
        w = World(root / field)
        w.commit_at('06:10', at='04:55')
        w.cron('05:00', dm='failed')
        if field == 'directory':
            w.marker.mkdir()
        elif field == 'fifo':
            os.mkfifo(w.marker)
        elif contents is not None:
            w.marker.write_bytes(contents)
        try:
            outcome = [level for level, _ in health(w)]
        except (AttributeError, UnicodeDecodeError, ReadTimedOut) as exc:
            outcome = type(exc).__name__
        expected = {'fifo': 'ReadTimedOut', 'invalid-utf8': 'UnicodeDecodeError',
                    'list': 'AttributeError', 'null': 'AttributeError', 'scalar': 'AttributeError',
                    'newer': [], 'same-quiet': [], 'same-unrecognized': []}.get(field, ['CRITICAL'])
        assert outcome == expected, (field, outcome)
        print('live', field, outcome)

    for field in ('missing', 'directory', 'fifo', 'list', 'utf8', 'oversize', 'date', 'tier', 'pick-fifo', 'pick-list', 'pick-oversize'):
        w = World(root / f'early-{field}')
        w.commit_at('06:10', at='04:55')
        w.cron('05:00', dm='failed')
        assert not health(w, at='06:04') and health(w, at='06:05')[0][0] == 'CRITICAL'
        raw = w.early.read_bytes()
        early = not field.startswith('pick-')
        path = w.early if early else w.picks / f'{DAY}.json'
        path.unlink()
        kind = field.removeprefix('pick-')
        if kind == 'fifo':
            os.mkfifo(path)
        elif kind == 'directory':
            path.mkdir()
        elif kind != 'missing':
            path.write_bytes({'list': b'[]', 'utf8': b'\xff', 'oversize': raw + b' ' * (1024 * 1024),
                              'date': raw.replace(DAY.encode(), b'2027-03-23'),
                              'tier': json.dumps({**json.loads(raw), 'escalations': [False]}).encode()}[kind])
        alerts = health(w)
        if early:
            assert alerts == [], (field, alerts)
        else:
            assert alerts[0][0] == 'CRITICAL' and 'unknown' in alerts[0][1]
            with __import__('review_support').deadline():
                assert w.cron('05:15')['receipt']['outcome'] == 'no_pick_file'
        print('early', field, alerts)


def seal_order(root):
    for overlap in (False, True):
        w = World(root / f'overlap-{overlap}')
        w.commit_at('06:10', at='04:40')
        before = World(w.root, day='2027-03-23')
        before.commit_at('19:10', at='17:50')
        before.invoke('18:00', manual=True, pending=[row()])
        old_live = w.marker.read_bytes()
        w.capture([RIGHT], f'{DAY}T084000Z')

        def early_failure(*_):
            failed = w.cron('05:00', dm='failed')
            assert failed['receipt']['marker_status'] == 'dm_failed'

        if overlap:
            from bts.cli import _atomic_write_json

            def interleaved_write(path, marker):
                if path == w.marker:
                    early_failure()
                    assert w.early.exists() and path.read_bytes() == old_live
                _atomic_write_json(path, marker)

            live = w.invoke('04:55', manual=True, pending=[row()],
                            patches={'bts.cli._atomic_write_json': interleaved_write})
        else:
            early_failure()
            live = w.invoke('05:15', manual=True, pending=[row()])
        assert live['receipt']['marker_status'] == 'confirmed' and live['exit'] == 0
        assert not health(w)
        # An operator performs a historical manual check that rolls the global live date back.
        before_marker = before.invoke('18:15', manual=True, pending=[row()])
        assert before_marker['marker']['date'] == before.day
        alerts = health(w)
        assert [level for level, _ in alerts] == (['CRITICAL'] if overlap else [])
        print('live write happened after early failure, overlap', overlap,
              'live response', live['receipt']['observation']['response_completed_at'],
              'early response', w.early_state()['observed_at'], 'after historical check', alerts)


if __name__ == '__main__':
    assert_source(ROOT / 'src')
    with TemporaryDirectory() as dirname:
        root = Path(dirname)
        input_matrix(root)
        seal_order(root)
