import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from review_support import DAY, HERE, ROOT, ReadTimedOut, World, assert_source, health


def health_contents(root, version):
    result = {}
    for label, data in {
        'missing': None, 'directory': None, 'fifo': None, 'json-list': b'[]', 'json-null': b'null',
        'json-scalar': b'5', 'bad-encoding': b'\xff', 'malformed': b'{', 'partial': b'{}',
        'confirmed': {'date': DAY, 'status': 'confirmed'}, 'unknown': {'date': DAY, 'status': ['?']},
        'unverified': {'date': DAY, 'status': 'present_unverified'},
        'alerted': {'date': DAY, 'status': 'alerted'}, 'dm-failed': {'date': DAY, 'status': 'dm_failed'},
        'stale': {'date': '2027-03-23', 'status': 'dm_failed'},
        'future': {'date': '2027-03-25', 'status': 'dm_failed'},
    }.items():
        w = World(root / label, version)
        w.commit_at('06:10', at='04:55')
        w.marker.parent.mkdir()
        if label == 'fifo':
            os.mkfifo(w.marker)
        elif label == 'directory':
            w.marker.mkdir()
        elif data is not None:
            w.marker.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
        values = []
        for at in ('06:04', '06:05', '23:59'):
            try:
                values.append(health(w, at))
            except (AttributeError, UnicodeDecodeError, ReadTimedOut) as exc:
                values.append(type(exc).__name__)
        result[label] = values
        print(label, values)
    return result


mode, scratch = sys.argv[1:]
if mode == 'compare':
    before, after = (json.loads((HERE / f'live-health-{which}.json').read_text()) for which in ('base', 'f5'))
    assert before == after
    print('16 live marker contents across three times, exact same outcomes')
else:
    assert_source((Path(scratch) / 'base' if mode == 'base' else ROOT) / 'src')
    with TemporaryDirectory() as dirname:
        results = health_contents(Path(dirname), mode)
        (HERE / f'live-health-{mode}.json').write_text(json.dumps(results, indent=2) + '\n')
