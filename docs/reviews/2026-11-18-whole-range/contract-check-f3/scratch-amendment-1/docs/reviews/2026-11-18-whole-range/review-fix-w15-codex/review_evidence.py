import hashlib
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts.features import probable_pitcher_cache as producer
from bts.probable_pitcher_status import read_record

from review_features import CACHE as cache, HEAD, RAW as raw, ROOT, SEED, feed


folder = cache.parent / 'rejected'
status = cache.parent.parent / 'health_state/probable_pitcher_cache'
original = b'{"00100":null,"100":{"away":1,"home":22,"away_tid":51,"home_tid":52},"101":null,"102":false,"103":[]}'


class Clock(datetime):
    at = datetime(2026, 12, 31, 23, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.at.astimezone(tz)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def wire(data):
    return json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode()


def evidence_name(data=original):
    return f'probable_pitcher_lookup.{Clock.at:%Y%m%dT%H%M%SZ}.{sha(data)[:12]}.json'


def records():
    return {path: json.loads(path.read_bytes()) for path in sorted(status.glob('*.json'))}


def seed(data=original, feeds=True):
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    if feeds:
        feed('101.json')
        feed('102.json', dict(SEED, away=False))


def build():
    stderr = StringIO()
    with redirect_stderr(stderr):
        result = producer.build(cache, raw.parent)
    return result, stderr.getvalue().splitlines()


def observation(data=original, preserved=None, reason='invalid_entries', scanned=True):
    if reason == 'invalid_entries':
        counts = (4, 1, int(scanned), int(scanned), 3 - 2 * int(scanned))
        outcomes = [[100, 1, 0, 0, 0, SEED],
                    [101, 0, int(scanned), 0, int(not scanned), SEED if scanned else None],
                    [102, 0, 0, int(scanned), int(not scanned), None],
                    [103, 0, 0, 0, 1, None]]
        ids = [100, 101, 102, 103]
    else:
        counts = (None, None, int(scanned), None, None)
        outcomes = [[101, SEED]] if scanned else []
        ids = []
    public = dict(reason=reason, cache_path=str(cache), original_sha256=sha(data), preserved=preserved,
                  **dict(zip(('dropped', 'recovered_from_cache', 'rescanned', 'unusable_feed', 'without_feed'), counts)))
    identity = sha(wire(dict(public, outcomes=outcomes)))
    day = Clock.at.astimezone(producer.ET).date().isoformat()
    filename = status / f'{day}.{reason}.{identity[:12]}.json'
    return filename, dict(public, identity=identity, game_ids=ids, recorded_date_et=day,
                          recorded_at=Clock.at.isoformat(), schema_version='bts_probable_pitcher_cache_status_v1')


def verify_observation(data=original, preserved=None, reason='invalid_entries', scanned=True):
    path, expected = observation(data, preserved, reason, scanned)
    row = records()[path]
    assert row == expected and read_record(path) == row
    return path, row


def identity_and_dates():
    seed(feeds=False)
    _, err1 = build()
    ref, first = verify_observation(scanned=False)
    Clock.at = Clock.at.replace(minute=5)
    _, err2 = build()
    assert len(records()) == 1 and records()[ref] == first and err1 == err2
    Clock.at = Clock.at.replace(minute=0, hour=5, day=1, month=1, year=2027)
    build()
    nxt, second = verify_observation(scanned=False)
    assert nxt != ref and first['identity'] == second['identity']
    Clock.at = datetime.fromisoformat(first['recorded_at'])
    seed()
    lookup, lines = build()
    copied, row = verify_observation(preserved=evidence_name())
    assert len(records()) == 3 and row['identity'] != first['identity'] and copied != ref
    assert (folder / row['preserved']).read_bytes() == original and lookup == {100: SEED, 101: SEED}
    assert lines == ['WARNING: probable-pitcher cache data/models/probable_pitcher_lookup.json: dropped 4 invalid entries '
                     '(1 recovered from valid cache entries, 1 rescanned from raw feeds, 1 with unusable feeds, '
                     '1 without feeds); first game ids: 100, 101, 102, 103']
    print('independent A4 hash, exact repeat, other date, no-copy then copy:', sorted(str(k.name) for k in records()))


def full_outcomes():
    data = wire({str(game): None for game in range(100, 112)})
    cache.parent.mkdir(parents=True)
    feed('111.json', dict(SEED, home=28))
    snapshots = []
    for rescanned in (111, 110):
        cache.write_bytes(data)
        build()
        outcomes = [[game, 0, int(game == rescanned), 0, int(game != rescanned),
                     dict(SEED, home=28) if game == rescanned else None] for game in range(100, 112)]
        all_rows = records()
        actual = next(row for row in all_rows.values() if row not in snapshots)
        public = {k: v for k, v in actual.items() if k not in ('identity', 'recorded_at', 'recorded_date_et',
                                                             'game_ids', 'schema_version')}
        assert actual['identity'] == sha(wire(dict(public, outcomes=outcomes)))
        assert actual['game_ids'] == list(range(100, 110))
        snapshots.append(actual)
        (raw / f'{rescanned}.json').rename(raw / '110.json')
    assert len(records()) == 2
    print('identical ten-id sample and totals, different unsampled outcomes:', [row['identity'] for row in snapshots])


def occupy(path, kind, payload=b'other'):
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == 'directory':
        path.mkdir()
    elif kind == 'fifo':
        os.mkfifo(path)
    elif kind == 'symlink':
        target = path.with_suffix('.target')
        target.write_bytes(payload)
        path.symlink_to(target.resolve())
    else:
        path.write_bytes(payload)


def failures(where, kind):
    seed()
    evidence = folder / evidence_name()
    status_with_copy, expected = observation(preserved=evidence.name)
    status_without_copy, _ = observation()
    if where in ('evidence', 'both'):
        payload = original if kind == 'symlink' else b'fake suffix match'
        occupy(evidence, kind, payload)
    if where in ('status', 'both'):
        target = status_with_copy if where == 'status' else status_without_copy
        if kind == 'conflict':
            expected['identity'] = expected['identity'][:12] + '0' * 52
        payload = wire(expected) if kind in ('symlink', 'conflict') else b'{'
        occupy(target, kind, payload)
    lookup, lines = build()
    assert cache.read_bytes() == original and lookup == {100: SEED, 101: SEED}
    if where in ('evidence', 'both'):
        assert lines[1] == f'WARNING: could not preserve probable-pitcher cache evidence at {evidence} (FileExistsError)'
    else:
        assert evidence.read_bytes() == original
    if where in ('status', 'both'):
        assert lines[-1].startswith(f'WARNING: could not write probable-pitcher cache status at {target} (')
    else:
        verify_observation()
    assert len(lines) == (3 if where == 'both' else 2)
    print(where, kind, 'active intact; committed evidence is verified only:', ' / '.join(lines[1:]))


def reuse():
    seed()
    folder.mkdir()
    names = [f'probable_pitcher_lookup.20261230T0{i}0000Z.{sha(original)[:12]}.json' for i in range(1, 5)]
    (folder / names[0]).write_bytes(b'wrong')
    (folder / names[1]).symlink_to((folder / names[2]).resolve())
    (folder / names[2]).write_bytes(original)
    (folder / names[3]).write_bytes(original)
    _, err = build()
    assert len(err) == 1 and len(list(folder.iterdir())) == 4
    verify_observation(preserved=names[2])
    print('earliest verified full digest selected:', names[2])


def undecodable():
    for data, error in [(b'\xff\x00\r\n', 'UnicodeDecodeError'), (b'[]', 'AttributeError'),
                        (b'{"100":null,"bad":{}}', 'ValueError'), ('{}'.encode('utf-16'), 'UnicodeDecodeError')]:
        seed(data)
        _, lines = build()
        assert lines == [f'WARNING: unreadable probable-pitcher cache {cache} ({error}); rebuilding it from raw feeds']
        preserved = evidence_name(data)
        assert (folder / preserved).read_bytes() == data
        verify_observation(data, preserved, reason='unreadable_cache')
        print('whole-cache unreadable, original bytes match:', repr(data), lines[0])


def size_limit():
    target, row = observation(scanned=False)
    row['cache_path'] = 'a'
    row['cache_path'] *= 1 + 65536 - len(wire(row))
    status.parent.parent.mkdir()
    producer.publish(target, row)
    assert len(target.read_bytes()) == 65536 and read_record(target) == row
    row['cache_path'] += 'a'
    other = target.with_name('oversized.json')
    try:
        producer.publish(other, row)
    except ValueError:
        assert not other.exists()
    else:
        raise AssertionError('65537 serialized bytes accepted')
    print('producer and reader accept 65536, producer rejects 65537')


def child(operation, side):
    import bts

    assert Path(bts.__file__).resolve() == HEAD / 'bts/__init__.py'
    producer.datetime = Clock
    replace = os.replace

    def interrupt(src, target):
        target = Path(target)
        kind = 'cache' if target == cache else 'evidence' if target.parent == folder else 'status'
        if kind == operation and side == 'before':
            os._exit(71)
        replace(src, target)
        if kind == operation and side == 'after':
            os._exit(71)

    with patch.object(os, 'replace', interrupt):
        result, lines = build()
    print(sorted(result), lines)


def renames():
    env = dict(os.environ, PYTHONPATH=str(HEAD), PYTHONDONTWRITEBYTECODE='1')
    args = [sys.executable, str(Path(__file__).resolve())]
    root = Path.cwd()
    for index, operation in enumerate(('evidence', 'status', 'cache')):
        for after in (False, True):
            work = root / f'{operation}-{after}'
            work.mkdir()
            os.chdir(work)
            seed()
            result = subprocess.run(args + [operation, 'after' if after else 'before'], env=env, timeout=5,
                                    capture_output=True, text=True)
            assert result.returncode == 71, result.stderr
            landed = index + int(after)
            evidence = folder / evidence_name()
            assert evidence.exists() == (landed >= 1)
            assert len(records()) == int(landed >= 2)
            assert (cache.read_bytes() == original) == (landed != 3)
            result = subprocess.run(args + ['restart', 'none'], env=env, timeout=5, capture_output=True, text=True)
            assert result.returncode == 0 and '100, 101' in result.stdout, result.stderr
            assert json.loads(cache.read_bytes()) == {'100': SEED, '101': SEED}
            assert evidence.read_bytes() == original and len(records()) == 1
            verify_observation(preserved=evidence.name)
            print('hard exit and fresh process:', operation, 'after' if after else 'before', 'commits at exit', landed)
    os.chdir(root)


def main():
    producer.datetime = Clock
    with TemporaryDirectory(dir=ROOT, prefix='evidence-run-') as temp:
        work = Path(temp)
        cases = [(case.__name__, case, ()) for case in (identity_and_dates, full_outcomes, reuse, undecodable, size_limit, renames)]
        cases += [(f'{where}-{kind}', failures, (where, kind)) for where, kinds in (
            ('evidence', ('directory', 'fifo', 'symlink', 'fake')),
            ('status', ('directory', 'fifo', 'symlink', 'malformed', 'conflict')),
            ('both', ('directory',)),
        ) for kind in kinds]
        try:
            for label, case, args in cases:
                directory = work / label
                directory.mkdir()
                os.chdir(directory)
                case(*args)
        finally:
            os.chdir(ROOT)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        child(*sys.argv[1:])
    else:
        main()
