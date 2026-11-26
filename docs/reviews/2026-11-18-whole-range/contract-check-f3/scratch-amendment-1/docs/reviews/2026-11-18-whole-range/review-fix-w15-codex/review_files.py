import json
import os
import resource
import subprocess
import sys
from contextlib import redirect_stderr
from datetime import date
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from review_features import BASE, CACHE, Clock, HEAD, RAW, ROOT, SEED, feed, pa, seed


def kind(path, form, payload, target=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if form == 'fifo':
        os.mkfifo(path)
    elif form == 'directory':
        path.mkdir()
    elif form in ('symlink', 'symlink-fifo', 'dangling'):
        if form == 'symlink':
            target.write_bytes(payload)
        elif form == 'symlink-fifo':
            os.mkfifo(target)
        path.symlink_to(target.resolve())
    elif form == 'binary':
        path.write_bytes(b'\xff')
    elif form == 'oversized':
        path.write_bytes(payload + b' ' * 65537)
    else:
        raise ValueError(form)


def unusual(area, form):
    from bts.features import probable_pitcher_cache as producer
    from bts.health import probable_pitcher_cache as health

    cache = CACHE
    status = cache.parent.parent / 'health_state/probable_pitcher_cache'
    rejected = cache.parent / 'rejected'
    other = Path('fixture-target').resolve()
    producer.datetime = Clock
    original = b'{"100":null}'
    new = 100
    permissions = None
    if area == 'cache':
        kind(cache, form, original, other)
        feed('100.json')
    else:
        cache.parent.mkdir(parents=True)
        cache.write_bytes(original)
        if area == 'feed':
            if form == 'unicode':
                feed('².json')
                feed('-100.json')
            elif form == 'unlistable':
                feed('100.json')
                permissions = RAW
            elif form == 'inaccessible':
                feed('100.json')
                permissions = RAW.parent
            elif form == 'file-root':
                RAW.parent.parent.mkdir(exist_ok=True)
                RAW.parent.write_text('{}')
            else:
                kind(RAW / '100.json', form, b'{}', other)
        else:
            if form == 'unwritable':
                artifact_dir = rejected if area == 'rejected' else status
                artifact_dir.mkdir(parents=True)
                permissions = artifact_dir
            elif area == 'rejected':
                kind(rejected, form, b'{}', other)
            else:
                kind(status, form, b'{}', other)
            feed('100.json')
    if permissions is not None:
        permissions.chmod(0)
    err = StringIO()
    try:
        with redirect_stderr(err):
            values = producer.build(cache, RAW.parent)
        health_alerts = health.check(status, date(2026, 12, 31))
    finally:
        if permissions is not None:
            permissions.chmod(0o700)
    if form == 'symlink' and area == 'cache':
        assert not cache.is_symlink() and other.read_bytes() == original
    stderr = err.getvalue().splitlines()
    records = []
    if status.is_dir() and not status.is_symlink():
        records = [json.loads(path.read_bytes()) for path in status.glob('*.json')]
    evidence = [] if not rejected.is_dir() or rejected.is_symlink() else sorted(path.name for path in rejected.iterdir())
    if area == 'cache':
        assert new in values and len(records) == 1
        if form in ('fifo', 'directory', 'symlink-fifo', 'dangling'):
            assert records[0]['reason'] == 'unreadable_cache' and records[0]['preserved'] is None and not evidence
        elif form == 'binary':
            assert records[0]['reason'] == 'unreadable_cache' and (rejected / evidence[0]).read_bytes() == b'\xff'
        else:
            assert records[0]['reason'] == 'invalid_entries' and records[0]['rescanned'] == 1 and evidence
            if form == 'oversized':
                assert len((rejected / evidence[0]).read_bytes()) == len(original) + 65537
    elif area == 'feed':
        usable = form in ('symlink', 'oversized')
        missing = form in ('unicode', 'file-root')
        assert (new in values) == usable and len(records) == 1
        assert records[0]['rescanned'] == int(usable) and records[0]['without_feed'] == int(missing)
        assert records[0]['unusable_feed'] == int(not usable and not missing)
    elif form != 'directory':
        assert new in values and cache.read_bytes() == original and len(stderr) == 2
    else:
        assert new in values and len(stderr) == 1 and evidence and len(records) == 1
    print(area, form, 'game', new in values, 'record',
          [(r['reason'], r['rescanned'], r['unusable_feed'], r['without_feed'], bool(r['preserved'])) for r in records],
          'stderr', ' / '.join(stderr), 'health', [a.incident_key for a in health_alerts])


def bad_stderr():
    from bts.features import probable_pitcher_cache as producer

    class Broken:
        def write(self, _):
            raise BrokenPipeError

    producer.datetime = Clock
    seed({'100': None})
    feed('100.json')
    (CACHE.parent / 'rejected').write_text('{}')
    (CACHE.parent.parent / 'health_state').mkdir()
    (CACHE.parent.parent / 'health_state/probable_pitcher_cache').write_text('{}')
    with patch.object(sys, 'stderr', Broken()):
        observed = producer.build(CACHE, RAW.parent)
    assert observed == {100: SEED} and json.loads(CACHE.read_bytes()) == {'100': None}
    print('closed stderr: incident and both artifact failures suppressed; usable lookup returned')


def descriptor_pressure(side):
    import bts
    import pandas
    from bts.features import compute, park_drag

    source = BASE if side == 'base' else HEAD
    assert Path(bts.__file__).resolve() == source / 'bts/__init__.py'
    compute._probable_pitcher_cache_path = CACHE
    dates = pandas.date_range('2025-04-01', periods=11)
    frame = pandas.DataFrame([pa(100 + i, day, 3 if i < 10 else 1, i % 2) for i, day in enumerate(dates)])
    seed({str(game): SEED for game in range(100, 111) if game != 109})
    feed('109.json')
    for game in range(1, 65):
        (RAW / f'{game:04}.json').mkdir()
    baseline = os.open('.', os.O_RDONLY)
    os.close(baseline)
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (48, hard))
    err = StringIO()
    with patch.object(park_drag, 'get_table', return_value=None), redirect_stderr(err):
        result = compute.compute_all_features(frame)
    value = result['opp_bullpen_hr_30g'].iloc[-1]
    try:
        after = os.open('.', os.O_RDONLY)
    except OSError as error:
        after = error.strerror
    else:
        os.close(after)
    print('directory feed descriptor pressure', side, 'bullpen', str(value), 'fd before', baseline,
          'open after', after, 'stderr', repr(err.getvalue()))
    if side == 'base':
        assert value == 0.5 and after == baseline
    else:
        assert pandas.isna(value) and isinstance(after, str) and not err.getvalue()


def run():
    cases = [(area, form) for area, forms in {
        'cache': ('fifo', 'directory', 'symlink', 'symlink-fifo', 'dangling', 'binary', 'oversized'),
        'feed': ('fifo', 'directory', 'symlink', 'symlink-fifo', 'dangling', 'binary', 'oversized',
                 'unicode', 'unlistable', 'inaccessible', 'file-root'),
        'rejected': ('fifo', 'directory', 'symlink', 'binary', 'oversized', 'unwritable'),
        'status': ('fifo', 'directory', 'symlink', 'binary', 'oversized', 'unwritable'),
    }.items() for form in forms]
    cases += [('stderr', 'broken'), ('descriptor', 'base'), ('descriptor', 'head')]
    with TemporaryDirectory(dir=ROOT, prefix='files-run-') as name:
        root = Path(name)
        for area, form in cases:
            directory = root / f'{area}-{form}'
            directory.mkdir()
            source = BASE if form == 'base' else HEAD
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), area, form], cwd=directory,
                                     env=dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1'),
                                     timeout=7, capture_output=True, text=True)
            assert process.returncode == 0, (area, form, process.stdout, process.stderr)
            print(process.stdout, end='', flush=True)
            if area == 'descriptor':
                on_disk = json.loads((directory / CACHE).read_bytes())
                assert ('109' in on_disk) == (source == BASE)
                print('game 109 committed', '109' in on_disk)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        run()
    elif sys.argv[1] == 'stderr':
        bad_stderr()
    elif sys.argv[1] == 'descriptor':
        descriptor_pressure(sys.argv[2])
    else:
        unusual(*sys.argv[1:])
