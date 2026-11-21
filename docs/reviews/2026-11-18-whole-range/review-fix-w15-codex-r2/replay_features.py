import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from review_features import BASE, CACHE, HEAD, RAW, Clock, SEED, feed, pa, seed


R1 = HERE / 'r1/src'
SCRIPT = str(Path(__file__).resolve())


def child(mode, source):
    import bts
    import pandas as pd
    from bts.features import compute, park_drag

    assert Path(bts.__file__).resolve() == source / 'bts/__init__.py'
    compute._probable_pitcher_cache_path = CACHE
    if source != BASE:
        compute.probable_pitcher_cache.datetime = Clock
    dates = pd.date_range('2025-04-01', periods=11)
    rows = [pa(100 + i, day, 3 if i < 10 else 1, i % 2) for i, day in enumerate(dates)]
    frame = pd.DataFrame(rows)
    base_cache = {str(i): dict(SEED) for i in range(100, 111)}
    if mode == 'fd':
        del base_cache['109']
        seed(base_cache)
        feed('109.json')
        for game in range(1, 65):
            (RAW / f'{game:04}.json').mkdir()
        _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (48, hard))
    elif mode == 'pipe':
        base_cache['100'] = None
        seed(base_cache)
        feed('100.json')
    elif mode == 'missing-key':
        del base_cache['100']['home_tid']
        seed(base_cache)
    elif mode == 'encoding':
        seed(base_cache)
        path = CACHE.with_name('café.json')
        CACHE.rename(path)
        path.write_bytes(b'{')
        compute._probable_pitcher_cache_path = path
    else:
        raise AssertionError(mode)

    before = len(os.listdir('/dev/fd'))
    with patch.object(park_drag, 'get_table', return_value=None):
        frame = compute.compute_all_features(frame)
    value = frame['opp_bullpen_hr_30g'].iloc[-1]
    try:
        after = len(os.listdir('/dev/fd'))
    except OSError as exc:
        after = exc.strerror
    print(json.dumps(dict(value=None if pd.isna(value) else float(value), before=before, after=after)), flush=True)


def run(work, mode, source, closed=False, encoding=None):
    folder = work / f'{mode}-{source.parent.name}'
    folder.mkdir()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(HERE.parent / 'offline'), str(source)]),
               PYTHONDONTWRITEBYTECODE='1', TZ='America/New_York')
    env.pop('PYTHONUNBUFFERED', None)
    if encoding:
        env['PYTHONIOENCODING'] = encoding
    options = dict(cwd=folder, env=env, timeout=9, stdout=subprocess.PIPE, text=True)
    command = [sys.executable, SCRIPT, mode, str(source)]
    if closed:
        reader, writer = os.pipe()
        os.close(reader)
        try:
            result = subprocess.run(command, stderr=writer, **options)
        finally:
            os.close(writer)
    else:
        result = subprocess.run(command, stderr=subprocess.PIPE, **options)
    values = json.loads(result.stdout)
    print(mode, source, 'rc', result.returncode, 'result', values, 'diagnostic', repr(result.stderr), flush=True)
    return result, values, folder


def main():
    print('Composer imports checked in every child.', flush=True)
    with TemporaryDirectory(dir=HERE, prefix='features-') as name:
        work = Path(name)
        for source in (BASE, R1, HEAD):
            result, values, folder = run(work, 'fd', source)
            assert result.returncode == 0 and result.stderr == ''
            assert ('109' in json.loads((folder / CACHE).read_bytes())) == (source != R1)
            if source == R1:
                assert values['value'] is None and 'Too many open files' in values['after']
            else:
                assert values['value'] == 0.5 and values['before'] == values['after']
        for source in (R1, HEAD):
            result, values, _ = run(work, 'pipe', source, closed=True)
            assert values['value'] == 0.5 and result.returncode == (0 if source == HEAD else 120)
        for source in (BASE, HEAD):
            result, values, _ = run(work, 'missing-key', source)
            assert result.returncode == 0 and values['value'] == (None if source == HEAD else 0.5)
        reports = [run(work, 'encoding', source, encoding='ascii')[0] for source in (R1, HEAD)]
        assert all(result.returncode == 0 for result in reports)
        assert r'caf\xe9.json' in reports[0].stderr and 'caf?.json' in reports[1].stderr


if __name__ == '__main__':
    if len(sys.argv) > 1:
        child(sys.argv[1], Path(sys.argv[2]))
    else:
        main()
