import json
import os
import resource
import shutil
import subprocess
import sys
from contextlib import redirect_stderr
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
HEAD = ROOT.parent.parent / 'src'
BASE = ROOT / 'base/src'
SEED = dict(away=1, home=22, away_tid=51, home_tid=52)
ALT = dict(home_tid=52, away_tid=51, home=23, away=2)
BIG = 10**120
CACHE = Path('data/models/probable_pitcher_lookup.json')
RAW = Path('data/raw/2025')
MTIME = 1_650_000_000_123_456_789


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 12, 31, 23, 0, tzinfo=timezone.utc).astimezone(tz)


def pa(game, day, pitcher, hit):
    return dict(
        pa_id=game, date=day, game_pk=game, batter_id=11, pitcher_id=pitcher,
        pitch_hand='R', venue_id=31, is_home=True, is_hit=hit, pitch_calls=['X'],
        pitch_types=['FF'], pitch_speeds=[93.0], pitch_spin_rates=[2300.0],
        pitch_extensions=[6.0], pitch_break_vertical=[15.0], pitch_break_horizontal=[5.0],
        pitch_px=[0.0], pitch_pz=[2.5], sz_top=3.5, sz_bottom=1.5,
        final_count_balls=0, final_count_strikes=0, launch_speed=91.0, launch_angle=10.0,
    )


def seed(entries):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(entries, indent=2))
    os.utime(CACHE, ns=(MTIME, MTIME))


def feed(name, entry=SEED):
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / name).write_text(json.dumps({'gameData': {
        'probablePitchers': {side: {'id': entry[side]} for side in ('away', 'home')},
        'teams': {side: {'id': entry[f'{side}_tid']} for side in ('away', 'home')},
    }}))


def build_snapshot(compute):
    before = CACHE.read_bytes()
    mtime = CACHE.stat().st_mtime_ns
    err = StringIO()
    with redirect_stderr(err):
        lookup = compute._build_probable_pitcher_lookup()
    signature = [
        [game, type(game).__name__, [[key, type(value).__name__, value] for key, value in entry.items()]]
        for game, entry in lookup.items()
    ]
    after = CACHE.read_bytes()
    result = dict(count=len(lookup), stderr=err.getvalue(), unchanged=after == before,
                  mtime_unchanged=CACHE.stat().st_mtime_ns == mtime,
                  evidence=list(CACHE.parent.glob('rejected/*')) != [],
                  records=(CACHE.parent.parent / 'health_state').exists())
    Path('signature.json').write_text(json.dumps(signature))
    return result


def feature(compute, park_drag, pandas, rows):
    err = StringIO()
    with redirect_stderr(err), patch.object(park_drag, 'get_table', return_value=None):
        try:
            frame = compute.compute_all_features(pandas.DataFrame(rows))
        except AttributeError as error:
            return f'{type(error).__name__}: {error}', err.getvalue()
    frame.to_pickle('features.pkl')
    result = frame['opp_bullpen_hr_30g'].iloc[-1]
    return None if pandas.isna(result) else float(result), err.getvalue()


def child(case, source):
    import bts
    import pandas
    from bts.features import compute, park_drag

    assert Path(bts.__file__).resolve() == source / 'bts/__init__.py'
    if source == HEAD:
        compute.probable_pitcher_cache.datetime = Clock
    compute._probable_pitcher_cache_path = CACHE
    dates = pandas.date_range('2025-04-01', periods=11)
    rows = [pa(100 + i, day, 3 if i < 10 else 1, i % 2) for i, day in enumerate(dates)]
    lookup = {str(game): SEED for game in range(100, 111)}
    if case.startswith(('live', 'frozen', 'generated')):
        if case.startswith('generated'):
            seed({'00100': SEED, ' 300 ': dict(ALT, away=None, home_tid=None),
                  '+100': ALT, str(BIG): dict(SEED, away=BIG), **{k: v for k, v in lookup.items() if k != '100'}})
        else:
            CACHE.parent.mkdir(parents=True)
            name = case.split('-')[0]
            shutil.copy2(ROOT / 'inputs' / f'{name}.json', CACHE)
        if case.endswith('-raw'):
            feed('99999999999999999999.json')
        elif case.endswith('-cached'):
            game = next(iter(json.loads(CACHE.read_bytes())))
            feed(f'{int(game)}.json', ALT)
        if case == 'generated-two':
            feed('9.json')
            feed('800.json')
        if case == 'generated-alias-raw':
            feed('9.json', ALT)
            feed('09.json')
        report = build_snapshot(compute)
        report['feature'] = feature(compute, park_drag, pandas, rows)
    elif case == 'restart':
        report = dict(feature=feature(compute, park_drag, pandas, rows))
    else:
        if case in ('defect-raw', 'defect-missing', 'extra', 'float', 'missing-probable', 'missing-team'):
            if case in ('defect-raw', 'defect-missing'):
                lookup['100'] = None
            elif case == 'extra':
                lookup['100'] = dict(SEED, ignored=True)
            elif case == 'float':
                lookup['100'] = dict(SEED, away_tid=51.0)
            elif case == 'missing-probable':
                lookup['100'] = {k: v for k, v in SEED.items() if k != 'away'}
            else:
                lookup['100'] = {k: v for k, v in SEED.items() if k != 'home_tid'}
            if case == 'defect-raw':
                feed('100.json')
        if case in ('scanned', 'fd-exhaustion'):
            del lookup['109']
            if case == 'scanned':
                lookup['999'] = None
            feed('109.json')
            if case == 'fd-exhaustion':
                for game in range(1, 65):
                    (RAW / f'{game:04}.json').mkdir()
                _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                resource.setrlimit(resource.RLIMIT_NOFILE, (48, hard))
        seed(lookup)
        report = dict(feature=feature(compute, park_drag, pandas, rows))
        report['committed'] = '109' in json.loads(CACHE.read_bytes()) if case != 'fd-exhaustion' else None
    print(json.dumps(report), flush=True)


def run(work, case, source):
    work.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1', TZ='America/New_York')
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), case, str(source)], cwd=work, env=env,
        capture_output=True, text=True, timeout=8, check=False,
    )
    if result.returncode:
        print(case, source, result.returncode, result.stdout, result.stderr)
        return None
    return json.loads(result.stdout)


def compare(work, case, exact=True):
    import pandas

    base, head = (work / f'{case}-{side}' for side in ('base', 'head'))
    old, new = run(base, case, BASE), run(head, case, HEAD)
    if old is None or new is None:
        raise RuntimeError(case)
    ordered = (base / 'signature.json').read_bytes() == (head / 'signature.json').read_bytes()
    same_bytes = (base / CACHE).read_bytes() == (head / CACHE).read_bytes()
    old_lookup, new_lookup = [
        {game: {k: v for k, _, v in entry} for game, _, entry in json.loads((side / 'signature.json').read_text())}
        for side in (base, head)
    ]
    pandas.testing.assert_frame_equal(pandas.read_pickle(base / 'features.pkl'), pandas.read_pickle(head / 'features.pkl'),
                                      check_exact=True)
    print(case, 'lookup equal', old_lookup == new_lookup, 'orders/types equal', ordered,
          'bytes equal', same_bytes, 'all feature values equal', old, new)
    if exact:
        assert ordered and old == new and same_bytes
        assert not new['stderr'] and not new['evidence'] and not new['records']


def main():
    with TemporaryDirectory(dir=ROOT, prefix='feature-run-') as temp:
        work = Path(temp)
        print('verified package imports:', BASE / 'bts/__init__.py', HEAD / 'bts/__init__.py', flush=True)
        for name in ('live', 'frozen', 'generated'):
            for suffix in ('empty', 'cached', 'raw'):
                compare(work, f'{name}-{suffix}')
        compare(work, 'generated-two', exact=False)
        compare(work, 'generated-alias-raw', exact=False)
        for case in ('defect-raw', 'defect-missing', 'extra', 'float', 'missing-probable', 'missing-team', 'fd-exhaustion'):
            old, new = [run(work / f'{case}-{side}', case, source) for side, source in (('base', BASE), ('head', HEAD))]
            print(case, old, new, flush=True)
        restarted = work / 'scanned'
        initial = run(restarted, 'scanned', HEAD)
        assert initial['feature'][0] == 0.5 and initial['committed']
        (restarted / RAW / '109.json').unlink()
        later = run(restarted, 'restart', HEAD)
        assert later['feature'] == [0.5, '']
        assert len(list((restarted / CACHE.parent / 'rejected').iterdir())) == 1
        assert len(list((restarted / CACHE.parent.parent / 'health_state/probable_pitcher_cache').iterdir())) == 1
        print('unrelated bad entry, real feature run and fresh process after feed removed:', initial, later)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        child(sys.argv[1], Path(sys.argv[2]))
    else:
        main()
