from contextlib import redirect_stderr
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import os
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent


def load_copy(name, filename):
    spec = spec_from_file_location(name, ROOT / 'reference' / filename)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compute = load_copy('w15_base_compute', 'compute.py')
from bts import features

park_drag = load_copy('bts.features.park_drag', 'park_drag.py')
features.park_drag = park_drag
capture_line = next(
    number for number, line in enumerate((ROOT / 'reference/compute.py').read_text().splitlines(), 1)
    if line.strip().startswith('reliever_pas = df[')
)
SIG = {'away': 1, 'home': 22, 'away_tid': 51, 'home_tid': 52}
GOOD_FEED = {'away': 2, 'home': 23, 'away_tid': 61, 'home_tid': 62}
COLS = ['_probable_pid', '_is_reliever_pa', '_pitcher_team_id']


def pa(game, day, pitcher, home, hit=0):
    return dict(pa_id=0, date=day, game_pk=game, batter_id=11, pitcher_id=pitcher,
                pitch_hand='R', venue_id=31, is_home=home, is_hit=hit, pitch_calls=['X'],
                pitch_types=['FF'], pitch_speeds=[93.0], pitch_spin_rates=[2300.0],
                pitch_extensions=[6.0], pitch_break_vertical=[15.0],
                pitch_break_horizontal=[5.0], pitch_px=[0.0], pitch_pz=[2.5],
                sz_top=3.5, sz_bottom=1.5, final_count_balls=0,
                final_count_strikes=0, launch_speed=91.0, launch_angle=10.0)


ROWS = [pa(100, '2025-04-01', pid, home) for pid, home in [(1, True), (22, False)]]


def feature(rows=ROWS):
    observed = []

    def trace(frame, event, arg):
        if frame.f_code is compute.compute_all_features.__code__:
            if event == 'line' and frame.f_lineno == capture_line:
                observed.extend(frame.f_locals['df'][COLS].itertuples(index=False, name=None))
            return trace
        return None

    with patch.object(park_drag, 'get_table', return_value=None):
        sys.settrace(trace)
        try:
            frame = pd.DataFrame(rows)
            frame['pa_id'] = range(len(frame))
            result = compute.compute_all_features(frame)
        except Exception as error:
            return observed, f'{type(error).__name__}: {error}', None
        finally:
            sys.settrace(None)
    return observed, 'ok', result


def feed(path, entry=GOOD_FEED):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'gameData': {
        'probablePitchers': {side: {'id': entry[side]} for side in ('away', 'home')},
        'teams': {side: {'id': entry[f'{side}_tid']} for side in ('away', 'home')},
    }}))


def put(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))


def exact_signature(lookup):
    return [(key, type(key).__name__, [(member, type(value).__name__, value)
                                     for member, value in entry.items()])
            for key, entry in lookup.items()]


def real_artifacts(work):
    raw = work / 'empty'
    raw.mkdir()
    for file in sorted((ROOT / 'inputs').glob('probable*.json')):
        before = file.read_bytes()
        stamp = file.stat().st_mtime_ns
        compute._probable_pitcher_cache_path = file
        stderr = StringIO()
        with redirect_stderr(stderr):
            lookup = compute._build_probable_pitcher_lookup(str(raw))
        original = json.loads(before)
        filtered = {int(k): v for k, v in original.items()
                    if isinstance(v, dict) and set(v) == set(SIG)
                    and all(item is None or type(item) is int for item in v.values())}
        nulls = {member: sum(v[member] is None for v in filtered.values()) for member in SIG}
        assert len(original) == len(lookup) == len(filtered)
        assert lookup == filtered and exact_signature(lookup) == exact_signature(filtered)
        assert file.read_bytes() == before and file.stat().st_mtime_ns == stamp and not stderr.getvalue()
        print('real cache', file.name, len(lookup), nulls, 'exact equality/order/types, unchanged bytes/mtime, stderr empty')
    compute._probable_pitcher_cache_path = work / 'feed-sample/models/cache.json'
    sample = compute._build_probable_pitcher_lookup(str(ROOT / 'inputs/real-feeds'))
    types = {name: sorted({type(entry[name]).__name__ for entry in sample.values()}) for name in SIG}
    print('copied official feeds scanned', len(sample), types)


def corpus(cache, raw):
    good = {'999': SIG}
    cases = {'valid': SIG, 'empty': {}, 'extra': dict(SIG, ballast={'unused': True}),
             'valid-null': dict(SIG, away=None, away_tid=None)}
    for field in SIG:
        cases[f'missing-{field}'] = {key: val for key, val in SIG.items() if key != field}
    for shape, value in [('string', '1'), ('integral-float', 1.0), ('fraction', 1.5),
                         ('true', True), ('false', False), ('list', [1]), ('object', {'id': 1}),
                         ('nan', float('nan')), ('infinity', float('inf'))]:
        cases[f'away-{shape}'] = dict(SIG, away=value)
        cases[f'away-tid-{shape}'] = dict(SIG, away_tid=value if shape != 'integral-float' else 51.0)
    for shape, value in [('null', None), ('number', 1), ('string', 'x'), ('list', [])]:
        cases[f'entry-{shape}'] = value
    good_path = raw / '2025/100.json'
    feed(good_path)
    for name, entry in cases.items():
        put(cache, {'100': entry, **good})
        stamp = cache.stat().st_mtime_ns
        assert compute._build_probable_pitcher_lookup()[100] == entry or name.endswith('nan')
        before, result, _ = feature()
        assert cache.stat().st_mtime_ns == stamp
        print('corpus', name, repr(before), result)
    put(cache, good)
    assert compute._build_probable_pitcher_lookup()[100] == GOOD_FEED
    print('after valid feed', *feature()[:2])
    good_path.unlink()
    put(cache, good)
    print('after absent feed, another lookup entry', *feature()[:2])
    put(cache, {})
    captured, outcome, frame = feature()
    assert not captured and 'opp_pitching_team_id' not in frame
    print('after absent feed, empty lookup', outcome, 'no intermediate or team column')


def edge_sequences(cache, raw):
    for payload in ('[1,2]', '{"bad":{}}', '{"100":', '{"100":null," 100 ":' + json.dumps(SIG) + '}',
                    '{"100":' + json.dumps(SIG) + ',"00100":null}',
                    '{"00100":' + json.dumps(SIG) + ',"99":' + json.dumps(SIG) + ',"+100":' + json.dumps(GOOD_FEED) + '}'):
        cache.write_text(payload)
        stderr = StringIO()
        stamp = cache.stat().st_mtime_ns
        with redirect_stderr(stderr):
            resolved = compute._build_probable_pitcher_lookup('missing-raw')
        print('keys/top-level; no raw', payload, resolved, 'mtime unchanged', stamp == cache.stat().st_mtime_ns,
              stderr.getvalue().strip())

    put(cache, {})
    path = raw / '2025/100.json'
    feed(path, dict(GOOD_FEED, away='2', home=True, away_tid=61.0, home_tid={'id': 62}))
    invalid_scan = compute._build_probable_pitcher_lookup()
    assert invalid_scan[100]['away'] == '2' and invalid_scan[100]['home'] is True
    print('producer writes exactly the rejected entry', invalid_scan, 'features:', *feature()[:2])
    for payload in ('{"gameData":{"probablePitchers":{"away":null}}}', '{', '{}'):
        put(cache, {})
        path.write_text(payload)
        print('present unusable vs valid all-null', payload, compute._build_probable_pitcher_lookup())
    path.unlink()

    put(cache, {'100': None, '999': SIG})
    for _ in range(2):
        assert compute._build_probable_pitcher_lookup()[100] is None
    print('missing feed no write, base still raises', *feature()[:2])
    feed(raw / '2025/101.json')
    print('unrelated scan base persists bad entry', compute._build_probable_pitcher_lookup())
    (raw / '2025/101.json').unlink()
    for initial in ('{"100":null}', '{'):
        cache.write_text(initial)
        feed(path)
        if initial != '{':
            put(cache, {})  # Stage the specified read-side removal before the real scanner writes.
        original_replace = os.replace

        def die_after_replace(source, target, **kwargs):
            original_replace(source, target, **kwargs)
            raise KeyboardInterrupt('at cache commit, before hypothetical status')

        try:
            with patch.object(os, 'replace', side_effect=die_after_replace):
                compute._build_probable_pitcher_lookup()
        except KeyboardInterrupt:
            recovered = compute._build_probable_pitcher_lookup()
            assert recovered == {100: GOOD_FEED}
            print('restart after successful cache replace', initial, recovered, 'no event remains on disk')
    path.unlink()


def bullpen(cache, raw):
    dates = pd.date_range('2025-04-01', periods=11)
    rows = [pa(100 + i, date, 3 if i < 10 else 1, True, i % 2) for i, date in enumerate(dates)]
    lookup = {str(game): SIG for game in range(100, 111)}
    lookup['100'] = dict(SIG, extra=0)
    put(cache, lookup)
    baseline = feature(rows)[2].iloc[-1]['opp_bullpen_hr_30g']
    del lookup['100']
    put(cache, lookup)
    absent = feature(rows)[2].iloc[-1]['opp_bullpen_hr_30g']
    feed(raw / '2025/100.json', SIG)
    restored = feature(rows)[2].iloc[-1]['opp_bullpen_hr_30g']
    assert baseline == restored == 0.5 and pd.isna(absent)
    print('strictly-later team bullpen feature with extra field / dropped / late feed:', baseline, absent, restored)


def main():
    with TemporaryDirectory(prefix='features-', dir=ROOT / 'runs') as name:
        work = Path(name)
        real_artifacts(work)
        os.chdir(work)
        raw = Path('data/raw')
        cache = Path('data/models/probable_pitcher_lookup.json')
        compute._probable_pitcher_cache_path = cache
        corpus(cache, raw)
        edge_sequences(cache, raw)
        bullpen(cache, raw)


if __name__ == '__main__':
    main()
