"""Exercise March cold refresh, disk cache recovery and the real bullpen feature."""

import json
import io
from types import SimpleNamespace

import pandas as pd

from bts.data import pull
from bts.features import compute, park_drag

from offline import instant, offline


DAY = '2027-03-24'
FEATURE = 'opp_bullpen_hr_30g'


def frame():
    dates = pd.date_range('2025-09-01', periods=11)
    return pd.DataFrame([
        dict(pa_id=i, game_pk=100 + i, date=day, batter_id=11, pitcher_id=22 if i < 10 else 21,
             pitch_hand='R', venue_id=31, is_home=True, is_hit=i % 2,
             pitch_calls=['X'], pitch_types=['FF'], pitch_speeds=[93.0],
             pitch_spin_rates=[2300.0], pitch_extensions=[6.0], pitch_break_vertical=[15.0],
             pitch_break_horizontal=[5.0], pitch_px=[0.0], pitch_pz=[2.5], sz_top=3.5, sz_bottom=1.5,
             final_count_balls=0, final_count_strikes=0, launch_speed=91.0, launch_angle=10.0)
        for i, day in enumerate(dates)
    ])


def feed(game_type):
    return {'gameData': {'game': {'type': game_type}, 'teams': {'away': {'id': 51}, 'home': {'id': 52}},
                         'probablePitchers': {'away': {'id': 21}, 'home': {'id': 23}}}}


def cold_features():
    from bts.model import predict

    with offline(instant(DAY, 3)) as (w, mp):
        raw = w.root / 'data' / 'raw'
        prior = raw / '2025'
        spring = raw / '2027'
        prior.mkdir(parents=True)
        spring.mkdir()
        (spring / '2027.json').write_text(json.dumps(feed('S')))
        proc = raw.parent / 'processed'
        proc.mkdir()
        frame().to_parquet(proc / 'pa_2025.parquet', index=False)

        cli = w.call('data', 'build', '--seasons', '2027')
        assert 'nothing written' in cli.output
        mp.setattr(pull, 'retry_urlopen', w.http)
        predict._refresh_season_data(DAY)
        assert not (proc / 'pa_2027.parquet').exists() and not list(proc.glob('.refreshed_2027_*'))
        assert [date for kind, _, date in w.events if kind == 'schedule'] == [
            '2027-03-20', '2027-03-21', '2027-03-22', '2027-03-23',
        ]

        mp.setattr(park_drag, 'get_table', lambda **_kwargs: None)
        for pk in range(100, 111):
            (prior / f'{pk}.json').write_text(json.dumps(feed('R')))
        df = pd.read_parquet(proc / 'pa_2025.parquet')
        before = compute.compute_all_features(df)
        assert before.loc[before['pa_id'] == 10, FEATURE].item() == 0.5
        assert predict._build_feature_lookups(before)['bullpen'][51] == 0.5
        cache = w.models / 'probable_pitcher_lookup.json'
        built = json.loads(cache.read_text())
        assert set(built) == {str(pk) for pk in range(100, 111)} | {'2027'}
        assert not list(w.models.glob('*.tmp'))

        # The restored host can have a historical parquet and cache without raw feeds.
        prior.rename(raw.parent / 'archived-raw')
        preserved = compute.compute_all_features(df)
        pd.testing.assert_series_equal(before[FEATURE], preserved[FEATURE])
        print('Historical cache preserves starter-only final observation and live lookup at 0.5')

        cache.write_text('{')
        rebuilt = compute.compute_all_features(df)
        assert set(json.loads(cache.read_text())) == {'2027'}
        assert rebuilt[FEATURE].isna().all()
        assert not predict._build_feature_lookups(rebuilt)['bullpen']
        print('Unreadable cache warns and rebuilds from spring only; historical bullpen coverage disappears')

        # The read validates outer JSON, but not entries. A shape-valid file can
        # permanently fail prediction before this spring-only raw root can repair it.
        poisoned = json.dumps(built | {'100': None})
        cache.write_text(poisoned)
        for _ in range(2):
            try:
                compute.compute_all_features(df)
            except AttributeError as exc:
                assert 'NoneType' in str(exc) and 'get' in str(exc)
                print('Malformed cache entry survives and fails feature computation:', exc)
            else:
                raise AssertionError('Malformed cache entry passed feature computation')
        assert cache.read_text() == poisoned


def feed_refusals():
    print('\n=== downloaded-feed and local build refusals ===')
    with offline(instant(DAY, 3)) as (w, mp):
        raw = w.root / 'data' / 'raw' / '2027'
        raw.mkdir(parents=True)
        proc = w.root / 'data' / 'processed'
        proc.mkdir()
        output = proc / 'pa_2027.parquet'
        for existing, raw_contents, expected in (
            (False, '{}', 'missing or unreadable gameData.game.type'),
            (True, json.dumps(feed('S')), 'season output already exists'),
            (False, '{', 'JSONDecodeError'),
        ):
            bad = raw / '123.json'
            bad.write_text(raw_contents)
            if existing:
                output.write_bytes(b'previous parquet')
            result = w.call('data', 'build', '--seasons', '2027', code=1, catch=True)
            assert expected in f'{result.output} {type(result.exception).__name__} {result.exception}'
            if existing:
                assert output.read_bytes() == b'previous parquet'
                output.unlink()
            bad.unlink()

        mp.setattr(pull, 'discover_games', lambda *_: [{'gamePk': 123, 'date': DAY}])
        calls = []

        def malformed(url, **kwargs):
            calls.append(url)
            return io.BytesIO(b'[]')

        mp.setattr(pull, 'urlopen', malformed)
        mp.setattr(pull, 'time', SimpleNamespace(sleep=lambda _: None))
        result = w.call('data', 'pull', '--start', DAY, '--end', DAY, '--delay', '0')
        assert len(calls) == 3 and not (raw / '123.json').exists()
        assert 'game feed 123 must be a JSON object' in result.output and '1 games failed' in result.output


if __name__ == '__main__':
    cold_features()
    feed_refusals()
