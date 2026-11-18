import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd

from bts.features import compute


DATA = Path('/Users/eric/projects/bts/data')
OUT = Path(__file__).resolve().parent
KEY = ['team', 'date']
FEATURE = 'opp_bullpen_hr_30g'
FIELDS = ['date', 'season', 'game_pk', 'is_home', 'pitcher_id', 'is_hit']


def tag(frame, lookup):
    result = frame[FIELDS].copy()
    result['date'] = pd.to_datetime(result['date'])
    for column, suffix in [('team', '_tid'), ('probable', '')]:
        result[column] = np.where(
            result['is_home'], result['game_pk'].map(lambda pk: lookup.get(pk, {}).get('away' + suffix)),
            result['game_pk'].map(lambda pk: lookup.get(pk, {}).get('home' + suffix)),
        )
    result['reliever'] = result['probable'].notna() & result['pitcher_id'].ne(result['probable'])
    return result


def daily_values(rows):
    known = rows.dropna(subset=KEY)
    dates = (known.groupby(KEY).agg(n=('is_hit', 'size'), hits=('is_hit', 'sum'),
                                  relievers=('reliever', 'sum')).reset_index())
    relief = (known.loc[known['reliever']].groupby(KEY)['is_hit']
              .agg(['sum', 'count']).reset_index())
    relief['rate'] = relief['sum'] / relief['count']
    rates = relief.groupby('team')['rate']
    relief['old'] = rates.transform(lambda x: x.shift().rolling(30, min_periods=10).mean())
    relief['unshifted'] = rates.transform(lambda x: x.rolling(30, min_periods=10).mean())
    dates = dates.merge(relief[KEY + ['old']], on=KEY, how='left', validate='1:1')
    dates['asof'] = pd.merge_asof(
        dates[KEY].sort_values('date', kind='stable'),
        relief[KEY + ['unshifted']].sort_values('date', kind='stable'), by='team', on='date',
        allow_exact_matches=False, direction='backward',
    ).set_index(KEY)['unshifted'].reindex(dates.set_index(KEY).index).to_numpy()

    dates['naive'] = dates.groupby('team')['old'].ffill()
    dates['carry'] = np.nan
    for team, group in dates.groupby('team'):
        source = relief.loc[relief['team'].eq(team)].set_index('date')['rate']
        if source.empty:
            continue
        # Shift on the reliever-date axis, including a virtual row after the final query.
        terminal = group['date'].max() + pd.Timedelta(days=1)
        shifted = source.reindex(source.index.append(pd.DatetimeIndex([terminal]))).shift()
        shifted = shifted.rolling(30, min_periods=10).mean()
        # Each shift-slot's value becomes available on the first team-date strictly
        # after its source reliever day. Reindex the effective dates positionally.
        queries = pd.DatetimeIndex(group['date'])
        effective = queries.append(pd.DatetimeIndex([terminal])).take(queries.searchsorted(source.index, side='right'))
        carry = pd.Series(shifted.iloc[1:].to_numpy(), index=effective)
        carry = carry[~carry.index.duplicated(keep='last')]
        dates.loc[group.index, 'carry'] = carry.reindex(queries, method='ffill').to_numpy()
    return dates


def same(left, right):
    return left.to_numpy(dtype='float64').view('uint64') == right.to_numpy(dtype='float64').view('uint64')


def compare(frame, left, right):
    a, b = frame[left], frame[right]
    both = a.notna() & b.notna()
    delta = (a[both] - b[both]).abs()
    print(left, right, 'both', int(both.sum()), 'both_missing', int((a.isna() & b.isna()).sum()),
          'left_only', int((a.notna() & b.isna()).sum()), 'right_only', int((a.isna() & b.notna()).sum()),
          'numeric_bit_diff', int((~same(a[both], b[both])).sum()), 'max_abs', delta.max(),
          'nonzero_mean_abs', delta[delta.ne(0)].mean())


def serve(dates, label):
    dates = dates.copy()
    for version in ('old', 'asof'):
        dates[f'serve_{version}'] = dates.groupby('team')[version].transform(lambda s: s.ffill().shift())
    dates['previous_date'] = dates.groupby('team')['date'].shift()
    dates['previous_relief'] = dates.groupby('team')['relievers'].shift()
    dates = dates[dates['date'].dt.year.eq(2025)].copy()
    print(label, 'team-dates', len(dates))
    compare(dates, 'asof', 'serve_asof')
    compare(dates, 'asof', 'serve_old')
    compare(dates, 'serve_asof', 'serve_old')
    for has_relief in (True, False):
        items = dates[dates['previous_relief'].gt(0) if has_relief else dates['previous_relief'].eq(0)]
        print('previous reliever date?', has_relief, len(items))
        compare(items, 'asof', 'serve_asof')
    dates.to_csv(OUT / f'parity-{label}.csv', index=False)
    return dates


def game_breakdown(tagged, dates, affected):
    games = []
    raw_ids = set()
    for game_pk, sides in tagged.groupby('game_pk'):
        side_rows = list(sides.groupby('is_home'))
        path = DATA / 'raw' / '2025' / f'{game_pk}.json'
        if not path.exists():
            continue
        raw_ids.add(game_pk)
        feed = json.loads(path.read_text())
        game = feed['gameData']
        for is_home, side in side_rows:
            pitchers = feed['liveData']['boxscore']['teams']['away' if is_home else 'home']['pitchers']
            probable = side['probable'].iloc[0]
            games.append(dict(game_pk=game_pk, date=side['date'].iloc[0], team=side['team'].iloc[0],
                              probable=probable, actual_first=pitchers[0] if pitchers else None,
                              actual_pitchers=len(pitchers), pa_pitchers=side['pitcher_id'].nunique(),
                              all_reliever=side['reliever'].all(), no_reliever=not side['reliever'].any(),
                              raw_status=game['status'].get('detailedState'),
                              innings=feed['liveData']['linescore'].get('currentInning'),
                              scheduled=feed['liveData']['linescore'].get('scheduledInnings'),
                              resume=game['datetime'].get('resumeDateTime'),
                              complete_games=sum(feed['liveData']['boxscore']['teams']['away' if is_home else 'home']
                                                 ['players'].get(f'ID{pitcher}', {}).get('stats', {}).get('pitching', {})
                                                 .get('completeGames', 0) for pitcher in pitchers)))
    sides = pd.DataFrame(games).merge(dates[KEY + ['relievers']], on=KEY, how='left')
    sides.to_csv(OUT / 'game-sides-2025.csv', index=False)
    impacted = sides.merge(affected[KEY], on=KEY)
    print('raw games present / seen', len(raw_ids), tagged['game_pk'].nunique())
    print('impacted raw sides grouping:')
    print(impacted.groupby(['actual_pitchers', 'pa_pitchers', 'raw_status', 'innings', 'complete_games'], dropna=False).size())
    for name, filtered in {
        'missing probable': sides[sides['probable'].isna()],
        'wrong probable, all relief': sides[sides['probable'].notna() & sides['actual_first'].ne(sides['probable']) & sides['all_reliever']],
        'wrong probable, mixed': sides[sides['probable'].notna() & sides['actual_first'].ne(sides['probable']) & ~sides['all_reliever']],
        'opener proxy, probable was first and faced <= 6 before handing off': sides.iloc[:0],
        'resumed': sides[sides['resume'].notna()],
        'shortened': sides[sides['innings'].lt(sides['scheduled'])],
    }.items():
        if name.startswith('opener'):
            first_pa = tagged[tagged['pitcher_id'].eq(tagged['probable'])].groupby(['game_pk', 'team']).size().rename('starter_pa')
            joined = sides.merge(first_pa, on=['game_pk', 'team'], how='left')
            filtered = joined[joined['probable'].eq(joined['actual_first']) & joined['starter_pa'].le(6) & joined['actual_pitchers'].gt(1)]
        print(name, 'sides', len(filtered), 'team-dates', len(filtered[KEY].drop_duplicates()),
              'no relief team-dates', len(filtered[filtered['relievers'].eq(0)][KEY].drop_duplicates()),
              'examples', filtered[['game_pk', 'team', 'date']].head(3).to_dict('records'))
    impacted.to_csv(OUT / 'affected-game-sides.csv', index=False)


if __name__ == '__main__':
    start = perf_counter()
    lookup = {int(k): v for k, v in json.loads((DATA / 'models/probable_pitcher_lookup.json').read_text()).items()}
    raw = pd.read_parquet(DATA / 'processed/pa_2025.parquet')
    for path in (DATA / 'raw/2025').glob('*.json'):
        if path.stem.isdigit() and int(path.stem) not in lookup:
            gd = json.loads(path.read_text())['gameData']
            lookup[int(path.stem)] = {
                **{side: gd.get('probablePitchers', {}).get(side, {}).get('id') for side in ('home', 'away')},
                **{side + '_tid': gd.get('teams', {}).get(side, {}).get('id') for side in ('home', 'away')},
            }
    tagged = tag(raw, lookup)
    with patch.object(compute, '_build_probable_pitcher_lookup', return_value=lookup):
        baseline = compute.compute_all_features(raw)
    print('source', compute.__file__, 'pandas', pd.__version__, '2025 full features seconds', perf_counter() - start)
    dates = daily_values(tagged)
    baseline = baseline.rename(columns={'opp_pitching_team_id': 'team'}).merge(
        dates[KEY + ['old', 'asof', 'naive', 'carry']], on=KEY, how='left', validate='m:1')
    compare(baseline, FEATURE, 'old')
    for option in ('asof', 'naive', 'carry'):
        compare(baseline, FEATURE, option)
    compare(baseline, 'asof', 'carry')
    dates['old_valued_before'] = dates.groupby('team')['old'].transform(lambda s: s.notna().cumsum() - s.notna())
    lead = dates[dates['old'].isna() & dates['old_valued_before'].gt(0)]
    affected = dates[dates['asof'].notna() & dates['old'].isna()]
    assert affected['relievers'].eq(0).all()
    for label, subset in [('all', dates), ('lead', lead), ('direct', affected), ('old valued', dates[dates['old'].notna()]),
                          ('other than direct', dates.drop(index=affected.index)), ('direct post warm-up', dates[dates['asof'].notna()])]:
        print(label, len(subset), subset['n'].sum(), subset['hits'].sum(), subset['hits'].sum() / subset['n'].sum())
    parity = serve(dates, '2025-only')
    next_games = parity.merge(affected[KEY].rename(columns={'date': 'previous_date'}), on=['team', 'previous_date'])
    next_games['delta_serving'] = next_games['serve_asof'] - next_games['serve_old']
    next_games.to_csv(OUT / 'following-games-2025.csv', index=False)
    print('next games', len(next_games), 'delta range', next_games['delta_serving'].min(), next_games['delta_serving'].max())
    game_breakdown(tagged, dates, affected)
    history = pd.concat([pd.read_parquet(DATA / f'processed/pa_{year}.parquet', columns=FIELDS)
                         for year in range(2017, 2025)] + [raw[FIELDS]], ignore_index=True)
    history_tagged = tag(history, lookup)
    print('history missing team by season', history_tagged.groupby('season')['team'].apply(lambda s: s.isna().sum()).to_dict())
    serve(daily_values(history_tagged), '2017-2025')
    print('total seconds', perf_counter() - start)
