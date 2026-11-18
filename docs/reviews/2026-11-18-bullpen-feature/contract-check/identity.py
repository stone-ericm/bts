import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import pandas as pd

from bts.features import compute
from measure import DATA, FEATURE, FIELDS, KEY, OUT, compare, daily_values, serve, tag


LOOKUP = Path('/Users/eric/bts-runs/inputs/probable_pitcher_lookup.2017-2025.json')


def make_candidate():
    file = Path(compute.__file__)
    source = file.read_text()
    start = source.index('        # Rolling 30-day average per team')
    end = source.index('        df.rename(columns={"_pitcher_team_id"', start)
    modified = source[:start] + '''        keys = ["_pitcher_team_id", "date"]
        output_column = "opp_bullpen_hr_30g"
        if daily_bullpen.empty:
            df[output_column] = np.nan
        else:
            daily_bullpen[output_column] = daily_bullpen.groupby("_pitcher_team_id")["bp_hr"].transform(
                lambda x: x.rolling(30, min_periods=10).mean()
            )
            known = df[keys].dropna().drop_duplicates().sort_values("date", kind="stable")
            prior = pd.merge_asof(
                known, daily_bullpen[keys + [output_column]].sort_values("date", kind="stable"),
                on="date", by="_pitcher_team_id", allow_exact_matches=False, direction="backward",
            )
            df = df.merge(prior, on=keys, how="left", validate="m:1")

''' + source[end:]
    namespace = dict(vars(compute))
    exec(compile(modified, '<in-memory bullpen probe>', 'exec'), namespace)
    return namespace


def read_lookup(path):
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


def compare_pipeline(cases, lookup):
    with patch.object(compute, '_build_probable_pitcher_lookup', return_value=lookup):
        scope = make_candidate()
        scope['_build_probable_pitcher_lookup'] = compute._build_probable_pitcher_lookup
        for label, frame in cases:
            start = perf_counter()
            old = compute.compute_all_features(frame)
            new = scope['compute_all_features'](frame)
            pd.testing.assert_frame_equal(old.drop(columns=FEATURE), new.drop(columns=FEATURE), check_exact=True)
            print(label, 'other columns exact', len(old), f'{perf_counter() - start:.2f}s')
            compare(pd.DataFrame({'old': old[FEATURE], 'new': new[FEATURE]}), 'old', 'new')


def gaps(lookup):
    # First eligible date is a gap; a later gap follows a changed reliever rate;
    # then all-null reliever dates force a finite rolling value to become NaN.
    series = pd.DataFrame({'date': pd.date_range('2025-01-01', periods=49), 'team': 1,
                           'is_hit': [0] * 10 + [0, 1, 0] + [float('nan')] * 36,
                           'reliever': [True] * 10 + [False, True, False] + [True] * 36})
    dates = daily_values(series)
    print(dates.iloc[[9, 10, 11, 12, 39, 40, 41, 48]][['date', 'relievers', 'old', 'naive', 'asof', 'carry']])
    compare(dates, 'asof', 'carry')


if __name__ == '__main__':
    frozen = read_lookup(LOOKUP)
    current = read_lookup(DATA / 'models/probable_pitcher_lookup.json')
    seasons = {year: pd.read_parquet(DATA / f'processed/pa_{year}.parquet') for year in (2024, 2025)}
    print('frozen/current lookup entries', len(frozen), len(current))
    files = [(str(year), frame) for year, frame in seasons.items()]
    files.append(('2024+2025', pd.concat(seasons.values(), ignore_index=True)))
    compare_pipeline(files, frozen)
    compare_pipeline([('2024 with current cache', seasons[2024]), ('2025 empty cache', seasons[2025])], current)
    compare_pipeline([('2025 empty lookup', seasons[2025])], {})
    gaps(frozen)
    print('2025 frozen vs current common-field changes', sum(frozen.get(pk) != current.get(pk) for pk in seasons[2025]['game_pk'].unique()))
    history = pd.concat([pd.read_parquet(DATA / f'processed/pa_{year}.parquet', columns=FIELDS)
                         for year in range(2017, 2024)] + [s[FIELDS] for s in seasons.values()], ignore_index=True)
    tagged = tag(history, frozen)
    print('frozen history missing team by season', tagged.groupby('season')['team'].apply(lambda s: s.isna().sum()).to_dict())
    dates = daily_values(tagged)
    targets = dates[dates['date'].dt.year.eq(2025)]
    compare(targets, 'old', 'asof')
    compare(targets, 'asof', 'naive')
    compare(targets, 'asof', 'carry')
    serve(dates, '2017-2025-frozen-lookup')
