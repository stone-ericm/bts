import json
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import pandas as pd

from bts.features import compute
from identity import LOOKUP, read_lookup
from measure import DATA, FEATURE, compare


def candidate():
    file = Path(compute.__file__)
    source = file.read_text()
    start = source.index('        # Rolling 30-day average per team')
    end = source.index('        df.rename(columns={"_pitcher_team_id"', start)
    replacement = '''        team = "_pitcher_team_id"
        match_team = "_bullpen_asof_team"
        keys = [team, "date"]
        output_column = "opp_bullpen_hr_30g"
        if daily_bullpen.empty:
            df[output_column] = np.nan
        else:
            daily_bullpen[output_column] = daily_bullpen.groupby(team)["bp_hr"].transform(
                lambda x: x.rolling(30, min_periods=10).mean()
            )
            known = df[keys].dropna().drop_duplicates().sort_values("date", kind="stable")
            known[match_team] = known[team].astype(object)
            right = daily_bullpen[keys + [output_column]].sort_values("date", kind="stable")
            right[match_team] = right.pop(team).astype(object)
            prior = pd.merge_asof(
                known, right, on="date", by=match_team, allow_exact_matches=False, direction="backward",
            ).drop(columns=match_team)
            df = df.merge(prior, on=keys, how="left", validate="m:1")

'''
    namespace = dict(vars(compute))
    exec(compile(source[:start] + replacement + source[end:], '<key-safe bullpen probe>', 'exec'), namespace)
    fn = namespace['compute_all_features']
    return type(fn)(fn.__code__, vars(compute), name=fn.__name__)


def real_controls():
    start = perf_counter()
    seasons = [pd.read_parquet(DATA / f'processed/pa_{year}.parquet') for year in (2024, 2025)]
    revised = candidate()
    with patch.object(compute, '_build_probable_pitcher_lookup', return_value=read_lookup(LOOKUP)):
        for label, frame in zip(('2024', '2025', '2024+2025'), [*seasons, pd.concat(seasons, ignore_index=True)]):
            old = compute.compute_all_features(frame)
            new = revised(frame)
            pd.testing.assert_frame_equal(old.drop(columns=FEATURE), new.drop(columns=FEATURE), check_exact=True)
            print(label, 'other columns exact', len(old))
            compare(pd.DataFrame({'old': old[FEATURE], 'new': new[FEATURE]}), 'old', 'new')
    print('seconds', perf_counter() - start)


if __name__ == '__main__':
    if sys.argv[1:] == ['--real']:
        real_controls()
    else:
        import pytest
        from bts.model import predict

        compute.compute_all_features = candidate()
        predict.compute_all_features = compute.compute_all_features
        raise SystemExit(pytest.main(sys.argv[1:]))
