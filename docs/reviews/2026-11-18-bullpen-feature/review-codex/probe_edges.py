"""Small independent regression probes for history and the strict as-of join."""

from unittest.mock import patch

import numpy as np
import pandas as pd
from bts.features import compute

from probe_common import (
    EXACT, FEATURE, TEAM, assert_aligned, assert_reference, compare_versions, load_base, run, toy,
)


def one_value(result, source_row):
    return result.set_index("review_row").at[source_row, FEATURE]


def history_edges():
    dates = pd.date_range("2024-12-22", periods=83)
    rows = [(day, 51, 21, 23, int(i < 6)) for i, day in enumerate(dates[:10])]
    rows += [
        (dates[9], 52, 21, 23, 1),       # team ID change, and same date across teams
        (dates[10], 51, 21, 21, 1),      # first eligible no-relief date, new season
        (dates[11], 51, 21, 22, 0),
        (dates[11], 51, 21, 23, 1),      # second game of the same team-date
        (dates[11], 51, 21, 23, 1),      # pool PAs rather than games
        (dates[12], 51, None, 22, 0),    # warmed known team with unknown probable
        (dates[12], 95, 21, 21, 0),      # known team with no reliever history at all
        (dates[12], 52, 21, 21, 0),      # no inherited history from 51
        (dates[12], None, 21, 23, 1),
        (dates[12], np.nan, 21, 23, 1),
        (pd.NaT, 51, 21, 23, 1),
    ]
    rows += [(day, 51, 21, 23, np.nan) for day in dates[13:43]]
    rows += [(dates[43], 51, 21, 21, 1)]  # prior defined mean expires, no stale carry
    rows += [(day, 51, 21, 23, int(i < 28)) for i, day in enumerate(dates[44:82])]
    rows += [(dates[-1], 51, 21, 21, 0)]  # trailing gap with over 30 historical buckets
    rows += [(dates[-1], 51, 21, 23, 1)]  # game is absent from the lookup
    frame, lookup = toy(rows)
    lookup.pop(frame.iloc[-1]["game_pk"])
    frame = frame.sample(frac=1, random_state=83)
    original_asof = pd.merge_asof
    seen = []

    def inspect_asof(left, right, *args, **kwargs):
        assert left["date"].is_monotonic_increasing and right["date"].is_monotonic_increasing
        assert not left.isna().any().any()
        assert not right[["_bullpen_asof_team", "date"]].isna().any().any()
        assert not left.duplicated(["_bullpen_asof_team", "date"]).any()
        assert not right.duplicated(["_bullpen_asof_team", "date"]).any()
        seen.append((str(left["_pitcher_team_id"].dtype), str(left["_bullpen_asof_team"].dtype),
                     str(right["_bullpen_asof_team"].dtype)))
        return original_asof(left, right, *args, **kwargs)

    base = load_base()
    for kind in ("float", "object"):
        if kind == "object":
            for item in lookup.values():
                item.pop("home_tid")
        with patch.object(pd, "merge_asof", side_effect=inspect_asof):
            actual = run(compute, frame, lookup)
        old = run(base, frame, lookup)
        compare_versions(old, actual)
        assert_aligned(actual, frame)
        assert actual[TEAM].dtype.kind == ("f" if kind == "float" else "O")
        assert len(set(actual.columns) - set(old.columns)) == 0
        error = assert_reference(actual, frame, lookup)
        assert np.isnan(one_value(actual, 9))
        assert one_value(actual, 11) == 0.6
        assert [one_value(actual, slot) for slot in (12, 13, 14)] == [0.6] * 3
        np.testing.assert_allclose(one_value(actual, 15), (6 + 2 / 3) / 11, atol=1e-12, rtol=0)
        assert np.isnan(actual.set_index("review_row").loc[list(range(16, 21)), FEATURE]).all()
        assert np.isnan(one_value(actual, 51))
        np.testing.assert_allclose(one_value(actual, len(frame) - 2), 20 / 30, atol=1e-12, rtol=0)
        assert np.isnan(one_value(actual, len(frame) - 1))
        print(kind, "as-of dtypes", seen[-1], "reference max error", error, flush=True)

    starters = {row.game_pk: {"away": row.pitcher_id, "away_tid": 51}
                for row in frame.itertuples()}
    for lookup_case in ({}, starters):
        actual = run(compute, frame, lookup_case)
        original = run(base, frame, lookup_case)
        pd.testing.assert_frame_equal(actual, original, **EXACT)
        assert actual[FEATURE].isna().all()
        assert (TEAM in actual) == bool(lookup_case)
        print("empty history; lookup populated", bool(lookup_case), "exact base schema", flush=True)

    def duplicate_attachment(*args, **kwargs):
        result = original_asof(*args, **kwargs)
        return pd.concat([result, result.iloc[:1]], ignore_index=True)

    with patch.object(pd, "merge_asof", side_effect=duplicate_attachment):
        try:
            run(compute, frame, lookup)
        except pd.errors.MergeError as exc:
            print("injected duplicate attachment right key rejected:", exc, flush=True)
        else:
            raise AssertionError("duplicate right-side key escaped validation")


def unusual_dates():
    dates = pd.date_range("2025-04-01", periods=11)
    rows = [(date, 51, 21, 23, int(i < 4)) for i, date in enumerate(dates[:10])]
    rows += [(dates[-1], 51, 21, 23, 1),
             (dates[-1] + pd.Timedelta(hours=4), 51, 21, 21, 0)]
    frame, lookup = toy(rows)
    for zone in (None, "UTC", "America/New_York"):
        if zone:
            frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(zone)
        before = run(compute, frame, lookup)
        assert_aligned(before, frame)
        assert_reference(before, frame, lookup)
        changed = frame.copy()
        changed.loc[changed["review_row"].eq(10), "is_hit"] = 0
        after = run(compute, changed, lookup)
        print("intraday", zone, "earlier PA", one_value(before, 10), one_value(after, 10),
              "later PA", one_value(before, 11), one_value(after, 11), flush=True)
        assert one_value(before, 10) == one_value(after, 10)
        assert one_value(before, 11) != one_value(after, 11)
        if zone:
            frame["date"] = frame["date"].dt.tz_localize(None)

    mixed = frame.assign(date=frame["date"].astype(str).astype(object))
    mixed.loc[0, "date"] = "2025-04-01T00:00:00Z"
    for module in (load_base(), compute):
        try:
            run(module, mixed, lookup)
        except ValueError as exc:
            print(module.__name__, "mixed date formats rejected during parsing:",
                  str(exc).splitlines()[0], flush=True)
        else:
            raise AssertionError("mixed date parsing behavior changed")


def oversized_keys():
    # These are outside MLB's ID domain. A subsequent unknown lookup entry can
    # still expose how pandas infers a common dtype before the new join runs.
    dates = pd.date_range("2025-04-01", periods=12)
    first, second = 2**53, 2**53 + 1
    rows = [(date, team, 21, 23, hit) for date in dates[:10]
            for team, hit in ((first, 0), (second, 1))]
    rows += [(dates[10], team, 21, 23, hit) for team, hit in ((first, 0), (second, 1))]
    rows += [(dates[11], None, 21, 21, 0)]
    frame, lookup = toy(rows)
    for module in (load_base(), compute):
        before = run(module, frame.iloc[:-1], lookup)
        after = run(module, frame, lookup)
        values_before = [one_value(before, i) for i in (20, 21)]
        values_after = [one_value(after, i) for i in (20, 21)]
        print(module.__name__, "oversized team IDs + future null team",
              before[TEAM].dtype, after[TEAM].dtype, values_before, values_after, flush=True)
        assert values_before == [0, 1] and values_after == [0.5, 0.5]


if __name__ == "__main__":
    history_edges()
    unusual_dates()
    oversized_keys()
