"""A starter PA should not depend on whether a later PA uses a reliever."""

from collections import defaultdict
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bts.features import compute, park_drag


def test_bullpen_availability_does_not_depend_on_same_day_reliever():
    dates = pd.date_range("2025-03-27", periods=11)
    observations = {
        "pitch_calls": ["X"],
        "pitch_types": ["FF"],
        "pitch_speeds": [93.0],
        "pitch_spin_rates": [2300.0],
        "pitch_extensions": [6.0],
        "pitch_break_vertical": [15.0],
        "pitch_break_horizontal": [5.0],
        "pitch_px": [0.0],
        "pitch_pz": [2.5],
        "sz_top": 3.5,
        "sz_bottom": 1.5,
        "final_count_balls": 0,
        "final_count_strikes": 0,
        "launch_speed": 91.0,
        "launch_angle": 10.0,
    }
    rows = [
        dict(observations, pa_id=day, date=date, game_pk=100 + day,
             batter_id=11, pitcher_id=22 if day < 10 else 21,
             pitch_hand="R", venue_id=31, is_home=True, is_hit=day % 2)
        for day, date in enumerate(dates)
    ]
    rows.append(dict(rows[-1], pa_id=11))
    starter_only = pd.DataFrame(rows)
    uses_reliever = starter_only.copy()
    uses_reliever.loc[uses_reliever["pa_id"] == 11, "pitcher_id"] = 22
    probables = {game: {"away": 21, "away_tid": 51} for game in range(100, 111)}

    with (
        patch.object(compute, "_build_probable_pitcher_lookup", return_value=probables),
        patch.object(park_drag, "get_table", return_value=None),
    ):
        before = compute.compute_all_features(starter_only).set_index("pa_id")
        after = compute.compute_all_features(uses_reliever).set_index("pa_id")

    pd.testing.assert_series_equal(
        before.loc[[10], "opp_bullpen_hr_30g"],
        after.loc[[10], "opp_bullpen_hr_30g"],
    )
    # Review note: equal is not enough (two missing values are equal). The ten
    # prior reliever dates alternate out and hit, so the value is exactly 0.5.
    assert before.loc[10, "opp_bullpen_hr_30g"] == 0.5
    assert after.loc[10, "opp_bullpen_hr_30g"] == 0.5

FEATURE = "opp_bullpen_hr_30g"


def _compute(frame, lookup):
    with (
        patch.object(compute, "_build_probable_pitcher_lookup", return_value=lookup),
        patch.object(park_drag, "get_table", return_value=None),
    ):
        return compute.compute_all_features(frame)


def _prior_values(frame, lookup):
    buckets = defaultdict(list)
    targets = {}
    for pa in frame.itertuples(index=False):
        side = "away" if pa.is_home else "home"
        game = lookup.get(pa.game_pk, {})
        team = game.get(f"{side}_tid")
        date = pd.Timestamp(pa.date)
        targets[pa.pa_id] = (team, date)
        if pd.isna(team) or pd.isna(date):
            continue
        probable = game.get(side)
        if pd.notna(probable) and pa.pitcher_id != probable:
            hits = buckets[team, date]
            if pd.notna(pa.is_hit):
                hits.append(pa.is_hit)

    dates_by_team = defaultdict(list)
    for (team, date), hits in sorted(buckets.items()):
        dates_by_team[team].append((date, sum(hits) / len(hits) if hits else np.nan))

    expected = {}
    for pa_id, (team, date) in targets.items():
        prior = [rate for day, rate in dates_by_team[team] if day < date]
        defined = [rate for rate in prior[-30:] if not np.isnan(rate)]
        expected[pa_id] = sum(defined) / len(defined) if len(defined) >= 10 else np.nan
    return pd.Series(expected, dtype=float)


def _assert_reference(frame, lookup):
    actual = _compute(frame, lookup).set_index("pa_id")[FEATURE]
    expected = _prior_values(frame, lookup).reindex(actual.index)
    assert actual.index.is_unique and len(actual) == len(frame)
    np.testing.assert_array_equal(actual.isna(), expected.isna())
    np.testing.assert_allclose(actual.dropna(), expected.dropna(), atol=1e-12, rtol=0)
    return actual


def _side(days):
    """One PA per game unless a date is repeated."""
    rows = [
        dict(pa_id=i, game_pk=100 + i, date=date, batter_id=11, pitcher_id=pitcher,
             pitch_hand="R", venue_id=31, is_home=True, is_hit=hit,
             pitch_calls=["X"], pitch_types=["FF"], pitch_speeds=[93.0],
             pitch_spin_rates=[2300.0], pitch_extensions=[6.0],
             pitch_break_vertical=[15.0], pitch_break_horizontal=[5.0],
             pitch_px=[0.0], pitch_pz=[2.5], sz_top=3.5, sz_bottom=1.5,
             final_count_balls=0, final_count_strikes=0, launch_speed=91.0,
             launch_angle=10.0)
        for i, (date, pitcher, hit) in enumerate(days)
    ]
    lookup = {row["game_pk"]: {"away": 21, "away_tid": 51} for row in rows}
    return pd.DataFrame(rows), lookup


def test_warmed_doubleheader_presence_and_outcomes():
    dates = pd.date_range("2024-09-15", periods=10)
    target = "2025-03-27"
    starters, lookup = _side(
        [(str(date.date()), 22, int(i < 3)) for i, date in enumerate(dates)]
        + [(target, 21, 0), (target, 21, 1)]
    )
    relievers = starters.copy()
    relievers.loc[10:, "pitcher_id"] = 22
    relievers.loc[10:, "is_hit"] = 1
    relievers_no_hits = relievers.copy()
    relievers_no_hits.loc[10:, "is_hit"] = 0
    for frame in (relievers, relievers_no_hits, starters):
        actual = _compute(frame.iloc[::-1], lookup).set_index("pa_id")[FEATURE]
        np.testing.assert_allclose(actual.loc[[10, 11]], [0.3, 0.3], atol=1e-12, rtol=0)


def test_bullpen_window_matches_loop():
    dates = pd.date_range("2024-12-22", periods=56)
    days = [(day, 22, 0) for day in dates[:10]]
    days += [(dates[10], 21, 0), (dates[11], 22, 1), (dates[11], 22, 0),
             (dates[11], 22, 0), (dates[12], 21, 0)]
    days += [(day, 22, np.nan) for day in dates[13:46]]
    days += [(day, 22, 1) for day in dates[46:]]
    days += [(dates[-1] + pd.Timedelta(days=1), 22, 0),
             (dates[-1] + pd.Timedelta(days=2), 22, 1),
             (dates[-1] + pd.Timedelta(days=3), 22, 1),
             (pd.NaT, 22, 1)]
    frame, lookup = _side(days)
    unknown_probable, new_team, unknown_team, _ = frame["game_pk"].iloc[-4:]
    lookup[unknown_probable].pop("away")
    lookup[new_team]["away_tid"] = 52
    del lookup[unknown_team]
    trailing_gap = dict(frame.iloc[0], pa_id=len(frame), game_pk=100 + len(frame),
                        date=dates[-1] + pd.Timedelta(days=4), pitcher_id=21)
    lookup[trailing_gap["game_pk"]] = {"away": 21, "away_tid": 51}
    frame = pd.concat([frame, pd.DataFrame([trailing_gap])], ignore_index=True)
    actual = _assert_reference(frame.sample(frac=1, random_state=7), lookup)
    assert actual.loc[:9].isna().all()
    assert actual.loc[10] == 0
    np.testing.assert_allclose(actual.loc[11:13], [0, 0, 0], atol=1e-12, rtol=0)
    assert actual.loc[14] == pytest.approx(1 / 33)
    assert actual.loc[37:47].isna().all()
    assert actual.loc[[58, trailing_gap["pa_id"]]].eq(1).all()
    assert actual.loc[[59, 60, 61]].isna().all()


@pytest.mark.parametrize("empty_lookup", [False, True])
def test_bullpen_empty_history(empty_lookup):
    frame, lookup = _side([("2025-04-01", 21, 1), (None, 22, 0)])
    if empty_lookup:
        lookup = {}
    result = _compute(frame, lookup)
    assert result[FEATURE].isna().all()
    assert ("opp_pitching_team_id" in result) == (not empty_lookup)
    _assert_reference(frame, lookup)


def test_bullpen_2025_slice_matches_loop():
    data = Path(os.environ.get("BULLPEN_TEST_DATA", "data/processed")) / "pa_2025.parquet"
    lookup_path = Path(os.environ.get(
        "BULLPEN_TEST_LOOKUP", "data/models/probable_pitcher_lookup.json"
    ))
    if not data.is_file() or not lookup_path.is_file():
        pytest.skip("2025 PA data and probable lookup are required")
    lookup = {int(pk): row for pk, row in json.loads(lookup_path.read_text()).items()}
    frame = pd.read_parquet(data)
    dates = pd.to_datetime(frame["date"])
    team_away = frame["game_pk"].map(lambda pk: lookup.get(pk, {}).get("away_tid"))
    team_home = frame["game_pk"].map(lambda pk: lookup.get(pk, {}).get("home_tid"))
    # Team 113 has a known gap on July 23. Include every earlier bucket that season,
    # including both games on any doubleheader date.
    sliced = frame.loc[
        (dates <= "2025-07-25") & (team_away.where(frame["is_home"], team_home) == 113)
    ].assign(pa_id=lambda rows: np.arange(len(rows)))
    assert len(sliced) > 100
    result = _assert_reference(sliced, lookup)
    gap = sliced.loc[pd.to_datetime(sliced["date"]).eq("2025-07-23"), "pa_id"]
    assert len(gap) and result.loc[gap].notna().all()
