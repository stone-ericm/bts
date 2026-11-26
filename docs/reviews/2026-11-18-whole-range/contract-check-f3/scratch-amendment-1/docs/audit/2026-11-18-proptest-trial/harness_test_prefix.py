"""Generated prefix and same-day outcome invariance for feature computation."""

import os
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd
from hypothesis import Phase, given, seed, settings, strategies as st

from bts.features import compute, park_drag


FEATURES = compute.FEATURE_COLS + compute.CONTEXT_COLS + compute.STATCAST_COLS
PITCH_PROFILES = (
    {
        "pitch_calls": ["C", "F", "X"],
        "pitch_types": ["FF", "SL", "FF"],
        "pitch_speeds": [94.0, 85.0, 92.0],
        "pitch_spin_rates": [2400.0, 2700.0, 2300.0],
        "pitch_extensions": [6.1, 5.8, 6.0],
        "pitch_break_vertical": [15.0, 4.0, 14.0],
        "pitch_break_horizontal": [6.0, -9.0, 5.0],
        "pitch_px": [0.8, -0.8, 0.0],
        "pitch_pz": [2.5, 2.0, 2.5],
        "sz_top": 3.5,
        "sz_bottom": 1.5,
        "final_count_balls": 0,
        "final_count_strikes": 2,
        "launch_speed": 101.0,
        "launch_angle": 29.0,
        "hardness": "hard",
    },
    {
        "pitch_calls": ["B", "S", "X"],
        "pitch_types": ["CH", "CH", "CH"],
        "pitch_speeds": [82.0, 84.0, 83.0],
        "pitch_spin_rates": [1700.0, None, 1600.0],
        "pitch_extensions": [5.2, 5.1, 5.0],
        "pitch_break_vertical": [1.0, 2.0, 3.0],
        "pitch_break_horizontal": [-13.0, -12.0, -11.0],
        "pitch_px": [0.9, None, -0.9],
        "pitch_pz": [3.3, 2.8, 1.5],
        "sz_top": 3.6,
        "sz_bottom": 1.6,
        "final_count_balls": 1,
        "final_count_strikes": 1,
        "launch_speed": 83.0,
        "launch_angle": 1.0,
        "hardness": "soft",
    },
)


@st.composite
def cases(draw):
    # Small examples shrink; longer histories activate the 10/20/30-date gates.
    size = draw(st.one_of(st.integers(2, 12), st.integers(13, 34)))
    fall_dates = draw(st.integers(1, min(size - 1, 5)))
    doubleheader = draw(st.integers(0, size - 1))
    pa_per_game = draw(st.integers(2, 4))
    alternating_batters = draw(st.booleans())
    lefty_reliever = draw(st.booleans())
    reverse_rows = draw(st.booleans())
    hits = draw(st.integers(0, 255))
    pitches = draw(st.integers(0, 255))
    cutoff = draw(st.integers(0, size - 1))
    return (size, fall_dates, doubleheader, pa_per_game, alternating_batters,
            lefty_reliever, reverse_rows, hits, pitches, cutoff)


def make_frame(case):
    (size, fall_dates, doubleheader, pa_per_game, alternating_batters,
     lefty_reliever, reverse_rows, hits, pitches, cutoff) = case
    fall = pd.Timestamp("2024-09-29")
    spring = pd.Timestamp("2025-03-27")
    dates = [fall - pd.Timedelta(days=2 * i) for i in reversed(range(fall_dates))]
    dates += [spring + pd.Timedelta(days=2 * i) for i in range(size - fall_dates)]
    rows = []
    for day, date in enumerate(dates):
        for game in range(1 + (day == doubleheader)):
            for pa in range(pa_per_game):
                pa_id = len(rows)
                pitcher = 21 + (pa == pa_per_game - 1)
                profile = (pitches >> (pa_id % 8)) & 1
                rows.append({
                    "pa_id": pa_id,
                    "date": date,
                    "season": date.year,
                    "game_pk": 100 + day * 2 + game,
                    "batter_id": 11 + (alternating_batters and (day + pa) % 3 == 1),
                    "pitcher_id": pitcher,
                    "pitch_hand": "L" if lefty_reliever and pitcher == 22 else "R",
                    "venue_id": 31 + (day % 4 == 1),
                    "is_home": True,
                    "hp_umpire_id": 41,
                    "weather_temp": 67,
                    "weather_wind_dir": "Out to CF",
                    "weather_wind_speed": 8,
                    "roof_type": "open",
                    "is_hit": (hits >> (pa_id % 8)) & 1,
                    "pitch_profile": profile,
                    **PITCH_PROFILES[profile],
                })
    return pd.DataFrame(rows[::-1] if reverse_rows else rows), dates[cutoff]


def changed_outcomes(frame, cutoff):
    changed = frame.copy()
    at_or_after = frame.index[frame["date"] >= cutoff]
    changed.loc[at_or_after, "is_hit"] = 1 - frame.loc[at_or_after, "is_hit"]
    for idx in at_or_after:
        profile = 1 - frame.at[idx, "pitch_profile"]
        changed.at[idx, "pitch_profile"] = profile
        for column, value in PITCH_PROFILES[profile].items():
            changed.at[idx, column] = value
    return changed


def compare(frame, cutoff, module=compute):
    dates = frame["date"]
    probables = {game: {"away": 21, "away_tid": 51} for game in frame["game_pk"]}
    parks = frame[["venue_id", "date"]].drop_duplicates().assign(park_drag_delta=0.01)
    with (
        patch.object(module, "_build_probable_pitcher_lookup", return_value=probables),
        patch.object(park_drag, "get_table", return_value=parks),
    ):
        full = module.compute_all_features(frame).set_index("pa_id").sort_index()
        prefix = module.compute_all_features(frame[dates <= cutoff]).set_index("pa_id").sort_index()
        changed = module.compute_all_features(changed_outcomes(frame, cutoff))
        changed = changed.set_index("pa_id").sort_index()

    def assert_same(other, mask, property_name):
        expected = full.loc[mask, FEATURES]
        try:
            pd.testing.assert_frame_equal(expected, other.loc[expected.index, FEATURES],
                                          check_dtype=False, rtol=1e-12, atol=1e-12)
        except AssertionError as error:
            raise AssertionError(f"{property_name}, cutoff {cutoff.date()}\n{error}") from error

    assert_same(prefix, full["date"] == cutoff, "prefix")
    assert_same(changed, full["date"] <= cutoff, "same-day and earlier outcome invariance")
    for _, appearances in frame.groupby("batter_id"):
        previous = None
        for date, day in appearances.groupby("date", sort=True):
            gap = np.nan if previous is None else (date - previous).days
            np.testing.assert_array_equal(full.loc[day["pa_id"], "days_rest"], gap)
            previous = date
    return full


MUTATIONS = {
    "rolling": (
        "x.shift(1).rolling(7, min_periods=3).mean()",
        "x.rolling(7, min_periods=3).mean()",
    ),
    "matchup": (
        '_grp["_day_hits"].cumsum() - _daily["_day_hits"]',
        '_grp["_day_hits"].cumsum()',
    ),
    "framing": (
        'lambda x: x.shift(1).expanding(min_periods=5).mean()',
        'lambda x: x.expanding(min_periods=5).mean()',
    ),
}


def scratch_module(mutation, tmp_path):
    source = Path(compute.__file__).read_text()
    old, new = MUTATIONS[mutation]
    assert source.count(old) == 1
    # The scratch source exists only in memory under a separate module name.
    module = ModuleType("scratch_compute")
    exec(compile(source.replace(old, new), str(tmp_path / "scratch_compute.py"), "exec"),
         module.__dict__)
    return module


def test_prefix(tmp_path):
    mutation = os.environ.get("PROPTEST_MUTATION")
    module = scratch_module(mutation, tmp_path) if mutation else compute

    @settings(max_examples=int(os.environ.get("PROPTEST_EXAMPLES", "400")),
              deadline=None, database=None, phases=(Phase.generate, Phase.shrink))
    @given(case=cases())
    def check(case):
        frame, cutoff = make_frame(case)
        compare(frame, cutoff, module)

    check()


if __name__ == "__main__":
    @seed(5118)
    @settings(max_examples=5, database=None, phases=(Phase.generate,))
    @given(case=cases())
    def preview(case):
        frame, cutoff = make_frame(case)
        print(f"case={case!r}; cutoff={cutoff.date()}")
        print(frame[["pa_id", "date", "game_pk", "batter_id", "pitcher_id",
                     "pitch_hand", "venue_id", "is_hit", "pitch_profile"]].to_string(index=False))
        print()

    preview()
