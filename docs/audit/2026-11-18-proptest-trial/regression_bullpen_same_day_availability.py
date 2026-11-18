"""A starter PA should not depend on whether a later PA uses a reliever."""

from unittest.mock import patch

import pandas as pd

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
