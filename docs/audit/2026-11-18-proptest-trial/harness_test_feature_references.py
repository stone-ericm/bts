"""Direct historical PA calculations for four feature families."""

from itertools import groupby
from statistics import mean

import numpy as np
import pytest
from hypothesis import Phase, given, settings

from bts.features import compute
from test_prefix import cases, compare, make_frame


def reference_values(row, history):
    prior = [
        pa for pa in history if pa.batter_id == row.batter_id and pa.date < row.date
    ]
    by_date = [
        [pa.is_hit for pa in game_date]
        for _, game_date in groupby(prior, key=lambda pa: pa.date)
    ]

    def date_mean(window, min_dates):
        return mean(mean(day) for day in window) if len(window) >= min_dates else np.nan

    expected = {"batter_hr_7g": date_mean(by_date[-7:], 3)}
    for size in (30, 60, 120):
        window = by_date[-size:]
        if compute.ROOKIE_GATE_K > 0 and len(prior) < compute.ROOKIE_THRESHOLD_PAS:
            observations = [hit for day in window for hit in day]
            k = compute.ROOKIE_GATE_K
            hit_rate = (sum(observations) + k * compute.LEAGUE_PA_HIT_RATE_PRIOR) / (
                len(observations) + k
            )
        else:
            hit_rate = date_mean(window, max(3, size // 4))
        expected[f"batter_hr_{size}g"] = hit_rate

    hand_history = [pa for pa in prior if pa.pitch_hand == row.pitch_hand]
    hand_dates = {pa.date for pa in hand_history}
    expected["platoon_hr"] = (
        mean(pa.is_hit for pa in hand_history)
        if len(hand_dates) >= 5 and len(hand_history) >= 30
        else np.nan
    )
    matchup_history = [pa.is_hit for pa in prior if pa.pitcher_id == row.pitcher_id]
    expected["batter_pitcher_shrunk_hr"] = (
        (sum(matchup_history) + 10 * 0.2195) / (len(matchup_history) + 10)
    )
    return expected


def check_reference(case):
    frame, cutoff = make_frame(case)
    actual = compare(frame, cutoff)
    history = list(frame.sort_values("date").itertuples(index=False))
    for row in history:
        expected = reference_values(row, history)
        np.testing.assert_allclose(
            actual.loc[row.pa_id, list(expected)].to_numpy(dtype=float),
            list(expected.values()),
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"PA {row.pa_id}, features {list(expected)}",
        )


@settings(max_examples=100, deadline=None, database=None, phases=(Phase.generate, Phase.shrink))
@given(case=cases())
def test_generated_references(case):
    check_reference(case)


@pytest.mark.parametrize(
    "case",
    [
        # Uneven date weights within an active window spanning the offseason.
        (8, 4, 2, 3, False, True, True, 173, 18, 4),
        # One batter reaches exactly 100 prior PAs, with enough game-dates to
        # activate every veteran baseline. The platoon PA gate also switches on.
        (34, 1, 0, 4, False, False, False, 146, 213, 24),
        # Opposite-handed reliever has distinct history from the starter.
        (34, 2, 1, 4, False, True, True, 214, 49, 32),
    ],
)
def test_reference_boundaries(case):
    check_reference(case)
