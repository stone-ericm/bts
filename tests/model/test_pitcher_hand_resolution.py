"""platoon_hr must resolve pitcher handedness at inference (audit M1).

At the game_time-45min production pick the live feed has no plays and the
schedule's probablePitcher hydrate returns only {id, fullName, link} (verified
against the live MLB API) — so slot["pitcher_hand"] is None and platoon_hr (a
production FEATURE_COL) is silently NaN, while it's fully populated in
training/backtest. Resolve from the pitcher's static handedness in history.
"""
import pytest

try:  # lightgbm is an optional extra; skip (not error) when it/libomp is absent
    import lightgbm  # noqa: F401
except (ImportError, OSError):
    pytest.skip(
        "lightgbm/libomp unavailable; skipping model tests",
        allow_module_level=True,
    )

import pandas as pd

from bts.model.predict import _pitcher_hand_lookup, _resolve_pitcher_hand


def test_pitcher_hand_lookup_from_history():
    df = pd.DataFrame([
        {"pitcher_id": 1, "pitch_hand": "L"},
        {"pitcher_id": 1, "pitch_hand": "L"},
        {"pitcher_id": 2, "pitch_hand": "R"},
        {"pitcher_id": 3, "pitch_hand": None},
    ])
    hand = _pitcher_hand_lookup(df)
    assert hand[1] == "L"
    assert hand[2] == "R"
    assert 3 not in hand  # all-NaN pitcher dropped


def test_resolve_prefers_live_hand():
    slot = {"pitcher_id": 1, "pitcher_hand": "R"}
    assert _resolve_pitcher_hand(slot, {"pitcher_hand": {1: "L"}}) == "R"


def test_resolve_falls_back_to_history_when_pregame():
    slot = {"pitcher_id": 7, "pitcher_hand": None}
    assert _resolve_pitcher_hand(slot, {"pitcher_hand": {7: "L"}}) == "L"


def test_resolve_none_for_true_debut():
    slot = {"pitcher_id": 999, "pitcher_hand": None}
    assert _resolve_pitcher_hand(slot, {"pitcher_hand": {}}) is None
