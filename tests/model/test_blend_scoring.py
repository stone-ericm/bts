"""Blend scores must be averaged per SLOT (row), not per batter (audit M5).

A batter in both games of a doubleheader has two rows with different game_pks
and different pitcher matchups. Keying the blend average by batter_id alone
collapses both to one shared value averaging two unrelated matchups; pick
selection between the DH games then ties arbitrarily on a wrong probability.
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

from bts.model.predict import _blend_average_by_slot


def test_blend_average_is_per_slot_not_per_batter():
    # batter 1 plays a doubleheader (2 rows, different game_pk); batter 2 one game.
    pred_df = pd.DataFrame({"batter_id": [1, 1, 2], "game_pk": [100, 200, 300]})
    m1 = pd.Series([0.30, 0.50, 0.40], index=pred_df.index)
    m2 = pd.Series([0.32, 0.52, 0.42], index=pred_df.index)

    out = _blend_average_by_slot(pred_df, [m1, m2])

    assert out.iloc[0] == pytest.approx(0.31)   # game 100
    assert out.iloc[1] == pytest.approx(0.51)   # game 200 — distinct from game 100
    assert out.iloc[2] == pytest.approx(0.41)


def test_blend_average_handles_nan_per_model():
    pred_df = pd.DataFrame({"batter_id": [1, 2], "game_pk": [100, 200]})
    m1 = pd.Series([0.40, float("nan")], index=pred_df.index)
    m2 = pd.Series([0.42, 0.60], index=pred_df.index)

    out = _blend_average_by_slot(pred_df, [m1, m2])

    assert out.iloc[0] == pytest.approx(0.41)   # both models contribute
    assert out.iloc[1] == pytest.approx(0.60)   # only the non-NaN model
