"""Tests for swing_daily aggregates + rolling features (campaign Stage 0)."""
import numpy as np
import pandas as pd

from bts.features.swing import (
    daily_swing_aggregates,
    rolling_swing_features,
    attach_swing_features,
    build_missingness_placebo,
    build_leaky_sentinel,
)


def _bronze(rows):
    base = {
        "game_date": "2025-06-01", "game_pk": 700001, "batter": 1, "pitcher": 9,
        "description": "swinging_strike", "miss_distance": 3.0,
        "swing_length": 7.0, "attack_angle": 10.0, "plate_z": 2.0,
        "sz_top": 3.4, "sz_bot": 1.6,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_daily_aggregates_keep_denominator_rows():
    # batter swings on a day but never whiffs -> row exists, whiff fields 0/NaN
    bronze = _bronze([
        {"description": "foul", "miss_distance": None},
        {"description": "hit_into_play", "miss_distance": None},
    ])
    daily = daily_swing_aggregates(bronze, entity="batter")
    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["n_swings"] == 2
    assert row["n_whiffs"] == 0
    assert row["n_whiffs_tracked"] == 0
    assert pd.isna(row["miss_sum"]) or row["miss_sum"] == 0


def test_daily_aggregates_distinguish_untracked_whiffs():
    bronze = _bronze([
        {"description": "swinging_strike", "miss_distance": 2.0},
        {"description": "swinging_strike", "miss_distance": None},  # whiff, no tracking
    ])
    daily = daily_swing_aggregates(bronze, entity="batter")
    row = daily.iloc[0]
    assert row["n_whiffs"] == 2
    assert row["n_whiffs_tracked"] == 1
    assert row["miss_sum"] == 2.0


def test_daily_aggregates_vertical_attack_on_whiffs():
    # plate_z above zone midline -> "over" attack
    bronze = _bronze([
        {"miss_distance": 2.0, "plate_z": 3.2},   # high
        {"miss_distance": 1.0, "plate_z": 1.7},   # low
    ])
    daily = daily_swing_aggregates(bronze, entity="pitcher")
    row = daily.iloc[0]
    assert row["n_whiff_high"] == 1
    assert row["n_whiff_low"] == 1


def test_rolling_features_are_shift1_leak_free():
    daily = pd.DataFrame({
        "batter": [1, 1, 1],
        "date": pd.to_datetime(["2025-06-01", "2025-06-02", "2025-06-03"]),
        "n_swings": [10, 10, 10],
        "n_whiffs": [2, 4, 6],
        "n_whiffs_tracked": [2, 4, 6],
        "miss_sum": [4.0, 12.0, 24.0],
        "miss_sumsq": [10.0, 40.0, 100.0],
        "swing_len_sum": [70.0, 70.0, 70.0],
        "n_swings_tracked": [10, 10, 10],
        "attack_angle_sum": [100.0, 100.0, 100.0],
        "n_whiff_high": [1, 2, 3],
        "n_whiff_low": [1, 2, 3],
    })
    feats = rolling_swing_features(daily, entity="batter", windows=[2], min_whiffs=1)
    # day 1: no prior data -> NaN
    assert pd.isna(feats.iloc[0]["batter_miss_dist_2g"])
    # day 2: only day 1 in window: 4.0/2 = 2.0
    assert feats.iloc[1]["batter_miss_dist_2g"] == 2.0
    # day 3: days 1+2: (4+12)/(2+4) = 16/6
    assert abs(feats.iloc[2]["batter_miss_dist_2g"] - 16 / 6) < 1e-9


def test_rolling_features_gate_on_min_whiffs():
    daily = pd.DataFrame({
        "batter": [1, 1],
        "date": pd.to_datetime(["2025-06-01", "2025-06-02"]),
        "n_swings": [10, 10],
        "n_whiffs": [1, 1],
        "n_whiffs_tracked": [1, 1],
        "miss_sum": [2.0, 2.0],
        "miss_sumsq": [4.0, 4.0],
        "swing_len_sum": [70.0, 70.0],
        "n_swings_tracked": [10, 10],
        "attack_angle_sum": [100.0, 100.0],
        "n_whiff_high": [1, 1],
        "n_whiff_low": [0, 0],
    })
    feats = rolling_swing_features(daily, entity="batter", windows=[2], min_whiffs=8)
    # only 1 tracked whiff in window < 8 -> gated to NaN
    assert feats["batter_miss_dist_2g"].isna().all()


def test_attach_joins_on_entity_and_date():
    pa = pd.DataFrame({
        "batter_id": [1], "pitcher_id": [9],
        "date": pd.to_datetime(["2025-06-03"]),
    })
    feats = pd.DataFrame({
        "batter": [1], "date": pd.to_datetime(["2025-06-03"]),
        "batter_miss_dist_2g": [2.5],
    })
    out = attach_swing_features(pa, batter_feats=feats, pitcher_feats=None)
    assert out.iloc[0]["batter_miss_dist_2g"] == 2.5


def test_missingness_placebo_is_boolean_flags_only():
    pa = pd.DataFrame({
        "batter_id": [1], "pitcher_id": [9],
        "date": pd.to_datetime(["2025-06-03"]),
        "batter_miss_dist_30g": [2.5],
        "pitcher_miss_dist_30g": [np.nan],
    })
    plc = build_missingness_placebo(pa, ["batter_miss_dist_30g", "pitcher_miss_dist_30g"])
    assert list(plc.columns) == ["has_batter_miss_dist_30g", "has_pitcher_miss_dist_30g"]
    assert plc.dtypes.map(lambda t: t == bool).all()
    assert plc.iloc[0]["has_batter_miss_dist_30g"] == True  # noqa: E712
    assert plc.iloc[0]["has_pitcher_miss_dist_30g"] == False  # noqa: E712


def test_leaky_sentinel_uses_same_day_data():
    daily = pd.DataFrame({
        "batter": [1, 1],
        "date": pd.to_datetime(["2025-06-01", "2025-06-02"]),
        "n_whiffs_tracked": [1, 2],
        "miss_sum": [2.0, 9.0],
    })
    pa = pd.DataFrame({
        "batter_id": [1], "pitcher_id": [9],
        "date": pd.to_datetime(["2025-06-02"]),
    })
    out = build_leaky_sentinel(pa, daily, entity="batter")
    # SAME-DAY mean miss = 9.0/2 — deliberately leaky, harness must flag it
    assert out.iloc[0]["LEAKY_same_day_miss"] == 4.5
