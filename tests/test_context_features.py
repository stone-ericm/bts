"""Tests for context_stack features added to compute_all_features."""

import numpy as np
import pandas as pd
import pytest

from bts.features.compute import compute_all_features, CONTEXT_COLS


def _make_pa_df(n=20):
    """Minimal PA DataFrame with columns needed for context features."""
    np.random.seed(42)
    dates = pd.date_range("2024-06-01", periods=5, freq="D")
    rows = []
    for i in range(n):
        d = dates[i % len(dates)]
        rows.append({
            "date": d, "season": 2024, "game_pk": 700000 + (i % 5),
            "batter_id": 100 + (i % 4), "pitcher_id": 200 + (i % 3),
            "is_hit": int(np.random.random() > 0.7), "is_home": i % 2 == 0,
            "venue_id": 1, "pitch_hand": "R",
            "final_count_balls": 2, "final_count_strikes": 1,
            "launch_speed": 90.0 + np.random.random() * 20,
            "launch_angle": 10.0 + np.random.random() * 20,
            "pitch_calls": ["B", "S", "X"], "pitch_types": ["FF", "SL"],
            "pitch_speeds": [92.0, 85.0], "pitch_spin_rates": [2200, 2500],
            "pitch_extensions": [6.2, 6.1],
            "pitch_break_vertical": [-15.0, -30.0],
            "pitch_break_horizontal": [8.0, 3.0],
            "pitch_px": [0.3, -0.8], "pitch_pz": [2.5, 3.1],
            "sz_top": 3.4, "sz_bottom": 1.6,
            "weather_temp": 72, "weather_wind_dir": "Out To CF",
            "weather_wind_speed": 12.0, "roof_type": "Open",
            "hp_umpire_id": 300 + (i % 2),
            "hardness": np.random.choice(["hard", "medium", "soft", None]),
        })
    return pd.DataFrame(rows)


class TestContextCols:
    def test_context_cols_has_4_entries(self):
        assert len(CONTEXT_COLS) == 4

    def test_context_cols_names(self):
        assert CONTEXT_COLS == [
            "ump_hr_30g", "wind_out_cf",
            "batter_hard_contact_30g", "is_indoor",
        ]


class TestUmpireHitRate:
    def test_column_present_after_compute(self):
        df = compute_all_features(_make_pa_df(50))
        assert "ump_hr_30g" in df.columns

    def test_values_between_0_and_1(self):
        df = compute_all_features(_make_pa_df(50))
        valid = df["ump_hr_30g"].dropna()
        if len(valid) > 0:
            assert valid.min() >= 0.0
            assert valid.max() <= 1.0


class TestWindVector:
    def test_column_present(self):
        df = compute_all_features(_make_pa_df())
        assert "wind_out_cf" in df.columns

    def test_out_to_cf_positive(self):
        df = _make_pa_df()
        df["weather_wind_dir"] = "Out To CF"
        df["weather_wind_speed"] = 10.0
        result = compute_all_features(df)
        assert (result["wind_out_cf"] > 0).all()

    def test_in_from_cf_negative(self):
        df = _make_pa_df()
        df["weather_wind_dir"] = "In From CF"
        df["weather_wind_speed"] = 10.0
        result = compute_all_features(df)
        assert (result["wind_out_cf"] < 0).all()

    def test_calm_is_zero(self):
        df = _make_pa_df()
        df["weather_wind_dir"] = "Calm"
        df["weather_wind_speed"] = 0.0
        result = compute_all_features(df)
        assert (result["wind_out_cf"] == 0).all()


class TestBatterHardContact:
    def test_column_present(self):
        df = compute_all_features(_make_pa_df(50))
        assert "batter_hard_contact_30g" in df.columns

    def test_values_between_0_and_1(self):
        df = compute_all_features(_make_pa_df(50))
        valid = df["batter_hard_contact_30g"].dropna()
        if len(valid) > 0:
            assert valid.min() >= 0.0
            assert valid.max() <= 1.0

    def test_does_not_sort_df_by_batter_id(self):
        # Regression: compute_all_features previously sorted df by batter_id
        # inside the hard-contact rolling block. LightGBM's subsample=0.8 with
        # random_state=42 picks rows by index, so a reordered training frame
        # silently picked different rows and produced a ~1pp P@1 regression.
        # Bisect closed at 30ed4fa on 2026-04-13.
        df = _make_pa_df(50)
        out = compute_all_features(df)
        assert not out["batter_id"].is_monotonic_increasing, (
            "compute_all_features output is sorted by batter_id — this "
            "scrambles LightGBM bagging picks and breaks reproducibility"
        )


class TestIsIndoor:
    def test_column_present(self):
        df = compute_all_features(_make_pa_df())
        assert "is_indoor" in df.columns

    def test_open_is_zero(self):
        df = _make_pa_df()
        df["roof_type"] = "Open"
        result = compute_all_features(df)
        assert (result["is_indoor"] == 0).all()

    def test_dome_is_one(self):
        df = _make_pa_df()
        df["roof_type"] = "Dome"
        result = compute_all_features(df)
        assert (result["is_indoor"] == 1).all()

    def test_retractable_is_one(self):
        df = _make_pa_df()
        df["roof_type"] = "Retractable"
        result = compute_all_features(df)
        assert (result["is_indoor"] == 1).all()
