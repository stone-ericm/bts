"""Tests for the Savant swing-data bronze pull layer (Stage 0)."""
import json

import pandas as pd

from bts.data.swing_pull import (
    BRONZE_COLUMNS,
    normalize_bronze,
    write_bronze_season,
)


def _raw_statcast_frame():
    # A minimal frame shaped like pybaseball.statcast() output
    return pd.DataFrame({
        "game_date": ["2025-06-01", "2025-06-01"],
        "game_pk": [700001, 700001],
        "at_bat_number": [1, 1],
        "pitch_number": [1, 2],
        "batter": [665742, 665742],
        "pitcher": [594798, 594798],
        "events": [None, "strikeout"],
        "description": ["ball", "swinging_strike"],
        "type": ["B", "S"],
        "pitch_type": ["FF", "SL"],
        "game_type": ["R", "R"],
        "balls": [0, 1],
        "strikes": [0, 0],
        "stand": ["L", "L"],
        "p_throws": ["R", "R"],
        "zone": [13, 6],
        "plate_x": [0.9, 0.1],
        "plate_z": [1.1, 2.4],
        "sz_top": [3.4, 3.4],
        "sz_bot": [1.6, 1.6],
        "miss_distance": [None, 2.5],
        "swing_length": [None, 7.2],
        "attack_angle": [None, 11.0],
        "attack_direction": [None, -4.0],
        "swing_path_tilt": [None, 32.0],
        "intercept_ball_minus_batter_pos_x_inches": [None, 28.0],
        "intercept_ball_minus_batter_pos_y_inches": [None, 33.0],
        "unrelated_savant_column": ["x", "y"],
    })


def test_normalize_keeps_bronze_columns_and_drops_rest():
    out = normalize_bronze(_raw_statcast_frame())
    assert "unrelated_savant_column" not in out.columns
    assert set(out.columns) <= set(BRONZE_COLUMNS)
    # core ids always present
    for col in ["game_date", "game_pk", "batter", "pitcher", "description", "miss_distance"]:
        assert col in out.columns


def test_normalize_tolerates_missing_columns():
    raw = _raw_statcast_frame().drop(columns=["swing_path_tilt", "sz_top"])
    out = normalize_bronze(raw)
    # absent columns are created as NA so season files share one schema
    assert "swing_path_tilt" in out.columns
    assert out["swing_path_tilt"].isna().all()


def test_normalize_filters_to_regular_season():
    raw = _raw_statcast_frame()
    raw.loc[0, "game_type"] = "S"  # spring training
    out = normalize_bronze(raw)
    assert (out["game_type"] == "R").all()
    assert len(out) == 1


def test_write_bronze_season_writes_parquet_and_manifest(tmp_path):
    df = normalize_bronze(_raw_statcast_frame())
    path = write_bronze_season(df, 2025, tmp_path, raw_columns=list(_raw_statcast_frame().columns))

    assert path == tmp_path / "swing_2025.parquet"
    back = pd.read_parquet(path)
    assert len(back) == len(df)
    manifest = json.loads((tmp_path / "swing_2025.manifest.json").read_text())
    assert manifest["season"] == 2025
    assert manifest["n_rows"] == len(df)
    assert "unrelated_savant_column" in manifest["raw_columns"]
    assert "pulled_at" in manifest


from bts.data.swing_pull import month_chunks


def test_month_chunks_cover_range_without_overlap():
    chunks = month_chunks("2023-07-14", "2023-09-02")
    assert chunks[0] == ("2023-07-14", "2023-07-31")
    assert chunks[1] == ("2023-08-01", "2023-08-31")
    assert chunks[-1] == ("2023-09-01", "2023-09-02")
    # contiguous, no overlap
    for (a_start, a_end), (b_start, b_end) in zip(chunks, chunks[1:]):
        assert pd.Timestamp(b_start) == pd.Timestamp(a_end) + pd.Timedelta(days=1)
