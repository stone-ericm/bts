"""Tests for per-PA prediction persistence in blend_walk_forward."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


class TestWritePAPredictionsChunk:
    def test_creates_parquet_with_expected_columns(self, tmp_path: Path):
        """When output_path doesn't exist, creates it with the 6 expected columns."""
        from bts.simulate.backtest_blend import _write_pa_predictions_chunk

        day_data = pd.DataFrame({
            "date": [pd.Timestamp("2024-04-01")] * 6,
            "game_pk": [100, 100, 100, 200, 200, 200],
            "batter_id": [1, 1, 1, 2, 2, 2],
            "p_hit_blend": [0.25, 0.30, 0.22, 0.18, 0.27, 0.31],
            "is_hit": [0, 1, 0, 0, 0, 1],
            # additional cols that should NOT be persisted:
            "n_pas": [3, 3, 3, 3, 3, 3],
        })
        out = tmp_path / "pa_predictions.parquet"
        _write_pa_predictions_chunk(day_data, out)

        assert out.exists()
        loaded = pd.read_parquet(out)
        assert set(loaded.columns) == {
            "date", "game_pk", "batter_id", "pa_index", "p_hit_blend", "is_hit",
        }
        assert len(loaded) == 6
        # pa_index should be 0..2 within each (batter_id, game_pk) group.
        first_group = loaded[(loaded["batter_id"] == 1) & (loaded["game_pk"] == 100)]
        assert sorted(first_group["pa_index"].tolist()) == [0, 1, 2]

    def test_appends_when_path_already_exists(self, tmp_path: Path):
        """Calling twice with different days produces a single concatenated file."""
        from bts.simulate.backtest_blend import _write_pa_predictions_chunk

        out = tmp_path / "pa_predictions.parquet"
        day1 = pd.DataFrame({
            "date": [pd.Timestamp("2024-04-01")] * 2,
            "game_pk": [100, 100],
            "batter_id": [1, 1],
            "p_hit_blend": [0.20, 0.25],
            "is_hit": [0, 1],
        })
        day2 = pd.DataFrame({
            "date": [pd.Timestamp("2024-04-02")] * 2,
            "game_pk": [101, 101],
            "batter_id": [2, 2],
            "p_hit_blend": [0.15, 0.18],
            "is_hit": [1, 0],
        })
        _write_pa_predictions_chunk(day1, out)
        _write_pa_predictions_chunk(day2, out)

        loaded = pd.read_parquet(out)
        assert len(loaded) == 4
        assert sorted(loaded["date"].astype(str).unique().tolist()) == [
            "2024-04-01", "2024-04-02",
        ]

    def test_creates_parent_directory_if_missing(self, tmp_path: Path):
        """When output_path's parent dir doesn't exist yet, helper creates it.

        Guards against the failure mode where a long-running backtest writes its
        first PA chunk into a path under a not-yet-created directory and crashes
        with FileNotFoundError after hours of work.
        """
        from bts.simulate.backtest_blend import _write_pa_predictions_chunk

        nested_path = tmp_path / "nested" / "subdir" / "pa_predictions.parquet"
        assert not nested_path.parent.exists()

        day_data = pd.DataFrame({
            "date": [pd.Timestamp("2024-04-01")] * 2,
            "game_pk": [100, 100],
            "batter_id": [1, 1],
            "p_hit_blend": [0.20, 0.25],
            "is_hit": [0, 1],
        })
        _write_pa_predictions_chunk(day_data, nested_path)

        assert nested_path.exists()
        loaded = pd.read_parquet(nested_path)
        assert len(loaded) == 2
