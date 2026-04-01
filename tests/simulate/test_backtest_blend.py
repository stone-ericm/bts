"""Tests for blend walk-forward backtest output."""

import pandas as pd
import pytest


class TestBlendBacktestOutput:
    def test_output_schema(self):
        """Verify the output parquet has the expected columns."""
        from bts.simulate.backtest_blend import PROFILE_COLUMNS
        assert PROFILE_COLUMNS == ["date", "rank", "batter_id", "p_game_hit", "actual_hit", "n_pas"]

    def test_load_saved_profiles(self, tmp_path):
        """Round-trip: save profiles, load them back."""
        from bts.simulate.backtest_blend import save_profiles
        from bts.simulate.monte_carlo import load_all_profiles

        df = pd.DataFrame({
            "date": ["2024-04-01"] * 10,
            "rank": list(range(1, 11)),
            "batter_id": [i * 1000 for i in range(1, 11)],
            "p_game_hit": [0.90 - i * 0.02 for i in range(10)],
            "actual_hit": [1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
            "n_pas": [4] * 10,
        })
        save_profiles(df, 2024, tmp_path)
        loaded = load_all_profiles(tmp_path)
        assert len(loaded) == 1  # 1 day
        assert loaded[0].top1_p == df.iloc[0]["p_game_hit"]
