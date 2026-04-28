"""Tests for Phase 2 forward selection's feature_cols handling.

Catches Bug 1 (discovered 2026-04-28): run_selection called blend_walk_forward
without passing experiment blend_configs, so feature additions were never used
in training. Fix introduces a `compose_blend_args` helper that derives
blend_configs/lgb_params/capture from a sequence of stacked experiments,
matching what run_single_screening already does in Phase 1.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from bts.experiment.registry import load_all_experiments, get_experiment


# Synthetic profiles dataframe matching blend_walk_forward output schema
def _synthetic_profiles(season: int) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range(f"{season}-04-01", periods=10).date,
        "rank": [1] * 10,
        "batter_id": [100 + i for i in range(10)],
        "p_game_hit": [0.75] * 10,
        "actual_hit": [1] * 5 + [0] * 5,
        "n_pas": [4] * 10,
    })


# Tiny synthetic PA frame so modify_features can run without crashing
@pytest.fixture
def tiny_pa_df():
    return pd.DataFrame({
        "date": pd.date_range("2024-04-01", periods=20).date,
        "batter_id": [100, 101] * 10,
        "pitcher_id": [200, 201] * 10,
        "is_hit": [1, 0] * 10,
        "weather_temp": [85.0] * 20,
        "ump_hr": [0.1] * 20,
        "wind": [5.0] * 20,
        "hardness": [0.3] * 20,
        "indoor": [0] * 20,
    })


class TestComposeBlendArgs:
    """The new helper: compose_blend_args(experiments) -> (blend_configs, lgb_params, capture)."""

    def test_no_experiments_returns_default_blend(self):
        from bts.experiment.runner import compose_blend_args
        from bts.model.predict import BLEND_CONFIGS
        from bts.model.predict import LGB_PARAMS

        blend_configs, lgb_params, capture = compose_blend_args([])
        # With zero experiments, blend_configs ≈ default BLEND_CONFIGS
        assert len(blend_configs) == len(BLEND_CONFIGS)
        # lgb_params = default
        assert lgb_params == LGB_PARAMS
        # No capture requested
        assert capture is False

    def test_single_feature_experiment_extends_blend_features(self):
        from bts.experiment.runner import compose_blend_args
        load_all_experiments()
        heat_dome = get_experiment("heat_dome")

        blend_configs, _, _ = compose_blend_args([heat_dome])
        # Each blend config's feature list should include "heat_dome"
        for config in blend_configs:
            cols = config[1]
            assert "heat_dome" in cols, f"heat_dome missing from blend_config {config[0]}"

    def test_multiple_feature_experiments_union_features(self):
        from bts.experiment.runner import compose_blend_args
        load_all_experiments()
        heat_dome = get_experiment("heat_dome")
        bp_match = get_experiment("batter_pitcher_matchup")

        blend_configs, _, _ = compose_blend_args([heat_dome, bp_match])
        # Both new features should appear in every blend config
        for config in blend_configs:
            cols = config[1]
            assert "heat_dome" in cols
            assert "batter_pitcher_shrunk_hr" in cols

    def test_no_op_experiment_doesnt_change_features(self):
        """An experiment that doesn't override feature_cols() doesn't extend FEATURE_COLS."""
        from bts.experiment.runner import compose_blend_args
        from bts.model.predict import BLEND_CONFIGS
        load_all_experiments()
        # decision_calibration is strategy-only (modifies post-hoc, not features)
        cal = get_experiment("decision_calibration")

        blend_configs, _, _ = compose_blend_args([cal])
        # Each config should have the same feature list as the default
        for new_config, old_config in zip(blend_configs, BLEND_CONFIGS):
            assert new_config[1] == old_config[1]


class TestRunSelectionUsesFeatureCols:
    """run_selection must pass experiment-derived blend_configs to blend_walk_forward."""

    def test_candidate_call_includes_blend_configs(self, tmp_path, tiny_pa_df, monkeypatch):
        """When testing a candidate, run_selection should call blend_walk_forward
        with blend_configs reflecting that candidate's feature additions."""
        from bts.experiment.runner import run_selection
        import bts.simulate.backtest_blend as bb

        load_all_experiments()
        heat_dome = get_experiment("heat_dome")

        captured_calls = []
        def fake_blend_wf(df, season, **kwargs):
            captured_calls.append({
                "season": season,
                "blend_configs": kwargs.get("blend_configs"),
                "df_cols": list(df.columns),
            })
            return _synthetic_profiles(season)

        monkeypatch.setattr(bb, "blend_walk_forward", fake_blend_wf)

        # Synthetic scorecard so run_selection doesn't blow up
        with patch("bts.validate.scorecard.compute_full_scorecard", return_value={"p_57_mdp": 0.5}):
            with patch("bts.validate.scorecard.diff_scorecards", return_value={}):
                with patch("bts.validate.scorecard.save_scorecard", return_value=None):
                    winners = [{"name": "heat_dome", "passed": True, "diff": {}}]
                    experiments_by_name = {"heat_dome": heat_dome}
                    run_selection(
                        winners,
                        experiments_by_name,
                        tiny_pa_df,
                        test_seasons=[2024],
                        results_dir=tmp_path,
                        retrain_every=7,
                    )

        # Should have made at least 2 calls: baseline + candidate
        assert len(captured_calls) >= 2
        # The candidate call (NOT baseline) should have a non-None blend_configs
        # that includes "heat_dome" in its feature list
        candidate_calls = captured_calls[1:]  # all except first (baseline)
        assert any(
            call["blend_configs"] is not None and any(
                "heat_dome" in (c[1] if isinstance(c, tuple) else c[1])
                for c in call["blend_configs"]
            )
            for call in candidate_calls
        ), f"No candidate call had blend_configs with heat_dome. Calls: {captured_calls}"
