"""Runner tests for the season-level selection/outer-evaluation split."""

from __future__ import annotations

import json

import pandas as pd

from bts.experiment.base import ExperimentDef


def _split_metadata() -> dict:
    return {
        "split_mode": "season_level_selection_outer_eval",
        "selection_seasons": [2023],
        "outer_eval_seasons": [2025],
        "lockbox_used": False,
        "lockbox_manifest": None,
        "production_deploy_claim": False,
    }


def test_run_selection_uses_outer_eval_only_after_selection(tmp_path, monkeypatch):
    from bts.experiment.runner import run_selection

    seasons_seen: list[int] = []
    p57_values = iter([0.10, 0.20, 0.10, 0.20, 0.30, 0.40])

    def fake_walk_forward(df, season, **kwargs):
        seasons_seen.append(season)
        return pd.DataFrame({
            "date": pd.date_range(f"{season}-04-01", periods=2).date,
            "rank": [1, 1],
            "batter_id": [100, 101],
            "p_game_hit": [0.6, 0.7],
            "actual_hit": [1, 0],
            "n_pas": [4, 4],
        })

    def fake_scorecard(profiles):
        return {
            "p_57_mdp": next(p57_values),
            "precision": {1: 0.5},
            "p_at_1_by_season": {
                int(profiles["season"].iloc[0]): 0.5,
            },
            "streak_metrics": {"mean_max_streak": 10.0},
        }

    monkeypatch.setattr("bts.simulate.backtest_blend.blend_walk_forward", fake_walk_forward)
    monkeypatch.setattr("bts.validate.scorecard.compute_full_scorecard", fake_scorecard)

    exp = ExperimentDef(
        name="heat_dome",
        phase=1,
        category="feature",
        description="fake",
    )
    result = run_selection(
        winners=[{"name": "heat_dome", "passed": True, "diff": {}}],
        experiments_by_name={"heat_dome": exp},
        pa_df=pd.DataFrame({"batter_id": [1]}),
        test_seasons=[2023],
        results_dir=tmp_path,
        outer_eval_seasons=[2025],
        split_metadata=_split_metadata(),
    )

    assert seasons_seen == [2023, 2023, 2023, 2023, 2025, 2025]
    assert result["included"] == ["heat_dome"]
    assert result["forward_log"][0]["validation_split"]["artifact_role"] == "selection_only"
    assert result["backward_log"][0]["validation_split"]["artifact_role"] == "selection_only"
    assert result["final_scorecard"]["validation_split"]["artifact_role"] == "selection_only"
    assert result["outer_eval_scorecard"]["validation_split"]["artifact_role"] == "outer_evaluation"
    assert result["outer_eval_scorecard"]["validation_split"]["selection_seasons"] == [2023]
    assert result["outer_eval_scorecard"]["validation_split"]["outer_eval_seasons"] == [2025]
    assert result["outer_eval_scorecard"]["validation_split"]["production_deploy_claim"] is False

    saved_outer = json.loads((tmp_path / "outer_eval_scorecard.json").read_text())
    assert saved_outer["validation_split"]["artifact_role"] == "outer_evaluation"
    saved_final = json.loads((tmp_path / "final_scorecard.json").read_text())
    assert saved_final["validation_split"]["artifact_role"] == "selection_only"


def test_run_single_screening_factored_path_receives_selection_seasons(
    tmp_path, monkeypatch,
):
    from bts.experiment import runner_factored as rf
    from bts.experiment.runner import run_single_screening

    seasons_seen: list[list[int]] = []

    def fake_model_swap(
        experiment, pa_df, baseline_scorecard, test_seasons, results_dir,
        retrain_every=7, cache_dir=None,
    ):
        seasons_seen.append(list(test_seasons))
        return {
            "name": experiment.name,
            "scorecard": {"p_57_mdp": 0.2},
            "diff": {"p_57_mdp": {"delta": 0.1}},
            "passed": True,
            "reason": "fake",
        }

    monkeypatch.setattr(rf, "_is_eligible_for_strategy_fast_path", lambda exp: (False, ""))
    monkeypatch.setattr(rf, "_is_eligible_for_model_swap_fast_path", lambda exp: (True, ""))
    monkeypatch.setattr(rf, "run_model_swap_experiment_fast", fake_model_swap)

    exp = ExperimentDef(
        name="fake_model_swap",
        phase=1,
        category="model",
        description="fake",
    )
    result = run_single_screening(
        exp,
        pa_df=pd.DataFrame(),
        baseline_scorecard={},
        test_seasons=[2023],
        results_dir=tmp_path,
        use_factored=True,
        split_metadata=_split_metadata(),
    )

    assert seasons_seen == [[2023]]
    assert result["scorecard"]["validation_split"]["artifact_role"] == "selection_only"
    saved = json.loads((tmp_path / "fake_model_swap" / "scorecard.json").read_text())
    assert saved["validation_split"]["selection_seasons"] == [2023]
