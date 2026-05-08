from __future__ import annotations

import json

import pandas as pd

from bts.experiment.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    PROFILE_SCHEMA_COLUMNS,
    compare_candidate_profile_pair,
    materialize_candidate_profile_pair,
)
from bts.experiment.models import DecisionWeightedLightGBMExperiment


def _fake_profiles(season: int, *, candidate: bool) -> pd.DataFrame:
    base = 0.72 if candidate else 0.62
    rows = []
    for day_idx, date in enumerate(pd.date_range(f"{season}-04-01", periods=2)):
        for rank in (1, 2):
            rows.append({
                "date": date.date(),
                "rank": rank,
                "batter_id": 1000 + day_idx * 10 + rank,
                "game_pk": 9000 + day_idx,
                "p_game_hit": base - rank * 0.02,
                "actual_hit": int(rank == 1),
                "n_pas": 4,
            })
    return pd.DataFrame(rows)


def test_materialize_candidate_profile_pair_writes_manifest_and_profiles(
    mini_pa_df,
    tmp_path,
    monkeypatch,
):
    import bts.simulate.backtest_blend as bb

    calls = []

    def fake_walk_forward(df, season, **kwargs):
        calls.append((season, kwargs))
        blend_configs = kwargs["blend_configs"]
        is_candidate = any(
            len(config) == 3
            and config[2].get("decision_weight_mode") == "top_slate_v0"
            for config in blend_configs
        )
        return _fake_profiles(season, candidate=is_candidate)

    monkeypatch.setattr(bb, "blend_walk_forward", fake_walk_forward)
    manifest = materialize_candidate_profile_pair(
        pa_df=mini_pa_df,
        candidate=DecisionWeightedLightGBMExperiment(),
        seasons=[2024],
        output_dir=tmp_path,
        retrain_every=3,
        top_n=2,
        data_dir="data/processed",
        git_commit="abc123",
        generated_at="2026-05-08T00:00:00+00:00",
    )

    assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["production_deploy_claim"] is False
    assert manifest["fresh_target_claim"] is False
    assert manifest["candidate_name"] == "decision_weighted_lgbm_v0"
    assert manifest["seasons"] == [2024]
    assert len(calls) == 2

    manifest_path = tmp_path / "manifest.json"
    assert json.loads(manifest_path.read_text())["schema_version"] == ARTIFACT_SCHEMA_VERSION

    production_path = tmp_path / manifest["profile_paths"]["production"]["2024"]
    candidate_path = tmp_path / manifest["profile_paths"]["candidate"]["2024"]
    production = pd.read_parquet(production_path)
    candidate = pd.read_parquet(candidate_path)

    assert list(production.columns) == PROFILE_SCHEMA_COLUMNS
    assert list(candidate.columns) == PROFILE_SCHEMA_COLUMNS
    assert production["variant"].unique().tolist() == ["production"]
    assert candidate["variant"].unique().tolist() == ["candidate"]
    assert production["model_name"].unique().tolist() == ["production_lgbm_v0"]
    assert candidate["model_name"].unique().tolist() == ["decision_weighted_lgbm_v0"]
    assert candidate["p_game_hit"].mean() > production["p_game_hit"].mean()


def test_compare_candidate_profile_pair_saves_scorecards_and_primary_delta(
    mini_pa_df,
    tmp_path,
    monkeypatch,
):
    import bts.simulate.backtest_blend as bb
    import bts.validate.scorecard as scorecard_mod

    def fake_walk_forward(df, season, **kwargs):
        blend_configs = kwargs["blend_configs"]
        is_candidate = any(
            len(config) == 3
            and config[2].get("decision_weight_mode") == "top_slate_v0"
            for config in blend_configs
        )
        return _fake_profiles(season, candidate=is_candidate)

    def fake_scorecard(profiles, mc_trials=10_000, season_length=180):
        is_candidate = profiles["variant"].iloc[0] == "candidate"
        return {
            "p_57_mdp": 0.25 if is_candidate else 0.10,
            "precision": {1: 0.9 if is_candidate else 0.8},
            "p_at_1_by_season": {2024: 0.9 if is_candidate else 0.8},
        }

    monkeypatch.setattr(bb, "blend_walk_forward", fake_walk_forward)
    monkeypatch.setattr(scorecard_mod, "compute_full_scorecard", fake_scorecard)
    materialize_candidate_profile_pair(
        pa_df=mini_pa_df,
        candidate=DecisionWeightedLightGBMExperiment(),
        seasons=[2024],
        output_dir=tmp_path,
        git_commit="abc123",
        generated_at="2026-05-08T00:00:00+00:00",
    )

    comparison = compare_candidate_profile_pair(
        artifact_dir=tmp_path,
        mc_trials=123,
        season_length=162,
        generated_at="2026-05-08T01:00:00+00:00",
    )

    assert comparison["primary_metric"] == "p_57_mdp"
    assert comparison["primary_delta"] == 0.15
    assert comparison["production_deploy_claim"] is False
    assert comparison["scorecards"]["production"]["p_57_mdp"] == 0.10
    assert comparison["scorecards"]["candidate"]["p_57_mdp"] == 0.25
    saved = json.loads((tmp_path / "comparison.json").read_text())
    assert saved["primary_delta"] == 0.15
