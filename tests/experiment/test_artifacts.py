from __future__ import annotations

import json

import pandas as pd
import pytest

from bts.experiment.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    OUTCOME_STATUS_RESOLVED,
    OUTCOME_STATUS_VOID_NO_PA,
    OUTCOME_STATUS_VOID_POSTPONEMENT,
    PROFILE_SCHEMA_COLUMNS,
    PRODUCTION_PICK_SNAPSHOT_VERSION,
    RESOLVED_ARTIFACT_SCHEMA_VERSION,
    RESOLVED_PROFILE_SCHEMA_COLUMNS,
    compare_candidate_profile_pair,
    materialize_candidate_profile_pair,
    materialize_live_candidate_profile_pair,
    resolve_live_candidate_artifact_pair,
    verify_candidate_artifact_pair,
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


def _write_live_preoutcome_artifact(artifact_dir, *, date: str = "2026-05-09") -> dict:
    profile_paths = {
        "production": {date: f"profiles/production/live_{date}.parquet"},
        "candidate": {date: f"profiles/candidate/live_{date}.parquet"},
    }
    for variant, base, model_name in (
        ("production", 0.64, "production_lgbm_v0"),
        ("candidate", 0.74, "decision_weighted_lgbm_v0"),
    ):
        frame = pd.DataFrame({
            "artifact_schema_version": [ARTIFACT_SCHEMA_VERSION, ARTIFACT_SCHEMA_VERSION],
            "run_kind": ["live_forward_preoutcome", "live_forward_preoutcome"],
            "variant": [variant, variant],
            "model_name": [model_name, model_name],
            "generated_at": [
                "2026-05-08T02:00:00+00:00",
                "2026-05-08T02:00:00+00:00",
            ],
            "git_commit": ["def456", "def456"],
            "date": [pd.Timestamp(date).date(), pd.Timestamp(date).date()],
            "season": [pd.Timestamp(date).year, pd.Timestamp(date).year],
            "rank": [1, 2],
            "batter_id": [11, 22],
            "game_pk": [1001, 1002],
            "p_game_hit": [base, base - 0.05],
            "actual_hit": [pd.NA, pd.NA],
            "n_pas": [pd.NA, pd.NA],
        })[PROFILE_SCHEMA_COLUMNS]
        path = artifact_dir / profile_paths[variant][date]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": "2026-05-08T02:00:00+00:00",
        "git_commit": "def456",
        "run_kind": "live_forward_preoutcome",
        "production_deploy_claim": False,
        "fresh_target_claim": True,
        "candidate_name": "decision_weighted_lgbm_v0",
        "baseline_name": "production_lgbm_v0",
        "date": date,
        "dates": [date],
        "seasons": [pd.Timestamp(date).year],
        "top_n": 2,
        "retrain_every": None,
        "profile_schema_columns": list(PROFILE_SCHEMA_COLUMNS),
        "profile_paths": profile_paths,
        "row_counts": {"production": {date: 2}, "candidate": {date: 2}},
        "day_counts": {"production": {date: 1}, "candidate": {date: 1}},
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _write_pick_file(path, *, date: str = "2026-05-09"):
    body = {
        "date": date,
        "run_time": f"{date}T18:00:00+00:00",
        "pick": {
            "batter_id": 11,
            "batter_name": "Primary Batter",
            "team": "AAA",
            "game_pk": 1001,
            "p_game_hit": 0.72,
            "projected_lineup": False,
        },
        "double_down": {
            "batter_id": 22,
            "batter_name": "Double Batter",
            "team": "BBB",
            "game_pk": 1002,
            "p_game_hit": 0.69,
            "projected_lineup": False,
        },
        "slot_results": {},
        "model_git_sha": "model-sha",
        "model_pickle_sha256": "pickle-sha",
        "policy_npz_sha256": "policy-sha",
        "production_lgbm_deterministic": False,
    }
    path.write_text(json.dumps(body))
    return path


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


def test_compare_candidate_profile_pair_uses_common_resolved_dates(
    tmp_path,
    monkeypatch,
):
    import bts.validate.scorecard as scorecard_mod

    generated_at = "2026-05-11T00:00:00+00:00"
    profile_paths = {
        "production": {"2026": "profiles/production/backtest_2026.parquet"},
        "candidate": {"2026": "profiles/candidate/backtest_2026.parquet"},
    }

    def frame_for(variant: str) -> pd.DataFrame:
        statuses = [
            OUTCOME_STATUS_RESOLVED,
            OUTCOME_STATUS_RESOLVED,
            OUTCOME_STATUS_RESOLVED,
            (
                OUTCOME_STATUS_VOID_NO_PA
                if variant == "production"
                else OUTCOME_STATUS_RESOLVED
            ),
        ]
        actual_hit = [1, 0, 1, pd.NA if variant == "production" else 1]
        n_pas = [4, 4, 4, pd.NA if variant == "production" else 4]
        return pd.DataFrame({
            "artifact_schema_version": [RESOLVED_ARTIFACT_SCHEMA_VERSION] * 4,
            "run_kind": ["live_forward_resolved"] * 4,
            "variant": [variant] * 4,
            "model_name": [
                "production_lgbm_v0" if variant == "production"
                else "decision_weighted_lgbm_v0"
            ] * 4,
            "generated_at": [generated_at] * 4,
            "git_commit": ["def456"] * 4,
            "date": [
                pd.Timestamp("2026-05-09").date(),
                pd.Timestamp("2026-05-09").date(),
                pd.Timestamp("2026-05-10").date(),
                pd.Timestamp("2026-05-10").date(),
            ],
            "season": [2026] * 4,
            "rank": [1, 2, 1, 2],
            "batter_id": [11, 22, 33, 44],
            "game_pk": [1001, 1002, 1003, 1004],
            "p_game_hit": [0.7, 0.65, 0.68, 0.64],
            "actual_hit": actual_hit,
            "n_pas": n_pas,
            "outcome_status": statuses,
        })[RESOLVED_PROFILE_SCHEMA_COLUMNS]

    for variant in ("production", "candidate"):
        path = tmp_path / profile_paths[variant]["2026"]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame_for(variant).to_parquet(path, index=False)

    manifest = {
        "schema_version": RESOLVED_ARTIFACT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "git_commit": "def456",
        "run_kind": "live_forward_resolved",
        "production_deploy_claim": False,
        "fresh_target_claim": True,
        "candidate_name": "decision_weighted_lgbm_v0",
        "baseline_name": "production_lgbm_v0",
        "date": "2026-05-09",
        "dates": ["2026-05-09", "2026-05-10"],
        "seasons": [2026],
        "top_n": 2,
        "profile_schema_columns": list(RESOLVED_PROFILE_SCHEMA_COLUMNS),
        "profile_paths": profile_paths,
        "row_counts": {"production": {"2026": 4}, "candidate": {"2026": 4}},
        "day_counts": {"production": {"2026": 2}, "candidate": {"2026": 2}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

    seen = {}

    def fake_scorecard(profiles, mc_trials=10_000, season_length=180):
        variant = profiles["variant"].iloc[0]
        seen[variant] = sorted(str(date) for date in profiles["date"].unique())
        return {
            "p_57_mdp": 0.25 if variant == "candidate" else 0.10,
        }

    monkeypatch.setattr(scorecard_mod, "compute_full_scorecard", fake_scorecard)
    comparison = compare_candidate_profile_pair(artifact_dir=tmp_path)

    assert seen == {
        "production": ["2026-05-09"],
        "candidate": ["2026-05-09"],
    }
    assert comparison["outcome_status_filter"]["paired_date_filter"] == {
        "common_scorecard_dates": 1,
        "production_only_dates_dropped": [],
        "candidate_only_dates_dropped": ["2026-05-10"],
        "policy": (
            "Candidate and production scorecards are evaluated on the same "
            "post-filter date set."
        ),
    }


def test_materialize_live_candidate_profile_pair_writes_preoutcome_profiles(
    tmp_path,
    monkeypatch,
):
    import bts.model.predict as predict_mod

    calls = []

    def fake_run_pipeline(date, **kwargs):
        calls.append(kwargs)
        is_candidate = kwargs.get("blend_configs_override") is not None
        base = 0.74 if is_candidate else 0.64
        return pd.DataFrame({
            "batter_id": [11, 22, 33],
            "game_pk": [1001, 1002, 1003],
            "p_game_hit": [base, base - 0.05, base - 0.10],
        })

    monkeypatch.setattr(predict_mod, "run_pipeline", fake_run_pipeline)
    pick_file = _write_pick_file(tmp_path / "2026-05-09.json")
    manifest = materialize_live_candidate_profile_pair(
        date="2026-05-09",
        candidate=DecisionWeightedLightGBMExperiment(),
        output_dir=tmp_path,
        data_dir="data/processed",
        top_n=2,
        refresh_data=False,
        production_pick_file=pick_file,
        git_commit="def456",
        generated_at="2026-05-08T02:00:00+00:00",
    )

    assert manifest["run_kind"] == "live_forward_preoutcome"
    assert manifest["fresh_target_claim"] is True
    assert manifest["production_deploy_claim"] is False
    assert manifest["dates"] == ["2026-05-09"]
    assert len(calls) == 2
    assert calls[0]["refresh_data"] is False
    assert calls[1]["refresh_data"] is False
    assert calls[1]["blend_configs_override"] is not None
    assert manifest["production_pick_snapshot"]["date"] == "2026-05-09"
    assert (
        manifest["production_pick_snapshot"]["snapshot_version"]
        == PRODUCTION_PICK_SNAPSHOT_VERSION
    )
    assert manifest["production_pick_snapshot"]["slots"]["pick"]["batter_id"] == 11
    assert manifest["production_pick_snapshot"]["policy_npz_sha256"] == "policy-sha"
    assert manifest["production_pick_snapshot"]["production_pick_json"]["date"] == "2026-05-09"
    assert manifest["production_pick_snapshot"]["production_lgbm_deterministic"] is False

    production = pd.read_parquet(tmp_path / manifest["profile_paths"]["production"]["2026-05-09"])
    candidate = pd.read_parquet(tmp_path / manifest["profile_paths"]["candidate"]["2026-05-09"])
    assert list(production.columns) == PROFILE_SCHEMA_COLUMNS
    assert production["actual_hit"].isna().all()
    assert production["n_pas"].isna().all()
    assert candidate["variant"].unique().tolist() == ["candidate"]
    assert candidate["p_game_hit"].mean() > production["p_game_hit"].mean()


def test_verify_candidate_artifact_pair_accepts_live_preoutcome_artifact(
    tmp_path,
    monkeypatch,
):
    import bts.model.predict as predict_mod

    def fake_run_pipeline(date, **kwargs):
        is_candidate = kwargs.get("blend_configs_override") is not None
        base = 0.74 if is_candidate else 0.64
        return pd.DataFrame({
            "batter_id": [11, 22],
            "game_pk": [1001, 1002],
            "p_game_hit": [base, base - 0.05],
        })

    monkeypatch.setattr(predict_mod, "run_pipeline", fake_run_pipeline)
    pick_file = _write_pick_file(tmp_path / "2026-05-09.json")
    materialize_live_candidate_profile_pair(
        date="2026-05-09",
        candidate=DecisionWeightedLightGBMExperiment(),
        output_dir=tmp_path,
        top_n=2,
        production_pick_file=pick_file,
        git_commit="def456",
        generated_at="2026-05-08T02:00:00+00:00",
    )

    report = verify_candidate_artifact_pair(
        artifact_dir=tmp_path,
        expected_run_kind="live_forward_preoutcome",
        expected_candidate="decision_weighted_lgbm_v0",
        expected_date="2026-05-09",
        expected_git_commit="def456",
        expected_top_n=2,
        require_live_preoutcome=True,
        require_production_pick_snapshot=True,
        generated_at="2026-05-09T00:00:00+00:00",
    )

    assert report["ok"] is True
    assert report["failure_count"] == 0
    assert report["manifest"]["git_commit"] == "def456"
    assert report["manifest"]["has_production_pick_snapshot"] is True
    assert report["variants"]["production"]["rows"] == 2
    assert report["variants"]["candidate"]["dates"] == ["2026-05-09"]


def test_verify_candidate_artifact_pair_flags_wrong_git_commit(
    tmp_path,
    monkeypatch,
):
    import bts.model.predict as predict_mod

    def fake_run_pipeline(date, **kwargs):
        return pd.DataFrame({
            "batter_id": [11, 22],
            "game_pk": [1001, 1002],
            "p_game_hit": [0.64, 0.59],
        })

    monkeypatch.setattr(predict_mod, "run_pipeline", fake_run_pipeline)
    materialize_live_candidate_profile_pair(
        date="2026-05-09",
        candidate=DecisionWeightedLightGBMExperiment(),
        output_dir=tmp_path,
        top_n=2,
        git_commit="def456",
        generated_at="2026-05-08T02:00:00+00:00",
    )

    report = verify_candidate_artifact_pair(
        artifact_dir=tmp_path,
        expected_git_commit="wrong",
        require_live_preoutcome=True,
    )

    assert report["ok"] is False
    failed_names = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert "expected_git_commit" in failed_names


def test_verify_candidate_artifact_pair_flags_missing_pick_snapshot(
    tmp_path,
    monkeypatch,
):
    import bts.model.predict as predict_mod

    def fake_run_pipeline(date, **kwargs):
        return pd.DataFrame({
            "batter_id": [11, 22],
            "game_pk": [1001, 1002],
            "p_game_hit": [0.64, 0.59],
        })

    monkeypatch.setattr(predict_mod, "run_pipeline", fake_run_pipeline)
    materialize_live_candidate_profile_pair(
        date="2026-05-09",
        candidate=DecisionWeightedLightGBMExperiment(),
        output_dir=tmp_path,
        top_n=2,
        git_commit="def456",
        generated_at="2026-05-08T02:00:00+00:00",
    )

    report = verify_candidate_artifact_pair(
        artifact_dir=tmp_path,
        require_live_preoutcome=True,
        require_production_pick_snapshot=True,
    )

    assert report["ok"] is False
    failed_names = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert "production_pick_snapshot_present" in failed_names


def test_verify_candidate_artifact_pair_reports_missing_manifest(tmp_path):
    report = verify_candidate_artifact_pair(artifact_dir=tmp_path)

    assert report["ok"] is False
    assert report["failure_count"] == 1
    assert report["checks"][0]["name"] == "manifest_exists"


def test_verify_candidate_artifact_pair_flags_non_null_live_outcomes(
    tmp_path,
    monkeypatch,
):
    import bts.model.predict as predict_mod

    def fake_run_pipeline(date, **kwargs):
        return pd.DataFrame({
            "batter_id": [11, 22],
            "game_pk": [1001, 1002],
            "p_game_hit": [0.64, 0.59],
        })

    monkeypatch.setattr(predict_mod, "run_pipeline", fake_run_pipeline)
    manifest = materialize_live_candidate_profile_pair(
        date="2026-05-09",
        candidate=DecisionWeightedLightGBMExperiment(),
        output_dir=tmp_path,
        top_n=2,
        git_commit="def456",
        generated_at="2026-05-08T02:00:00+00:00",
    )
    production_path = tmp_path / manifest["profile_paths"]["production"]["2026-05-09"]
    production = pd.read_parquet(production_path)
    production.loc[0, "actual_hit"] = 1
    production.to_parquet(production_path, index=False)

    report = verify_candidate_artifact_pair(
        artifact_dir=tmp_path,
        require_live_preoutcome=True,
    )

    assert report["ok"] is False
    failed_names = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert "production_2026-05-09_actual_hit_null" in failed_names


def test_resolve_live_candidate_artifact_pair_writes_resolved_copy(
    tmp_path,
):
    artifact_dir = tmp_path / "preoutcome"
    resolved_dir = tmp_path / "resolved"
    data_dir = tmp_path / "processed"
    data_dir.mkdir()

    manifest = _write_live_preoutcome_artifact(artifact_dir)
    pd.DataFrame([
        {"date": "2026-05-09", "batter_id": 11, "game_pk": 1001, "is_hit": 0},
        {"date": "2026-05-09", "batter_id": 11, "game_pk": 1001, "is_hit": 1},
        {"date": "2026-05-09", "batter_id": 22, "game_pk": 1002, "is_hit": 0},
    ]).to_parquet(data_dir / "pa_2026.parquet", index=False)

    report = resolve_live_candidate_artifact_pair(
        artifact_dir=artifact_dir,
        output_dir=resolved_dir,
        data_dir=data_dir,
        generated_at="2026-05-10T00:00:00+00:00",
        save_path=resolved_dir / "resolution.json",
    )

    assert report["complete"] is True
    assert report["missing_count"] == 0
    assert report["resolution_path"] == str(resolved_dir / "resolution.json")

    resolved_manifest = json.loads((resolved_dir / "manifest.json").read_text())
    assert resolved_manifest["schema_version"] == RESOLVED_ARTIFACT_SCHEMA_VERSION
    assert resolved_manifest["run_kind"] == "live_forward_resolved"
    assert resolved_manifest["source_run_kind"] == "live_forward_preoutcome"
    assert resolved_manifest["source_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert resolved_manifest["outcome_missing_total"] == 0
    assert resolved_manifest["outcome_status_counts"] == {
        "resolved": 4,
        "void_postponement": 0,
        "void_cancellation": 0,
        "void_no_pa": 0,
        "pending": 0,
    }
    assert "never coerced to actual_hit=0" in resolved_manifest["outcome_missing_semantics"]
    assert "never coerced to actual_hit=0" in report["missing_semantics"]

    source_production = pd.read_parquet(
        artifact_dir / manifest["profile_paths"]["production"]["2026-05-09"]
    )
    assert source_production["actual_hit"].isna().all()

    resolved_production = pd.read_parquet(
        resolved_dir / manifest["profile_paths"]["production"]["2026-05-09"]
    )
    assert resolved_production["run_kind"].unique().tolist() == ["live_forward_resolved"]
    assert resolved_production["artifact_schema_version"].unique().tolist() == [
        RESOLVED_ARTIFACT_SCHEMA_VERSION
    ]
    assert resolved_production["outcome_status"].tolist() == ["resolved", "resolved"]
    assert resolved_production["actual_hit"].astype(int).tolist() == [1, 0]
    assert resolved_production["n_pas"].astype(int).tolist() == [2, 1]
    assert list(resolved_production.columns) == RESOLVED_PROFILE_SCHEMA_COLUMNS


def test_resolve_live_candidate_artifact_pair_fails_missing_outcome(
    tmp_path,
):
    artifact_dir = tmp_path / "preoutcome"
    resolved_dir = tmp_path / "resolved"
    data_dir = tmp_path / "processed"
    data_dir.mkdir()

    _write_live_preoutcome_artifact(artifact_dir)
    pd.DataFrame([
        {"date": "2026-05-09", "batter_id": 11, "game_pk": 1001, "is_hit": 1},
    ]).to_parquet(data_dir / "pa_2026.parquet", index=False)

    with pytest.raises(ValueError, match="missing outcomes"):
        resolve_live_candidate_artifact_pair(
            artifact_dir=artifact_dir,
            output_dir=resolved_dir,
            data_dir=data_dir,
        )

    assert not (resolved_dir / "manifest.json").exists()

    partial_dir = tmp_path / "partial"
    report = resolve_live_candidate_artifact_pair(
        artifact_dir=artifact_dir,
        output_dir=partial_dir,
        data_dir=data_dir,
        allow_partial=True,
        save_path=partial_dir / "resolution.json",
    )

    assert report["complete"] is False
    assert report["missing_count"] == 2
    saved_report = json.loads((partial_dir / "resolution.json").read_text())
    assert isinstance(saved_report["missing_examples"][0]["date"], str)


def test_resolve_live_candidate_artifact_pair_terminal_void_status(
    tmp_path,
):
    artifact_dir = tmp_path / "preoutcome"
    resolved_dir = tmp_path / "resolved"
    data_dir = tmp_path / "processed"
    data_dir.mkdir()

    manifest = _write_live_preoutcome_artifact(artifact_dir)
    pd.DataFrame([
        {"date": "2026-05-09", "batter_id": 11, "game_pk": 1001, "is_hit": 1},
    ]).to_parquet(data_dir / "pa_2026.parquet", index=False)

    report = resolve_live_candidate_artifact_pair(
        artifact_dir=artifact_dir,
        output_dir=resolved_dir,
        data_dir=data_dir,
        treat_void_games_as_terminal=True,
        detailed_statuses_by_date={
            "2026-05-09": {
                1002: {"abstract": "F", "detailed": "Postponed"},
            }
        },
        save_path=resolved_dir / "resolution.json",
    )

    assert report["complete"] is True
    assert report["missing_count"] == 0
    assert report["terminal_void_count"] == 2
    assert report["outcome_status_counts"] == {
        "resolved": 2,
        "void_postponement": 2,
        "void_cancellation": 0,
        "void_no_pa": 0,
        "pending": 0,
    }

    resolved_manifest = json.loads((resolved_dir / "manifest.json").read_text())
    assert resolved_manifest["schema_version"] == RESOLVED_ARTIFACT_SCHEMA_VERSION
    assert resolved_manifest["outcome_missing_total"] == 0
    assert resolved_manifest["outcome_terminal_void_total"] == 2
    assert resolved_manifest["outcome_terminal_void_enabled"] is True
    assert resolved_manifest["outcome_status_counts"] == {
        "resolved": 2,
        "void_postponement": 2,
        "void_cancellation": 0,
        "void_no_pa": 0,
        "pending": 0,
    }
    assert resolved_manifest["profile_schema_columns"] == RESOLVED_PROFILE_SCHEMA_COLUMNS

    resolved_production = pd.read_parquet(
        resolved_dir / manifest["profile_paths"]["production"]["2026-05-09"]
    )
    void_row = resolved_production.loc[resolved_production["rank"] == 2].iloc[0]
    assert void_row["outcome_status"] == "void_postponement"
    assert pd.isna(void_row["actual_hit"])
    assert pd.isna(void_row["n_pas"])
    assert list(resolved_production.columns) == RESOLVED_PROFILE_SCHEMA_COLUMNS

    verification = verify_candidate_artifact_pair(
        artifact_dir=resolved_dir,
        expected_run_kind="live_forward_resolved",
        expected_candidate="decision_weighted_lgbm_v0",
        expected_date="2026-05-09",
        expected_git_commit="def456",
        expected_top_n=2,
    )
    assert verification["ok"] is True


def test_verify_resolved_artifact_accepts_legacy_zero_outcome_status_count(
    tmp_path,
):
    artifact_dir = tmp_path / "preoutcome"
    resolved_dir = tmp_path / "resolved"
    data_dir = tmp_path / "processed"
    data_dir.mkdir()

    _write_live_preoutcome_artifact(artifact_dir)
    pd.DataFrame([
        {"date": "2026-05-09", "batter_id": 11, "game_pk": 1001, "is_hit": 0},
        {"date": "2026-05-09", "batter_id": 11, "game_pk": 1001, "is_hit": 1},
        {"date": "2026-05-09", "batter_id": 22, "game_pk": 1002, "is_hit": 0},
    ]).to_parquet(data_dir / "pa_2026.parquet", index=False)

    resolve_live_candidate_artifact_pair(
        artifact_dir=artifact_dir,
        output_dir=resolved_dir,
        data_dir=data_dir,
    )
    resolved_manifest_path = resolved_dir / "manifest.json"
    resolved_manifest = json.loads(resolved_manifest_path.read_text())
    resolved_manifest["outcome_status_counts"].pop("void_no_pa")
    resolved_manifest["outcome_status_values"].remove("void_no_pa")
    for counts in resolved_manifest["outcome_status_counts_by_variant"].values():
        counts.pop("void_no_pa")
    resolved_manifest_path.write_text(json.dumps(resolved_manifest, indent=2))

    verification = verify_candidate_artifact_pair(
        artifact_dir=resolved_dir,
        expected_run_kind="live_forward_resolved",
        expected_candidate="decision_weighted_lgbm_v0",
        expected_date="2026-05-09",
        expected_git_commit="def456",
        expected_top_n=2,
    )

    assert verification["ok"] is True


def test_resolve_live_candidate_artifact_pair_final_no_pa_void_status(
    tmp_path,
):
    artifact_dir = tmp_path / "preoutcome"
    resolved_dir = tmp_path / "resolved"
    data_dir = tmp_path / "processed"
    data_dir.mkdir()

    manifest = _write_live_preoutcome_artifact(artifact_dir)
    pd.DataFrame([
        {"date": "2026-05-09", "batter_id": 11, "game_pk": 1001, "is_hit": 1},
        {"date": "2026-05-09", "batter_id": 44, "game_pk": 1002, "is_hit": 0},
    ]).to_parquet(data_dir / "pa_2026.parquet", index=False)

    report = resolve_live_candidate_artifact_pair(
        artifact_dir=artifact_dir,
        output_dir=resolved_dir,
        data_dir=data_dir,
        treat_void_games_as_terminal=True,
        detailed_statuses_by_date={
            "2026-05-09": {
                1002: {"abstract": "F", "detailed": "Final"},
            }
        },
        save_path=resolved_dir / "resolution.json",
    )

    assert report["complete"] is True
    assert report["missing_count"] == 0
    assert report["terminal_void_count"] == 2
    assert report["outcome_status_counts"] == {
        "resolved": 2,
        "void_postponement": 0,
        "void_cancellation": 0,
        "void_no_pa": 2,
        "pending": 0,
    }

    resolved_manifest = json.loads((resolved_dir / "manifest.json").read_text())
    assert "no PA for the player" in resolved_manifest["outcome_terminal_void_semantics"]

    resolved_production = pd.read_parquet(
        resolved_dir / manifest["profile_paths"]["production"]["2026-05-09"]
    )
    void_row = resolved_production.loc[resolved_production["rank"] == 2].iloc[0]
    assert void_row["outcome_status"] == OUTCOME_STATUS_VOID_NO_PA
    assert pd.isna(void_row["actual_hit"])
    assert pd.isna(void_row["n_pas"])

    verification = verify_candidate_artifact_pair(
        artifact_dir=resolved_dir,
        expected_run_kind="live_forward_resolved",
        expected_candidate="decision_weighted_lgbm_v0",
        expected_date="2026-05-09",
        expected_git_commit="def456",
        expected_top_n=2,
    )
    assert verification["ok"] is True

    resolved_manifest_path = resolved_dir / "manifest.json"
    resolved_manifest = json.loads(resolved_manifest_path.read_text())
    resolved_manifest["outcome_status_counts"].pop("void_no_pa")
    resolved_manifest_path.write_text(json.dumps(resolved_manifest, indent=2))

    failed_verification = verify_candidate_artifact_pair(
        artifact_dir=resolved_dir,
        expected_run_kind="live_forward_resolved",
        expected_candidate="decision_weighted_lgbm_v0",
        expected_date="2026-05-09",
        expected_git_commit="def456",
        expected_top_n=2,
    )
    failed_names = {
        check["name"] for check in failed_verification["checks"]
        if check["status"] == "fail"
    }
    assert "outcome_status_counts" in failed_names


def test_resolve_live_candidate_artifact_pair_mixed_void_and_pending(
    tmp_path,
):
    artifact_dir = tmp_path / "preoutcome"
    resolved_dir = tmp_path / "resolved"
    data_dir = tmp_path / "processed"
    data_dir.mkdir()

    manifest = _write_live_preoutcome_artifact(artifact_dir)
    for variant in ("production", "candidate"):
        rel_path = manifest["profile_paths"][variant]["2026-05-09"]
        path = artifact_dir / rel_path
        frame = pd.read_parquet(path)
        extra = frame.iloc[[0]].copy()
        extra["rank"] = 3
        extra["batter_id"] = 33
        extra["game_pk"] = 1003
        extra["p_game_hit"] = 0.55
        frame = pd.concat([frame, extra], ignore_index=True)
        frame.to_parquet(path, index=False)
        manifest["row_counts"][variant]["2026-05-09"] = 3
    manifest["top_n"] = 3
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    pd.DataFrame([
        {"date": "2026-05-09", "batter_id": 11, "game_pk": 1001, "is_hit": 1},
    ]).to_parquet(data_dir / "pa_2026.parquet", index=False)

    report = resolve_live_candidate_artifact_pair(
        artifact_dir=artifact_dir,
        output_dir=resolved_dir,
        data_dir=data_dir,
        allow_partial=True,
        treat_void_games_as_terminal=True,
        detailed_statuses_by_date={
            "2026-05-09": {
                1002: {"abstract": "F", "detailed": "Postponed"},
                1003: {"abstract": "F", "detailed": "Final"},
            }
        },
    )

    assert report["complete"] is False
    assert report["missing_count"] == 2
    assert report["terminal_void_count"] == 2
    assert report["outcome_status_counts"] == {
        "resolved": 2,
        "void_postponement": 2,
        "void_cancellation": 0,
        "void_no_pa": 0,
        "pending": 2,
    }

    verification = verify_candidate_artifact_pair(
        artifact_dir=resolved_dir,
        expected_run_kind="live_forward_resolved",
        expected_candidate="decision_weighted_lgbm_v0",
        expected_date="2026-05-09",
        expected_git_commit="def456",
        expected_top_n=3,
    )
    failed_names = {
        check["name"] for check in verification["checks"]
        if check["status"] == "fail"
    }
    assert "production_2026-05-09_pending_outcomes_absent" in failed_names
    assert "candidate_2026-05-09_pending_outcomes_absent" in failed_names
