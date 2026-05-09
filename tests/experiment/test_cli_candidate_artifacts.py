from __future__ import annotations

from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner


def test_export_candidate_artifacts_cli_routes_to_materializer(tmp_path, monkeypatch):
    from bts.cli import cli
    import bts.experiment.artifacts as artifacts_mod

    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    pd.DataFrame({"batter_id": [1], "is_hit": [1]}).to_parquet(data_dir / "pa_2024.parquet")
    output_dir = tmp_path / "artifact"
    captured = {}

    def fake_materialize(**kwargs):
        captured.update(kwargs)
        return {
            "profile_paths": {
                "production": {"2024": "profiles/production/backtest_2024.parquet"},
                "candidate": {"2024": "profiles/candidate/backtest_2024.parquet"},
            }
        }

    monkeypatch.setattr(artifacts_mod, "materialize_candidate_profile_pair", fake_materialize)
    with patch("bts.features.compute.compute_all_features", side_effect=lambda x: x):
        result = CliRunner().invoke(cli, [
            "experiment", "export-candidate-artifacts",
            "--data-dir", str(data_dir),
            "--candidate", "decision_weighted_lgbm_v0",
            "--seasons", "2024",
            "--output-dir", str(output_dir),
            "--retrain-every", "3",
            "--top-n", "2",
        ])

    assert result.exit_code == 0, result.output
    assert captured["candidate"].name == "decision_weighted_lgbm_v0"
    assert captured["seasons"] == [2024]
    assert captured["retrain_every"] == 3
    assert captured["top_n"] == 2
    assert "Saved manifest" in result.output


def test_compare_candidate_artifacts_cli_reports_primary_delta(tmp_path, monkeypatch):
    from bts.cli import cli
    import bts.experiment.artifacts as artifacts_mod

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    def fake_compare(**kwargs):
        return {
            "comparison_path": str(artifact_dir / "comparison.json"),
            "primary_metric": "p_57_mdp",
            "primary_delta": 0.0123,
        }

    monkeypatch.setattr(artifacts_mod, "compare_candidate_profile_pair", fake_compare)
    result = CliRunner().invoke(cli, [
        "experiment", "compare-candidate-artifacts",
        "--artifact-dir", str(artifact_dir),
        "--mc-trials", "123",
        "--season-length", "162",
    ])

    assert result.exit_code == 0, result.output
    assert "Saved comparison" in result.output
    assert "Primary delta (p_57_mdp): +0.012300" in result.output


def test_verify_candidate_artifacts_cli_reports_pass(tmp_path, monkeypatch):
    from bts.cli import cli
    import bts.experiment.artifacts as artifacts_mod

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    captured = {}

    def fake_verify(**kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "failure_count": 0,
            "checks": [],
        }

    monkeypatch.setattr(artifacts_mod, "verify_candidate_artifact_pair", fake_verify)
    result = CliRunner().invoke(cli, [
        "experiment", "verify-candidate-artifacts",
        "--artifact-dir", str(artifact_dir),
        "--expected-run-kind", "live_forward_preoutcome",
        "--expected-candidate", "decision_weighted_lgbm_v0",
        "--expected-date", "2026-05-09",
        "--expected-git-commit", "abc123",
        "--expected-top-n", "10",
        "--require-live-preoutcome",
    ])

    assert result.exit_code == 0, result.output
    assert captured["expected_run_kind"] == "live_forward_preoutcome"
    assert captured["expected_candidate"] == "decision_weighted_lgbm_v0"
    assert captured["expected_date"] == "2026-05-09"
    assert captured["expected_git_commit"] == "abc123"
    assert captured["expected_top_n"] == 10
    assert captured["require_live_preoutcome"] is True
    assert "Candidate artifact verification: PASS" in result.output


def test_verify_candidate_artifacts_cli_fails_on_failed_check(tmp_path, monkeypatch):
    from bts.cli import cli
    import bts.experiment.artifacts as artifacts_mod

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    def fake_verify(**kwargs):
        return {
            "ok": False,
            "failure_count": 1,
            "checks": [
                {
                    "name": "expected_git_commit",
                    "status": "fail",
                    "detail": "expected abc123, found def456",
                }
            ],
        }

    monkeypatch.setattr(artifacts_mod, "verify_candidate_artifact_pair", fake_verify)
    result = CliRunner().invoke(cli, [
        "experiment", "verify-candidate-artifacts",
        "--artifact-dir", str(artifact_dir),
    ])

    assert result.exit_code != 0
    assert "Candidate artifact verification: FAIL" in result.output
    assert "FAIL expected_git_commit" in result.output


def test_export_live_candidate_artifacts_cli_routes_to_materializer(tmp_path, monkeypatch):
    from bts.cli import cli
    import bts.experiment.artifacts as artifacts_mod

    output_dir = tmp_path / "live-artifact"
    captured = {}

    def fake_materialize_live(**kwargs):
        captured.update(kwargs)
        return {
            "profile_paths": {
                "production": {"2026-05-09": "profiles/production/live_2026-05-09.parquet"},
                "candidate": {"2026-05-09": "profiles/candidate/live_2026-05-09.parquet"},
            }
        }

    monkeypatch.setattr(
        artifacts_mod,
        "materialize_live_candidate_profile_pair",
        fake_materialize_live,
    )
    result = CliRunner().invoke(cli, [
        "experiment", "export-live-candidate-artifacts",
        "--date", "2026-05-09",
        "--candidate", "decision_weighted_lgbm_v0",
        "--output-dir", str(output_dir),
        "--data-dir", "data/processed",
        "--top-n", "2",
        "--no-refresh-data",
    ])

    assert result.exit_code == 0, result.output
    assert captured["date"] == "2026-05-09"
    assert captured["candidate"].name == "decision_weighted_lgbm_v0"
    assert captured["top_n"] == 2
    assert captured["refresh_data"] is False
    assert "Saved manifest" in result.output
