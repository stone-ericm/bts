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
