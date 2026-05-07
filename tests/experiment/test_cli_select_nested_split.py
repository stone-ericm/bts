"""Tests for `bts experiment select` season-level split flags."""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner


@pytest.fixture
def fake_results(tmp_path, monkeypatch):
    from bts.experiment import cli as cli_mod

    monkeypatch.setattr(cli_mod, "RESULTS_BASE", tmp_path)
    p1 = tmp_path / "phase1" / "heat_dome"
    p1.mkdir(parents=True)
    (p1 / "summary.txt").write_text("PASS | reason")
    (p1 / "diff.json").write_text(json.dumps({
        "streak_metrics": {"mean_max_streak": {"delta": 1.0}},
        "p_at_1_by_season": {"2023": {"delta": 0.01}},
    }))
    return tmp_path


@pytest.fixture
def fake_pa_data(tmp_path):
    proc = tmp_path / "data_processed"
    proc.mkdir()
    pd.DataFrame({"batter_id": [1, 2], "is_hit": [0, 1]}).to_parquet(proc / "pa_2024.parquet")
    return proc


@pytest.fixture
def stub_features():
    with patch("bts.features.compute.compute_all_features", side_effect=lambda x: x):
        yield


@pytest.fixture
def stub_run_selection():
    with patch("bts.experiment.runner.run_selection") as mock_run:
        mock_run.return_value = {
            "included": [],
            "forward_log": [],
            "backward_log": [],
            "final_scorecard": {},
            "final_diff": {},
        }
        yield mock_run


def test_select_rejects_overlapping_split_seasons(fake_results):
    from bts.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        "experiment", "select",
        "--selection-seasons", "2024",
        "--outer-eval-seasons", "2024",
    ])

    assert result.exit_code != 0
    assert "disjoint" in result.output.lower()


def test_select_routes_decisions_to_selection_and_final_to_outer(
    fake_results, fake_pa_data, stub_features, stub_run_selection,
):
    from bts.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        "experiment", "select",
        "--data-dir", str(fake_pa_data),
        "--selection-seasons", "2023",
        "--outer-eval-seasons", "2025",
    ])

    assert result.exit_code == 0, result.output
    assert stub_run_selection.call_args.args[3] == [2023]
    assert stub_run_selection.call_args.kwargs["outer_eval_seasons"] == [2025]
    split = stub_run_selection.call_args.kwargs["split_metadata"]
    assert split["selection_seasons"] == [2023]
    assert split["outer_eval_seasons"] == [2025]
    assert split["production_deploy_claim"] is False
