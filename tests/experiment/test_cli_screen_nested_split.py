"""Tests for `bts experiment screen` season-level split flags."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner


@pytest.fixture
def fake_results(tmp_path, monkeypatch):
    from bts.experiment import cli as cli_mod

    monkeypatch.setattr(cli_mod, "RESULTS_BASE", tmp_path)
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


def test_screen_rejects_overlapping_split_seasons(fake_results):
    from bts.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        "experiment", "screen",
        "--selection-seasons", "2024",
        "--outer-eval-seasons", "2024",
    ])

    assert result.exit_code != 0
    assert "disjoint" in result.output.lower()


def test_screen_rejects_test_seasons_with_split_flags(fake_results):
    from bts.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        "experiment", "screen",
        "--test-seasons", "2024",
        "--selection-seasons", "2023",
        "--outer-eval-seasons", "2025",
    ])

    assert result.exit_code != 0
    assert "--test-seasons cannot be combined" in result.output


def test_screen_routes_phase1_to_selection_seasons_only(
    fake_results, fake_pa_data, stub_features,
):
    from bts.cli import cli

    seasons_seen: list[int] = []

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

    with patch("bts.simulate.backtest_blend.blend_walk_forward", side_effect=fake_walk_forward), \
         patch("bts.validate.scorecard.compute_full_scorecard", return_value={"p_57_mdp": 0.5}), \
         patch("bts.validate.scorecard.save_scorecard") as mock_save, \
         patch("bts.experiment.runner.run_screening", return_value=[]) as mock_screen:
        runner = CliRunner()
        result = runner.invoke(cli, [
            "experiment", "screen",
            "--data-dir", str(fake_pa_data),
            "--subset", "heat_dome",
            "--selection-seasons", "2023",
            "--outer-eval-seasons", "2025",
            "--no-use-factored",
        ])

    assert result.exit_code == 0, result.output
    assert seasons_seen == [2023]
    assert mock_screen.call_args.args[3] == [2023]
    assert mock_screen.call_args.kwargs["split_metadata"]["selection_seasons"] == [2023]
    assert mock_screen.call_args.kwargs["split_metadata"]["outer_eval_seasons"] == [2025]
    saved_scorecard = mock_save.call_args.args[0]
    assert saved_scorecard["validation_split"]["artifact_role"] == "selection_only"
    assert saved_scorecard["validation_split"]["outer_eval_seasons"] == [2025]
