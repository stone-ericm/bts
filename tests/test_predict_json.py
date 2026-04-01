"""Tests for bts predict-json CLI command."""

import json
import pytest
from unittest.mock import patch
from click.testing import CliRunner

from bts.cli import cli


def _mock_predictions():
    """Build a mock predictions DataFrame."""
    import pandas as pd
    return pd.DataFrame([
        {
            "batter_name": "Jacob Wilson",
            "batter_id": 700363,
            "team": "ATH",
            "lineup": 1,
            "pitcher_name": "Jose Suarez",
            "pitcher_id": 660761,
            "p_game_hit": 0.763,
            "p_hit_pa": 0.312,
            "flags": "",
            "game_pk": 778899,
            "game_time": "2026-04-01T23:10:00Z",
        },
        {
            "batter_name": "Jake Mangum",
            "batter_id": 700100,
            "team": "NYM",
            "lineup": 2,
            "pitcher_name": "Logan Webb",
            "pitcher_id": 657277,
            "p_game_hit": 0.726,
            "p_hit_pa": 0.295,
            "flags": "PROJECTED",
            "game_pk": 778900,
            "game_time": "2026-04-01T23:10:00Z",
        },
    ])


class TestPredictJson:
    @patch("bts.model.predict.run_pipeline")
    def test_outputs_valid_json(self, mock_pipeline):
        mock_pipeline.return_value = _mock_predictions()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "predict-json", "--date", "2026-04-01",
            "--data-dir", "data/processed",
        ])

        assert result.exit_code == 0
        # Use result.stdout (pure stdout) — result.output mixes stdout+stderr in Click 8.2+
        data = json.loads(result.stdout)
        assert len(data) == 2
        assert data[0]["batter_name"] == "Jacob Wilson"
        assert data[0]["p_game_hit"] == 0.763

    @patch("bts.model.predict.run_pipeline")
    def test_includes_all_required_fields(self, mock_pipeline):
        mock_pipeline.return_value = _mock_predictions()

        runner = CliRunner()
        result = runner.invoke(cli, ["predict-json", "--date", "2026-04-01"])
        data = json.loads(result.stdout)

        required = ["batter_name", "batter_id", "team", "lineup",
                     "pitcher_name", "pitcher_id", "game_pk",
                     "game_time", "p_game_hit", "flags"]
        for field in required:
            assert field in data[0], f"Missing field: {field}"

    @patch("bts.model.predict.run_pipeline")
    def test_empty_predictions(self, mock_pipeline):
        import pandas as pd
        mock_pipeline.return_value = pd.DataFrame()

        runner = CliRunner()
        result = runner.invoke(cli, ["predict-json", "--date", "2026-04-01"])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data == []

    @patch("bts.model.predict.run_pipeline")
    def test_error_exits_nonzero(self, mock_pipeline):
        mock_pipeline.side_effect = RuntimeError("No data")

        runner = CliRunner()
        result = runner.invoke(cli, ["predict-json", "--date", "2026-04-01"])

        assert result.exit_code != 0
