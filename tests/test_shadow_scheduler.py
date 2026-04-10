"""Test shadow model integration in scheduler."""

from unittest.mock import patch, MagicMock
from pathlib import Path

from bts.scheduler import _run_shadow_prediction


class TestRunShadowPrediction:
    def test_saves_shadow_pick(self, tmp_path):
        mock_predictions = MagicMock()
        mock_result = MagicMock()
        mock_result.daily.date = "2026-04-10"
        mock_result.daily.pick.batter_name = "Luis Arraez"
        mock_result.daily.pick.p_game_hit = 0.767

        with patch("bts.scheduler.predict_local_shadow", return_value=mock_predictions), \
             patch("bts.scheduler.select_pick", return_value=mock_result), \
             patch("bts.scheduler.save_shadow_pick") as mock_save:
            _run_shadow_prediction(
                config={"orchestrator": {"picks_dir": str(tmp_path)}},
                date="2026-04-10",
                production_pick_name="Luis Arraez",
            )
            mock_save.assert_called_once()

    def test_logs_agreement(self, tmp_path, capsys):
        mock_predictions = MagicMock()
        mock_result = MagicMock()
        mock_result.daily.pick.batter_name = "Luis Arraez"
        mock_result.daily.pick.team = "SF"
        mock_result.daily.pick.p_game_hit = 0.767

        with patch("bts.scheduler.predict_local_shadow", return_value=mock_predictions), \
             patch("bts.scheduler.select_pick", return_value=mock_result), \
             patch("bts.scheduler.save_shadow_pick"):
            _run_shadow_prediction(
                config={"orchestrator": {"picks_dir": str(tmp_path)}},
                date="2026-04-10",
                production_pick_name="Luis Arraez",
            )
        captured = capsys.readouterr()
        assert "AGREES" in captured.err

    def test_logs_disagreement(self, tmp_path, capsys):
        mock_predictions = MagicMock()
        mock_result = MagicMock()
        mock_result.daily.pick.batter_name = "Steven Kwan"
        mock_result.daily.pick.team = "CLE"
        mock_result.daily.pick.p_game_hit = 0.720

        with patch("bts.scheduler.predict_local_shadow", return_value=mock_predictions), \
             patch("bts.scheduler.select_pick", return_value=mock_result), \
             patch("bts.scheduler.save_shadow_pick"):
            _run_shadow_prediction(
                config={"orchestrator": {"picks_dir": str(tmp_path)}},
                date="2026-04-10",
                production_pick_name="Luis Arraez",
            )
        captured = capsys.readouterr()
        assert "DISAGREES" in captured.err

    def test_failure_does_not_raise(self, tmp_path):
        with patch("bts.scheduler.predict_local_shadow", side_effect=RuntimeError("boom")):
            _run_shadow_prediction(
                config={"orchestrator": {"picks_dir": str(tmp_path)}},
                date="2026-04-10",
                production_pick_name="Luis Arraez",
            )
