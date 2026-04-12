"""Test shadow model integration in scheduler."""

import json
from unittest.mock import patch, MagicMock
from pathlib import Path

import pandas as pd
import pytest

from bts.picks import DailyPick, Pick, save_pick
from bts.scheduler import _run_shadow_prediction


@pytest.fixture(autouse=True)
def _disable_mdp():
    """Force heuristic mode — MDP policy file may exist on dev machines."""
    with patch("bts.strategy._load_mdp", return_value=None):
        yield


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

    def test_writes_independent_shadow_file_when_production_locked(self, tmp_path):
        """Integration test: when production is locked + bluesky_posted=True, the
        shadow prediction pipeline must still write its OWN pick (not a copy of
        production) to {date}.shadow.json.

        Regression for the select_pick short-circuit bug: before the fix, the
        scheduler called select_pick without for_shadow=True, so the function
        loaded production from disk, saw bluesky_posted=True, and returned
        production's DailyPick. The shadow predictions were silently discarded.

        This test deliberately does NOT mock select_pick — it uses the real
        function to catch any future regression in this integration.
        """
        # Arrange: production is locked and posted
        prod = DailyPick(
            date="2026-04-12",
            run_time="2026-04-12T17:19:41.741015+00:00",
            pick=Pick(
                batter_name="Brendan Donovan", batter_id=680977, team="SEA",
                lineup_position=1, pitcher_name="Cody Bolton", pitcher_id=675989,
                p_game_hit=0.7169, flags=[], projected_lineup=False,
                game_pk=823154, game_time="2026-04-12T20:10:00Z",
                pitcher_team="HOU",
            ),
            double_down=None, runner_up=None,
            bluesky_posted=True,
            bluesky_uri="at://did:plc:test/app.bsky.feed.post/abc",
        )
        save_pick(prod, tmp_path)

        # Shadow predictions: a completely different top batter in a different
        # game. Probabilities are above SKIP_THRESHOLD (0.80) so heuristic mode
        # (used when MDP policy file is absent in tests) will actually pick.
        shadow_preds = pd.DataFrame([
            {
                "batter_name": "Nico Hoerner", "batter_id": 663538, "team": "CHC",
                "lineup": 1, "pitcher_name": "Opposing SP", "pitcher_id": 111111,
                "game_pk": 824696, "game_time": "2026-04-12T18:20:00Z",
                "p_game_hit": 0.82, "p_hit_pa": 0.30, "flags": "",
                "projected_lineup": False,
            },
            {
                "batter_name": "Steven Kwan", "batter_id": 680757, "team": "CLE",
                "lineup": 1, "pitcher_name": "Opposing SP2", "pitcher_id": 222222,
                "game_pk": 824938, "game_time": "2026-04-12T23:20:00Z",
                "p_game_hit": 0.81, "p_hit_pa": 0.29, "flags": "",
                "projected_lineup": False,
            },
        ])

        statuses = {823154: "P", 824696: "P", 824938: "P"}

        # Act: run shadow with REAL select_pick (only mocking the predict call
        # and the game-status HTTP lookup)
        with patch("bts.scheduler.predict_local_shadow", return_value=shadow_preds), \
             patch("bts.strategy.get_game_statuses", return_value=statuses):
            _run_shadow_prediction(
                config={"orchestrator": {"picks_dir": str(tmp_path)}},
                date="2026-04-12",
                production_pick_name="Brendan Donovan",
            )

        # Assert: the written shadow file reflects the shadow predictions,
        # NOT the production pick that's locked on disk.
        shadow_path = tmp_path / "2026-04-12.shadow.json"
        assert shadow_path.exists(), "shadow file was not written"
        shadow_data = json.loads(shadow_path.read_text())
        assert shadow_data["pick"]["batter_name"] == "Nico Hoerner"
        assert shadow_data["pick"]["p_game_hit"] == pytest.approx(0.82)
        assert shadow_data["bluesky_posted"] is False
        assert shadow_data["bluesky_uri"] is None
        # Shadow should also inherit the double-down computation (same MDP
        # logic as production, different predictions)
        assert shadow_data["double_down"] is not None
        assert shadow_data["double_down"]["batter_name"] == "Steven Kwan"
        # And the production file must NOT be clobbered
        prod_data = json.loads((tmp_path / "2026-04-12.json").read_text())
        assert prod_data["pick"]["batter_name"] == "Brendan Donovan"
        assert prod_data["bluesky_posted"] is True

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
