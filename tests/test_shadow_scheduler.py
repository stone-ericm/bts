"""Test shadow model integration in scheduler."""

import json
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path

import pandas as pd
import pytest

from bts.picks import DailyPick, Pick, save_pick
from bts.scheduler import (
    _live_candidate_artifacts_config,
    _run_live_candidate_artifacts,
    _run_shadow_prediction,
)


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

    def test_threads_config_data_dir_and_models_dir(self, tmp_path):
        """Per Codex bus #172/#174: a non-default orchestrator data_dir/models_dir
        in the TOML must be passed through to predict_local_shadow AND used to
        compute the artifact hash, so the recorded sha matches the loaded artifact.
        """
        custom_models_dir = tmp_path / "custom_models"
        custom_models_dir.mkdir()
        date = "2026-04-10"
        blend_artifact = custom_models_dir / f"blend_{date}_shadow.pkl"
        blend_artifact.write_bytes(b"dummy shadow blend content")

        custom_data_dir = tmp_path / "custom_processed"
        custom_data_dir.mkdir()

        # Use a real DailyPick (not MagicMock) so attach_provenance can mutate
        # real fields and the assertions inspect real state.
        sample_pick = Pick(
            batter_name="Luis Arraez", batter_id=650333, team="SF",
            lineup_position=1, pitcher_name="Test Pitcher", pitcher_id=12345,
            p_game_hit=0.75, flags=[], projected_lineup=False,
            game_pk=999999, game_time=f"{date}T20:00:00Z",
        )
        real_daily = DailyPick(
            date=date, run_time=f"{date}T15:00:00+00:00",
            pick=sample_pick, double_down=None, runner_up=None,
        )
        mock_result = MagicMock()
        mock_result.daily = real_daily

        with patch("bts.scheduler.predict_local_shadow") as mock_predict, \
             patch("bts.scheduler.select_pick", return_value=mock_result), \
             patch("bts.scheduler.save_shadow_pick"):
            mock_predict.return_value = MagicMock()  # truthy -> path proceeds

            _run_shadow_prediction(
                config={
                    "orchestrator": {
                        "picks_dir": str(tmp_path / "picks"),
                        "data_dir": str(custom_data_dir),
                        "models_dir": str(custom_models_dir),
                    }
                },
                date=date,
                production_pick_name="Luis Arraez",
            )

            # 1. predict_local_shadow received the configured paths.
            mock_predict.assert_called_once()
            call_kwargs = mock_predict.call_args.kwargs
            assert call_kwargs.get("data_dir") == str(custom_data_dir)
            assert call_kwargs.get("models_dir") == str(custom_models_dir)

            # 2. Provenance attached AND artifact hash reflects the file at
            #    the configured models_dir (not the default path).
            assert real_daily.model_pickle_sha256 is not None
            assert len(real_daily.model_pickle_sha256) == 64
            from bts.picks import _sha256_file
            expected = _sha256_file(blend_artifact)
            assert real_daily.model_pickle_sha256 == expected

    def test_failure_does_not_raise(self, tmp_path):
        with patch("bts.scheduler.predict_local_shadow", side_effect=RuntimeError("boom")):
            _run_shadow_prediction(
                config={"orchestrator": {"picks_dir": str(tmp_path)}},
                date="2026-04-10",
                production_pick_name="Luis Arraez",
            )


class TestRunLiveCandidateArtifacts:
    def _config(self, tmp_path, **overrides):
        live_config = {
            "enabled": True,
            "command": ["/fake/bin/bts"],
            "worktree_dir": str(tmp_path / "frozen-worktree"),
            "output_dir": str(tmp_path / "live" / "{date}"),
            "data_dir": str(tmp_path / "processed"),
            "top_n": 7,
            "refresh_data": False,
            "timeout_sec": 123,
        }
        live_config.update(overrides)
        return {
            "orchestrator": {
                "picks_dir": str(tmp_path / "picks"),
                "heartbeat_path": str(tmp_path / ".heartbeat"),
            },
            "scheduler": {"live_candidate_artifacts": live_config},
        }

    def test_runs_frozen_cli_with_preoutcome_defaults(self, tmp_path):
        config = self._config(tmp_path)

        with patch("bts.scheduler.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="",
                stderr="",
            )

            _run_live_candidate_artifacts(config, "2026-05-09")

        mock_run.assert_called_once()
        args = mock_run.call_args.args[0]
        kwargs = mock_run.call_args.kwargs
        assert args == [
            "/fake/bin/bts",
            "experiment",
            "export-live-candidate-artifacts",
            "--date", "2026-05-09",
            "--candidate", "decision_weighted_lgbm_v0",
            "--output-dir", str(tmp_path / "live" / "2026-05-09"),
            "--data-dir", str(tmp_path / "processed"),
            "--top-n", "7",
            "--no-refresh-data",
        ]
        assert kwargs["cwd"] == tmp_path / "frozen-worktree"
        assert kwargs["timeout"] == 123
        assert kwargs["env"]["BTS_LGBM_DETERMINISTIC"] == "1"
        assert kwargs["env"]["BTS_LGBM_RANDOM_STATE"] == "42"
        assert kwargs["env"]["UV_CACHE_DIR"] == "/tmp/uv-cache"

    def test_worktree_defaults_to_its_own_venv_binary(self, tmp_path):
        worktree = tmp_path / "frozen-worktree"
        config = self._config(tmp_path, command=None, worktree_dir=str(worktree))
        del config["scheduler"]["live_candidate_artifacts"]["command"]

        live_config = _live_candidate_artifacts_config(config)

        assert live_config is not None
        assert live_config["command"] == str(worktree / ".venv" / "bin" / "bts")

    def test_skips_existing_manifest(self, tmp_path, capsys):
        out_dir = tmp_path / "live" / "2026-05-09"
        out_dir.mkdir(parents=True)
        (out_dir / "manifest.json").write_text("{}")
        config = self._config(tmp_path)

        with patch("bts.scheduler.subprocess.run") as mock_run:
            _run_live_candidate_artifacts(config, "2026-05-09")

        mock_run.assert_not_called()
        captured = capsys.readouterr()
        assert "already logged" in captured.err

    def test_disabled_does_not_run(self, tmp_path):
        config = self._config(tmp_path, enabled=False)

        with patch("bts.scheduler.subprocess.run") as mock_run:
            _run_live_candidate_artifacts(config, "2026-05-09")

        mock_run.assert_not_called()

    def test_absent_section_does_not_run(self, tmp_path):
        config = {
            "orchestrator": {
                "picks_dir": str(tmp_path / "picks"),
                "heartbeat_path": str(tmp_path / ".heartbeat"),
            },
            "scheduler": {},
        }

        with patch("bts.scheduler.subprocess.run") as mock_run:
            _run_live_candidate_artifacts(config, "2026-05-09")

        mock_run.assert_not_called()

    def test_failure_does_not_raise(self, tmp_path, capsys):
        config = self._config(tmp_path)

        with patch("bts.scheduler.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=2,
                stdout="",
                stderr="failure details\n",
            )

            _run_live_candidate_artifacts(config, "2026-05-09")

        captured = capsys.readouterr()
        assert "Failed: exit 2" in captured.err
        assert "failure details" in captured.err

    def test_missing_binary_failure_does_not_fall_back(self, tmp_path, capsys):
        config = self._config(tmp_path, command="/missing/frozen/bts")

        with patch("bts.scheduler.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("missing frozen bts")

            _run_live_candidate_artifacts(config, "2026-05-09")

        mock_run.assert_called_once()
        assert mock_run.call_args.args[0][0] == "/missing/frozen/bts"
        captured = capsys.readouterr()
        assert "Failed: missing frozen bts" in captured.err
