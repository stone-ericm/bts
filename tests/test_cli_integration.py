"""Integration tests for bts run and bts check-results CLI commands."""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from click.testing import CliRunner

from bts.cli import cli
from bts.picks import (
    Pick, DailyPick, save_pick, save_streak,
    save_shadow_pick, load_shadow_pick,
)


def _sample_pick(**overrides):
    defaults = dict(
        batter_name="Jacob Wilson",
        batter_id=700363,
        team="ATH",
        lineup_position=1,
        pitcher_name="Jose Suarez",
        pitcher_id=660761,
        p_game_hit=0.83,
        flags=[],
        projected_lineup=False,
        game_pk=778899,
        game_time="2026-04-01T23:10:00Z",
    )
    defaults.update(overrides)
    return Pick(**defaults)


def _sample_daily(**overrides):
    defaults = dict(
        date="2026-04-01",
        run_time="2026-04-01T15:00:00+00:00",
        pick=_sample_pick(),
        double_down=None,
        runner_up=None,
        bluesky_posted=False,
        bluesky_uri=None,
    )
    defaults.update(overrides)
    return DailyPick(**defaults)


def _mock_predictions():
    """Build a mock predictions DataFrame matching run_pipeline output."""
    import pandas as pd
    return pd.DataFrame([
        {
            "batter_name": "Jacob Wilson",
            "batter_id": 700363,
            "team": "ATH",
            "lineup": 1,
            "pitcher_name": "Jose Suarez",
            "pitcher_id": 660761,
            "p_game_hit": 0.83,
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
            "p_game_hit": 0.81,
            "flags": "",
            "game_pk": 778900,
            "game_time": "2026-04-01T23:10:00Z",
        },
    ])


class TestBtsRun:
    @patch("bts.posting.post_to_bluesky")
    @patch("bts.posting.should_post_now", return_value=True)
    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    @patch("bts.model.predict.run_pipeline")
    def test_run_saves_pick_and_posts(
        self, mock_pipeline, mock_statuses, mock_should_post, mock_post, tmp_path,
    ):
        mock_pipeline.return_value = _mock_predictions()
        mock_post.return_value = "at://did:plc:test/post/123"

        picks_dir = tmp_path / "picks"
        models_dir = tmp_path / "models"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
            "--models-dir", str(models_dir),
            "--data-dir", "data/processed",
        ])

        assert result.exit_code == 0
        assert "Jacob Wilson" in result.output
        assert "Posted to Bluesky" in result.output

        # Verify pick file was saved
        pick_file = picks_dir / "2026-04-01.json"
        assert pick_file.exists()
        data = json.loads(pick_file.read_text())
        assert data["pick"]["batter_name"] == "Jacob Wilson"
        assert data["bluesky_posted"] is True

    @patch("bts.posting.should_post_now", return_value=False)
    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    @patch("bts.model.predict.run_pipeline")
    def test_run_dry_run_skips_posting(
        self, mock_pipeline, mock_statuses, mock_should_post, tmp_path,
    ):
        mock_pipeline.return_value = _mock_predictions()

        picks_dir = tmp_path / "picks"
        models_dir = tmp_path / "models"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
            "--models-dir", str(models_dir),
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "dry-run" in result.output

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    @patch("bts.model.predict.run_pipeline")
    def test_run_no_games_reports_empty(
        self, mock_pipeline, mock_statuses, tmp_path,
    ):
        import pandas as pd
        mock_pipeline.return_value = pd.DataFrame()

        picks_dir = tmp_path / "picks"
        models_dir = tmp_path / "models"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
            "--models-dir", str(models_dir),
        ])

        assert result.exit_code == 0
        assert "No games found" in result.output


class TestBtsCheckResults:
    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_check_results_hit_updates_streak(self, mock_check, _mock_statuses, tmp_path):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        save_pick(_sample_daily(), picks_dir)
        save_streak(3, picks_dir)
        mock_check.return_value = True

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check-results", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
        ])

        assert result.exit_code == 0
        assert "HIT!" in result.output
        assert "Streak: 4" in result.output

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_check_results_miss_resets_streak(self, mock_check, _mock_statuses, tmp_path):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        save_pick(_sample_daily(), picks_dir)
        save_streak(5, picks_dir)
        mock_check.return_value = False

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check-results", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
        ])

        assert result.exit_code == 0
        assert "MISS" in result.output
        assert "Streak reset to 0" in result.output

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_check_results_none_warns_scratched(self, mock_check, _mock_statuses, tmp_path):
        """Scratched player (None result) should warn, not change streak."""
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        save_pick(_sample_daily(), picks_dir)
        save_streak(3, picks_dir)
        mock_check.return_value = None

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check-results", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
        ])

        assert result.exit_code == 0
        assert "WARNING" in result.output
        # Streak should be unchanged
        from bts.picks import load_streak
        assert load_streak(picks_dir) == 3

    @patch("bts.picks.get_game_statuses_detailed")
    @patch("bts.picks.check_hit")
    def test_check_results_voids_primary_and_scores_double_once(
        self, mock_check, mock_statuses, tmp_path,
    ):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        double = _sample_pick(
            batter_name="Carlos Cortes",
            batter_id=666126,
            team="ATH",
            game_pk=778900,
        )
        save_pick(_sample_daily(double_down=double), picks_dir)
        save_streak(3, picks_dir)
        mock_statuses.return_value = {
            778899: {"abstract": "F", "detailed": "Postponed"},
            778900: {"abstract": "F", "detailed": "Final"},
        }
        mock_check.return_value = True

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check-results", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
        ])

        assert result.exit_code == 0
        assert "VOID: Jacob Wilson." in result.output
        assert "HIT! Carlos Cortes. Streak: 4" in result.output

        from bts.picks import load_pick, load_streak
        loaded = load_pick("2026-04-01", picks_dir)
        assert loaded.result == "hit"
        assert loaded.slot_results == {"pick": "void", "double_down": "hit"}
        assert load_streak(picks_dir) == 4
        mock_check.assert_called_once()

    @patch("bts.picks.get_game_statuses_detailed")
    @patch("bts.picks.check_hit")
    def test_check_results_voids_double_and_scores_primary_once(
        self, mock_check, mock_statuses, tmp_path,
    ):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        double = _sample_pick(
            batter_name="Carlos Cortes",
            batter_id=666126,
            team="ATH",
            game_pk=778900,
        )
        save_pick(_sample_daily(double_down=double), picks_dir)
        save_streak(3, picks_dir)
        mock_statuses.return_value = {
            778899: {"abstract": "F", "detailed": "Final"},
            778900: {"abstract": "F", "detailed": "Postponed"},
        }
        mock_check.return_value = True

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check-results", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
        ])

        assert result.exit_code == 0
        assert "VOID: Carlos Cortes." in result.output
        assert "HIT! Jacob Wilson. Streak: 4" in result.output

        from bts.picks import load_pick, load_streak
        loaded = load_pick("2026-04-01", picks_dir)
        assert loaded.result == "hit"
        assert loaded.slot_results == {"pick": "hit", "double_down": "void"}
        assert load_streak(picks_dir) == 4
        mock_check.assert_called_once()

    @patch("bts.picks.get_game_statuses_detailed")
    @patch("bts.picks.check_hit")
    def test_check_results_all_void_keeps_streak(
        self, mock_check, mock_statuses, tmp_path,
    ):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        double = _sample_pick(
            batter_name="Carlos Cortes",
            batter_id=666126,
            team="ATH",
            game_pk=778900,
        )
        save_pick(_sample_daily(double_down=double), picks_dir)
        save_streak(3, picks_dir)
        mock_statuses.return_value = {
            778899: {"abstract": "F", "detailed": "Postponed"},
            778900: {"abstract": "F", "detailed": "Canceled"},
        }

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check-results", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
        ])

        assert result.exit_code == 0
        assert "VOID: Jacob Wilson, Carlos Cortes." in result.output
        assert "All picks void. Streak unchanged: 3" in result.output

        from bts.picks import load_pick, load_streak
        loaded = load_pick("2026-04-01", picks_dir)
        assert loaded.result == "void"
        assert loaded.slot_results == {"pick": "void", "double_down": "void"}
        assert load_streak(picks_dir) == 3
        mock_check.assert_not_called()

    def test_check_results_skips_already_resolved(self, tmp_path):
        """Scheduler already set result — check-results should not double-count streak."""
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        daily = _sample_daily(result="hit")
        save_pick(daily, picks_dir)
        save_streak(2, picks_dir)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check-results", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
        ])

        assert result.exit_code == 0
        assert "Already resolved" in result.output
        # Streak must NOT be incremented
        from bts.picks import load_streak
        assert load_streak(picks_dir) == 2

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_check_results_resolves_shadow_when_production_already_resolved(
        self, mock_check, _mock_statuses, tmp_path,
    ):
        """Already-resolved production picks should still reconcile shadow results."""
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        save_pick(_sample_daily(result="hit"), picks_dir)
        save_streak(2, picks_dir)
        save_shadow_pick(_sample_daily(
            pick=_sample_pick(
                batter_name="Shadow Batter",
                batter_id=111,
                team="BOS",
                game_pk=999,
            ),
            result=None,
        ), picks_dir)
        mock_check.return_value = True

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check-results", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
        ])

        assert result.exit_code == 0
        assert "Shadow: Shadow Batter — HIT" in result.output
        assert "Already resolved" in result.output
        assert load_shadow_pick("2026-04-01", picks_dir).result == "hit"
        mock_check.assert_called_once_with(
            999, 111,
            batter_name="Shadow Batter", date="2026-04-01", team="BOS",
        )
        from bts.picks import load_streak
        assert load_streak(picks_dir) == 2

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_check_results_writes_shadow_status_artifact(
        self, mock_check, _mock_statuses, tmp_path,
    ):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()
        status_path = tmp_path / "shadow_status.json"

        save_pick(_sample_daily(result="hit"), picks_dir)
        save_streak(2, picks_dir)
        save_shadow_pick(_sample_daily(
            pick=_sample_pick(
                batter_name="Shadow Batter",
                batter_id=111,
                team="BOS",
                game_pk=999,
            ),
            result=None,
        ), picks_dir)
        mock_check.return_value = True

        result = CliRunner().invoke(cli, [
            "check-results", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
            "--shadow-status-output", str(status_path),
        ])

        assert result.exit_code == 0
        assert "Shadow status:" in result.output
        payload = json.loads(status_path.read_text())
        assert payload["schema_version"] == "bts_shadow_cycle_status_v1"
        assert payload["counts"]["resolved_shadow_results"] == 1
        assert payload["coverage"]["unresolved_shadow_dates"] == []

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_check_results_shadow_status_failure_does_not_crash(
        self, mock_check, _mock_statuses, tmp_path,
    ):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        save_pick(_sample_daily(result="hit"), picks_dir)
        save_streak(2, picks_dir)
        save_shadow_pick(_sample_daily(
            pick=_sample_pick(
                batter_name="Shadow Batter",
                batter_id=111,
                team="BOS",
                game_pk=999,
            ),
            result=None,
        ), picks_dir)
        mock_check.return_value = True

        with patch(
            "bts.shadow_eval.build_shadow_cycle_status",
            side_effect=RuntimeError("status failed"),
        ):
            result = CliRunner().invoke(cli, [
                "check-results", "--date", "2026-04-01",
                "--picks-dir", str(picks_dir),
            ])

        assert result.exit_code == 0
        assert "WARNING: Failed to write shadow status" in result.output
        assert load_shadow_pick("2026-04-01", picks_dir).result == "hit"

    @patch("bts.picks.check_hit")
    def test_check_results_skips_already_resolved_shadow(self, mock_check, tmp_path):
        """Shadow reconciliation should be idempotent for resolved shadow files."""
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        save_pick(_sample_daily(result="hit"), picks_dir)
        save_streak(2, picks_dir)
        save_shadow_pick(_sample_daily(
            pick=_sample_pick(
                batter_name="Shadow Batter",
                batter_id=111,
                team="BOS",
                game_pk=999,
            ),
            result="hit",
        ), picks_dir)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check-results", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
        ])

        assert result.exit_code == 0
        assert "Already resolved" in result.output
        assert load_shadow_pick("2026-04-01", picks_dir).result == "hit"
        mock_check.assert_not_called()

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_check_results_resolves_shadow_with_unresolved_production(
        self, mock_check, _mock_statuses, tmp_path,
    ):
        """The normal unresolved-production path should still resolve shadow results."""
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        save_pick(_sample_daily(), picks_dir)
        save_streak(2, picks_dir)
        save_shadow_pick(_sample_daily(
            pick=_sample_pick(
                batter_name="Shadow Batter",
                batter_id=111,
                team="BOS",
                game_pk=999,
            ),
            result=None,
        ), picks_dir)
        mock_check.side_effect = [True, True]

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check-results", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
        ])

        assert result.exit_code == 0
        assert "HIT!" in result.output
        assert "Shadow: Shadow Batter — HIT" in result.output
        assert load_shadow_pick("2026-04-01", picks_dir).result == "hit"

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_check_results_shadow_double_down_is_dd_aware(
        self, mock_check, _mock_statuses, tmp_path,
    ):
        """Shadow double-down days only count as a hit when both picks hit."""
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        save_pick(_sample_daily(result="hit"), picks_dir)
        save_streak(2, picks_dir)
        save_shadow_pick(_sample_daily(
            pick=_sample_pick(
                batter_name="Shadow Primary",
                batter_id=111,
                team="BOS",
                game_pk=999,
            ),
            double_down=_sample_pick(
                batter_name="Shadow Double",
                batter_id=222,
                team="LAD",
                game_pk=1000,
            ),
            result=None,
        ), picks_dir)
        mock_check.side_effect = [True, False]

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check-results", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
        ])

        assert result.exit_code == 0
        assert "Shadow: Shadow Primary + Shadow Double — MISS" in result.output
        assert load_shadow_pick("2026-04-01", picks_dir).result == "miss"

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_check_results_shadow_error_leaves_shadow_unresolved(
        self, mock_check, _mock_statuses, tmp_path,
    ):
        """Shadow API failures should not overwrite an unresolved shadow file."""
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        save_pick(_sample_daily(result="hit"), picks_dir)
        save_shadow_pick(_sample_daily(
            pick=_sample_pick(
                batter_name="Shadow Batter",
                batter_id=111,
                team="BOS",
                game_pk=999,
            ),
            result=None,
        ), picks_dir)
        mock_check.side_effect = RuntimeError("boxscore unavailable")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check-results", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
        ])

        assert result.exit_code == 0
        assert "ERROR: Failed to check shadow result" in result.output
        assert load_shadow_pick("2026-04-01", picks_dir).result is None

    def test_check_results_no_pick_found(self, tmp_path):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "check-results", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
        ])

        assert result.exit_code == 0
        assert "No pick found" in result.output
