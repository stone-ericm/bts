"""Integration tests for bts run and bts check-results CLI commands."""

import json
import pytest
import sys
import types
from unittest.mock import patch, MagicMock
from pathlib import Path
from click.testing import CliRunner

import bts.model

from bts.cli import cli
from bts.picks import (
    Pick, DailyPick, save_pick, save_streak,
    save_shadow_pick, load_shadow_pick, load_streak,
)
from bts.daily_decision import write_decision


@pytest.fixture(autouse=True)
def fake_predict_module(monkeypatch):
    fake_predict = types.ModuleType("bts.model.predict")
    fake_predict.run_pipeline = MagicMock(name="run_pipeline")
    fake_predict.save_blend = MagicMock(name="save_blend")
    fake_predict.load_blend = MagicMock(name="load_blend", return_value=None)
    monkeypatch.setitem(sys.modules, "bts.model.predict", fake_predict)
    monkeypatch.setattr(bts.model, "predict", fake_predict, raising=False)


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
    @patch("bts.picks.get_game_statuses_detailed", return_value={
        778899: {"abstract": "P", "detailed": "Pre-Game"},
        778900: {"abstract": "P", "detailed": "Pre-Game"},
    })
    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    @patch("bts.model.predict.run_pipeline")
    def test_run_saves_pick_and_posts(
        self, mock_pipeline, mock_statuses, _detailed_statuses,
        mock_should_post, mock_post, tmp_path,
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

    @patch("bts.posting.should_post_now", return_value=False)
    @patch("bts.picks.get_game_statuses_detailed", return_value={
        778899: {"abstract": "P", "detailed": "Pre-Game"},
        778900: {"abstract": "P", "detailed": "Pre-Game"},
    })
    @patch("bts.strategy._mdp_action_from")
    @patch("bts.model.predict.run_pipeline")
    @patch("bts.strategy._load_mdp", return_value=None)
    def test_run_uses_fresh_contest_streak_for_live_action(
        self, _load_mdp_mock, mock_pipeline, mock_mdp, _detailed_statuses, _should_post, tmp_path,
    ):
        def action(_mdp, _p, streak, _date, saver):
            assert streak == 7
            assert saver is False
            return "skip"

        mock_pipeline.return_value = _mock_predictions()
        mock_mdp.side_effect = action
        picks_dir = tmp_path / "picks"
        models_dir = tmp_path / "models"
        picks_dir.mkdir()
        save_streak(4, picks_dir, saver_available=True)
        state_dir = picks_dir / "account_state"
        state_dir.mkdir()
        (state_dir / "contest_streak.manual.json").write_text(json.dumps({
            "active_streak": 7,
            "source": "manual_screenshot",
            "source_date": "2026-04-01",
        }))

        result = CliRunner().invoke(cli, [
            "run", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
            "--models-dir", str(models_dir),
        ])

        mock_mdp.assert_called()  # streak-threading side_effect must actually fire
        assert result.exit_code == 0
        assert "Streak holds at 7" in result.output


class TestBtsPreview:
    @patch("bts.picks.get_game_statuses_detailed", return_value={
        778899: {"abstract": "P", "detailed": "Pre-Game"},
        778900: {"abstract": "P", "detailed": "Pre-Game"},
    })
    @patch("bts.strategy._mdp_action_from")
    @patch("bts.model.predict.run_pipeline")
    @patch("bts.strategy._load_mdp", return_value=None)
    def test_preview_uses_fresh_contest_streak_for_projected_pick(
        self, _load_mdp_mock, mock_pipeline, mock_mdp, _detailed_statuses, tmp_path,
    ):
        def action(_mdp, _p, streak, _date, saver):
            assert streak == 7
            assert saver is False
            return "skip"

        mock_pipeline.return_value = _mock_predictions()
        mock_mdp.side_effect = action
        picks_dir = tmp_path / "picks"
        models_dir = tmp_path / "models"
        picks_dir.mkdir()
        save_streak(4, picks_dir, saver_available=True)
        state_dir = picks_dir / "account_state"
        state_dir.mkdir()
        (state_dir / "contest_streak.manual.json").write_text(json.dumps({
            "active_streak": 7,
            "source": "manual_screenshot",
            "source_date": "2026-04-01",
        }))

        result = CliRunner().invoke(cli, [
            "preview", "--date", "2026-04-01",
            "--picks-dir", str(picks_dir),
            "--models-dir", str(models_dir),
        ])

        mock_mdp.assert_called()  # streak-threading side_effect must actually fire
        assert result.exit_code == 0
        assert "Skip day" in result.output
        assert not (picks_dir / "2026-04-01.json").exists()


class TestSetContestStreak:
    def test_set_contest_streak_writes_manual_state(self, tmp_path):
        picks_dir = tmp_path / "picks"

        result = CliRunner().invoke(cli, [
            "set-contest-streak",
            "--streak", "7",
            "--best-streak", "7",
            "--saver-unavailable",
            "--source-date", "2026-05-29",
            "--source", "manual_screenshot",
            "--username", "stonehengee",
            "--picks-dir", str(picks_dir),
        ])

        assert result.exit_code == 0
        path = picks_dir / "account_state" / "contest_streak.manual.json"
        data = json.loads(path.read_text())
        assert data["schema_version"] == "bts_contest_streak_manual_v2"
        assert "override_expires_at" in data
        assert data["active_streak"] == 7
        assert data["best_streak"] == 7
        assert "saver_available" not in data    # deprecated -> no longer written to the manual file
        assert "deprecated" in result.output     # the CLI notes the deprecation
        assert data["source_date"] == "2026-05-29"
        assert data["source"] == "manual_screenshot"
        assert data["username"] == "stonehengee"
        assert "recorded_at" in data

        from bts.contest_state import load_contest_streak_state
        state = load_contest_streak_state(picks_dir)
        assert state is not None
        assert state.streak == 7
        assert state.best_streak == 7
        assert state.saver_available is None    # deprecated: set-contest-streak no longer writes it
        assert state.source == "manual_screenshot"
        assert state.source_date is not None
        assert state.source_date.isoformat() == "2026-05-29"

    def test_set_contest_streak_rejects_best_below_active(self, tmp_path):
        result = CliRunner().invoke(cli, [
            "set-contest-streak",
            "--streak", "7",
            "--best-streak", "6",
            "--picks-dir", str(tmp_path / "picks"),
        ])

        assert result.exit_code != 0
        assert "best streak must be at least the active streak" in result.output


class TestBtsCheckResults:
    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_check_results_hit_updates_streak(self, mock_check, _mock_statuses, tmp_path):
        picks_dir = tmp_path / "picks"
        picks_dir.mkdir()

        save_pick(_sample_daily(bluesky_posted=True), picks_dir)
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

        save_pick(_sample_daily(bluesky_posted=True), picks_dir)
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

        save_pick(_sample_daily(bluesky_posted=True), picks_dir)
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
        save_pick(_sample_daily(double_down=double, bluesky_posted=True), picks_dir)
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
        save_pick(_sample_daily(double_down=double, bluesky_posted=True), picks_dir)
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
        save_pick(_sample_daily(double_down=double, bluesky_posted=True), picks_dir)
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
            return_status=True,
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

        save_pick(_sample_daily(bluesky_posted=True), picks_dir)
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

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_check_results_skips_unscoreable_skip_record(self, mock_check, _s, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        save_pick(_sample_daily(bluesky_posted=False), picks_dir)   # stale preview-style file
        save_streak(5, picks_dir)
        write_decision("2026-04-01", picks_dir, action="skip", source="mdp",
                       delivery_status="not_applicable", scoreable=False)
        result = CliRunner().invoke(cli, ["check-results", "--date", "2026-04-01", "--picks-dir", str(picks_dir)])
        assert result.exit_code == 0
        assert "not scoring" in result.output.lower()
        assert load_streak(picks_dir) == 5            # untouched
        mock_check.assert_not_called()

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit", return_value=True)
    def test_check_results_scores_scoreable_decision(self, _c, _s, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        save_pick(_sample_daily(bluesky_posted=True), picks_dir)
        save_streak(3, picks_dir)
        write_decision("2026-04-01", picks_dir, action="single", source="mdp",
                       delivery_status="delivered", scoreable=True)
        result = CliRunner().invoke(cli, ["check-results", "--date", "2026-04-01", "--picks-dir", str(picks_dir)])
        assert "Streak: 4" in result.output

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit", return_value=True)
    def test_check_results_missing_decision_falls_back_to_delivered(self, _c, _s, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        save_pick(_sample_daily(bluesky_posted=True), picks_dir)   # delivered, no decision.json (legacy)
        save_streak(3, picks_dir)
        result = CliRunner().invoke(cli, ["check-results", "--date", "2026-04-01", "--picks-dir", str(picks_dir)])
        assert "Streak: 4" in result.output

    @patch("bts.picks.get_game_statuses_detailed", return_value={})
    @patch("bts.picks.check_hit")
    def test_check_results_missing_decision_undelivered_not_scored(self, mock_check, _s, tmp_path):
        # the core #144 case: a stale preview <date>.json on a skip day, no decision.json, undelivered
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        save_pick(_sample_daily(bluesky_posted=False), picks_dir)
        save_streak(5, picks_dir)
        result = CliRunner().invoke(cli, ["check-results", "--date", "2026-04-01", "--picks-dir", str(picks_dir)])
        assert load_streak(picks_dir) == 5
        mock_check.assert_not_called()


class TestFetchContestStreak:
    @staticmethod
    def _patch_auth(monkeypatch, username="stonehengee", user_id=50311):
        import bts.leaderboard.auth as auth
        from bts.leaderboard.auth import AuthSession
        monkeypatch.setattr(auth, "load_session_cookies", lambda: {"oktaid": "uid"})
        monkeypatch.setattr(auth, "extract_uid", lambda c: "uid")
        monkeypatch.setattr(auth, "fetch_login_session",
                            lambda *a, **k: AuthSession(xsid="x_1", user_id=user_id, username=username))

    def test_happy_path_writes_auto(self, monkeypatch, tmp_path):
        import datetime as dt
        import bts.contest_fetch as cf
        import bts.cli as climod
        self._patch_auth(monkeypatch)
        monkeypatch.setattr(cf, "fetch_profile", lambda *a, **k: {
            "activeStreak": 0, "seasonBestStreak": 9,
            "predictions": [{"roundId": 1, "result": "hit"}]})
        monkeypatch.setattr(climod, "_fetch_rounds", lambda *a, **k: {1: dt.date(2026, 6, 6)})
        picks = tmp_path / "picks"; picks.mkdir()
        (picks / "2026-06-05.json").write_text(json.dumps({"result": "hit"}))
        r = CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks),
                                     "--expected-username", "stonehengee"])
        assert r.exit_code == 0, r.output
        data = json.loads((picks / "account_state" / "contest_streak.json").read_text())
        assert data["active_streak"] == 0 and data["best_streak"] == 9
        assert data["schema_version"] == "bts_contest_streak_auto_v1"
        assert data["source_date"] == "2026-06-06"
        assert not list((picks / "account_state").glob("*.tmp"))   # atomic, no temp left

    def test_identity_mismatch_no_write_and_alerts(self, monkeypatch, tmp_path):
        import bts.contest_fetch as cf
        import bts.dm
        self._patch_auth(monkeypatch, username="someone_else")
        monkeypatch.setattr(cf, "fetch_profile", lambda *a, **k: {
            "activeStreak": 0, "seasonBestStreak": 9, "predictions": []})
        dm_calls = []
        monkeypatch.setattr(bts.dm, "send_dm", lambda h, m: dm_calls.append((h, m)))
        picks = tmp_path / "picks"; picks.mkdir()
        r = CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks),
                                     "--expected-username", "stonehengee", "--dm-recipient", "x.bsky.social"])
        assert r.exit_code != 0
        assert not (picks / "account_state" / "contest_streak.json").exists()
        assert len(dm_calls) == 1

    def test_genuinely_stale_profile_still_writes_snapshot(self, monkeypatch, tmp_path):
        # source 6/1 with FOUR settled picks newer (6/2..6/5). The activeStreak counter
        # is still current, so the snapshot/coverage split WRITES it (exit 0, no fetch DM);
        # contest_state + the level-aware health check surface the >=2-pick staleness.
        import datetime as dt
        import bts.contest_fetch as cf
        import bts.cli as climod
        import bts.dm
        self._patch_auth(monkeypatch)
        monkeypatch.setattr(cf, "fetch_profile", lambda *a, **k: {
            "activeStreak": 0, "seasonBestStreak": 9,
            "predictions": [{"roundId": 1, "result": "hit"}]})
        monkeypatch.setattr(climod, "_fetch_rounds", lambda *a, **k: {1: dt.date(2026, 6, 1)})
        dm_calls = []
        monkeypatch.setattr(bts.dm, "send_dm", lambda h, m: dm_calls.append((h, m)))
        picks = tmp_path / "picks"; picks.mkdir()
        for d in ("2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"):
            (picks / f"{d}.json").write_text(json.dumps({"result": "hit"}))
        r = CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks),
                                     "--dm-recipient", "x.bsky.social"])
        assert r.exit_code == 0, r.output
        data = json.loads((picks / "account_state" / "contest_streak.json").read_text())
        assert data["active_streak"] == 0 and data["source_date"] == "2026-06-01"
        assert dm_calls == []   # no fetch-level staleness DM; health check handles gap>=2

    def test_expected_settlement_lag_writes_snapshot_no_alert(self, monkeypatch, tmp_path):
        # source 6/4 with ONE settled pick newer (6/5) == the expected overnight lag.
        # The current activeStreak is WRITTEN (snapshot/coverage split); no DM noise.
        import datetime as dt
        import bts.contest_fetch as cf
        import bts.cli as climod
        import bts.dm
        self._patch_auth(monkeypatch)
        monkeypatch.setattr(cf, "fetch_profile", lambda *a, **k: {
            "activeStreak": 5, "seasonBestStreak": 9,
            "predictions": [{"roundId": 1, "result": "hit"}]})
        monkeypatch.setattr(climod, "_fetch_rounds", lambda *a, **k: {1: dt.date(2026, 6, 4)})
        dm_calls = []
        monkeypatch.setattr(bts.dm, "send_dm", lambda h, m: dm_calls.append((h, m)))
        picks = tmp_path / "picks"; picks.mkdir()
        (picks / "2026-06-05.json").write_text(json.dumps({"result": "miss"}))  # latest 6/5, gap=1
        r = CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks),
                                     "--dm-recipient", "x.bsky.social"])
        assert r.exit_code == 0, r.output
        data = json.loads((picks / "account_state" / "contest_streak.json").read_text())
        assert data["active_streak"] == 5 and data["source_date"] == "2026-06-04"
        assert dm_calls == []  # no noise DM on the lag

    def test_auth_failure_dm_throttled(self, monkeypatch, tmp_path):
        import bts.leaderboard.auth as auth
        import bts.dm
        def _raise(*a, **k):
            raise auth.AuthError("cookies expired")
        monkeypatch.setattr(auth, "load_session_cookies", _raise)
        dm_calls = []
        monkeypatch.setattr(bts.dm, "send_dm", lambda h, m: dm_calls.append(m))
        picks = tmp_path / "picks"; picks.mkdir()
        args = ["fetch-contest-streak", "--picks-dir", str(picks), "--dm-recipient", "x.bsky.social"]
        assert CliRunner().invoke(cli, args).exit_code != 0
        assert CliRunner().invoke(cli, args).exit_code != 0   # within cooldown -> no second DM
        assert len(dm_calls) == 1

    def test_malformed_profile_alerts_not_silent(self, monkeypatch, tmp_path):
        import bts.contest_fetch as cf
        import bts.cli as climod
        import bts.dm
        self._patch_auth(monkeypatch)
        monkeypatch.setattr(cf, "fetch_profile", lambda *a, **k: None)   # success=null
        monkeypatch.setattr(climod, "_fetch_rounds", lambda *a, **k: {})
        dm_calls = []
        monkeypatch.setattr(bts.dm, "send_dm", lambda h, m: dm_calls.append(m))
        picks = tmp_path / "picks"; picks.mkdir()
        r = CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks),
                                     "--expected-username", "stonehengee", "--dm-recipient", "x.bsky.social"])
        assert r.exit_code != 0
        assert not (picks / "account_state" / "contest_streak.json").exists()
        assert len(dm_calls) == 1   # alerted, not a silent crash

    def test_prior_auto_identity_mismatch_no_overwrite(self, monkeypatch, tmp_path):
        import datetime as dt
        import bts.contest_fetch as cf
        import bts.cli as climod
        import bts.dm
        self._patch_auth(monkeypatch, username="stonehengee", user_id=50311)
        monkeypatch.setattr(cf, "fetch_profile", lambda *a, **k: {
            "activeStreak": 0, "seasonBestStreak": 9, "predictions": [{"roundId": 1, "result": "hit"}]})
        monkeypatch.setattr(climod, "_fetch_rounds", lambda *a, **k: {1: dt.date(2026, 6, 6)})
        dm_calls = []
        monkeypatch.setattr(bts.dm, "send_dm", lambda h, m: dm_calls.append(m))
        picks = tmp_path / "picks"; (picks / "account_state").mkdir(parents=True)
        (picks / "account_state" / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1", "active_streak": 3, "best_streak": 9,
            "source": "mlb_bts_profile", "source_date": "2026-06-06",
            "user_id": 99999, "username": "someone_else"}))
        r = CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks),
                                     "--dm-recipient", "x.bsky.social"])
        assert r.exit_code != 0
        data = json.loads((picks / "account_state" / "contest_streak.json").read_text())
        assert data["username"] == "someone_else" and data["active_streak"] == 3   # untouched
        assert len(dm_calls) == 1


class TestCheckPickEntered:
    """check-pick-entered: pre-first-pitch 'did the pick get into the MLB app' DM."""

    @staticmethod
    def _patch_auth(monkeypatch, username="stonehengee", user_id=50311):
        import bts.leaderboard.auth as auth
        from bts.leaderboard.auth import AuthSession
        monkeypatch.setattr(auth, "load_session_cookies", lambda: {"oktaid": "uid"})
        monkeypatch.setattr(auth, "extract_uid", lambda c: "uid")
        monkeypatch.setattr(auth, "fetch_login_session",
                            lambda *a, **k: AuthSession(xsid="x_1", user_id=user_id, username=username))

    @staticmethod
    def _patch_crosswalk(monkeypatch, mapping):
        """Patch the BTS-playerId -> MLB-feedId crosswalk fetch."""
        import bts.cli as climod
        monkeypatch.setattr(climod, "_fetch_bts_to_mlb", lambda *a, **k: dict(mapping))

    @staticmethod
    def _save_pick(picks_dir, date_str, game_time_iso, batter_id=1, dd_batter_id=None,
                   delivered=True):
        """Save a pick. delivered=True marks it committed (dm-delivered) so the
        entry check runs; delivered=False models an undelivered preview/deferred
        pick that was never committed/locked."""
        from bts.picks import DailyPick, Pick, save_pick

        def _mk(name, bid, gpk):
            return Pick(batter_name=name, batter_id=bid, team="NYY",
                        lineup_position=1, pitcher_name="P", pitcher_id=2,
                        p_game_hit=0.8, flags=[], projected_lineup=False,
                        game_pk=gpk, game_time=game_time_iso)

        daily = DailyPick(
            date=date_str, run_time=f"{date_str}T15:00:00+00:00",
            pick=_mk("Test Batter", batter_id, 1),
            double_down=_mk("DD Batter", dd_batter_id, 2) if dd_batter_id else None,
            runner_up=None,
            notification_sent=delivered,
            notification_channel="bluesky_dm" if delivered else None,
            notification_id="dm_x" if delivered else None,
        )
        save_pick(daily, picks_dir)

    @staticmethod
    def _write_decision(picks_dir, date_str, *, action, scoreable, delivery_status):
        """Write a decision.json commit record (the authoritative gate source)."""
        from bts.daily_decision import write_decision
        write_decision(date_str, picks_dir, action=action, source="test",
                       delivery_status=delivery_status, scoreable=scoreable)

    def _run(self, picks, now_et, extra=None):
        return CliRunner().invoke(cli, [
            "check-pick-entered", "--picks-dir", str(picks),
            "--dm-recipient", "x.bsky.social", "--now-et", now_et,
        ] + (extra or []))

    def _setup(self, monkeypatch, *, profile_preds=(), pending=(), crosswalk=None):
        import datetime as dt
        import bts.contest_fetch as cf
        import bts.cli as climod
        import bts.dm
        self._patch_auth(monkeypatch)
        monkeypatch.setattr(cf, "fetch_profile",
                            lambda *a, **k: {"predictions": list(profile_preds)})
        monkeypatch.setattr(cf, "fetch_pending_predictions", lambda *a, **k: list(pending))
        monkeypatch.setattr(climod, "_fetch_rounds", lambda *a, **k: {7: dt.date(2026, 6, 12)})
        self._patch_crosswalk(monkeypatch, crosswalk or {})
        dms = []
        monkeypatch.setattr(bts.dm, "send_dm", lambda h, m: dms.append((h, m)))
        return dms

    @staticmethod
    def _pending(player_id, round_id=7, number=1):
        return {"roundId": round_id, "unitId": 1, "playerId": player_id,
                "number": number, "result": None}

    def _status(self, tmp_path):
        return json.loads((tmp_path / "health_state" / "pick_entry_check.json").read_text())

    def test_missing_entry_in_window_dms_and_exits_nonzero(self, monkeypatch, tmp_path):
        dms = self._setup(monkeypatch, pending=[], crosswalk={100: 1})
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00", batter_id=1)

        r = self._run(picks, "2026-06-12T18:30:00")  # 40 min before pitch
        assert r.exit_code != 0
        assert len(dms) == 1 and "NOT entered" in dms[0][1]
        status = self._status(tmp_path)
        assert status["date"] == "2026-06-12" and status["status"] == "alerted"
        assert status["reason"] == "no_pick"

    def test_uncommitted_pick_in_window_no_dm(self, monkeypatch, tmp_path):
        # A {date}.json exists (preview/deferred) but the pick was never
        # committed/locked (no decision.json, not delivered). The scheduler
        # rewrites {date}.json all day with projections, so its mere existence
        # must NOT trigger a "you didn't enter your pick" nag. Regression for the
        # 2026-07-06 premature DM on a deferred double-down.
        dms = self._setup(monkeypatch, pending=[], crosswalk={100: 1})
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00",
                        batter_id=1, delivered=False)

        r = self._run(picks, "2026-06-12T18:30:00")  # 40 min before pitch, in window
        assert r.exit_code == 0, r.output
        assert dms == []
        assert not (tmp_path / "health_state" / "pick_entry_check.json").exists()

    def test_no_alert_inside_submission_cutoff(self, monkeypatch, tmp_path):
        # Within 5 min of first pitch the pick can no longer be submitted, so
        # "Fix it now!" is useless (and the cutoff countdown would go negative).
        # The firing window must exclude the un-submittable final 5 minutes.
        dms = self._setup(monkeypatch, pending=[], crosswalk={100: 1})
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00", batter_id=1)

        r = self._run(picks, "2026-06-12T19:06:00")  # 4 min to first pitch (inside cutoff)
        assert dms == [], r.output
        assert not (tmp_path / "health_state" / "pick_entry_check.json").exists()

    def test_dm_countdown_uses_submission_cutoff(self, monkeypatch, tmp_path):
        # BTS rejects submissions within 5 min of first pitch, so the true
        # deadline is first pitch - 5. The DM countdown must report minutes to
        # that cutoff, not minutes to first pitch.
        dms = self._setup(monkeypatch, pending=[], crosswalk={100: 1})
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00", batter_id=1)

        r = self._run(picks, "2026-06-12T18:30:00")  # 40 min to first pitch -> 35 to cutoff
        assert r.exit_code != 0
        assert len(dms) == 1
        msg = dms[0][1]
        assert "35 min to submit" in msg, msg
        assert "(40 min" not in msg

    def test_committed_via_decision_record_alerts(self, monkeypatch, tmp_path):
        # Exercises the real decision.json gate, NOT the delivered fallback: the
        # pick file carries no delivery flags, but a scoreable commit record
        # exists (e.g. private_locked / locked_unconfirmed), so the alert fires.
        dms = self._setup(monkeypatch, pending=[], crosswalk={100: 1})
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00",
                        batter_id=1, delivered=False)
        self._write_decision(picks, "2026-06-12", action="single",
                             scoreable=True, delivery_status="private_locked")

        r = self._run(picks, "2026-06-12T18:30:00")
        assert r.exit_code != 0
        assert len(dms) == 1 and "NOT entered" in dms[0][1]

    def test_skip_day_decision_no_dm(self, monkeypatch, tmp_path):
        # A non-scoreable decision (skip day) must not alert, even though a
        # {date}.json preview exists.
        dms = self._setup(monkeypatch, pending=[], crosswalk={100: 1})
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00",
                        batter_id=1, delivered=False)
        self._write_decision(picks, "2026-06-12", action="skip",
                             scoreable=False, delivery_status="not_applicable")

        r = self._run(picks, "2026-06-12T18:30:00")
        assert r.exit_code == 0, r.output
        assert dms == []
        assert not (tmp_path / "health_state" / "pick_entry_check.json").exists()

    def test_matching_entry_confirms_without_dm(self, monkeypatch, tmp_path):
        # Entered pending pick (BTS id 100) maps to the delivered MLB id 1 -> match.
        dms = self._setup(monkeypatch, pending=[self._pending(100)], crosswalk={100: 1})
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00", batter_id=1)

        r = self._run(picks, "2026-06-12T18:30:00")
        assert r.exit_code == 0, r.output
        assert dms == []
        assert self._status(tmp_path)["reason"] == "match"

    def test_wrong_player_entered_alerts_mismatch(self, monkeypatch, tmp_path):
        # Delivered MLB id 1, but the entered pick (BTS 200) maps to MLB 2 -> mismatch.
        dms = self._setup(monkeypatch, pending=[self._pending(200)], crosswalk={100: 1, 200: 2})
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00", batter_id=1)

        r = self._run(picks, "2026-06-12T18:30:00")
        assert r.exit_code != 0
        assert len(dms) == 1 and "does NOT match" in dms[0][1]
        assert self._status(tmp_path)["reason"] == "mismatch"

    def test_double_down_slot_missing_alerts(self, monkeypatch, tmp_path):
        # Delivered primary(1)+DD(2); only the primary is entered -> mismatch.
        dms = self._setup(monkeypatch, pending=[self._pending(100)], crosswalk={100: 1, 200: 2})
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00",
                        batter_id=1, dd_batter_id=2)

        r = self._run(picks, "2026-06-12T18:30:00")
        assert r.exit_code != 0
        assert self._status(tmp_path)["reason"] == "mismatch"

    def test_both_double_down_slots_entered_confirms(self, monkeypatch, tmp_path):
        dms = self._setup(monkeypatch,
                          pending=[self._pending(100, number=1), self._pending(200, number=2)],
                          crosswalk={100: 1, 200: 2})
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00",
                        batter_id=1, dd_batter_id=2)

        r = self._run(picks, "2026-06-12T18:30:00")
        assert r.exit_code == 0, r.output
        assert dms == []
        assert self._status(tmp_path)["reason"] == "match"

    def test_unresolved_crosswalk_falls_back_to_present(self, monkeypatch, tmp_path):
        # A pick IS entered but our crosswalk can't map it -> don't false-alarm;
        # confirm as present-but-unverified (OUR gap must never page Eric).
        dms = self._setup(monkeypatch, pending=[self._pending(999)], crosswalk={})
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00", batter_id=1)

        r = self._run(picks, "2026-06-12T18:30:00")
        assert r.exit_code == 0, r.output
        assert dms == []
        assert self._status(tmp_path)["reason"] == "present_unverified"

    def test_profile_nested_roundpredictions_resolve_identity(self, monkeypatch, tmp_path):
        # A settled-but-same-day pick can arrive via the nested profile shape.
        dms = self._setup(
            monkeypatch,
            profile_preds=[{"roundId": 7, "result": "hit",
                            "roundPredictions": [{"number": 1, "playerId": 100}]}],
            pending=[], crosswalk={100: 1})
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00", batter_id=1)

        r = self._run(picks, "2026-06-12T18:30:00")
        assert r.exit_code == 0, r.output
        assert self._status(tmp_path)["reason"] == "match"

    def test_dm_failure_leaves_retryable_marker_and_realerts(self, monkeypatch, tmp_path):
        # THE Codex #1 fix: if the DM send throws, we must NOT burn the daily
        # marker as 'alerted' — the next run has to retry, or the one alert this
        # feature exists for is silently lost.
        import bts.dm
        self._setup(monkeypatch, pending=[], crosswalk={100: 1})
        def _boom(h, m):
            raise RuntimeError("bluesky down")
        monkeypatch.setattr(bts.dm, "send_dm", _boom)
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00", batter_id=1)

        r1 = self._run(picks, "2026-06-12T18:30:00")
        assert r1.exit_code != 0
        assert self._status(tmp_path)["status"] == "dm_failed"

        # Second run: the DM now succeeds -> it must re-alert (marker was retryable)
        dms = []
        monkeypatch.setattr(bts.dm, "send_dm", lambda h, m: dms.append((h, m)))
        r2 = self._run(picks, "2026-06-12T18:45:00")
        assert r2.exit_code != 0
        assert len(dms) == 1
        assert self._status(tmp_path)["status"] == "alerted"

    def test_outside_window_is_silent_no_network(self, monkeypatch, tmp_path):
        import bts.contest_fetch as cf
        def _boom(*a, **k):
            raise AssertionError("network must not be touched outside the window")
        monkeypatch.setattr(cf, "fetch_profile", _boom)
        monkeypatch.setattr(cf, "fetch_pending_predictions", _boom)
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00")

        r = self._run(picks, "2026-06-12T12:00:00")  # 7 hours early
        assert r.exit_code == 0, r.output

    def test_marker_dedupes_second_alert(self, monkeypatch, tmp_path):
        dms = self._setup(monkeypatch, pending=[], crosswalk={100: 1})
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00", batter_id=1)

        assert self._run(picks, "2026-06-12T18:30:00").exit_code != 0
        r2 = self._run(picks, "2026-06-12T18:45:00")
        assert r2.exit_code == 0  # already alerted; quiet
        assert len(dms) == 1

    def test_pending_fetch_failure_skips_quietly_no_marker(self, monkeypatch, tmp_path):
        # A transient /predictions failure must NOT produce a false "not
        # entered" DM (the exact false-alarm class that killed v1) and must
        # NOT consume the once-per-day marker — the next 15-min cron retries.
        import httpx
        import bts.contest_fetch as cf
        dms = self._setup(monkeypatch, pending=[], crosswalk={100: 1})
        def _fail(*a, **k):
            raise httpx.ConnectError("transient")
        monkeypatch.setattr(cf, "fetch_pending_predictions", _fail)
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00")

        r = self._run(picks, "2026-06-12T18:30:00")
        assert r.exit_code == 0, r.output
        assert dms == []
        assert not (tmp_path / "health_state" / "pick_entry_check.json").exists()

    def test_schema_drift_skips_quietly_no_false_alarm(self, monkeypatch, tmp_path):
        # A drifted 200 from /predictions raises ContestFetchError -> quiet skip,
        # NOT a false "not entered" alert, and no marker consumed.
        import bts.contest_fetch as cf
        dms = self._setup(monkeypatch, pending=[], crosswalk={100: 1})
        def _drift(*a, **k):
            raise cf.ContestFetchError("success.predictions missing")
        monkeypatch.setattr(cf, "fetch_pending_predictions", _drift)
        picks = tmp_path / "picks"; picks.mkdir()
        self._save_pick(picks, "2026-06-12", "2026-06-12T23:10:00+00:00")

        r = self._run(picks, "2026-06-12T18:30:00")
        assert r.exit_code == 0, r.output
        assert dms == []
        assert not (tmp_path / "health_state" / "pick_entry_check.json").exists()

    def test_no_pick_today_is_silent(self, tmp_path):
        picks = tmp_path / "picks"; picks.mkdir()
        r = self._run(picks, "2026-06-12T18:30:00")
        assert r.exit_code == 0


class TestSaverStateCli:
    def _run(self, picks_dir, *args):
        return CliRunner().invoke(cli, ["saver-state", "--picks-dir", str(picks_dir),
                                        "--season", "2026", *args])

    def test_show_uninitialized(self, tmp_path):
        r = self._run(tmp_path, "--show")
        assert r.exit_code == 0 and "uninitialized" in r.output

    def test_init_active(self, tmp_path):
        assert self._run(tmp_path, "--init", "active").exit_code == 0
        assert "active" in self._run(tmp_path, "--show").output

    def test_init_guarded_requires_force(self, tmp_path):
        self._run(tmp_path, "--init", "not_earned")
        r = self._run(tmp_path, "--init", "active")     # already initialized -> rejected
        assert r.exit_code != 0 and "force" in r.output.lower()
        assert "not_earned" in self._run(tmp_path, "--show").output   # unchanged
        assert self._run(tmp_path, "--init", "active", "--force").exit_code == 0
        assert "active" in self._run(tmp_path, "--show").output

    def test_use_and_undo(self, tmp_path):
        self._run(tmp_path, "--init", "active")
        assert self._run(tmp_path, "--use").exit_code == 0
        assert "used" in self._run(tmp_path, "--show").output
        assert self._run(tmp_path, "--undo").exit_code == 0
        assert "active" in self._run(tmp_path, "--show").output

    def test_use_noop_when_not_active(self, tmp_path):
        self._run(tmp_path, "--init", "not_earned")
        self._run(tmp_path, "--use")     # not active -> no-op
        assert "not_earned" in self._run(tmp_path, "--show").output
