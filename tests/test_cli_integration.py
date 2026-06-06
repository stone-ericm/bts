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
    save_shadow_pick, load_shadow_pick,
)


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
    @patch("bts.strategy._mdp_action")
    @patch("bts.model.predict.run_pipeline")
    def test_run_uses_fresh_contest_streak_for_live_action(
        self, mock_pipeline, mock_mdp, _detailed_statuses, _should_post, tmp_path,
    ):
        def action(_p, streak, _date, saver):
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

        assert result.exit_code == 0
        assert "Streak holds at 7" in result.output


class TestBtsPreview:
    @patch("bts.picks.get_game_statuses_detailed", return_value={
        778899: {"abstract": "P", "detailed": "Pre-Game"},
        778900: {"abstract": "P", "detailed": "Pre-Game"},
    })
    @patch("bts.strategy._mdp_action")
    @patch("bts.model.predict.run_pipeline")
    def test_preview_uses_fresh_contest_streak_for_projected_pick(
        self, mock_pipeline, mock_mdp, _detailed_statuses, tmp_path,
    ):
        def action(_p, streak, _date, saver):
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
        assert data["saver_available"] is False
        assert data["source_date"] == "2026-05-29"
        assert data["source"] == "manual_screenshot"
        assert data["username"] == "stonehengee"
        assert "recorded_at" in data

        from bts.contest_state import load_contest_streak_state
        state = load_contest_streak_state(picks_dir)
        assert state is not None
        assert state.streak == 7
        assert state.best_streak == 7
        assert state.saver_available is False
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

    def test_stale_profile_no_write(self, monkeypatch, tmp_path):
        import datetime as dt
        import bts.contest_fetch as cf
        import bts.cli as climod
        self._patch_auth(monkeypatch)
        monkeypatch.setattr(cf, "fetch_profile", lambda *a, **k: {
            "activeStreak": 0, "seasonBestStreak": 9,
            "predictions": [{"roundId": 1, "result": "hit"}]})
        monkeypatch.setattr(climod, "_fetch_rounds", lambda *a, **k: {1: dt.date(2026, 6, 1)})
        picks = tmp_path / "picks"; picks.mkdir()
        (picks / "2026-06-05.json").write_text(json.dumps({"result": "hit"}))   # latest 6/5 > source 6/1
        r = CliRunner().invoke(cli, ["fetch-contest-streak", "--picks-dir", str(picks)])
        assert r.exit_code != 0
        assert not (picks / "account_state" / "contest_streak.json").exists()

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
