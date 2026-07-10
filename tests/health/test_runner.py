"""Tests for the health-check runner aggregator.

The runner's job: call each check, isolate failures, log, dispatch DM.
We verify (a) it calls every check, (b) per-check exceptions are isolated,
(c) DM dispatcher is called once with the aggregated alert list.
"""

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

from bts.health.alert import Alert
from bts.health.runner import run_all_checks


def _set_up_picks_dir(picks_dir, models_dir):
    """Set up a minimal valid picks state so most checks return clean."""
    # streak.json
    (picks_dir / "streak.json").write_text(json.dumps({"streak": 7, "saver_available": True}))
    # tomorrow's blend pkl exists (no blend_training alert)
    (models_dir / "blend_2026-04-28.pkl").write_text("")
    # today's pick file with bluesky posted
    (picks_dir / "2026-04-27.json").write_text(json.dumps({
        "date": "2026-04-27",
        "pick": {"batter_name": "X", "p_game_hit": 0.75, "projected_lineup": False},
        "double_down": {"batter_name": "Y", "p_game_hit": 0.72, "projected_lineup": False},
        "result": "hit",
        "bluesky_posted": True,
        "bluesky_uri": "at://abc/def",
    }))


class TestRunAllChecks:
    def test_clean_state_no_critical_no_dm(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        # Mock disk_usage to a clean 50% — these tests must be host-state-independent
        # (a dev box at 95% disk is real but not what these integration tests exercise).
        from collections import namedtuple
        _Usage = namedtuple("_Usage", ["total", "used", "free"])
        clean_usage = _Usage(total=100 * 1024 ** 3, used=50 * 1024 ** 3, free=50 * 1024 ** 3)
        with patch("bts.health.disk_fill.shutil.disk_usage", return_value=clean_usage), \
             patch("bts.health.runner.dispatch_dm_for_health_alerts") as mock_dm:
            mock_dm.return_value = False
            alerts = run_all_checks(
                picks_dir=picks_dir, models_dir=models_dir,
                dm_recipient="x.bsky.social",
                today=date(2026, 4, 27),
            )
            # Clean state — should produce no CRITICAL
            crits = [a for a in alerts if a.level == "CRITICAL"]
            assert crits == []
            mock_dm.assert_called_once()

    def test_missing_blend_triggers_critical(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        # Remove tomorrow's pkl
        (models_dir / "blend_2026-04-28.pkl").unlink()
        alerts = run_all_checks(
            picks_dir=picks_dir, models_dir=models_dir,
            dm_recipient=None, today=date(2026, 4, 27),
        )
        sources = [a.source for a in alerts if a.level == "CRITICAL"]
        assert "blend_training" in sources

    def test_missing_streak_triggers_critical(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        (picks_dir / "streak.json").unlink()
        alerts = run_all_checks(
            picks_dir=picks_dir, models_dir=models_dir,
            dm_recipient=None, today=date(2026, 4, 27),
        )
        assert any(a.source == "streak_validation" and a.level == "CRITICAL" for a in alerts)

    def test_expected_contest_state_missing_triggers_critical(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)

        alerts = run_all_checks(
            picks_dir=picks_dir,
            models_dir=models_dir,
            dm_recipient=None,
            today=date(2026, 4, 27),
            contest_state_expected=True,
        )

        assert any(a.source == "contest_state" and a.level == "CRITICAL" for a in alerts)

    def test_per_check_failure_isolated(self, tmp_path):
        # If one check raises, the others still run
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        with patch("bts.health.runner.calibration.check", side_effect=RuntimeError("boom")):
            # The other checks (especially streak_validation) should still pass
            alerts = run_all_checks(
                picks_dir=picks_dir, models_dir=models_dir,
                dm_recipient=None, today=date(2026, 4, 27),
            )
            # No alerts from calibration (it failed) but no exception either
            cal_alerts = [a for a in alerts if a.source == "calibration_drift"]
            assert cal_alerts == []
            # Other checks ran cleanly
            assert isinstance(alerts, list)

    def test_crashed_check_emits_critical(self, tmp_path):
        """A check that raises must surface a CRITICAL from the runner — not
        vanish into the logs (dead-smoke-detector guard, audit H3)."""
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        with patch("bts.health.runner.calibration.check",
                   side_effect=RuntimeError("boom")):
            alerts = run_all_checks(
                picks_dir=picks_dir, models_dir=models_dir,
                dm_recipient=None, today=date(2026, 4, 27),
            )
        crash = [a for a in alerts
                 if a.source == "health_runner" and a.level == "CRITICAL"]
        assert len(crash) == 1, alerts
        assert "calibration" in crash[0].message and "boom" in crash[0].message

    def test_runs_postponed_pick_check(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        expected = Alert(level="INFO", source="postponed_pick", message="called")
        with patch("bts.health.runner.postponed_pick.check", return_value=[expected]) as mock_check:
            alerts = run_all_checks(
                picks_dir=picks_dir, models_dir=models_dir,
                dm_recipient=None, today=date(2026, 4, 27),
            )

        mock_check.assert_called_once_with(picks_dir, today=date(2026, 4, 27))
        assert expected in alerts

    def test_runs_fallback_defer_check(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        expected = Alert(level="INFO", source="fallback_defer", message="called")
        with patch(
            "bts.health.runner.fallback_defer.check",
            return_value=[expected],
        ) as mock_check:
            alerts = run_all_checks(
                picks_dir=picks_dir, models_dir=models_dir,
                dm_recipient=None, today=date(2026, 4, 27),
            )

        mock_check.assert_called_once_with(picks_dir, today=date(2026, 4, 27))
        assert expected in alerts

    def test_runs_backup_freshness_check(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        expected = Alert(level="INFO", source="backup_freshness", message="called")
        with patch(
            "bts.health.runner.backup_freshness.check",
            return_value=[expected],
        ) as mock_check:
            alerts = run_all_checks(
                picks_dir=picks_dir, models_dir=models_dir,
                dm_recipient=None, today=date(2026, 4, 27),
            )

        mock_check.assert_called_once_with(
            tmp_path / "health_state", thresholds=None,
        )
        assert expected in alerts

    def test_backup_freshness_is_always_attention(self):
        from bts.health.attention import ALWAYS_ATTENTION_WARN_SOURCES
        assert "backup_freshness" in ALWAYS_ATTENTION_WARN_SOURCES

    def test_dm_dispatcher_called_with_full_alert_list(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        # Force a CRITICAL via missing streak
        (picks_dir / "streak.json").unlink()
        with patch("bts.health.runner.dispatch_dm_for_health_alerts") as mock_dm:
            run_all_checks(
                picks_dir=picks_dir, models_dir=models_dir,
                dm_recipient="x.bsky.social", today=date(2026, 4, 27),
            )
            mock_dm.assert_called_once()
            args = mock_dm.call_args.args
            alerts_arg = args[0]
            # Includes the streak_validation CRITICAL
            assert any(a.source == "streak_validation" for a in alerts_arg)
            assert mock_dm.call_args.kwargs["status_path"] == (
                tmp_path / "health_state" / "health_dm_delivery_status.json"
            )

    def test_skips_restart_check_when_nrestarts_none(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        alerts = run_all_checks(
            picks_dir=picks_dir, models_dir=models_dir,
            dm_recipient=None, today=date(2026, 4, 27),
            current_nrestarts=None,  # not provided
        )
        # restart_spike not in any alert source
        assert all(a.source != "restart_spike" for a in alerts)

    def test_runs_restart_check_when_nrestarts_provided(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        # First call records baseline
        run_all_checks(
            picks_dir=picks_dir, models_dir=models_dir,
            dm_recipient=None, today=date(2026, 4, 27),
            current_nrestarts=52,
        )
        # Second call with spike
        alerts = run_all_checks(
            picks_dir=picks_dir, models_dir=models_dir,
            dm_recipient=None, today=date(2026, 4, 27),
            current_nrestarts=99,
        )
        assert any(a.source == "restart_spike" and a.level == "CRITICAL" for a in alerts)

    def test_skips_memory_check_when_pid_none(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        alerts = run_all_checks(
            picks_dir=picks_dir, models_dir=models_dir,
            dm_recipient=None, today=date(2026, 4, 27),
            scheduler_pid=None,
        )
        assert all(a.source != "memory_growth" for a in alerts)

    def test_passes_shadow_unit_to_analytics_artifact_check(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        with patch(
            "bts.health.runner.analytics_artifacts_missing.check",
            return_value=[],
        ) as mock_check:
            run_all_checks(
                picks_dir=picks_dir,
                models_dir=models_dir,
                dm_recipient=None,
                today=date(2026, 4, 27),
                shadow_model_enabled=True,
                shadow_unit="bts-shadow-prediction.service",
            )

        assert mock_check.call_args.kwargs["shadow_expected"] is True
        assert mock_check.call_args.kwargs["shadow_unit"] == (
            "bts-shadow-prediction.service"
        )
        assert mock_check.call_args.kwargs["scheduler_unit"] == "bts-scheduler.service"

    def test_runs_live_forward_resolution_when_capture_enabled(self, tmp_path):
        repo_root = tmp_path
        picks_dir = repo_root / "data" / "picks"; picks_dir.mkdir(parents=True)
        models_dir = repo_root / "data" / "models"; models_dir.mkdir(parents=True)
        _set_up_picks_dir(picks_dir, models_dir)
        expected = Alert("WARN", "live_forward_resolution", "stalled")
        with patch(
            "bts.health.runner.analytics_artifacts_missing.check",
            return_value=[],
        ), patch(
            "bts.health.runner.live_forward_resolution.check",
            return_value=[expected],
        ) as mock_resolve, patch(
            "bts.health.runner.dispatch_dm_for_health_alerts"
        ):
            alerts = run_all_checks(
                picks_dir=picks_dir,
                models_dir=models_dir,
                dm_recipient=None,
                today=date(2026, 5, 24),
                live_forward_capture_enabled=True,
                live_forward_capture_artifact_root=Path("data/validation/pre"),
                live_forward_resolve_status_root=Path("data/validation/status"),
                thresholds_overrides={
                    "live_forward_resolution": {"grace_days": 3},
                },
            )

        mock_resolve.assert_called_once()
        assert mock_resolve.call_args.kwargs["preoutcome_root"] == (
            repo_root / "data" / "validation" / "pre"
        )
        assert mock_resolve.call_args.kwargs["status_root"] == (
            repo_root / "data" / "validation" / "status"
        )
        assert mock_resolve.call_args.kwargs["thresholds"] == {"grace_days": 3}
        assert expected in alerts

    def test_path_overrides_are_normalized_to_path_objects(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        memory_history = tmp_path / "health" / "memory_growth_history.jsonl"
        warn_state = tmp_path / "health" / "warn_attention_state.json"
        dm_status = tmp_path / "health" / "health_dm_delivery_status.json"
        with patch(
            "bts.health.runner.memory_growth.check",
            return_value=[],
        ) as mock_memory, patch(
            "bts.health.runner.apply_warn_attention_policy",
            return_value=([], False),
        ) as mock_attention, patch(
            "bts.health.runner.dispatch_dm_for_health_alerts",
        ) as mock_dm:
            run_all_checks(
                picks_dir=picks_dir,
                models_dir=models_dir,
                dm_recipient=None,
                scheduler_pid=123,
                today=date(2026, 4, 27),
                thresholds_overrides={
                    "memory_growth_history": str(memory_history),
                    "warn_attention_state": str(warn_state),
                    "health_dm_delivery_status": str(dm_status),
                },
            )

        history_arg = mock_memory.call_args.kwargs["history_path"]
        state_arg = mock_attention.call_args.kwargs["state_path"]
        dm_status_arg = mock_dm.call_args.kwargs["status_path"]
        assert isinstance(history_arg, Path)
        assert isinstance(state_arg, Path)
        assert isinstance(dm_status_arg, Path)
        assert history_arg == memory_history
        assert state_arg == warn_state
        assert dm_status_arg == dm_status
