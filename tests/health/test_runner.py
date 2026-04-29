"""Tests for the health-check runner aggregator.

The runner's job: call each check, isolate failures, log, dispatch DM.
We verify (a) it calls every check, (b) per-check exceptions are isolated,
(c) DM dispatcher is called once with the aggregated alert list.
"""

import json
from datetime import date
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
             patch("bts.health.runner.dispatch_dm_for_critical") as mock_dm:
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

    def test_dm_dispatcher_called_with_full_alert_list(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        models_dir = tmp_path / "models"; models_dir.mkdir()
        _set_up_picks_dir(picks_dir, models_dir)
        # Force a CRITICAL via missing streak
        (picks_dir / "streak.json").unlink()
        with patch("bts.health.runner.dispatch_dm_for_critical") as mock_dm:
            run_all_checks(
                picks_dir=picks_dir, models_dir=models_dir,
                dm_recipient="x.bsky.social", today=date(2026, 4, 27),
            )
            mock_dm.assert_called_once()
            args = mock_dm.call_args.args
            alerts_arg = args[0]
            # Includes the streak_validation CRITICAL
            assert any(a.source == "streak_validation" for a in alerts_arg)

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
