"""Tests for calibration drift checking on production picks.

Two-layer module:
- compute_drift_metrics(): pure I/O on picks_dir, returns DriftMetrics
- evaluate_drift(): pure function (metrics, thresholds) -> list[Alert]

Thresholds are data-grounded against 28 days of 2026-04 picks (n=28):
  Top-1 P observed: mean=0.7508 std=0.0273 min=0.7022 max=0.7918
  Single-day floor: 0.70 (= mean - 2σ; never observed historically)
  Sustained floor:  0.71 (= mean - 1.5σ; for 3+ consecutive days)
  Drift: 0.02 INFO / 0.04 WARN / 0.06 CRITICAL (vs 14d baseline)
"""

import json
from datetime import date
from pathlib import Path

import pytest

from unittest.mock import patch, MagicMock

from bts.calibration_check import (
    Alert,
    DriftMetrics,
    DEFAULT_THRESHOLDS,
    compute_drift_metrics,
    evaluate_drift,
    run_calibration_check,
)


def _write_pick(picks_dir: Path, date_iso: str, top1: float | None, dd: float | None = None):
    """Write a minimal pick JSON with the fields we care about."""
    data = {
        "date": date_iso,
        "run_time": f"{date_iso}T17:15:00+00:00",
    }
    if top1 is not None:
        data["pick"] = {"batter_name": "X", "p_game_hit": top1}
    if dd is not None:
        data["double_down"] = {"batter_name": "Y", "p_game_hit": dd}
    (picks_dir / f"{date_iso}.json").write_text(json.dumps(data))


# --- compute_drift_metrics ---

class TestComputeDriftMetrics:
    def test_empty_dir(self, tmp_path):
        m = compute_drift_metrics(tmp_path)
        assert m.daily_top1 == {}
        assert m.daily_dd == {}
        assert m.rolling_7d_mean is None
        assert m.rolling_14d_mean is None
        assert m.drift is None

    def test_single_day(self, tmp_path):
        _write_pick(tmp_path, "2026-04-15", 0.75, 0.72)
        m = compute_drift_metrics(tmp_path, today=date(2026, 4, 15))
        assert m.daily_top1 == {"2026-04-15": 0.75}
        assert m.daily_dd == {"2026-04-15": 0.72}
        # Need >=7 days for 7d mean and >=14 for 14d
        assert m.rolling_7d_mean is None
        assert m.rolling_14d_mean is None
        assert m.drift is None

    def test_seven_days(self, tmp_path):
        # 7 days, all top1 = 0.75 → 7d mean is 0.75, 14d still None
        for i in range(1, 8):
            _write_pick(tmp_path, f"2026-04-{i:02d}", 0.75)
        m = compute_drift_metrics(tmp_path, today=date(2026, 4, 7))
        assert len(m.daily_top1) == 7
        assert m.rolling_7d_mean == pytest.approx(0.75, abs=1e-9)
        assert m.rolling_14d_mean is None
        assert m.drift is None

    def test_fourteen_days_no_drift(self, tmp_path):
        # 14 days, all top1 = 0.75 → both means 0.75, drift = 0
        for i in range(1, 15):
            _write_pick(tmp_path, f"2026-04-{i:02d}", 0.75)
        m = compute_drift_metrics(tmp_path, today=date(2026, 4, 14))
        assert m.rolling_7d_mean == pytest.approx(0.75, abs=1e-9)
        assert m.rolling_14d_mean == pytest.approx(0.75, abs=1e-9)
        assert m.drift == pytest.approx(0.0, abs=1e-9)

    def test_drift_negative(self, tmp_path):
        # First 7 days at 0.78, next 7 at 0.72 → 14d mean 0.75, 7d mean 0.72, drift -0.03
        for i in range(1, 8):
            _write_pick(tmp_path, f"2026-04-{i:02d}", 0.78)
        for i in range(8, 15):
            _write_pick(tmp_path, f"2026-04-{i:02d}", 0.72)
        m = compute_drift_metrics(tmp_path, today=date(2026, 4, 14))
        assert m.rolling_7d_mean == pytest.approx(0.72, abs=1e-9)
        assert m.rolling_14d_mean == pytest.approx(0.75, abs=1e-9)
        assert m.drift == pytest.approx(-0.03, abs=1e-9)

    def test_skips_shadow_files(self, tmp_path):
        _write_pick(tmp_path, "2026-04-15", 0.75)
        # shadow file should be ignored even if it has the same date pattern
        (tmp_path / "2026-04-15.shadow.json").write_text(
            json.dumps({"date": "2026-04-15", "pick": {"p_game_hit": 0.50}})
        )
        m = compute_drift_metrics(tmp_path, today=date(2026, 4, 15))
        assert m.daily_top1 == {"2026-04-15": 0.75}  # the 0.50 from shadow not loaded

    def test_handles_missing_dd(self, tmp_path):
        # Pick file without a double_down field → daily_dd has no entry for that date
        _write_pick(tmp_path, "2026-04-15", 0.75, dd=None)
        m = compute_drift_metrics(tmp_path, today=date(2026, 4, 15))
        assert m.daily_top1 == {"2026-04-15": 0.75}
        assert m.daily_dd == {}

    def test_skips_corrupt_files(self, tmp_path):
        _write_pick(tmp_path, "2026-04-15", 0.75)
        (tmp_path / "2026-04-16.json").write_text("not valid json{{{")
        m = compute_drift_metrics(tmp_path, today=date(2026, 4, 16))
        assert m.daily_top1 == {"2026-04-15": 0.75}

    def test_skips_files_outside_lookback(self, tmp_path):
        _write_pick(tmp_path, "2026-03-01", 0.99)  # outside 30-day lookback
        _write_pick(tmp_path, "2026-04-15", 0.75)
        m = compute_drift_metrics(tmp_path, today=date(2026, 4, 15), lookback_days=30)
        assert "2026-03-01" not in m.daily_top1
        assert "2026-04-15" in m.daily_top1


# --- evaluate_drift ---

class TestEvaluateDrift:
    def _metrics(self, daily_top1, daily_dd=None, top1_today=None, drift=None,
                rolling_7d=None, rolling_14d=None) -> DriftMetrics:
        if daily_dd is None:
            daily_dd = {}
        return DriftMetrics(
            daily_top1=daily_top1,
            daily_dd=daily_dd,
            rolling_7d_mean=rolling_7d,
            rolling_14d_mean=rolling_14d,
            drift=drift,
        )

    def test_no_alerts_normal_range(self):
        # All top-1 in healthy range, drift small
        daily = {f"2026-04-{i:02d}": 0.75 for i in range(1, 15)}
        m = self._metrics(daily, drift=0.001, rolling_7d=0.751, rolling_14d=0.75)
        alerts = evaluate_drift(m)
        assert alerts == []

    def test_warn_single_day_floor(self):
        daily = {f"2026-04-{i:02d}": 0.75 for i in range(1, 14)}
        daily["2026-04-14"] = 0.69  # below 0.70 floor
        m = self._metrics(daily, drift=-0.005, rolling_7d=0.75, rolling_14d=0.755)
        alerts = evaluate_drift(m)
        assert any(a.level == "WARN" and "0.69" in a.message for a in alerts)

    def test_warn_sustained_low(self):
        # 3 consecutive days below 0.71 → WARN
        daily = {f"2026-04-{i:02d}": 0.75 for i in range(1, 12)}
        daily["2026-04-12"] = 0.705
        daily["2026-04-13"] = 0.708
        daily["2026-04-14"] = 0.704
        m = self._metrics(daily, drift=-0.015, rolling_7d=0.74, rolling_14d=0.755)
        alerts = evaluate_drift(m)
        assert any(a.level == "WARN" and "3 consecutive" in a.message.lower() for a in alerts)

    def test_no_sustained_alert_under_3_days(self):
        # Only 2 consecutive low days → no sustained alert (single-day still applies if any < 0.70)
        daily = {f"2026-04-{i:02d}": 0.75 for i in range(1, 13)}
        daily["2026-04-13"] = 0.705
        daily["2026-04-14"] = 0.708
        m = self._metrics(daily, drift=0.001, rolling_7d=0.75, rolling_14d=0.749)
        alerts = evaluate_drift(m)
        # No 3-day sustained, no single-day < 0.70, no big drift
        assert not any("consecutive" in a.message.lower() for a in alerts)

    def test_info_drift(self):
        # Drift 0.025 → INFO
        m = self._metrics(
            {f"2026-04-{i:02d}": 0.75 for i in range(1, 15)},
            drift=-0.025, rolling_7d=0.73, rolling_14d=0.755,
        )
        alerts = evaluate_drift(m)
        assert any(a.level == "INFO" and "drift" in a.message.lower() for a in alerts)

    def test_warn_drift(self):
        # Drift 0.045 → WARN
        m = self._metrics(
            {f"2026-04-{i:02d}": 0.75 for i in range(1, 15)},
            drift=-0.045, rolling_7d=0.71, rolling_14d=0.755,
        )
        alerts = evaluate_drift(m)
        assert any(a.level == "WARN" and "drift" in a.message.lower() for a in alerts)

    def test_critical_drift(self):
        # Drift 0.065 → CRITICAL
        m = self._metrics(
            {f"2026-04-{i:02d}": 0.75 for i in range(1, 15)},
            drift=-0.065, rolling_7d=0.69, rolling_14d=0.755,
        )
        alerts = evaluate_drift(m)
        assert any(a.level == "CRITICAL" and "drift" in a.message.lower() for a in alerts)

    def test_no_drift_alert_when_metrics_unavailable(self):
        # rolling_7d / rolling_14d None (insufficient data) → no drift alert (but single-day still works)
        m = self._metrics(
            {"2026-04-14": 0.65},  # below floor
            drift=None, rolling_7d=None, rolling_14d=None,
        )
        alerts = evaluate_drift(m)
        assert any(a.level == "WARN" and "0.65" in a.message for a in alerts)
        assert not any("drift" in a.message.lower() for a in alerts)

    def test_multiple_alerts_fire_simultaneously(self):
        # Critical drift AND 3-day sustained low
        daily = {f"2026-04-{i:02d}": 0.75 for i in range(1, 12)}
        daily["2026-04-12"] = 0.69
        daily["2026-04-13"] = 0.69
        daily["2026-04-14"] = 0.69
        m = self._metrics(
            daily, drift=-0.07, rolling_7d=0.685, rolling_14d=0.755,
        )
        alerts = evaluate_drift(m)
        levels = [a.level for a in alerts]
        assert "CRITICAL" in levels  # drift
        assert "WARN" in levels  # single-day OR sustained

    def test_custom_thresholds(self):
        # Use a tighter threshold; previously-quiet metrics now alert
        daily = {f"2026-04-{i:02d}": 0.75 for i in range(1, 15)}
        m = self._metrics(daily, drift=-0.011, rolling_7d=0.745, rolling_14d=0.756)
        alerts = evaluate_drift(m, {**DEFAULT_THRESHOLDS, "drift_info": 0.01})
        assert any(a.level == "INFO" for a in alerts)

    def test_alerts_are_frozen(self):
        # Alerts are immutable (frozen dataclass) so they can be used as keys / hashed
        a = Alert(level="INFO", message="test")
        with pytest.raises(Exception):
            a.level = "WARN"  # type: ignore


# --- run_calibration_check (integration wrapper) ---

class TestRunCalibrationCheck:
    """The wrapper called from scheduler.py — handles logging + Bluesky DM + exception suppression."""

    def test_no_alerts_no_dm(self, tmp_path):
        # Empty picks dir → no alerts → no DM
        with patch("bts.calibration_check.send_dm") as mock_dm:
            alerts = run_calibration_check(picks_dir=tmp_path, dm_recipient="x.bsky.social")
            assert alerts == []
            mock_dm.assert_not_called()

    def test_warn_alert_logs_but_no_dm(self, tmp_path):
        # WARN-level alerts log but DON'T send DM (CRITICAL only)
        _write_pick(tmp_path, "2026-04-15", 0.65)  # below 0.70 floor → WARN
        with patch("bts.calibration_check.send_dm") as mock_dm:
            alerts = run_calibration_check(
                picks_dir=tmp_path, dm_recipient="x.bsky.social",
                today=date(2026, 4, 15),
            )
            assert any(a.level == "WARN" for a in alerts)
            mock_dm.assert_not_called()

    def test_critical_alert_sends_dm(self, tmp_path):
        # CRITICAL drift → DM sent with full alert content
        # Set up: 7 days at 0.78, 7 days at 0.70 → drift = -0.08 (above CRITICAL=0.06)
        # 7 days at 0.84, 7 at 0.70 → 14d mean=0.77, 7d mean=0.70, drift=-0.07 (CRITICAL)
        for i in range(1, 8):
            _write_pick(tmp_path, f"2026-04-{i:02d}", 0.84)
        for i in range(8, 15):
            _write_pick(tmp_path, f"2026-04-{i:02d}", 0.70)
        with patch("bts.calibration_check.send_dm") as mock_dm:
            mock_dm.return_value = "msg-id-123"
            alerts = run_calibration_check(
                picks_dir=tmp_path, dm_recipient="x.bsky.social",
                today=date(2026, 4, 14),
            )
            assert any(a.level == "CRITICAL" for a in alerts)
            mock_dm.assert_called_once()
            call_args = mock_dm.call_args
            assert call_args.args[0] == "x.bsky.social"
            # Message should contain the CRITICAL alert text
            assert "CRITICAL" in call_args.args[1]

    def test_dm_failure_does_not_propagate(self, tmp_path):
        # If send_dm raises, the wrapper swallows it (so a DM bug never breaks scheduler)
        # 7 days at 0.84, 7 at 0.70 → 14d mean=0.77, 7d mean=0.70, drift=-0.07 (CRITICAL)
        for i in range(1, 8):
            _write_pick(tmp_path, f"2026-04-{i:02d}", 0.84)
        for i in range(8, 15):
            _write_pick(tmp_path, f"2026-04-{i:02d}", 0.70)
        with patch("bts.calibration_check.send_dm", side_effect=RuntimeError("network down")):
            alerts = run_calibration_check(
                picks_dir=tmp_path, dm_recipient="x.bsky.social",
                today=date(2026, 4, 14),
            )
            # Alert was still detected even though DM failed
            assert any(a.level == "CRITICAL" for a in alerts)

    def test_corrupt_picks_does_not_propagate(self, tmp_path):
        # Even if compute_drift_metrics encounters issues, run_calibration_check returns []
        # (compute_drift_metrics already skips corrupt files; this just verifies no surprises)
        (tmp_path / "2026-04-15.json").write_text("not json")
        alerts = run_calibration_check(picks_dir=tmp_path, dm_recipient="x.bsky.social",
                                       today=date(2026, 4, 15))
        assert alerts == []

    def test_no_dm_when_recipient_unset(self, tmp_path):
        # If dm_recipient is None or empty, DM never attempted
        # 7 days at 0.84, 7 at 0.70 → 14d mean=0.77, 7d mean=0.70, drift=-0.07 (CRITICAL)
        for i in range(1, 8):
            _write_pick(tmp_path, f"2026-04-{i:02d}", 0.84)
        for i in range(8, 15):
            _write_pick(tmp_path, f"2026-04-{i:02d}", 0.70)
        with patch("bts.calibration_check.send_dm") as mock_dm:
            alerts = run_calibration_check(
                picks_dir=tmp_path, dm_recipient=None,
                today=date(2026, 4, 14),
            )
            # CRITICAL alert detected but no DM sent
            assert any(a.level == "CRITICAL" for a in alerts)
            mock_dm.assert_not_called()
