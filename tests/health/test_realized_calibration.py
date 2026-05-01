"""Tests for the realized_calibration health check (2026-04-29).

Detects ABSOLUTE-LEVEL overconfidence in the 75-80% predicted-P bucket of
production picks vs realized outcomes. Distinct from predicted_vs_realized,
which detects DRIFT in the gap over time. F's analysis on 2026-04-29 found
+14pp overconfidence in this bucket sitting unaddressed for weeks because
no one ran the analysis manually until then.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from bts.health.realized_calibration import check, SOURCE


def _write_pick(picks_dir: Path, d: date, predicted_p: float, result: str | None,
                run_time: str | None = None):
    picks_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "date": d.isoformat(),
        "pick": {"p_game_hit": predicted_p, "batter_name": "X"},
    }
    if result is not None:
        body["result"] = result
    if run_time is not None:
        body["run_time"] = run_time
    (picks_dir / f"{d.isoformat()}.json").write_text(json.dumps(body))


def _well_calibrated_75_80(picks_dir: Path, today: date, n: int):
    """n picks at 0.78 predicted with 78% realized hit rate (well-calibrated)."""
    for i in range(n):
        d = today - timedelta(days=i + 1)
        result = "hit" if i < int(0.78 * n) else "miss"
        _write_pick(picks_dir, d, 0.78, result)


def _overconfident_75_80(picks_dir: Path, today: date, n: int, realized_rate: float):
    """n picks at 0.78 predicted with `realized_rate` actual hit rate."""
    for i in range(n):
        d = today - timedelta(days=i + 1)
        result = "hit" if i < int(realized_rate * n) else "miss"
        _write_pick(picks_dir, d, 0.78, result)


class TestRealizedCalibration:
    def test_well_calibrated_no_alert(self, tmp_path):
        picks_dir = tmp_path / "picks"
        today = date(2026, 4, 29)
        _well_calibrated_75_80(picks_dir, today, n=20)
        assert check(picks_dir, today=today) == []

    def test_info_at_8pp_overconfidence(self, tmp_path):
        # Thresholds raised 2026-05-01 after attribution-bias finding (DD picks were
        # over-counted as misses, inflating apparent overconfidence). New floor: 8pp.
        # n=20 at 0.78 predicted, 70% realized → 8pp gap → INFO
        picks_dir = tmp_path / "picks"
        today = date(2026, 4, 29)
        _overconfident_75_80(picks_dir, today, n=20, realized_rate=0.70)
        alerts = check(picks_dir, today=today)
        assert len(alerts) == 1
        assert alerts[0].level == "INFO"
        assert alerts[0].source == SOURCE
        assert "75-80" in alerts[0].message

    def test_warn_at_15pp_overconfidence(self, tmp_path):
        # 0.78 predicted, 0.63 realized → 15pp gap → WARN
        picks_dir = tmp_path / "picks"
        today = date(2026, 4, 29)
        _overconfident_75_80(picks_dir, today, n=20, realized_rate=0.63)
        alerts = check(picks_dir, today=today)
        assert alerts[0].level == "WARN"

    def test_critical_at_25pp_overconfidence(self, tmp_path):
        # 0.78 predicted, 0.50 realized → 28pp gap → CRITICAL (the true distribution-shift signal)
        picks_dir = tmp_path / "picks"
        today = date(2026, 4, 29)
        _overconfident_75_80(picks_dir, today, n=20, realized_rate=0.50)
        alerts = check(picks_dir, today=today)
        assert alerts[0].level == "CRITICAL"

    def test_no_alert_with_empty_picks_dir(self, tmp_path):
        picks_dir = tmp_path / "picks"; picks_dir.mkdir()
        assert check(picks_dir, today=date(2026, 4, 29)) == []

    def test_no_alert_with_nonexistent_picks_dir(self, tmp_path):
        picks_dir = tmp_path / "no_picks_yet"
        assert check(picks_dir, today=date(2026, 4, 29)) == []

    def test_no_alert_when_only_unresolved_picks(self, tmp_path):
        # Picks without a `result` field (e.g. tomorrow's preview) are excluded
        picks_dir = tmp_path / "picks"
        today = date(2026, 4, 29)
        for i in range(10):
            _write_pick(picks_dir, today + timedelta(days=i + 1), 0.78, None)
        assert check(picks_dir, today=today) == []

    def test_picks_outside_75_80_bucket_dont_trigger(self, tmp_path):
        # Picks in 70-75% or 80%+ buckets shouldn't drive the 75-80% alert
        picks_dir = tmp_path / "picks"
        today = date(2026, 4, 29)
        for i in range(20):
            d = today - timedelta(days=i + 1)
            # All picks at 0.72 (in 70-75% bucket) — wildly overconfident if all miss
            _write_pick(picks_dir, d, 0.72, "miss" if i < 15 else "hit")
        # No alert: the overconfident bucket is 70-75, not 75-80. The check
        # specifically targets the 75-80% bucket per F's finding.
        assert check(picks_dir, today=today) == []

    def test_n_too_low_no_alert(self, tmp_path):
        # With <5 picks in the bucket, the calibration estimate is too noisy
        # to alert on. The check should defer until enough data exists.
        picks_dir = tmp_path / "picks"
        today = date(2026, 4, 29)
        _overconfident_75_80(picks_dir, today, n=4, realized_rate=0.50)
        # 0.78 predicted, 50% realized = +28pp gap, but n=4 → no alert
        assert check(picks_dir, today=today) == []

    def test_lookback_window_30d(self, tmp_path):
        # Old picks (>30d) should not influence the check
        picks_dir = tmp_path / "picks"
        today = date(2026, 4, 29)
        # Recent: well-calibrated
        _well_calibrated_75_80(picks_dir, today, n=20)
        # Old (60d ago): wildly miscalibrated — should be IGNORED
        for i in range(20):
            d = today - timedelta(days=60 + i)
            _write_pick(picks_dir, d, 0.78, "miss")
        # Should not alert because recent (within 30d) data is fine
        assert check(picks_dir, today=today) == []

    def test_threshold_overrides(self, tmp_path):
        # Custom thresholds override defaults.
        # n=20 picks at 0.78 predicted, realized_rate=0.755 → int(15.1)=15 hits
        # → 0.75 realized → 3pp gap (under default 8pp INFO, over 2pp override)
        picks_dir = tmp_path / "picks"
        today = date(2026, 4, 29)
        _overconfident_75_80(picks_dir, today, n=20, realized_rate=0.755)
        # Default: 8pp INFO threshold, 3pp gap → no alert
        assert check(picks_dir, today=today) == []
        # Override: 2pp INFO threshold → fires INFO
        alerts = check(picks_dir, today=today, thresholds={"info_pp": 2, "warn_pp": 8, "critical_pp": 13})
        assert len(alerts) == 1
        assert alerts[0].level == "INFO"


class TestSinceDeployFilter:
    """Filter out pre-deploy picks (per project_bts_production_realized_contaminated.md).

    Without this filter the alert pools picks generated by many model iterations
    and produces noise rather than signal about the current model.
    """

    def test_pre_deploy_picks_excluded(self, tmp_path):
        picks_dir = tmp_path / "picks"
        today = date(2026, 4, 29)
        # 20 pre-deploy bad picks (would alert if counted)
        for i in range(20):
            d = today - timedelta(days=i + 1)
            result = "miss"  # all miss → 0% realized vs 0.78 predicted, 78pp gap
            _write_pick(picks_dir, d, 0.78, result, run_time="2026-03-15T00:00:00+00:00")
        # since_deploy 2026-04-01 → all 20 filtered out → no alert
        deploy_iso = "2026-04-01T00:00:00+00:00"
        assert check(picks_dir, today=today, since_deploy_iso=deploy_iso) == []

    def test_post_deploy_picks_counted(self, tmp_path):
        picks_dir = tmp_path / "picks"
        today = date(2026, 4, 29)
        # 20 post-deploy bad picks → should still alert
        for i in range(20):
            d = today - timedelta(days=i + 1)
            result = "miss"
            _write_pick(picks_dir, d, 0.78, result, run_time="2026-04-15T00:00:00+00:00")
        deploy_iso = "2026-04-01T00:00:00+00:00"
        alerts = check(picks_dir, today=today, since_deploy_iso=deploy_iso)
        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"
        assert "since-deploy" in alerts[0].message

    def test_filter_message_includes_skip_count(self, tmp_path):
        picks_dir = tmp_path / "picks"
        today = date(2026, 4, 29)
        # 5 pre-deploy + 20 post-deploy, all miss
        for i in range(5):
            d = today - timedelta(days=i + 1)
            _write_pick(picks_dir, d, 0.78, "miss", run_time="2026-03-15T00:00:00+00:00")
        for i in range(20):
            d = today - timedelta(days=i + 6)  # different dates
            _write_pick(picks_dir, d, 0.78, "miss", run_time="2026-04-15T00:00:00+00:00")
        alerts = check(picks_dir, today=today,
                       since_deploy_iso="2026-04-01T00:00:00+00:00")
        assert len(alerts) == 1
        assert "skipped 5 pre-deploy" in alerts[0].message

    def test_no_filter_when_since_deploy_iso_none(self, tmp_path):
        # Backward compat: when since_deploy_iso=None, no filter applied
        picks_dir = tmp_path / "picks"
        today = date(2026, 4, 29)
        for i in range(20):
            d = today - timedelta(days=i + 1)
            _write_pick(picks_dir, d, 0.78, "miss", run_time="2026-03-15T00:00:00+00:00")
        # All 20 counted; alert fires
        alerts = check(picks_dir, today=today, since_deploy_iso=None)
        assert len(alerts) == 1
        # Message should flag the unfiltered (iteration-contaminated) state
        assert "ALL-PICKS" in alerts[0].message or "iteration-contaminated" in alerts[0].message

    def test_pick_without_run_time_field_excluded(self, tmp_path):
        # Old pick files might lack run_time. With since_deploy_iso set,
        # they're excluded (we can't verify they're post-deploy).
        picks_dir = tmp_path / "picks"
        today = date(2026, 4, 29)
        for i in range(20):
            d = today - timedelta(days=i + 1)
            _write_pick(picks_dir, d, 0.78, "miss")  # no run_time
        # All 20 missing run_time → all filtered out → no alert
        assert check(picks_dir, today=today,
                     since_deploy_iso="2026-04-01T00:00:00+00:00") == []
