"""Health sources must let unexpected crashes reach the runner (audit F4).

`_safe_run` promises a crashing check becomes a CRITICAL health_runner alert
("a dead smoke detector is worse than a noisy one", audit H3). Four sources
defeated that by catching Exception internally and returning [] — an evaluation
failure looked identical to healthy. Contract: expected data-absence stays
quiet; unexpected exceptions PROPAGATE.
"""
import json
from unittest.mock import patch

import pytest

from bts.health import calibration, disk_fill, post_failure, projected_lineup
from bts.health.runner import _safe_run


# --- calibration -----------------------------------------------------------

def test_calibration_internal_crash_propagates(tmp_path):
    with patch("bts.health.calibration.compute_drift_metrics",
               side_effect=RuntimeError("schema drift")):
        with pytest.raises(RuntimeError):
            calibration.check(tmp_path)


def test_calibration_empty_picks_dir_stays_quiet(tmp_path):
    assert calibration.check(tmp_path) == []


# --- projected_lineup ------------------------------------------------------

def _plant_pick(picks_dir, date_str, projected=False):
    (picks_dir / f"{date_str}.json").write_text(json.dumps({
        "pick": {"batter_name": "X", "projected_lineup": projected},
        "double_down": None,
    }))


def test_projected_lineup_internal_crash_propagates(tmp_path):
    _plant_pick(tmp_path, "2026-07-01")
    with patch("bts.health.projected_lineup.json.loads",
               side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            projected_lineup.check(tmp_path)


def test_projected_lineup_corrupt_file_still_skipped_quietly(tmp_path):
    (tmp_path / "2026-07-01.json").write_text("{not json")
    assert projected_lineup.check(tmp_path) == []


# --- post_failure ----------------------------------------------------------

def test_post_failure_unreadable_pick_json_propagates(tmp_path):
    from datetime import date
    today = date(2026, 7, 9)
    (tmp_path / "2026-07-09.json").write_text("{definitely not json")
    with pytest.raises(json.JSONDecodeError):
        post_failure.check(tmp_path, today=today)


def test_post_failure_missing_pick_file_stays_quiet(tmp_path):
    from datetime import date
    assert post_failure.check(tmp_path, today=date(2026, 7, 9)) == []


# --- disk_fill ---------------------------------------------------------------

def test_disk_fill_usage_failure_propagates(tmp_path):
    with patch("bts.health.disk_fill.shutil.disk_usage",
               side_effect=PermissionError("denied")):
        with pytest.raises(PermissionError):
            disk_fill.check(tmp_path)


# --- runner conversion (the promise these sources must not defeat) ----------

def test_safe_run_converts_propagated_crash_to_critical(tmp_path):
    with patch("bts.health.calibration.compute_drift_metrics",
               side_effect=RuntimeError("schema drift")):
        alerts = _safe_run("calibration", lambda: calibration.check(tmp_path))
    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL"
    assert alerts[0].source == "health_runner"
    assert "calibration" in alerts[0].message
