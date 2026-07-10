"""Tests for the backup_freshness health source (audit F5).

Reads data/health_state/backup_status.json written by `bts backup run`.
Absent file = backups not armed on this machine (local dev) — silent, same
convention as park_drag_freshness. Once armed, staleness or a failed last
run must surface: a dead backup is a dead smoke detector.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bts.health import backup_freshness


NOW = datetime(2026, 7, 10, 16, 0, 0, tzinfo=timezone.utc)


def _write_status(root: Path, *, ops_age_h=1.0, archive_age_h=2.0,
                  ops_ok=True, archive_ok=True, ops_error=None,
                  drop=()):
    root.mkdir(parents=True, exist_ok=True)
    entries = {}
    if "ops" not in drop:
        entries["ops"] = {
            "set": "ops", "ok": ops_ok,
            "last_success_at": (NOW - timedelta(hours=ops_age_h)).isoformat(),
            **({"error": ops_error} if ops_error else {}),
        }
    if "archive" not in drop:
        entries["archive"] = {
            "set": "archive", "ok": archive_ok,
            "last_success_at": (NOW - timedelta(hours=archive_age_h)).isoformat(),
        }
    (root / "backup_status.json").write_text(json.dumps(entries))
    return root


def test_absent_status_file_is_silent(tmp_path):
    assert backup_freshness.check(tmp_path / "nope", now=NOW) == []


def test_fresh_backups_healthy(tmp_path):
    root = _write_status(tmp_path / "hs")
    assert backup_freshness.check(root, now=NOW) == []


def test_ops_stale_warns(tmp_path):
    root = _write_status(tmp_path / "hs", ops_age_h=8.0)
    alerts = backup_freshness.check(root, now=NOW)
    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert alerts[0].source == "backup_freshness"
    assert "ops" in alerts[0].message


def test_ops_very_stale_critical(tmp_path):
    root = _write_status(tmp_path / "hs", ops_age_h=27.0)
    alerts = backup_freshness.check(root, now=NOW)
    assert [a.level for a in alerts] == ["CRITICAL"]


def test_archive_stale_warns_critical_later(tmp_path):
    root = _write_status(tmp_path / "hs", archive_age_h=31.0)
    assert [a.level for a in backup_freshness.check(root, now=NOW)] == ["WARN"]
    root = _write_status(tmp_path / "hs", archive_age_h=80.0)
    assert [a.level for a in backup_freshness.check(root, now=NOW)] == ["CRITICAL"]


def test_failed_last_run_warns_even_when_fresh(tmp_path):
    root = _write_status(
        tmp_path / "hs", ops_ok=False, ops_error="Fatal: locked", ops_age_h=1.0,
    )
    alerts = backup_freshness.check(root, now=NOW)
    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert "locked" in alerts[0].message


def test_missing_set_entry_warns(tmp_path):
    root = _write_status(tmp_path / "hs", drop=("archive",))
    alerts = backup_freshness.check(root, now=NOW)
    assert len(alerts) == 1
    assert "archive" in alerts[0].message
    assert "never" in alerts[0].message.lower()


def test_unparseable_status_warns(tmp_path):
    root = tmp_path / "hs"
    root.mkdir()
    (root / "backup_status.json").write_text("{corrupt")
    alerts = backup_freshness.check(root, now=NOW)
    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert "unparseable" in alerts[0].message


def test_thresholds_override(tmp_path):
    root = _write_status(tmp_path / "hs", ops_age_h=2.0)
    alerts = backup_freshness.check(
        root, now=NOW, thresholds={"ops_warn_hours": 1.0},
    )
    assert [a.level for a in alerts] == ["WARN"]


def test_stale_and_failed_do_not_double_alert_per_set(tmp_path):
    # one alert per set at the highest applicable severity
    root = _write_status(
        tmp_path / "hs", ops_age_h=30.0, ops_ok=False, ops_error="boom",
    )
    alerts = backup_freshness.check(root, now=NOW)
    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL"
