"""Tests for the pooled-training daily-status health check (item #6 / Component D).

The pooled-training cron writes data/models_pooled/<DATE>_status.json with
n_complete out of N seeds. This check reads it on day N at end-of-day for the
N+1 status, alerts if the pool is under-filled.

Severity ladder:
  n_complete >= n_seeds:           no alert
  n_complete >= 0.8 * n_seeds:     INFO  (still a usable pool)
  n_complete >= 0.5 * n_seeds:     WARN  (degraded pooled inference)
  n_complete <  0.5 * n_seeds:     CRITICAL (would force fallback to single seed)
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from bts.health.pooled_training import check, SOURCE


def _write_status(pooled_dir: Path, target_date: date, n_complete: int, n_seeds: int = 10):
    pooled_dir.mkdir(parents=True, exist_ok=True)
    (pooled_dir / f"{target_date.isoformat()}_status.json").write_text(json.dumps({
        "date": target_date.isoformat(),
        "provider": "vultr",
        "seed_set": "canonical-n10",
        "n_seeds": n_seeds,
        "n_complete": n_complete,
        "per_seed": {},
    }))


class TestPooledTraining:
    def test_full_pool_no_alert(self, tmp_path):
        today = date(2026, 4, 29)
        tomorrow = today + timedelta(days=1)
        _write_status(tmp_path, tomorrow, n_complete=10, n_seeds=10)
        assert check(pooled_dir=tmp_path, today=today) == []

    def test_info_at_8_of_10(self, tmp_path):
        today = date(2026, 4, 29)
        tomorrow = today + timedelta(days=1)
        _write_status(tmp_path, tomorrow, n_complete=8, n_seeds=10)
        alerts = check(pooled_dir=tmp_path, today=today)
        assert len(alerts) == 1
        assert alerts[0].level == "INFO"
        assert alerts[0].source == SOURCE

    def test_warn_at_5_of_10(self, tmp_path):
        today = date(2026, 4, 29)
        tomorrow = today + timedelta(days=1)
        _write_status(tmp_path, tomorrow, n_complete=5, n_seeds=10)
        alerts = check(pooled_dir=tmp_path, today=today)
        assert alerts[0].level == "WARN"

    def test_critical_at_4_of_10(self, tmp_path):
        today = date(2026, 4, 29)
        tomorrow = today + timedelta(days=1)
        _write_status(tmp_path, tomorrow, n_complete=4, n_seeds=10)
        alerts = check(pooled_dir=tmp_path, today=today)
        assert alerts[0].level == "CRITICAL"

    def test_critical_when_status_file_missing(self, tmp_path):
        # No status file → cron didn't run or failed before writing
        today = date(2026, 4, 29)
        alerts = check(pooled_dir=tmp_path, today=today)
        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"
        assert "no status file" in alerts[0].message.lower() or \
               "missing" in alerts[0].message.lower()

    def test_critical_when_status_malformed(self, tmp_path):
        today = date(2026, 4, 29)
        tomorrow = today + timedelta(days=1)
        (tmp_path / f"{tomorrow.isoformat()}_status.json").write_text("not valid json")
        alerts = check(pooled_dir=tmp_path, today=today)
        assert alerts[0].level == "CRITICAL"

    def test_critical_when_n_complete_missing(self, tmp_path):
        today = date(2026, 4, 29)
        tomorrow = today + timedelta(days=1)
        (tmp_path / f"{tomorrow.isoformat()}_status.json").write_text(json.dumps({
            "date": tomorrow.isoformat(), "n_seeds": 10,  # n_complete absent
        }))
        alerts = check(pooled_dir=tmp_path, today=today)
        assert alerts[0].level == "CRITICAL"

    def test_no_alert_when_pooled_dir_doesnt_exist(self, tmp_path):
        # Pre-Phase-1 state: pooled training not yet enabled. Don't false-alert.
        nonexistent = tmp_path / "never_created"
        assert check(pooled_dir=nonexistent, today=date(2026, 4, 29)) == []

    def test_severity_ladder_at_n_seeds_5(self, tmp_path):
        """Smaller seed set (n_seeds=5) — thresholds scale by ratio not absolute."""
        today = date(2026, 4, 29)
        tomorrow = today + timedelta(days=1)
        # 4 of 5 = 80% → INFO
        _write_status(tmp_path, tomorrow, n_complete=4, n_seeds=5)
        assert check(pooled_dir=tmp_path, today=today)[0].level == "INFO"
        # 3 of 5 = 60% → WARN
        _write_status(tmp_path, tomorrow, n_complete=3, n_seeds=5)
        assert check(pooled_dir=tmp_path, today=today)[0].level == "WARN"
        # 2 of 5 = 40% → CRITICAL
        _write_status(tmp_path, tomorrow, n_complete=2, n_seeds=5)
        assert check(pooled_dir=tmp_path, today=today)[0].level == "CRITICAL"
