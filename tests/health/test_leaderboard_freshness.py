"""Tests for the leaderboard_freshness health check."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from bts.health.leaderboard_freshness import check, SOURCE


def _write_snapshot(dir_path: Path, hours_ago: float):
    dir_path.mkdir(parents=True, exist_ok=True)
    p = dir_path / "2026-05-01.parquet"
    p.write_bytes(b"PAR1placeholder")
    mtime = time.time() - (hours_ago * 3600)
    os.utime(p, (mtime, mtime))


class TestLeaderboardFreshness:
    def test_no_alert_when_recent(self, tmp_path):
        snaps = tmp_path / "leaderboard_snapshots"
        _write_snapshot(snaps, hours_ago=2)
        assert check(tmp_path) == []

    def test_warn_when_12h_to_36h(self, tmp_path):
        snaps = tmp_path / "leaderboard_snapshots"
        _write_snapshot(snaps, hours_ago=20)
        alerts = check(tmp_path)
        assert len(alerts) == 1
        assert alerts[0].level == "WARN"
        assert alerts[0].source == SOURCE

    def test_critical_when_more_than_36h(self, tmp_path):
        snaps = tmp_path / "leaderboard_snapshots"
        _write_snapshot(snaps, hours_ago=40)
        alerts = check(tmp_path)
        assert alerts[0].level == "CRITICAL"

    def test_warn_when_no_snapshots_at_all(self, tmp_path):
        (tmp_path / "leaderboard_snapshots").mkdir()
        alerts = check(tmp_path)
        assert alerts[0].level == "WARN"

    def test_no_alert_when_dir_missing_entirely(self, tmp_path):
        # Watcher not yet deployed — silent (don't alarm pre-launch)
        assert check(tmp_path) == []
