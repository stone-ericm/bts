"""Tests for scripts/check_heartbeat.py staleness decision logic."""
from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import sys
sys.path.insert(0, "scripts")
from check_heartbeat import is_stale


def _write_hb(path: Path, **kv) -> None:
    path.write_text(json.dumps(kv))


def test_fresh_running_state(tmp_path):
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    _write_hb(hb_path, state="running", timestamp=now.isoformat())
    stale, _ = is_stale(hb_path, now=now)
    assert not stale


def test_stale_running_state(tmp_path):
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    old = now - timedelta(minutes=8)
    _write_hb(hb_path, state="running", timestamp=old.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
    assert "running" in reason.lower()


def test_sleeping_with_future_wakeup_is_fresh(tmp_path):
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    wake = now + timedelta(hours=2)
    _write_hb(hb_path, state="sleeping",
              timestamp=(now - timedelta(hours=1)).isoformat(),
              sleeping_until=wake.isoformat())
    stale, _ = is_stale(hb_path, now=now)
    assert not stale


def test_sleeping_past_wakeup_is_stale(tmp_path):
    """If sleeping_until is in the past by >10 min, daemon should have woken."""
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    wake = now - timedelta(minutes=15)  # past due
    _write_hb(hb_path, state="sleeping",
              timestamp=(now - timedelta(hours=2)).isoformat(),
              sleeping_until=wake.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
    assert "sleeping" in reason.lower()


def test_missing_file_is_stale(tmp_path):
    hb_path = tmp_path / "missing.json"
    stale, reason = is_stale(hb_path)
    assert stale
    assert "not found" in reason.lower()


def test_corrupt_file_is_stale(tmp_path):
    hb_path = tmp_path / "hb.json"
    hb_path.write_text("{not valid json")
    stale, reason = is_stale(hb_path)
    assert stale


def test_idle_end_of_day_after_1am_is_stale(tmp_path):
    """Daemon should transition off idle_end_of_day by 1 AM ET."""
    from zoneinfo import ZoneInfo
    hb_path = tmp_path / "hb.json"
    # Construct a "now" at 01:30 ET
    et = ZoneInfo("America/New_York")
    now = datetime(2026, 4, 21, 1, 30, tzinfo=et).astimezone(timezone.utc)
    ts = datetime(2026, 4, 20, 23, 30, tzinfo=timezone.utc)  # 2h old
    _write_hb(hb_path, state="idle_end_of_day", timestamp=ts.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
