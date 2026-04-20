"""Tests for scripts/check_heartbeat.py staleness decision logic."""
from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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


def test_idle_end_of_day_stale_after_max_age(tmp_path):
    """idle_end_of_day should be stale after 90 min — it's a brief transitional state."""
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    old = now - timedelta(minutes=95)
    _write_hb(hb_path, state="idle_end_of_day", timestamp=old.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
    assert "idle_end_of_day" in reason.lower() or "stuck" in reason.lower()


def test_idle_end_of_day_fresh_within_max_age(tmp_path):
    """idle_end_of_day with recent timestamp is still fresh."""
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    old = now - timedelta(minutes=30)
    _write_hb(hb_path, state="idle_end_of_day", timestamp=old.isoformat())
    stale, _ = is_stale(hb_path, now=now)
    assert not stale


def test_unknown_state_is_stale(tmp_path):
    """Unknown states (new writer state not yet supported by monitor) trigger stale."""
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    _write_hb(hb_path, state="teleporting", timestamp=now.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
    assert "unknown" in reason.lower()


def test_sleeping_without_wake_target_is_stale(tmp_path):
    """Malformed sleeping heartbeat (no sleeping_until field) is stale."""
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    _write_hb(hb_path, state="sleeping", timestamp=now.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
    assert "sleeping_until" in reason.lower()


def test_fresh_waiting_for_games(tmp_path):
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    _write_hb(hb_path, state="waiting_for_games", timestamp=now.isoformat())
    stale, _ = is_stale(hb_path, now=now)
    assert not stale


def test_stale_waiting_for_games(tmp_path):
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    old = now - timedelta(minutes=15)
    _write_hb(hb_path, state="waiting_for_games", timestamp=old.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
    assert "waiting" in reason.lower()
