"""Tests for heartbeat module."""
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bts.heartbeat import (
    HeartbeatState,
    heartbeat_watchdog,
    is_heartbeat_fresh,
    read_heartbeat,
    write_heartbeat,
)


def test_write_and_read_heartbeat(tmp_path: Path):
    hb_path = tmp_path / ".heartbeat"
    now = datetime(2026, 4, 9, 15, 30, tzinfo=timezone.utc)

    write_heartbeat(hb_path, state="running", now_utc=now)
    hb = read_heartbeat(hb_path)

    assert hb is not None
    assert hb["state"] == "running"
    assert hb["timestamp"] == now.isoformat()


def test_read_missing_heartbeat_returns_none(tmp_path: Path):
    assert read_heartbeat(tmp_path / "nonexistent") is None


def test_is_fresh_true_when_recent(tmp_path: Path):
    hb_path = tmp_path / ".heartbeat"
    now = datetime(2026, 4, 9, 15, 30, tzinfo=timezone.utc)
    write_heartbeat(hb_path, state="running", now_utc=now)

    check_time = now + timedelta(minutes=2)
    assert is_heartbeat_fresh(hb_path, max_age_sec=180, now_utc=check_time) is True


def test_is_stale_when_old(tmp_path: Path):
    hb_path = tmp_path / ".heartbeat"
    now = datetime(2026, 4, 9, 15, 30, tzinfo=timezone.utc)
    write_heartbeat(hb_path, state="running", now_utc=now)

    check_time = now + timedelta(minutes=10)
    assert is_heartbeat_fresh(hb_path, max_age_sec=180, now_utc=check_time) is False


def test_sleeping_state_is_fresh_even_if_old(tmp_path: Path):
    hb_path = tmp_path / ".heartbeat"
    now = datetime(2026, 4, 9, 10, 0, tzinfo=timezone.utc)
    wake = now + timedelta(hours=5)

    write_heartbeat(hb_path, state="sleeping", now_utc=now, sleeping_until=wake)

    check_time = now + timedelta(hours=2)
    assert is_heartbeat_fresh(hb_path, max_age_sec=180, now_utc=check_time) is True


def test_sleeping_past_wake_time_is_stale(tmp_path: Path):
    hb_path = tmp_path / ".heartbeat"
    now = datetime(2026, 4, 9, 10, 0, tzinfo=timezone.utc)
    wake = now + timedelta(hours=1)

    write_heartbeat(hb_path, state="sleeping", now_utc=now, sleeping_until=wake)

    check_time = now + timedelta(hours=2)
    assert is_heartbeat_fresh(hb_path, max_age_sec=180, now_utc=check_time) is False


def test_atomic_write_uses_tmp_rename(tmp_path: Path):
    hb_path = tmp_path / ".heartbeat"
    now = datetime(2026, 4, 9, 15, 30, tzinfo=timezone.utc)
    write_heartbeat(hb_path, state="running", now_utc=now)

    tmp_file = hb_path.with_suffix(".tmp")
    assert not tmp_file.exists()
    assert hb_path.exists()
    data = json.loads(hb_path.read_text())
    assert data["state"] == "running"


def test_heartbeat_watchdog_refreshes_during_block(tmp_path: Path):
    """heartbeat_watchdog refreshes the heartbeat periodically while its body runs.

    Motivation: the scheduler's prediction loop can run 5+ minutes without
    natural state transitions. Wrapping long calls with this context keeps
    the external staleness monitor from false-alerting mid-prediction.
    """
    hb_path = tmp_path / ".heartbeat"
    write_heartbeat(hb_path, state=HeartbeatState.IDLE_END_OF_DAY)
    initial = read_heartbeat(hb_path)
    assert initial["state"] == HeartbeatState.IDLE_END_OF_DAY

    with heartbeat_watchdog(hb_path, interval_sec=0.05):
        time.sleep(0.25)

    final = read_heartbeat(hb_path)
    assert final["state"] == HeartbeatState.RUNNING
    assert final["timestamp"] > initial["timestamp"]


def test_heartbeat_watchdog_stops_writing_after_exit(tmp_path: Path):
    """After the context exits, no further writes happen."""
    hb_path = tmp_path / ".heartbeat"

    with heartbeat_watchdog(hb_path, interval_sec=0.05):
        time.sleep(0.15)

    ts_after_exit = read_heartbeat(hb_path)["timestamp"]
    time.sleep(0.2)
    ts_later = read_heartbeat(hb_path)["timestamp"]
    assert ts_after_exit == ts_later
