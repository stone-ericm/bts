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


def test_stalled_state_is_never_fresh(tmp_path: Path):
    hb_path = tmp_path / ".heartbeat"
    now = datetime(2026, 6, 11, 15, 30, tzinfo=timezone.utc)
    write_heartbeat(hb_path, state="stalled", now_utc=now,
                    extra={"stage": "computing_features", "stalled_for_s": 1000})

    # timestamp is brand new — age check alone would call this fresh
    assert is_heartbeat_fresh(hb_path, max_age_sec=180, now_utc=now) is False


import time as _time

from bts import progress
from bts.heartbeat import heartbeat_watchdog


def _read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_watchdog_running_heartbeat_carries_stage(tmp_path: Path):
    hb = tmp_path / ".heartbeat"
    with heartbeat_watchdog(hb, interval_sec=0.01, kind="primary",
                            date="2026-06-11", stall_after_sec=999):
        progress.mark("computing_features")
        _time.sleep(0.08)
        payload = read_heartbeat(hb)
        assert payload["state"] == "running"
        assert payload["stage"] == "computing_features"
        assert "run_id" in payload


def test_watchdog_flips_to_stalled_and_keeps_sd_notify(tmp_path: Path, monkeypatch):
    pings = []
    monkeypatch.setattr("bts.sd_notify.notify_watchdog", lambda: pings.append(1))
    hb = tmp_path / ".heartbeat"
    dur = tmp_path / "durations.jsonl"
    with heartbeat_watchdog(hb, interval_sec=0.01, kind="primary",
                            date="2026-06-11", stall_after_sec=0.0,
                            durations_path=dur):
        progress.mark("computing_features")
        _time.sleep(0.08)
        payload = read_heartbeat(hb)
        assert payload["state"] == "stalled"
        assert payload["stage"] == "computing_features"
        assert payload["stalled_for_s"] >= 0
    assert len(pings) >= 2  # sd_notify NEVER stops during a stall (no systemd kill)
    rows = _read_jsonl(dur)
    incomplete = [r for r in rows if r["status"] == "stalled_incomplete"
                  and r["stage"] == "computing_features"]
    assert len(incomplete) == 1  # once per stage instance, despite many ticks
    assert incomplete[0]["kind"] == "primary"
    assert incomplete[0]["date"] == "2026-06-11"
    assert incomplete[0]["threshold_used_s"] == 0.0
    assert incomplete[0]["pid"] > 0


def test_watchdog_recovery_flips_back_to_running_and_marks_ok_after_stall(tmp_path: Path):
    hb = tmp_path / ".heartbeat"
    dur = tmp_path / "durations.jsonl"
    with heartbeat_watchdog(hb, interval_sec=0.01, kind="primary",
                            date="2026-06-11", stall_after_sec=0.05,
                            durations_path=dur):
        progress.mark("stage_a")
        _time.sleep(0.12)            # stage_a crosses 0.05s -> stalled
        assert read_heartbeat(hb)["state"] == "stalled"
        progress.mark("stage_b")     # recovery: stage_a completes late
        _time.sleep(0.04)            # stage_b still under threshold
        assert read_heartbeat(hb)["state"] == "running"
    rows = _read_jsonl(dur)
    a_rows = [r for r in rows if r["stage"] == "stage_a"]
    assert {r["status"] for r in a_rows} == {"stalled_incomplete", "ok_after_stall"}


def test_watchdog_second_stall_in_later_stage_gets_own_row(tmp_path: Path):
    hb = tmp_path / ".heartbeat"
    dur = tmp_path / "durations.jsonl"
    with heartbeat_watchdog(hb, interval_sec=0.01, kind="primary",
                            date="2026-06-11", stall_after_sec=0.05,
                            durations_path=dur):
        progress.mark("stage_a")
        _time.sleep(0.1)
        progress.mark("stage_b")
        _time.sleep(0.1)
    incomplete = [r["stage"] for r in _read_jsonl(dur)
                  if r["status"] == "stalled_incomplete" and r["stage"] != "cascade_starting"]
    assert incomplete == ["stage_a", "stage_b"]


def test_watchdog_exit_retires_run_and_drains_final_stage(tmp_path: Path):
    hb = tmp_path / ".heartbeat"
    dur = tmp_path / "durations.jsonl"
    with heartbeat_watchdog(hb, interval_sec=0.01, kind="shadow",
                            date="2026-06-11", stall_after_sec=999,
                            durations_path=dur) as run_id:
        progress.mark("selecting_pick")
    assert progress.snapshot(run_id) is None  # retired: leaked pulse writes nothing
    rows = _read_jsonl(dur)
    assert "selecting_pick" in [r["stage"] for r in rows]  # final drain got the last stage
    assert all(r["kind"] == "shadow" for r in rows)
    # post-exit scheduler heartbeats are not overwritten
    write_heartbeat(hb, state="sleeping")
    _time.sleep(0.05)
    assert read_heartbeat(hb)["state"] == "sleeping"


def test_watchdog_stall_latch_not_flipped_on_failed_append(tmp_path: Path):
    hb = tmp_path / ".heartbeat"
    blocked = tmp_path / "blocked"
    blocked.write_text("")  # durations_path parent mkdir will fail under a file
    dur = blocked / "durations.jsonl"
    with heartbeat_watchdog(hb, interval_sec=0.01, kind="primary",
                            date="2026-06-11", stall_after_sec=0.0,
                            durations_path=dur):
        progress.mark("stage_a")
        _time.sleep(0.06)  # several ticks; every append fails; must not raise
    assert not dur.exists()


def test_watchdog_backward_compatible_defaults(tmp_path: Path):
    # existing callers pass only (path, interval_sec) — must still work
    hb = tmp_path / ".heartbeat"
    with heartbeat_watchdog(hb, interval_sec=0.01):
        _time.sleep(0.03)
        assert read_heartbeat(hb)["state"] == "running"
