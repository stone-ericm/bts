"""scheduler_state.json torn-write / corrupt-parse resilience (audit F3).

The daemon previously (a) wrote scheduler_state.json with bare write_text —
a crash mid-write could tear the file; (b) parsed it crash-unprotected at
startup; and (c) signaled READY + wrote the heartbeat BEFORE touching state.
Torn file → 30s systemd crash-restart loop that every external monitor scored
healthy (heartbeat refreshed pre-crash each cycle).

Contract now: saves are atomic; a corrupt state file is quarantined
(*.corrupt-*) and treated as absent, with the daemon proceeding; state is
loaded BEFORE the daemon advertises readiness.
"""
import json
from unittest.mock import patch

import pytest

from bts.scheduler import SchedulerState, load_state, run_day, save_state

MINIMAL_STATE = dict(
    date="2026-07-09",
    schedule_fetched_at="2026-07-09T10:00:00-04:00",
    games=[],
    confirmed_game_pks=[],
    runs_completed=[],
    pick_locked=False,
    pick_locked_at=None,
    result_status=None,
    next_wakeup=None,
)


def _state_path(picks_dir, date="2026-07-09"):
    return picks_dir / date / "scheduler_state.json"


# --- (a) atomic save ---------------------------------------------------------

def test_save_state_failure_preserves_existing_file(tmp_path, monkeypatch):
    state = SchedulerState(**MINIMAL_STATE)
    save_state(state, tmp_path)
    good = _state_path(tmp_path).read_text()

    # Crash at the atomic-commit step: the prior good file must survive.
    monkeypatch.setattr(
        "bts.util.os.replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    state.pick_locked = True
    with pytest.raises(OSError):
        save_state(state, tmp_path)

    assert _state_path(tmp_path).read_text() == good
    assert not list((tmp_path / "2026-07-09").glob("*.tmp"))


# --- (b) corrupt-state quarantine -------------------------------------------

def test_load_state_corrupt_json_quarantines_and_returns_none(tmp_path):
    p = _state_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text('{"date": "2026-07-09", "games": [TORN')

    assert load_state("2026-07-09", tmp_path) is None
    assert not p.exists(), "corrupt file must be moved aside, not left to crash the next start"
    quarantined = list(p.parent.glob("scheduler_state.json.corrupt-*"))
    assert len(quarantined) == 1
    assert "TORN" in quarantined[0].read_text()  # evidence preserved for diagnosis


def test_load_state_wrong_shape_quarantines_and_returns_none(tmp_path):
    p = _state_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"date": "2026-07-09", "not_a_real_field": 1}))

    assert load_state("2026-07-09", tmp_path) is None
    assert list(p.parent.glob("scheduler_state.json.corrupt-*"))


def test_load_state_valid_roundtrip_unaffected(tmp_path):
    save_state(SchedulerState(**MINIMAL_STATE), tmp_path)
    loaded = load_state("2026-07-09", tmp_path)
    assert loaded is not None
    assert loaded.date == "2026-07-09"
    assert not list((tmp_path / "2026-07-09").glob("*.corrupt-*"))


# --- (c) READY only after state initialization -------------------------------

def _run_day_harness(tmp_path, monkeypatch, events):
    """Drive run_day down the cheap no-games path with collaborators recorded."""
    monkeypatch.setattr("bts.scheduler.fetch_schedule",
                        lambda d: events.append("fetch_schedule") or [])
    monkeypatch.setattr("bts.scheduler._idle_until_next_wakeup",
                        lambda *a, **k: events.append("idle"))
    monkeypatch.setattr("bts.scheduler._next_day_wakeup",
                        lambda *a, **k: __import__("datetime").datetime(2026, 7, 10, 10, 0))
    monkeypatch.setattr("bts.scheduler.notify_ready",
                        lambda: events.append("notify_ready"))
    monkeypatch.setattr("bts.scheduler.notify_watchdog", lambda: None)
    real_load = load_state

    def recording_load(date, picks_dir):
        events.append("load_state")
        return real_load(date, picks_dir)

    monkeypatch.setattr("bts.scheduler.load_state", recording_load)
    config = {"orchestrator": {"picks_dir": str(tmp_path),
                               "heartbeat_path": str(tmp_path / ".heartbeat")},
              "scheduler": {}}
    run_day("2026-07-09", config, dry_run=False)


def test_run_day_loads_state_before_signaling_ready(tmp_path, monkeypatch):
    events = []
    _run_day_harness(tmp_path, monkeypatch, events)
    assert "load_state" in events, "state must be initialized on every startup path"
    assert "notify_ready" in events
    assert events.index("load_state") < events.index("notify_ready"), (
        "READY before state init advertises a daemon that may be about to "
        "crash-loop on a torn state file"
    )


def test_run_day_recovers_from_corrupt_state_and_still_readies(tmp_path, monkeypatch):
    p = _state_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text('{"date": "2026-07-09", TORN')

    events = []
    _run_day_harness(tmp_path, monkeypatch, events)  # must not raise

    assert "notify_ready" in events
    assert list(p.parent.glob("scheduler_state.json.corrupt-*")), (
        "corrupt state must be quarantined during startup"
    )


def test_dry_run_does_not_quarantine_corrupt_state(tmp_path, monkeypatch):
    # Codex review #12: --dry-run promises read-only behavior; it must not
    # rename a corrupt live state file into quarantine.
    p = _state_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text('{"date": "2026-07-09", TORN')

    events = []
    monkeypatch.setattr("bts.scheduler.fetch_schedule", lambda d: [])
    monkeypatch.setattr("bts.scheduler._idle_until_next_wakeup",
                        lambda *a, **k: events.append("idle"))
    monkeypatch.setattr("bts.scheduler._next_day_wakeup",
                        lambda *a, **k: __import__("datetime").datetime(2026, 7, 10, 10, 0))
    monkeypatch.setattr("bts.scheduler.notify_ready", lambda: None)
    monkeypatch.setattr("bts.scheduler.notify_watchdog", lambda: None)
    config = {"orchestrator": {"picks_dir": str(tmp_path),
                               "heartbeat_path": str(tmp_path / ".heartbeat")},
              "scheduler": {}}
    run_day("2026-07-09", config, dry_run=True)

    assert p.exists(), "dry-run must not mutate live state"
    assert not list(p.parent.glob("*.corrupt-*"))
