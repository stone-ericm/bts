"""Heartbeat file read/write for scheduler health monitoring.

The scheduler writes a heartbeat every 30 seconds with its current state.
The dashboard's /health endpoint reads the heartbeat and decides whether
the scheduler is alive, making Fly's HTTP health check work without
needing IPC between processes.

During long sleeps between lineup checks, the heartbeat's 'state' field
indicates sleeping_until_X so the staleness check knows the scheduler
is intentionally quiet, not hung.
"""
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Iterator, Optional

log = logging.getLogger(__name__)


class HeartbeatState:
    """Constants for well-known heartbeat state values."""
    RUNNING = "running"
    SLEEPING = "sleeping"
    WAITING_FOR_GAMES = "waiting_for_games"
    IDLE_END_OF_DAY = "idle_end_of_day"
    STALLED = "stalled"  # process alive but cascade progress stopped (H5b)


def write_heartbeat(
    path: Path,
    state: str,
    now_utc: Optional[datetime] = None,
    sleeping_until: Optional[datetime] = None,
    extra: Optional[dict] = None,
) -> None:
    """Write a heartbeat JSON file atomically (via .tmp + rename)."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    payload = {
        "state": state,
        "timestamp": now_utc.isoformat(),
    }
    if sleeping_until is not None:
        payload["sleeping_until"] = sleeping_until.isoformat()
    if extra:
        payload.update(extra)

    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload))
    tmp.rename(path)


def read_heartbeat(path: Path) -> Optional[dict]:
    """Read the current heartbeat. Returns None if missing or unreadable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def is_heartbeat_fresh(
    path: Path,
    max_age_sec: int = 180,
    now_utc: Optional[datetime] = None,
) -> bool:
    """Return True if the heartbeat is fresh enough to indicate a live scheduler.

    A heartbeat in state='sleeping' with sleeping_until in the future is
    considered fresh regardless of age.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    hb = read_heartbeat(path)
    if hb is None:
        return False

    # Stalled = process alive but no cascade progress; timestamps stay fresh
    # because the pulse keeps writing, so the age check below must not see it.
    if hb.get("state") == HeartbeatState.STALLED:
        return False

    # If sleeping, trust sleeping_until
    if hb.get("state") == HeartbeatState.SLEEPING:
        sleeping_until = hb.get("sleeping_until")
        if sleeping_until:
            wake = datetime.fromisoformat(sleeping_until)
            if wake > now_utc:
                return True

    # Otherwise, check age
    ts = datetime.fromisoformat(hb["timestamp"])
    age_sec = (now_utc - ts).total_seconds()
    return age_sec <= max_age_sec


@contextmanager
def heartbeat_watchdog(
    path: Path,
    interval_sec: float = 60,
    *,
    kind: str = "primary",
    date: Optional[str] = None,
    stall_after_sec: float = 900,
    durations_path: Optional[Path] = None,
) -> Iterator[str]:
    """Refresh the heartbeat while executing the body — truthfully (H5b).

    Owns a progress-beacon run token. Each tick: ALWAYS sd_notify (the unit
    has WatchdogSec=1800 — silence means systemd kills the scheduler, which
    Phase 1 forbids), then heartbeat state by progress age: fresh marks →
    RUNNING with {stage, stage_age_s, run_id}; older than ``stall_after_sec``
    → STALLED with {stage, stalled_for_s, run_id} (re-snapshotted just before
    writing to shrink the stall-vs-recovery race to one tick). The pulse is
    the single writer of ``durations_path`` (stage-transition jsonl, the
    Phase-2 threshold dataset): completed stages get status ok /
    ok_after_stall; the first tick of each stalled stage instance appends one
    stalled_incomplete row (latch flips only on successful append).

    Yields the run_id (tests use it; callers may ignore). On exit the run is
    retired — a leaked pulse's snapshot returns None, so it can never
    overwrite the scheduler's post-cascade state heartbeats.

    Spec: docs/superpowers/specs/2026-06-11-h5b-truthful-heartbeat-design.md
    """
    from bts import progress
    from bts import sd_notify

    run_id = progress.begin_run(kind)
    stop = Event()
    stalled_gens: set = set()      # stage generations observed stalled
    persisted_gens: set = set()    # generations whose stalled_incomplete row is on disk

    def _append_rows(rows: list) -> bool:
        if durations_path is None or not rows:
            return False
        try:
            durations_path.parent.mkdir(parents=True, exist_ok=True)
            with durations_path.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            return True
        except Exception as e:
            log.warning(f"stage-durations append failed (non-fatal): {e}")
            return False

    def _enrich(rec: dict, status: str) -> dict:
        return {
            "run_id": run_id, "pid": os.getpid(), "kind": kind, "date": date,
            "threshold_used_s": stall_after_sec, "status": status, **rec,
        }

    def _drain_completed() -> None:
        rows = [
            _enrich(r, "ok_after_stall" if r["generation"] in stalled_gens else "ok")
            for r in progress.drain_transitions(run_id)
        ]
        _append_rows(rows)

    def _tick() -> None:
        sd_notify.notify_watchdog()  # ALWAYS — never let systemd kill on a stall
        snap = progress.snapshot(run_id)
        if snap is None:
            return  # run retired/superseded: write nothing, ever
        if snap["stage_age_s"] >= stall_after_sec:
            snap = progress.snapshot(run_id)  # revalidate: recovery may have marked
            if snap is None:
                return
        if snap["stage_age_s"] < stall_after_sec:
            write_heartbeat(path, state=HeartbeatState.RUNNING, extra={
                "stage": snap["stage"],
                "stage_age_s": round(snap["stage_age_s"], 1),
                "run_id": run_id,
            })
        else:
            write_heartbeat(path, state=HeartbeatState.STALLED, extra={
                "stage": snap["stage"],
                "stalled_for_s": round(snap["stage_age_s"], 1),
                "run_id": run_id,
            })
            gen = snap["generation"]
            stalled_gens.add(gen)
            if gen not in persisted_gens:
                row = _enrich({
                    "stage": snap["stage"],
                    "generation": gen,
                    "started_at": snap["stage_started_wall"],
                    "ended_at": None,
                    "duration_s": round(snap["stage_age_s"], 1),
                }, "stalled_incomplete")
                if _append_rows([row]):
                    persisted_gens.add(gen)
        _drain_completed()

    def _pulse() -> None:
        while not stop.is_set():
            try:
                _tick()
            except Exception:
                pass
            stop.wait(interval_sec)

    thread = Thread(target=_pulse, daemon=True)
    thread.start()
    try:
        yield run_id
    finally:
        stop.set()
        thread.join(timeout=2)
        final = [
            _enrich(r, "ok_after_stall" if r["generation"] in stalled_gens else "ok")
            for r in progress.end_run(run_id)
        ]
        _append_rows(final)
