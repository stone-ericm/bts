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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class HeartbeatState:
    """Constants for well-known heartbeat state values."""
    RUNNING = "running"
    SLEEPING = "sleeping"
    WAITING_FOR_GAMES = "waiting_for_games"
    IDLE_END_OF_DAY = "idle_end_of_day"


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
