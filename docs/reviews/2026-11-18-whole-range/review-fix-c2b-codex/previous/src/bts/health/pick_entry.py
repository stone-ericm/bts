"""EOD record for a committed pick whose MLB entry was never confirmed (audit F1).

check-pick-entered's escalation DMs are the live alarms; this source is the
audit-trail backstop: once the submission cutoff has passed, a marker still in
"alerted" means the day ended with no verified entry (WARN), and "dm_failed"
means the alert itself never reached the operator (CRITICAL — unentered AND
unreachable). Runs in the EOD health suite; stays quiet while a late game's
entry window is still open.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from bts.health.alert import Alert
from bts.picks import SUBMISSION_CUTOFF_MIN

log = logging.getLogger(__name__)

SOURCE = "pick_entry"

SUBMIT_CUTOFF_MIN = SUBMISSION_CUTOFF_MIN  # single definition lives in bts.picks


def _earliest_cutoff_utc(picks_dir: Path, today: date) -> datetime | None:
    """Earliest (first_pitch - 5min) across the day's pick slots, UTC-aware.

    None when the pick file or its game times are unavailable — callers treat
    that as 'window unknowable, assume closed' (by EOD it is)."""
    path = Path(picks_dir) / f"{today.isoformat()}.json"
    try:
        body = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    times = []
    for slot_key in ("pick", "double_down"):
        slot = body.get(slot_key) or {}
        gt = slot.get("game_time")
        if not gt:
            continue
        try:
            t = datetime.fromisoformat(gt.replace("Z", "+00:00"))
        except ValueError:
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        times.append(t)
    if not times:
        return None
    return min(times) - timedelta(minutes=SUBMIT_CUTOFF_MIN)


def check(
    picks_dir: Path,
    today: date | None = None,
    now: datetime | None = None,
) -> list[Alert]:
    """WARN/CRITICAL when today's entry marker never reached 'confirmed'."""
    if today is None:
        today = date.today()
    if now is None:
        now = datetime.now(timezone.utc)

    marker_path = Path(picks_dir).parent / "health_state" / "pick_entry_check.json"
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if marker.get("date") != today.isoformat():
        return []
    status = marker.get("status")
    if status not in ("alerted", "dm_failed"):
        return []

    cutoff = _earliest_cutoff_utc(picks_dir, today)
    if cutoff is not None and now < cutoff:
        return []  # entry window still open — the live escalations own this

    if status == "dm_failed":
        return [Alert(
            level="CRITICAL",
            source=SOURCE,
            message=(
                f"{today.isoformat()}: committed pick entry never confirmed AND "
                f"the not-entered alert DM failed to send — the operator was "
                f"never reached (reason={marker.get('reason')})"
            ),
        )]
    return [Alert(
        level="WARN",
        source=SOURCE,
        message=(
            f"{today.isoformat()}: committed pick entry never confirmed by the "
            f"submission cutoff (last status=alerted, "
            f"reason={marker.get('reason')}) — check the contest account and "
            f"the settled result"
        ),
    )]
