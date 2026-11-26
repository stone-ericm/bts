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

from bts import entry_receipt, receipt_io
from bts.health.alert import Alert
from bts.picks import SUBMISSION_CUTOFF_MIN
from bts.util import read_bounded_file_bytes

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


def _live_receipt_follows(picks_dir: Path, et_date: str, observed: datetime) -> bool:
    root = entry_receipt.receipts_root(picks_dir) / et_date
    for path in receipt_io.discover(root):
        try:
            receipt = json.loads(read_bounded_file_bytes(path, entry_receipt.STATE_READ_LIMIT).decode("utf-8"))
        except (OSError, ValueError, RecursionError):
            continue
        if (not isinstance(receipt, dict) or receipt.get("schema") != entry_receipt.SCHEMA
                or receipt.get("et_date") != et_date or receipt.get("early_cron_run") is True
                or receipt.get("outcome") != "observed" or receipt.get("exit_code") not in (0, 1)
                or receipt.get("marker_status") not in ("alerted", "dm_failed", "confirmed", "present_unverified")):
            continue
        # A normal exit establishes that the marker write returned. The response
        # timestamp establishes a lower bound on when that write could occur.
        observation = receipt.get("observation")
        if isinstance(observation, dict):
            completed = entry_receipt.aware_timestamp(observation.get("response_completed_at"))
            if completed is not None and completed > observed:
                return True
    return False


def _early_alerts(picks_dir: Path, today: date, now: datetime) -> list[Alert]:
    et_date = today.isoformat()
    marker = entry_receipt.read_early_marker(picks_dir, et_date)
    status = marker.get("status")
    if status not in ("alerted", "dm_failed"):
        return []
    observed_at = marker.get("observed_at", marker["checked_at"])
    observed = entry_receipt.aware_timestamp(observed_at)
    if observed is None or _live_receipt_follows(picks_dir, et_date, observed):
        return []
    daily, _, first_pitch = entry_receipt.read_early_pick(picks_dir, et_date)
    if daily is None:
        if status == "dm_failed":
            return [Alert(level="CRITICAL", source=SOURCE,
                          message=f"{et_date}: early entry warning DM failed at {observed_at}; "
                                  "the current selection is unknown")]
        return []
    if (entry_receipt.slot_identities(entry_receipt.selection_slots(daily))
            != entry_receipt.slot_identities(marker["slots"])):
        return []
    if now < first_pitch - timedelta(minutes=SUBMIT_CUTOFF_MIN):
        return []
    return _marker_alerts(et_date, marker)


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
        return _early_alerts(picks_dir, today, now)
    if marker.get("date") != today.isoformat():
        try:
            live_date = date.fromisoformat(marker.get("date"))
        except (TypeError, ValueError):
            live_date = None
        return [] if live_date is not None and live_date >= today else _early_alerts(picks_dir, today, now)
    status = marker.get("status")
    if status not in ("alerted", "dm_failed"):
        return []

    cutoff = _earliest_cutoff_utc(picks_dir, today)
    if cutoff is not None and now < cutoff:
        return []  # entry window still open — the live escalations own this

    return _marker_alerts(today.isoformat(), marker)


def _marker_alerts(et_date: str, marker: dict) -> list[Alert]:
    if marker["status"] == "dm_failed":
        return [Alert(
            level="CRITICAL",
            source=SOURCE,
            message=(
                f"{et_date}: committed pick entry never confirmed AND "
                f"the not-entered alert DM failed to send — the operator was "
                f"never reached (reason={marker.get('reason')})"
            ),
        )]
    return [Alert(
        level="WARN",
        source=SOURCE,
        message=(
            f"{et_date}: committed pick entry never confirmed by the "
            f"submission cutoff (last status=alerted, "
            f"reason={marker.get('reason')}) — check the contest account and "
            f"the settled result"
        ),
    )]
