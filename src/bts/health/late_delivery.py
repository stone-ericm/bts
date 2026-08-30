"""EOD backstop for the delivery-cutoff guard (2026-08-30 incident).

CRITICAL when the day's pick was sent at/after first pitch − 5 min (unenterable
when delivered), or when the scheduler REFUSED a late delivery (nothing was sent
by that path). WARN when the send landed inside the operator reserve — the pick
was enterable but the operator had less than `operator_reserve_min` to act.

Why a separate source from fallback_defer: that check validates "never miss"
purely as "a delivered pick file exists", which is exactly how the 2026-08-30
13:36 DM for a 13:35 cutoff read as healthy.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from bts.health.alert import Alert
from bts.picks import load_pick, pick_was_delivered, submission_cutoff_et

SOURCE = "late_delivery"
ET = ZoneInfo("America/New_York")


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=ET)


def _state(picks_dir: Path, day: date) -> dict:
    try:
        body = json.loads((Path(picks_dir) / day.isoformat() / "scheduler_state.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return body if isinstance(body, dict) else {}


def check(picks_dir: Path, today: date | None = None, now: datetime | None = None,
          operator_reserve_min: float = 10) -> list[Alert]:
    """Return alerts for `today` (default: the current ET date)."""
    day = today or (now or datetime.now(ET)).astimezone(ET).date()
    alerts: list[Alert] = []
    state = _state(picks_dir, day)
    for r in state.get("delivery_refusals") or []:
        alerts.append(Alert("CRITICAL", SOURCE, (
            f"late delivery REFUSED for {day.isoformat()}: {r.get('batter')} ({r.get('label')}) "
            f"was {r.get('late_min')} min past the {r.get('cutoff_et')} cutoff; nothing was "
            f"delivered by that path")))
    try:
        daily = load_pick(day.isoformat(), picks_dir)
    except Exception:
        daily = None
    if daily is None or not pick_was_delivered(daily):
        return alerts
    delivered_at = _parse(getattr(daily, "delivered_at", None)) or _parse(state.get("pick_locked_at"))
    if delivered_at is None:
        return alerts
    cutoff = submission_cutoff_et(daily)
    names = daily.pick.batter_name + (
        f" + {daily.double_down.batter_name}" if daily.double_down else "")
    sent = delivered_at.astimezone(ET)
    if delivered_at >= cutoff:
        alerts.append(Alert("CRITICAL", SOURCE, (
            f"late delivery for {day.isoformat()}: {names} sent {sent:%H:%M} ET, cutoff was "
            f"{cutoff:%H:%M} ET — unenterable when delivered")))
    elif delivered_at > cutoff - timedelta(minutes=operator_reserve_min):
        left = (cutoff - delivered_at).total_seconds() / 60
        alerts.append(Alert("WARN", SOURCE, (
            f"tight delivery for {day.isoformat()}: {names} sent {sent:%H:%M} ET, only "
            f"{left:.0f} min before the {cutoff:%H:%M} ET cutoff")))
    return alerts
