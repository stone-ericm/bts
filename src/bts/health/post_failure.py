"""Tier 1: pick delivery failure check.

Reads today's pick file. If a pick was locked but neither public Bluesky
posting nor private notification delivery is recorded, publication failed
silently and the human operator may not see today's pick.

**Time guard**: the alert is suppressed before 22:00 ET because Bluesky
delivery fires at lineup confirmation (45min before each game's first pitch)
or via the 1 AM safety-net cron the next day. Pre-cutoff alerts are daily
false positives — the post window hasn't closed yet.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bts.daily_decision import load_decision
from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "pick_delivery"

ET = ZoneInfo("America/New_York")
EARLIEST_HOUR_ET = 22  # well after the latest typical first-pitch (~7-9pm ET)


def check(
    picks_dir: Path,
    today: date | None = None,
    now: datetime | None = None,
) -> list[Alert]:
    """Return CRITICAL if today's pick was not publicly posted or privately delivered."""
    if today is None:
        today = date.today()
    pick_path = picks_dir / f"{today.isoformat()}.json"
    if not pick_path.exists():
        return []
    # An unreadable pick file on a day a pick EXISTS is itself alarming — let
    # it propagate to _safe_run → CRITICAL rather than silently skipping the
    # delivery check (audit F4).
    data = json.loads(pick_path.read_text())

    pick = data.get("pick")
    if not pick:
        # No pick (e.g., all games skipped). Nothing to post.
        return []
    # Decision gate (D6 / GH #144): a deliberate MDP skip is not a post failure.
    # A non-scoreable decision (action=="skip" or scoreable==False) means no pick
    # was committed, so its absence from the public feed is expected, not a failure.
    dec = load_decision(today.isoformat(), picks_dir)
    if dec is not None and (dec.get("action") == "skip" or not dec.get("scoreable")):
        return []
    posted = data.get("bluesky_posted")
    uri = data.get("bluesky_uri")
    notified = data.get("notification_sent")
    notification_id = data.get("notification_id")
    if notified is True and notification_id:
        return []
    if posted is True and uri:
        return []
    # Time guard: suppress before 22:00 ET — post window may still be open.
    if now is None:
        now = datetime.now(ET)
    now_et = now.astimezone(ET) if now.tzinfo is not None else now.replace(tzinfo=ET)
    if now_et.date() == today and now_et.hour < EARLIEST_HOUR_ET:
        return []
    return [Alert(
        level="CRITICAL",
        source=SOURCE,
        message=(
            f"pick locked for {today.isoformat()} but pick delivery failed: "
            f"bluesky_posted={posted}, bluesky_uri={uri}, "
            f"notification_sent={notified}, notification_id={notification_id}. "
            f"No public post or private notification is recorded."
        ),
    )]
