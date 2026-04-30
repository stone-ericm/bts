"""Tier 1: Bluesky post failure check.

Reads today's pick file. If a pick was locked but bluesky_posted is false
or the URI is missing, post-publication failed silently — followers don't
see today's pick.

**Time guard**: the alert is suppressed before 22:00 ET because Bluesky
posts fire at lineup confirmation (45min before each game's first pitch)
or via the 1 AM safety-net cron the next day. Pre-cutoff alerts are daily
false positives — the post window hasn't closed yet.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "bluesky_post"

ET = ZoneInfo("America/New_York")
EARLIEST_HOUR_ET = 22  # well after the latest typical first-pitch (~7-9pm ET)


def check(
    picks_dir: Path,
    today: date | None = None,
    now: datetime | None = None,
) -> list[Alert]:
    """Returns CRITICAL alert if today's pick was locked but Bluesky post failed (post 22:00 ET)."""
    if today is None:
        today = date.today()
    pick_path = picks_dir / f"{today.isoformat()}.json"
    if not pick_path.exists():
        return []
    try:
        data = json.loads(pick_path.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning(f"could not parse {pick_path}; skipping bluesky_post check")
        return []

    pick = data.get("pick")
    if not pick:
        # No pick (e.g., all games skipped). Nothing to post.
        return []
    posted = data.get("bluesky_posted")
    uri = data.get("bluesky_uri")
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
            f"pick locked for {today.isoformat()} but Bluesky post failed: "
            f"bluesky_posted={posted}, bluesky_uri={uri}. Followers don't see today's pick."
        ),
    )]
