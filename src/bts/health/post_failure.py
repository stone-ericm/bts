"""Tier 1: Bluesky post failure check.

Reads today's pick file. If a pick was locked but bluesky_posted is false
or the URI is missing, post-publication failed silently — followers don't
see today's pick.

Run at end-of-day, well after first-pitch lock. If the pick file doesn't
exist (no games today) or no pick was made (rare), no alert.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "bluesky_post"


def check(picks_dir: Path, today: date | None = None) -> list[Alert]:
    """Returns CRITICAL alert if today's pick was locked but Bluesky post failed."""
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
    return [Alert(
        level="CRITICAL",
        source=SOURCE,
        message=(
            f"pick locked for {today.isoformat()} but Bluesky post failed: "
            f"bluesky_posted={posted}, bluesky_uri={uri}. Followers don't see today's pick."
        ),
    )]
