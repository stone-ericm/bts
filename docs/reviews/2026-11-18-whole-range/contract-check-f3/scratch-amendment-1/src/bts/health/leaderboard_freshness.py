"""Tier 2 health check: detect a stale or absent leaderboard scrape.

Watches `data/leaderboard/leaderboard_snapshots/` mtimes. Fires WARN
when the last successful scrape is between 12h and 36h old, CRITICAL
beyond 36h.

The 36h threshold is intentional: with twice-daily scrapes (10:00 ET
and 01:00 ET), a healthy gap is at most ~9h. A 36h gap means we
missed both slots — almost certainly auth-cookie expiry or persistent
network issue.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "leaderboard_freshness"

DEFAULT_THRESHOLDS = {
    "warn_hours": 12.0,
    "critical_hours": 36.0,
}


def check(leaderboard_dir: Path, thresholds: dict | None = None) -> list[Alert]:
    """Returns INFO/WARN/CRITICAL when leaderboard scrape is stale."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    snaps_dir = leaderboard_dir / "leaderboard_snapshots"
    # Watcher not yet deployed -> silent (don't alarm pre-launch)
    if not snaps_dir.exists():
        return []
    snaps = sorted(snaps_dir.glob("*.parquet"))
    if not snaps:
        return [Alert(
            level="WARN", source=SOURCE,
            message="leaderboard_snapshots directory empty - no successful scrapes recorded",
        )]
    latest = max(snaps, key=lambda p: p.stat().st_mtime)
    age_h = (datetime.now().timestamp() - latest.stat().st_mtime) / 3600
    if age_h >= t["critical_hours"]:
        return [Alert(
            level="CRITICAL", source=SOURCE,
            message=(f"leaderboard scrape stale by {age_h:.1f}h "
                     f"(latest: {latest.name}). Auth cookies likely expired - "
                     f"refresh via scripts/capture_bts_cookies.py on Mac."),
        )]
    if age_h >= t["warn_hours"]:
        return [Alert(
            level="WARN", source=SOURCE,
            message=f"leaderboard scrape lagging: latest {age_h:.1f}h ago ({latest.name})",
        )]
    return []
