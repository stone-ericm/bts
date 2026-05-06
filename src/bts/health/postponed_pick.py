"""Tier 1: unposted pick committed to a postponed or missing game.

This catches the same incident class as the 2026-05-05 postponed-game
production failure before an unposted stale pick can silently survive to
the posting window.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from bts.health.alert import Alert
from bts.picks import classify_pick_lock_state, load_pick

log = logging.getLogger(__name__)

SOURCE = "postponed_pick"


def check(picks_dir: Path, today: date | None = None) -> list[Alert]:
    """Return an alert if today's unposted pick is stale due to game status."""
    if today is None:
        today = date.today()
    date_iso = today.isoformat()
    try:
        daily = load_pick(date_iso, picks_dir)
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
        log.warning(f"could not parse pick for {date_iso}; skipping postponed_pick check")
        return []
    if daily is None or daily.bluesky_posted:
        return []

    lock_state = classify_pick_lock_state(daily, date_iso)
    if lock_state.stale:
        details = _format_lock_state(lock_state.reason, lock_state.game_pk,
                                     lock_state.abstract, lock_state.detailed)
        return [Alert(
            level="CRITICAL",
            source=SOURCE,
            message=(
                f"unposted pick for {date_iso} is stale ({details}). "
                "Regenerate before any Bluesky post."
            ),
        )]
    if lock_state.reason == "status_lookup_failed":
        return [Alert(
            level="WARN",
            source=SOURCE,
            message=(
                f"could not verify game status for unposted pick on {date_iso}; "
                "postponed_pick health check failed closed."
            ),
        )]
    return []


def _format_lock_state(
    reason: str,
    game_pk: int | None,
    abstract: str | None,
    detailed: str | None,
) -> str:
    parts = [f"reason={reason}"]
    if game_pk is not None:
        parts.append(f"game_pk={game_pk}")
    if abstract:
        parts.append(f"abstract={abstract}")
    if detailed:
        parts.append(f"detailed={detailed}")
    return ", ".join(parts)
