"""Surface quarantined scheduler state files (audit F3).

load_state quarantines a corrupt/torn scheduler_state.json to
scheduler_state.json.corrupt-<ts> and starts with fresh state — the daemon
keeps operating, but the day-state (pick_locked, skip context, finalization
tracking) was reset and the corruption itself may indicate disk/deploy
trouble. This source WARNs while the evidence is recent so an operator
actually looks at it.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "scheduler_state_integrity"

DEFAULT_THRESHOLDS = {
    # 7 covers a multi-day break (All-Star ~4 days) during which no EOD suite
    # runs — a quarantine early in a break must still be visible at the first
    # post-break evaluation (Codex review #4).
    "lookback_days": 7,
}


def check(
    picks_dir: Path,
    today: date | None = None,
    thresholds: dict | None = None,
) -> list[Alert]:
    """WARN if any scheduler_state quarantine files exist in the lookback window."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if today is None:
        today = date.today()

    found: list[str] = []
    for offset in range(int(t["lookback_days"]) + 1):
        d = today - timedelta(days=offset)
        date_dir = Path(picks_dir) / d.isoformat()
        if not date_dir.is_dir():
            continue
        for q in sorted(date_dir.glob("scheduler_state.json.corrupt-*")):
            found.append(f"{d.isoformat()}/{q.name}")

    if not found:
        return []
    return [Alert(
        level="WARN",
        source=SOURCE,
        message=(
            f"{len(found)} quarantined scheduler state file(s) in the last "
            f"{t['lookback_days']} day(s) — a torn/corrupt scheduler_state.json "
            f"was recovered at startup (day-state such as pick_locked/skip "
            f"context reset to fresh): {', '.join(found)}"
        ),
    )]
