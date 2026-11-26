"""Tier 1: pooled-training daily status check (item #6 / Component D).

The pooled-training cron writes data/models_pooled/<DATE>_status.json with
the count of successful seeds out of N. This check reads the status for
tomorrow's date at end-of-day and alerts when the pool is under-filled —
production falls back to single-seed=42 if too few seeds train, which
re-introduces the seed=42 overconfidence we shipped pooling to fix.

Severity ladder (by ratio):
  n_complete >= n_seeds:           no alert (full pool)
  n_complete >= 0.8 * n_seeds:     INFO  (still a usable pool)
  n_complete >= 0.5 * n_seeds:     WARN  (degraded inference quality)
  n_complete <  0.5 * n_seeds:     CRITICAL (would force fallback to single)

Pre-deployment behavior: if data/models_pooled/ doesn't exist, the check
returns [] (don't false-alert before Phase 1 ships).
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "pooled_training"

INFO_RATIO = 0.8
WARN_RATIO = 0.5


def check(pooled_dir: Path, today: date | None = None) -> list[Alert]:
    """Returns severity-graded alert reading <TOMORROW>_status.json from pooled_dir."""
    if today is None:
        today = date.today()
    if not pooled_dir.exists():
        # Pre-launch — pooled training not yet enabled
        return []

    tomorrow = today + timedelta(days=1)
    status_path = pooled_dir / f"{tomorrow.isoformat()}_status.json"
    if not status_path.exists():
        return [Alert(
            level="CRITICAL",
            source=SOURCE,
            message=(
                f"no status file for {tomorrow.isoformat()} at {status_path}. "
                f"Pooled-training cron likely failed before writing status; "
                f"prediction will fall back to single seed=42."
            ),
        )]

    try:
        body = json.loads(status_path.read_text())
    except json.JSONDecodeError as e:
        return [Alert(
            level="CRITICAL",
            source=SOURCE,
            message=f"malformed {status_path.name}: {e}",
        )]

    n_complete = body.get("n_complete")
    n_seeds = body.get("n_seeds")
    if not isinstance(n_complete, int) or not isinstance(n_seeds, int) or n_seeds <= 0:
        return [Alert(
            level="CRITICAL",
            source=SOURCE,
            message=(
                f"{status_path.name} missing or invalid n_complete / n_seeds "
                f"(n_complete={n_complete!r}, n_seeds={n_seeds!r})."
            ),
        )]

    if n_complete >= n_seeds:
        return []

    ratio = n_complete / n_seeds
    if ratio >= INFO_RATIO:
        level = "INFO"
    elif ratio >= WARN_RATIO:
        level = "WARN"
    else:
        level = "CRITICAL"

    return [Alert(
        level=level,
        source=SOURCE,
        message=(
            f"pooled training for {tomorrow.isoformat()}: {n_complete}/{n_seeds} "
            f"seeds complete ({ratio:.0%})."
            + (" Production will fall back to single seed=42." if level == "CRITICAL" else "")
        ),
    )]
