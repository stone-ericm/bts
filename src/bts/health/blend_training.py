"""Tier 1: blend training cron miss check.

The 3 AM ET cron on day N generates blend_<N+1>.pkl. If that file is
missing at end-of-day on day N, tomorrow's prediction will fall back
to a stale blend, silently degrading picks.

Pattern observed in production:
  blend_2026-04-26.pkl generated 2026-04-25 03:06 ET  ← N's cron → N+1's blend
  blend_2026-04-27.pkl generated 2026-04-26 03:06 ET
  blend_2026-04-28.pkl generated 2026-04-27 03:06 ET

So at end-of-day on day N, blend_<N+1>.pkl must exist.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "blend_training"


def check(models_dir: Path, today: date | None = None) -> list[Alert]:
    """Returns CRITICAL alert if tomorrow's blend pkl is missing on day N at end-of-day."""
    if today is None:
        today = date.today()
    tomorrow = today + timedelta(days=1)
    expected = models_dir / f"blend_{tomorrow.isoformat()}.pkl"
    if expected.exists():
        return []
    return [Alert(
        level="CRITICAL",
        source=SOURCE,
        message=(
            f"missing tomorrow's blend pkl: {expected.name} not found in {models_dir}. "
            f"3 AM cron on {today.isoformat()} likely failed; predictions on "
            f"{tomorrow.isoformat()} will fall back to stale blend."
        ),
    )]
