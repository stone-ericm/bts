"""Tier 2: projected lineup frequency.

% of recent days locked with `projected_lineup: true` on pick or DD.

Higher than ~30% in a rolling 14-day window suggests either:
  - MLB API delays in publishing confirmed lineups, OR
  - A scheduler-side gating bug (we should be waiting longer for confirmation),
  - OR rampant doubleheader/late-game scheduling forcing projection use.

Yesterday's case: the shadow-det watcher's morning capture had Perdomo (DD)
flagged PROJECTED — and Perdomo's game didn't materialize (NO_GAME). Catching
this pattern early lets us tighten the lineup-confirmation gate before it
costs streaks.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "projected_lineup"

DEFAULT_THRESHOLDS = {
    "info_pct": 0.30,
    "warn_pct": 0.50,
    "min_days": 7,  # require at least 7 days for stat power
    "lookback_days": 14,
}


def check(picks_dir: Path, today: date | None = None,
          thresholds: dict | None = None) -> list[Alert]:
    """Returns INFO/WARN if rolling-14d projected_lineup % exceeds threshold."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if today is None:
        today = date.today()
    cutoff = today - timedelta(days=int(t["lookback_days"]))

    n_total = 0
    n_projected = 0
    try:
        for p in sorted(picks_dir.glob("*.json")):
            if "." in p.stem:
                continue
            try:
                file_date = date.fromisoformat(p.stem)
            except ValueError:
                continue
            if file_date < cutoff or file_date > today:
                continue
            try:
                data = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            pick = data.get("pick") or {}
            dd = data.get("double_down") or {}
            if not pick:
                continue
            n_total += 1
            pick_proj = bool(pick.get("projected_lineup"))
            dd_proj = bool(dd.get("projected_lineup")) if dd else False
            if pick_proj or dd_proj:
                n_projected += 1
    except Exception as e:
        log.exception(f"projected_lineup check failed: {e}")
        return []

    if n_total < int(t["min_days"]):
        return []
    pct = n_projected / n_total
    if pct < t["info_pct"]:
        return []
    if pct >= t["warn_pct"]:
        level = "WARN"
    else:
        level = "INFO"
    return [Alert(
        level=level,
        source=SOURCE,
        message=(
            f"projected_lineup frequency {pct:.1%} ({n_projected}/{n_total}) "
            f"in last {int(t['lookback_days'])} days — exceeds {t['info_pct']:.0%} info threshold"
        ),
    )]
