"""Tier 2: pitcher data sparsity tracker.

Counts how often the morning's pick (or double-down) has a 'LIMITED pitcher
data' flag, which means `pitcher_hr_30g` fell back to the league-average
starter rate (0.22) because the actual pitcher had < min_periods PAs.

Persistent firing indicates production is leaning on the fallback heavily.
Possible causes:
  - Wave of pitcher debuts/call-ups (early-season, post-trade-deadline)
  - Tommy John returns disproportionately on the schedule
  - Calibration issue with min_periods=7 (was 10 before 2026-04-14)

This is the diagnostic gap memo'd in `project_bts_strategic_gaps_2026_04_30.md`
gap #5 — also forms the empirical basis for evaluating MiLB transfer (gap #4).

Thresholds chosen to be quiet during normal operation but fire when the
fallback rate becomes a meaningful share of picks.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "pitcher_sparsity"

DEFAULT_THRESHOLDS = {
    "lookback_days": 14,
    "min_examined": 5,    # need at least this many days with pick data before alerting
    "warn_count": 5,      # 5/14 days = 35%
    "critical_count": 8,  # 8/14 days = 57%
}

# The flag string lives in src/bts/orchestrator.py (predict_local). Keep this
# in sync with that producer; if it changes, update here AND add a redirect.
FLAG_NEEDLE = "LIMITED pitcher data"


def check(
    picks_dir: Path,
    today: date | None = None,
    thresholds: dict | None = None,
) -> list[Alert]:
    """Returns WARN/CRITICAL when LIMITED pitcher data flag fires too often."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if today is None:
        today = date.today()
    if not picks_dir.exists():
        return []

    cutoff = today - timedelta(days=t["lookback_days"])
    flagged_days = 0
    examined = 0

    for p in sorted(picks_dir.glob("*.json")):
        try:
            pick_date = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if pick_date < cutoff or pick_date > today:
            continue
        try:
            body = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning(f"could not parse {p}; skipping pitcher_sparsity check")
            continue
        # Only count days with an actual pick (skip skip-days / corrupted).
        pick = body.get("pick")
        if not pick:
            continue
        examined += 1
        pick_flags = pick.get("flags") or []
        dd = body.get("double_down") or {}
        dd_flags = dd.get("flags") or []
        if any(FLAG_NEEDLE in str(f) for f in pick_flags + dd_flags):
            flagged_days += 1

    if examined < t["min_examined"]:
        return []  # insufficient history; quiet during ramp-up

    if flagged_days >= t["critical_count"]:
        return [Alert(
            level="CRITICAL",
            source=SOURCE,
            message=(
                f"{flagged_days}/{examined} days in last {t['lookback_days']}d had a "
                f"{FLAG_NEEDLE!r} flag — pitcher_hr_30g is falling back to league avg "
                f"on a majority of picks. Investigate: pitcher debuts, min_periods, or "
                f"data pipeline gaps."
            ),
        )]
    if flagged_days >= t["warn_count"]:
        return [Alert(
            level="WARN",
            source=SOURCE,
            message=(
                f"{flagged_days}/{examined} days in last {t['lookback_days']}d had a "
                f"{FLAG_NEEDLE!r} flag — fallback rate elevated."
            ),
        )]
    return []
