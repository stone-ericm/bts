"""Health visibility for context-stack shadow result resolution.

The wait loop in `bts check-results --wait-deadline-et` settles the common
same-night cases (West Coast extras not final at the 01:00 ET cron); this
check surfaces the residual stranded ones — box down overnight, multi-day
suspensions, a deadline exhausted on transient API failures — instead of
leaving them silent. The 2026-07-10 shadow sat unresolved for a month with
no signal anywhere.

Unresolved dates are derived in-memory from the shadow files themselves via
build_shadow_cycle_status, NOT read from the status JSON artifact: that file
is written non-atomically by another process and can be stale or torn.
Legacy v1 (unstamped) shadow files are excluded by the same version filter
the status artifact uses.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "shadow_resolution"

DEFAULT_THRESHOLDS = {
    # Yesterday's shadow may legitimately still be pending until tonight's
    # cron (and its wait loop) has had its chance — alert from age 2.
    "grace_days": 2,
    "critical_age_days": 7,
}


def check(
    picks_dir: Path | str,
    *,
    today: date | None = None,
    thresholds: dict | None = None,
) -> list[Alert]:
    picks_path = Path(picks_dir)
    if not picks_path.exists() or not any(picks_path.glob("*.shadow.json")):
        return []  # silent pre-shadow / off-season

    limits = dict(DEFAULT_THRESHOLDS)
    limits.update(thresholds or {})
    today = today or date.today()

    from bts.shadow_eval import build_shadow_cycle_status

    status = build_shadow_cycle_status(picks_path)
    unresolved = status.get("coverage", {}).get("unresolved_shadow_dates", [])

    alerts: list[Alert] = []
    for date_str in unresolved:
        try:
            shadow_date = date.fromisoformat(date_str)
        except ValueError:
            log.warning("unparseable unresolved shadow date %r", date_str)
            continue
        age = (today - shadow_date).days
        if age < limits["grace_days"]:
            continue
        level = "CRITICAL" if age >= limits["critical_age_days"] else "WARN"
        alerts.append(Alert(
            level=level,
            source=SOURCE,
            message=(
                f"shadow result for {date_str} unresolved for {age} days — "
                f"the check-results wait loop did not settle it; run "
                f"`bts check-results --date {date_str}` manually (safe: a "
                f"terminal production result short-circuits to shadow-only)"
            ),
            # One incident per stranded date: two different stranded dates
            # must both reach the operator, not dedup as "already seen".
            incident_key=f"{SOURCE}:{date_str}",
        ))
    return alerts
