"""Tier 2 health check: park_drag external table freshness (arming item 3).

The park_drag_delta context feature reads data/external/park_drag/. A stale
or broken table degrades silently to NaN/stderr inside the pick path (by
design — it must never break picks), so THIS check is the visible alarm:

- root absent            -> silent (feature not armed on this box; pre-launch)
- producer_status ok=False -> WARN (last refresh run failed; includes error)
- manifest data gap      -> WARN at >= warn_data_days, CRITICAL at
                            >= critical_data_days behind today
- manifest generated_at  -> WARN when older than warn_generated_hours
                            (cron liveness, distinct from data staleness)
- manifest/export absent while producer dir exists -> WARN

Off-season months (Oct-Feb) are silent: the table legitimately stops at the
end of the regular season and the cron may be disabled.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "park_drag_freshness"

DEFAULT_THRESHOLDS = {
    "warn_data_days": 3.0,
    "critical_data_days": 6.0,
    "warn_generated_hours": 30.0,
}
OFFSEASON_MONTHS = {10, 11, 12, 1, 2}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def check(root: Path, *, today: date, thresholds: dict | None = None) -> list[Alert]:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    root = Path(root)
    if not root.exists():
        return []  # not armed on this box — silent by design
    if today.month in OFFSEASON_MONTHS:
        return []

    alerts: list[Alert] = []

    status = _read_json(root / "producer_status.json")
    if status is not None and status.get("ok") is False:
        alerts.append(Alert(
            level="WARN", source=SOURCE,
            message=(f"park_drag refresh last run FAILED: "
                     f"{status.get('error', 'unknown error')}"),
        ))

    manifest = _read_json(root / "park_drag_manifest.json")
    export_exists = (root / "park_drag_export.csv").exists()
    if manifest is None or not export_exists:
        alerts.append(Alert(
            level="WARN", source=SOURCE,
            message=("park_drag root exists but manifest/export missing - "
                     "feature is serving NaN; seed or run bts park-drag-refresh"),
        ))
        return alerts

    max_src = manifest.get("max_source_game_date")
    if max_src:
        try:
            gap_days = (today - date.fromisoformat(str(max_src))).days
        except ValueError:
            gap_days = None
            alerts.append(Alert(
                level="WARN", source=SOURCE,
                message=f"park_drag manifest max_source_game_date unparseable: {max_src!r}",
            ))
        if gap_days is not None and gap_days >= t["critical_data_days"]:
            alerts.append(Alert(
                level="CRITICAL", source=SOURCE,
                message=(f"park_drag table data {gap_days}d behind "
                         f"(source through {max_src}); serving suppresses to NaN — "
                         f"check park_drag.log / producer_status.json on the box"),
            ))
        elif gap_days is not None and gap_days >= t["warn_data_days"]:
            alerts.append(Alert(
                level="WARN", source=SOURCE,
                message=f"park_drag table data lagging {gap_days}d (source through {max_src})",
            ))

    gen = manifest.get("generated_at")
    if gen:
        try:
            age_h = (datetime.now() - datetime.fromisoformat(str(gen))).total_seconds() / 3600
            if age_h >= t["warn_generated_hours"]:
                alerts.append(Alert(
                    level="WARN", source=SOURCE,
                    message=(f"park_drag refresh has not completed in {age_h:.0f}h "
                             f"(manifest generated_at {gen}) - cron liveness"),
                ))
        except ValueError:
            pass
    return alerts
