"""Tier 3 health check: restic backup freshness (audit F5).

Reads data/health_state/backup_status.json written by `bts backup run`
(cron: ops every 3h, archive daily). A backup that silently stops is a
dead smoke detector for the exact scenario it exists to survive, so:

- status file absent      -> silent (backups not armed — local dev)
- file unparseable        -> WARN
- set entry missing       -> WARN (never succeeded on this box)
- last run ok=False       -> WARN (includes restic's error tail)
- last_success_at stale   -> WARN at warn_hours, CRITICAL at critical_hours

One alert per set at the highest applicable severity — staleness measures
real data-loss exposure, so it outranks a fresh-but-failed run.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from bts.health.alert import Alert
from bts.data.backup import STATUS_FILENAME

log = logging.getLogger(__name__)

SOURCE = "backup_freshness"

DEFAULT_THRESHOLDS = {
    "ops_warn_hours": 7.0,        # two missed 3h cycles + margin
    "ops_critical_hours": 26.0,   # >1 day of decisions/saver state exposed
    "archive_warn_hours": 30.0,   # one missed daily run + margin
    "archive_critical_hours": 78.0,
}


def _age_hours(entry: dict, now: datetime) -> float | None:
    stamp = entry.get("last_success_at")
    if not stamp:
        return None
    try:
        then = datetime.fromisoformat(str(stamp))
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (now - then).total_seconds() / 3600


def check(
    health_state_dir: Path,
    *,
    now: datetime | None = None,
    thresholds: dict | None = None,
) -> list[Alert]:
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    now = now or datetime.now(timezone.utc)
    path = Path(health_state_dir) / STATUS_FILENAME
    if not path.exists():
        return []  # not armed on this box — silent by design

    try:
        status = json.loads(path.read_text())
    except (OSError, ValueError):
        return [Alert(
            level="WARN", source=SOURCE,
            message=f"{STATUS_FILENAME} unparseable — backup state unknown",
        )]

    alerts: list[Alert] = []
    for set_name in ("ops", "archive"):
        entry = status.get(set_name)
        if not isinstance(entry, dict):
            alerts.append(Alert(
                level="WARN", source=SOURCE,
                message=(f"backup set '{set_name}' has never succeeded on this "
                         f"box (no status entry) — run: bts backup run --set {set_name}"),
            ))
            continue

        age_h = _age_hours(entry, now)
        warn_h = t[f"{set_name}_warn_hours"]
        crit_h = t[f"{set_name}_critical_hours"]

        if age_h is None:
            alerts.append(Alert(
                level="WARN", source=SOURCE,
                message=(f"backup set '{set_name}' has no recorded success "
                         f"(last run ok={entry.get('ok')}, "
                         f"error: {entry.get('error', 'n/a')})"),
            ))
        elif age_h >= crit_h:
            alerts.append(Alert(
                level="CRITICAL", source=SOURCE,
                message=(f"backup set '{set_name}' last succeeded {age_h:.0f}h ago "
                         f"(>= {crit_h:.0f}h) — operational state unprotected; "
                         f"check ~/logs/backup.log on the box"),
            ))
        elif age_h >= warn_h:
            alerts.append(Alert(
                level="WARN", source=SOURCE,
                message=(f"backup set '{set_name}' last succeeded {age_h:.0f}h ago "
                         f"(>= {warn_h:.0f}h)"),
            ))
        elif entry.get("ok") is False:
            alerts.append(Alert(
                level="WARN", source=SOURCE,
                message=(f"backup set '{set_name}' last run FAILED "
                         f"(fresh success {age_h:.1f}h ago remains): "
                         f"{str(entry.get('error', 'unknown'))[:200]}"),
            ))
    return alerts
