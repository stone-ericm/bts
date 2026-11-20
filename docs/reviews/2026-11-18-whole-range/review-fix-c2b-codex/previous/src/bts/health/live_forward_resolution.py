"""Health visibility for canonical live-forward artifact resolution."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from bts.health.alert import Alert

log = logging.getLogger(__name__)

SOURCE = "live_forward_resolution"

SUCCESS_STATUSES = {
    "existing_verified",
    "existing_verified_with_voids",
    "resolved_verified",
    "resolved_with_voids",
}
STALE_PICK_SNAPSHOT_MARKER = ".stale_pick_snapshot."
DEFAULT_THRESHOLDS = {
    "grace_days": 3,
    "critical_age_days": 7,
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        body = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read live-forward resolution status %s: %s", path, exc)
        return None
    return body if isinstance(body, dict) else None


def _canonical_artifact_dates(preoutcome_root: Path) -> list[date]:
    if not preoutcome_root.exists():
        return []
    dates: list[date] = []
    for path in preoutcome_root.iterdir():
        if not path.is_dir() or not (path / "manifest.json").exists():
            continue
        if STALE_PICK_SNAPSHOT_MARKER in path.name:
            continue
        try:
            dates.append(date.fromisoformat(path.name))
        except ValueError:
            continue
    return sorted(dates)


def _format_stall(
    *,
    artifact_date: date,
    age_days: int,
    status_path: Path,
    status_body: dict[str, Any] | None,
) -> str:
    if status_body is None:
        if status_path.exists():
            return (
                f"{artifact_date.isoformat()} age={age_days}d status=malformed_status_json "
                f"path={status_path}"
            )
        return (
            f"{artifact_date.isoformat()} age={age_days}d status=missing_status_json "
            f"path={status_path}"
        )

    status = str(status_body.get("status") or "unknown")
    generated_at = status_body.get("generated_at") or "unknown_generated_at"
    message = str(status_body.get("message") or "").replace("\n", " ").strip()
    if len(message) > 180:
        message = message[:177] + "..."
    if message:
        message = f" message={message}"
    return (
        f"{artifact_date.isoformat()} age={age_days}d status={status} "
        f"generated_at={generated_at}{message}"
    )


def check(
    *,
    preoutcome_root: Path,
    status_root: Path,
    today: date | None = None,
    thresholds: dict | None = None,
) -> list[Alert]:
    """Alert on past-grace canonical live-forward dates that are unresolved.

    Same-day and recent `pending_outcomes` are normal, so this check only
    reports canonical artifact dates older than the empirical grace window.
    It is observation-only: it never re-runs the resolver.
    """
    if today is None:
        today = date.today()
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    grace_days = int(t["grace_days"])
    critical_age_days = int(t["critical_age_days"])

    stale: list[tuple[int, str]] = []
    for artifact_date in _canonical_artifact_dates(preoutcome_root):
        age_days = (today - artifact_date).days
        if age_days <= grace_days:
            continue

        status_path = status_root / f"{artifact_date.isoformat()}.json"
        status_body = _read_json(status_path) if status_path.exists() else None
        status = status_body.get("status") if status_body is not None else None
        if status in SUCCESS_STATUSES:
            continue

        stale.append((
            age_days,
            _format_stall(
                artifact_date=artifact_date,
                age_days=age_days,
                status_path=status_path,
                status_body=status_body,
            ),
        ))

    if not stale:
        return []

    level = "CRITICAL" if any(age >= critical_age_days for age, _ in stale) else "WARN"
    max_items = 5
    details = "; ".join(item for _, item in stale[:max_items])
    if len(stale) > max_items:
        details += f"; +{len(stale) - max_items} more"
    return [Alert(
        level=level,
        source=SOURCE,
        message=(
            f"canonical live-forward resolution stalled for {len(stale)} date(s); "
            f"grace_days={grace_days}, critical_age_days={critical_age_days}; "
            "realized-data n-growth for calibration/gate checks may stall. "
            f"{details}"
        ),
    )]
