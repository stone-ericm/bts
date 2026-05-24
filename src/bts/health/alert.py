"""Shared Alert type + DM dispatcher for production health checks.

All health-check modules import Alert from here, so a runner can
collect alerts of uniform type from independent checks.

The DM-on-CRITICAL dispatcher is centralized so it's wrapped in a single
try/except — a notification failure never propagates back to the caller
(the scheduler's pick lifecycle must not be blocked by an alerting bug).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bts.dm import send_dm

log = logging.getLogger(__name__)
HEALTH_DM_STATUS_SCHEMA_VERSION = "bts_health_dm_delivery_status_v1"


@dataclass(frozen=True)
class Alert:
    level: str  # "INFO" | "WARN" | "CRITICAL"
    source: str  # name of the check, e.g. "calibration_drift", "blend_training"
    message: str


def log_alerts(alerts: list[Alert]) -> None:
    """Log all alerts at the appropriate level."""
    for a in alerts:
        if a.level == "CRITICAL":
            log.error(f"[{a.source} {a.level}] {a.message}")
        elif a.level == "WARN":
            log.warning(f"[{a.source} {a.level}] {a.message}")
        else:
            log.info(f"[{a.source} {a.level}] {a.message}")


def _write_health_dm_delivery_status(
    status_path: Path | str | None,
    *,
    status: str,
    dm_recipient: str | None,
    critical_count: int,
    warn_attention_count: int,
    body: str,
    error: str | None = None,
) -> None:
    if status_path is None:
        return

    payload = {
        "schema_version": HEALTH_DM_STATUS_SCHEMA_VERSION,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "recipient_configured": bool(dm_recipient),
        "recipient": dm_recipient,
        "critical_count": critical_count,
        "warn_attention_count": warn_attention_count,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "body_first_line": body.splitlines()[0] if body else "",
    }
    if error is not None:
        payload["error"] = error

    try:
        path = Path(status_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        log.warning(
            "could not write health DM delivery status at %s: %s",
            status_path,
            exc,
        )


def dispatch_dm_for_critical(alerts: list[Alert], dm_recipient: str | None) -> bool:
    """Send a single Bluesky DM summarizing all CRITICAL alerts.

    Returns True if a DM was attempted, False otherwise. Any send_dm
    failure is logged and suppressed — the caller never sees it. No DM
    is sent if there are no CRITICAL alerts or dm_recipient is unset.
    """
    critical = [a for a in alerts if a.level == "CRITICAL"]
    if not critical or not dm_recipient:
        return False
    body_lines = ["BTS health CRITICAL alert(s):"]
    for a in critical:
        body_lines.append(f"- [{a.source}] {a.message}")
    body = "\n".join(body_lines)
    try:
        send_dm(dm_recipient, body)
        log.info(f"sent CRITICAL DM to {dm_recipient} ({len(critical)} alert(s))")
        return True
    except Exception as e:
        log.exception(f"send_dm failed (alerts detected but DM not delivered): {e}")
        return True


def format_health_dm_body(
    critical: list[Alert],
    warn_attention: list[Alert] | None = None,
) -> str | None:
    """Build the single daily health DM body.

    CRITICAL alerts keep their existing headline. Selected WARNs are folded
    into the same message when CRITICALs exist, or sent as a WARN-attention
    digest on days without CRITICALs.
    """
    warn_attention = warn_attention or []
    if not critical and not warn_attention:
        return None

    if critical:
        lines = ["BTS health CRITICAL alert(s):"]
        for a in critical:
            lines.append(f"- [{a.source}] {a.message}")
        if warn_attention:
            lines.append("")
            lines.append("WARN attention:")
            for a in warn_attention:
                lines.append(f"- [{a.source}] {a.message}")
        return "\n".join(lines)

    lines = ["BTS health WARN attention:"]
    for a in warn_attention:
        lines.append(f"- [{a.source}] {a.message}")
    return "\n".join(lines)


def dispatch_dm_for_health_alerts(
    alerts: list[Alert],
    dm_recipient: str | None,
    warn_attention: list[Alert] | None = None,
    status_path: Path | str | None = None,
) -> bool:
    """Send one Bluesky DM for CRITICALs and selected WARN attention.

    Returns True if a DM was attempted, False otherwise. Send failures are
    logged at ERROR/exception level and suppressed so health reporting cannot
    break the scheduler lifecycle.
    """
    warn_attention = warn_attention or []
    critical = [a for a in alerts if a.level == "CRITICAL"]
    body = format_health_dm_body(critical, warn_attention)
    if body is None:
        return False
    if not dm_recipient:
        _write_health_dm_delivery_status(
            status_path,
            status="skipped_no_recipient",
            dm_recipient=dm_recipient,
            critical_count=len(critical),
            warn_attention_count=len(warn_attention),
            body=body,
            error="dm_recipient is not configured",
        )
        log.error("[health_dm_delivery CRITICAL] health alert DM skipped: no recipient")
        return False
    try:
        send_dm(dm_recipient, body)
        _write_health_dm_delivery_status(
            status_path,
            status="sent",
            dm_recipient=dm_recipient,
            critical_count=len(critical),
            warn_attention_count=len(warn_attention),
            body=body,
        )
        log.info(
            "sent health DM to %s (%d CRITICAL, %d WARN attention)",
            dm_recipient,
            len(critical),
            len(warn_attention),
        )
        return True
    except Exception as e:
        _write_health_dm_delivery_status(
            status_path,
            status="failed",
            dm_recipient=dm_recipient,
            critical_count=len(critical),
            warn_attention_count=len(warn_attention),
            body=body,
            error=str(e),
        )
        log.exception(
            "send_dm failed for health alerts; CRITICAL/WARN visibility may be lost: %s",
            e,
        )
        log.error("[health_dm_delivery CRITICAL] health alert DM delivery failed")
        return True
