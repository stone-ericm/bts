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
import os
import stat
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from bts.dm import send_dm

log = logging.getLogger(__name__)
HEALTH_DM_STATUS_SCHEMA_VERSION = "bts_health_dm_delivery_status_v1"


@dataclass(frozen=True)
class Alert:
    level: str  # "INFO" | "WARN" | "CRITICAL"
    source: str  # name of the check, e.g. "calibration_drift", "blend_training"
    message: str
    # Dedup identity when `source` is too coarse (round-2 review #6): two
    # DIFFERENT checks crashing both carry source="health_runner" — without a
    # distinct key the second crash would be suppressed as "already seen
    # today". None → the source is the identity (the common case).
    incident_key: str | None = None


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
    sent_day: str | None = None,
    sent_sources: list[str] | None = None,
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
    if sent_day is not None:
        payload["sent_day"] = sent_day
    if sent_sources is not None:
        payload["sent_sources"] = sent_sources

    try:
        path = Path(status_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_regular_file_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        log.warning(
            "could not write health DM delivery status at %s: %s",
            status_path,
            exc,
        )


def _write_regular_file_text(path: Path, text: str) -> None:
    """``path.write_text(text)`` that cannot block on, or write into, a special file.

    The open is non-blocking and the type is checked on the OPENED descriptor, before
    anything is truncated, so a FIFO put at the path after any earlier check (cs6b2
    review, finding 2) raises ``OSError`` here instead of waiting for a reader after
    the DM has already gone out. Creation mode, symlink following and the text
    encoding are those of ``write_text``.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NONBLOCK, 0o666)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("not a regular file")
        os.ftruncate(fd, 0)
        with open(fd, "w", closefd=False) as handle:
            handle.write(text)
    finally:
        os.close(fd)


def _read_regular_file_text(path: Path) -> str:
    """``path.read_text()`` that cannot block on a special file (review finding 1)."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("not a regular file")
        with open(fd, "r", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(fd)


def _persist_health_dm_delivery_status(status_path: Path | str | None, **payload) -> None:
    # Keep special files out of the writer as well as the reader. In particular,
    # opening a FIFO for writing waits for a reader, even after a successful DM.
    if status_path is not None:
        try:
            mode = Path(status_path).stat().st_mode
        except (OSError, ValueError):
            # The existing writer handles absent paths and reports write errors.
            pass
        else:
            if not stat.S_ISREG(mode):
                log.warning(
                    "could not write health DM delivery status at %s: not a regular file",
                    status_path,
                )
                return
    _write_health_dm_delivery_status(status_path, **payload)


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


def _read_health_dm_delivery_status(status_path: Path | str | None) -> dict | None:
    if status_path is None:
        return None
    try:
        path = Path(status_path)
        if not stat.S_ISREG(path.stat().st_mode):
            return None
        data = json.loads(_read_regular_file_text(path))
    except (OSError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _resurface_note(data: dict | None) -> str | None:
    """H6: if the PREVIOUS health DM failed to send, return a note to prepend so
    the missed alerts resurface on the next DM. Day-keyed one-shot CRITICALs
    (e.g. pick_delivery for day D) are otherwise lost on a transient Bluesky
    outage during that EOD run."""
    if data is None:
        return None
    n = data.get("critical_count")
    if data.get("status") == "failed" and type(n) is int and n > 0:
        return (f"⚠️ Previous health alert DM FAILED to send "
                f"({n} CRITICAL) — it may have been missed.")
    return None


def _unknown_alert_error_text(error: Exception) -> str:
    try:
        return str(error)
    except Exception:
        return type(error).__name__


def report_unknown_outcome_alert_failure(detail: str, error: Exception | None = None) -> None:
    try:
        if error is not None:
            detail = f"{detail}: {_unknown_alert_error_text(error)}"
        print(f"Unknown-outcome alert failed: {ascii(detail)[1:-1][:200]}", file=sys.stderr)
    except Exception:
        pass


def _check_receipt_file(info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise OSError("receipt must be an unaliased regular file")


def _check_receipt_entry(directory_fd: int, name: str) -> None:
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    _check_receipt_file(info)


def _unique_receipt_members(pairs: list[tuple[str, object]]) -> dict:
    members = {}
    for key, value in pairs:
        if key in members:
            raise ValueError(f"Duplicate receipt member: {key}")
        members[key] = value
    return members


def _read_receipt(directory_fd: int, name: str) -> dict | None:
    _check_receipt_entry(directory_fd, name)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    try:
        _check_receipt_file(os.fstat(fd))
        with open(fd, "r", encoding="utf-8", closefd=False) as handle:
            data = json.load(handle, object_pairs_hook=_unique_receipt_members)
    except (ValueError, RecursionError):
        return None
    finally:
        os.close(fd)
    return data if isinstance(data, dict) else None


def _replace_receipt(directory_fd: int, name: str, payload: dict) -> None:
    temporary = f".{name}.{uuid4().hex}"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory_fd)
    try:
        with open(fd, "w", encoding="utf-8", closefd=False) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(fd)
        _check_receipt_entry(directory_fd, name)
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    finally:
        os.close(fd)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except OSError:
            pass


def _usable_message_id(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _dispatch_with_acknowledgement(
    body: str,
    dm_recipient: str | None,
    status_path: Path | str | None,
    day: str,
    alert_key: list[str],
) -> bool:
    """Use a dedicated receipt and keep the DM eligible if storage is unavailable."""
    body_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
    directory_fd = None
    storage_error = None
    send_error = None
    attempted = False
    try:
        prior = None
        if status_path is not None:
            stage = "open receipt directory"
            try:
                path = Path(status_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                stage = "read receipt"
                prior = _read_receipt(directory_fd, path.name)
            except Exception as error:
                storage_error = f"{stage} at {status_path}: {_unknown_alert_error_text(error)}"
        if (
            prior is not None
            and prior.keys() == {"schema_version", "status", "sent_day", "sent_sources", "message_id", "body_sha256"}
            and prior.get("schema_version") == HEALTH_DM_STATUS_SCHEMA_VERSION
            and prior.get("status") == "sent"
            and prior.get("sent_day") == day
            and prior.get("sent_sources") == alert_key
            and _usable_message_id(prior.get("message_id"))
            and prior.get("body_sha256") == body_sha256
        ):
            return False

        payload = {"schema_version": HEALTH_DM_STATUS_SCHEMA_VERSION}
        if not dm_recipient:
            send_error = "dm_recipient is not configured"
            payload.update(status="skipped_no_recipient", error=send_error)
        else:
            attempted = True
            try:
                message_id = send_dm(dm_recipient, body)
                if not _usable_message_id(message_id):
                    raise ValueError("send_dm returned no usable message id")
            except Exception as error:
                detail = _unknown_alert_error_text(error)
                send_error = f"transport: {detail}"
                payload.update(status="failed", error=detail)
            else:
                payload.update(
                    status="sent", message_id=message_id, sent_day=day,
                    sent_sources=alert_key, body_sha256=body_sha256,
                )

        if directory_fd is not None and storage_error is None:
            try:
                _replace_receipt(directory_fd, path.name, payload)
            except Exception as error:
                storage_error = f"write receipt at {status_path}: {_unknown_alert_error_text(error)}"
    finally:
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError as error:
                if storage_error is None:
                    storage_error = f"close receipt directory at {status_path}: {_unknown_alert_error_text(error)}"
        failures = [detail for detail in (storage_error, send_error) if detail is not None]
        if failures:
            report_unknown_outcome_alert_failure(". ".join(failures))
    return attempted


def dispatch_dm_for_health_alerts(
    alerts: list[Alert],
    dm_recipient: str | None,
    warn_attention: list[Alert] | None = None,
    status_path: Path | str | None = None,
    now_et_date: date | None = None,
    *,
    require_message_id: bool = False,
) -> bool:
    """Send one Bluesky DM for CRITICALs and selected WARN attention.

    Returns True if a DM was attempted, False otherwise. Send failures are
    logged at ERROR/exception level and suppressed so health reporting cannot
    break the scheduler lifecycle.

    Per-day source-set dedup (2026-07-12 incident): this dispatcher was
    designed to run once per day, but a Restart=always thrash re-walked EOD
    every ~48s and re-sent the same CRITICAL 47 times. If today (ET) already
    sent a DM covering every (level, source) in this call, suppress the
    resend. Keyed on the source SET, not the body hash — a growing metric
    (restart_spike's delta) changes the body every run while remaining the
    same problem. A new source same-day still sends (escalation visible);
    a prior FAILED send is never suppressed (H6 resurface flow).

    With require_message_id, status_path is a dedicated receipt and only a
    nonblank transport ID acknowledges delivery. The message body stays fixed
    across retries.
    """
    warn_attention = warn_attention or []
    critical = [a for a in alerts if a.level == "CRITICAL"]
    body = format_health_dm_body(critical, warn_attention)
    prior = None
    if not require_message_id:
        prior = _read_health_dm_delivery_status(status_path)
        resurface = _resurface_note(prior)  # H6: resurface a prior failed DM
        if resurface:
            body = resurface + ("\n\n" + body if body else "")
    if body is None:
        return False

    now_day = (now_et_date or datetime.now(ZoneInfo("America/New_York")).date()).isoformat()
    alert_key = sorted(
        {f"CRITICAL:{a.incident_key or a.source}" for a in critical}
        | {f"WARN_ATTENTION:{a.incident_key or a.source}" for a in warn_attention}
    )
    if require_message_id:
        return _dispatch_with_acknowledgement(body, dm_recipient, status_path, now_day, alert_key)

    prior_sent_sources: list[str] = []
    if prior is not None and prior.get("status") == "sent" and prior.get("sent_day") == now_day:
        sources = prior.get("sent_sources")
        if isinstance(sources, list):
            prior_sent_sources = [source for source in sources if isinstance(source, str)]
        if alert_key and set(alert_key) <= set(prior_sent_sources):
            log.info(
                "health DM suppressed: %s already covered by today's DM (day=%s)",
                alert_key, now_day,
            )
            return False

    if not dm_recipient:
        _persist_health_dm_delivery_status(
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
        _persist_health_dm_delivery_status(
            status_path,
            status="sent",
            dm_recipient=dm_recipient,
            critical_count=len(critical),
            warn_attention_count=len(warn_attention),
            body=body,
            sent_day=now_day,
            sent_sources=sorted(set(prior_sent_sources) | set(alert_key)),
        )
        log.info(
            "sent health DM to %s (%d CRITICAL, %d WARN attention)",
            dm_recipient,
            len(critical),
            len(warn_attention),
        )
        return True
    except Exception as e:
        _persist_health_dm_delivery_status(
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
