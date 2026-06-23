"""End-of-day visibility for shadow and live-forward validation artifacts."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import date, datetime
from pathlib import Path

from bts.daily_decision import is_scoreable_commit
from bts.health.alert import Alert
from bts.picks import load_pick

log = logging.getLogger(__name__)

SOURCE = "analytics_artifacts_missing"

DEFAULT_CAPTURE_ARTIFACT_ROOT = Path(
    "data/validation/decision_weighted_lgbm_v0_live_forward"
)
CAPTURE_OK_STATUSES = {
    "existing_verified",
    "exported_verified",
    "recaptured_due_to_snapshot_drift",
}
BENIGN_SHADOW_MISSING_REASONS = {
    "select_pick_returned_none",
}
INLINE_SHADOW_FATAL_REASONS = {
    "prior_dispatched_without_artifact",
}


def _state_path(picks_dir: Path, date_iso: str) -> Path:
    return picks_dir / date_iso / "scheduler_state.json"


def _load_scheduler_state(picks_dir: Path, date_iso: str) -> dict:
    path = _state_path(picks_dir, date_iso)
    if not path.exists():
        return {}
    try:
        body = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read scheduler state at %s: %s", path, exc)
        return {}
    return body if isinstance(body, dict) else {}


def _locked_pick_exists(picks_dir: Path, date_iso: str, state: dict) -> bool:
    """Return True only for a GENUINE scoreable commit.

    Replaces the old ``pick_locked or pick_was_delivered`` heuristic with
    ``is_scoreable_commit`` so a classification-lock on a skip day (pick_locked=True
    but decision.json has action=="skip" or scoreable==False) does NOT trigger
    shadow/capture artifact alerts (D6 / GH #144).
    """
    try:
        daily = load_pick(date_iso, picks_dir)
    except Exception as exc:
        log.warning("could not read pick for analytics artifact check: %s", exc)
        return False
    return is_scoreable_commit(date_iso, picks_dir, daily)


def _repo_root_from_picks_dir(picks_dir: Path) -> Path:
    if picks_dir.name == "picks" and picks_dir.parent.name == "data":
        return picks_dir.parent.parent
    return picks_dir.parent


def _resolve_capture_root(picks_dir: Path, capture_artifact_root: Path | None) -> Path:
    root = capture_artifact_root or DEFAULT_CAPTURE_ARTIFACT_ROOT
    if root.is_absolute():
        return root
    return _repo_root_from_picks_dir(picks_dir) / root


def read_systemd_unit_summary(unit: str | None) -> str | None:
    """Return a short systemd unit summary, or None when unavailable."""
    if not unit:
        return None
    try:
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "-p",
                "Result",
                "-p",
                "ExecMainStatus",
                "-p",
                "ExecMainCode",
                "-p",
                "MemoryPeak",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        log.warning("could not query systemd unit %s: %s", unit, exc)
        return None
    if result.returncode != 0:
        return None
    fields = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value and value != "[not set]":
            fields[key] = value
    if not fields:
        return None
    parts = []
    for key in ("Result", "ExecMainStatus", "ExecMainCode", "MemoryPeak"):
        if key in fields:
            parts.append(f"{key}={fields[key]}")
    return ", ".join(parts) if parts else None


def _journal_since_arg(since: str | None) -> str | None:
    if not since:
        return None
    text = str(since).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text.replace("T", " ")
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _fatal_scheduler_journal_line(unit: str | None, since: str | None) -> str | None:
    """Return scheduler journal evidence for an inline shadow process death."""
    since_arg = _journal_since_arg(since)
    if not unit or not since_arg:
        return None
    try:
        result = subprocess.run(
            [
                "journalctl",
                "--user",
                "-u",
                unit,
                "--since",
                since_arg,
                "--no-pager",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        log.warning("could not query scheduler journal for %s: %s", unit, exc)
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        lower = line.lower()
        if "oom" in lower or "out of memory" in lower:
            return line.strip()[:240]
        if "code=killed" in lower and (
            "signal=kill" in lower
            or "status=9/kill" in lower
            or "status=kill" in lower
        ):
            return line.strip()[:240]
        if "status=137" in lower:
            return line.strip()[:240]
    return None


def _job_status(state: dict, job: str) -> dict:
    jobs = state.get("analytics_jobs")
    if not isinstance(jobs, dict):
        return {}
    status = jobs.get(job)
    return status if isinstance(status, dict) else {}


def _status_detail(state: dict, job: str) -> str:
    status = _job_status(state, job)
    if not status:
        return "status=unrecorded"
    parts = []
    for key in ("status", "reason", "unit", "updated_at"):
        value = status.get(key)
        if value:
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "status=unrecorded"


def _check_shadow(
    picks_dir: Path,
    date_iso: str,
    state: dict,
    shadow_unit: str | None,
    scheduler_unit: str | None,
) -> list[Alert]:
    shadow_path = picks_dir / f"{date_iso}.shadow.json"
    if shadow_path.exists():
        try:
            json.loads(shadow_path.read_text())
            return []
        except (json.JSONDecodeError, OSError) as exc:
            return [Alert(
                level="WARN",
                source=SOURCE,
                message=f"shadow artifact malformed for {date_iso}: {exc}",
            )]

    status = _job_status(state, "shadow")
    reason = status.get("reason")
    if status.get("status") == "failed" and reason in BENIGN_SHADOW_MISSING_REASONS:
        return [Alert(
            level="INFO",
            source=SOURCE,
            message=(
                f"shadow artifact absent for {date_iso}; shadow model abstained "
                f"({_status_detail(state, 'shadow')})."
            ),
        )]

    detail = _status_detail(state, "shadow")
    unit_summary = read_systemd_unit_summary(shadow_unit)
    if unit_summary:
        detail = f"{detail}; {unit_summary}"
    inline_shadow_died = (
        shadow_unit is None
        and (
            status.get("status") == "dispatched"
            or reason in INLINE_SHADOW_FATAL_REASONS
        )
    )
    if inline_shadow_died:
        fatal_line = _fatal_scheduler_journal_line(
            scheduler_unit,
            str(status.get("dispatched_at") or status.get("updated_at") or ""),
        )
        if fatal_line:
            return [Alert(
                level="CRITICAL",
                source=SOURCE,
                message=(
                    f"shadow artifact missing for {date_iso} after locked pick "
                    f"({detail}); scheduler death evidence: {fatal_line}"
                ),
            )]
    return [Alert(
        level="WARN",
        source=SOURCE,
        message=(
            f"shadow artifact missing for {date_iso} after locked pick "
            f"({detail})."
        ),
    )]


def _read_capture_status(status_path: Path) -> dict | None:
    if not status_path.exists():
        return None
    try:
        body = json.loads(status_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"status": "malformed_capture_status", "message": str(exc)}
    return body if isinstance(body, dict) else {"status": "malformed_capture_status"}


def _check_capture(
    picks_dir: Path,
    date_iso: str,
    state: dict,
    capture_artifact_root: Path | None,
    capture_unit: str | None,
) -> list[Alert]:
    artifact_root = _resolve_capture_root(picks_dir, capture_artifact_root)
    artifact_dir = artifact_root / date_iso
    status_path = artifact_dir / "capture_status.json"
    status = _read_capture_status(status_path)
    unit_summary = read_systemd_unit_summary(capture_unit)
    detail = _status_detail(state, "live_forward_capture")
    if unit_summary:
        detail = f"{detail}; {unit_summary}"

    if status is None:
        return [Alert(
            level="CRITICAL",
            source=SOURCE,
            message=(
                f"live-forward capture status missing for {date_iso} at "
                f"{status_path} ({detail})."
            ),
        )]

    capture_status = str(status.get("status", "unknown"))
    stale = status.get("stale_pick_snapshot") is True
    if capture_status in CAPTURE_OK_STATUSES and not stale:
        return []

    reason = status.get("message") or "no message"
    stale_text = " stale_pick_snapshot=true;" if stale else ""
    return [Alert(
        level="CRITICAL",
        source=SOURCE,
        message=(
            f"live-forward capture unhealthy for {date_iso}: "
            f"status={capture_status};{stale_text} {reason} ({detail})."
        ),
    )]


def check(
    picks_dir: Path,
    today: date | None = None,
    *,
    shadow_expected: bool = False,
    capture_expected: bool = False,
    capture_artifact_root: Path | None = None,
    capture_unit: str | None = "bts-live-forward-capture.service",
    shadow_unit: str | None = None,
    scheduler_unit: str | None = "bts-scheduler.service",
) -> list[Alert]:
    """Return alerts when expected analytics artifacts are missing/unhealthy."""
    if today is None:
        today = date.today()
    date_iso = today.isoformat()
    state = _load_scheduler_state(picks_dir, date_iso)
    if not _locked_pick_exists(picks_dir, date_iso, state):
        return []

    alerts: list[Alert] = []
    if shadow_expected:
        alerts.extend(_check_shadow(
            picks_dir,
            date_iso,
            state,
            shadow_unit,
            scheduler_unit,
        ))
    if capture_expected:
        alerts.extend(_check_capture(
            picks_dir,
            date_iso,
            state,
            capture_artifact_root,
            capture_unit,
        ))
    return alerts
