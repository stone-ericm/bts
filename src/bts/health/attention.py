"""Selected WARN attention policy for daily BTS health alerts."""

from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path

from bts.health.alert import Alert

log = logging.getLogger(__name__)

STATE_SCHEMA_VERSION = "bts_warn_attention_state_v1"

ALWAYS_ATTENTION_WARN_SOURCES = {
    "analytics_artifacts_missing",
    "backup_freshness",            # a dead backup must not rot silently (audit F5)
    "disk_fill",
    "live_forward_resolution",
    "pick_entry",                  # committed pick never confirmed (audit F1)
    "postponed_pick",
    "scheduler_state_integrity",   # quarantined day-state (audit F3)
}

REPEATED_ATTENTION_WARN_SOURCES = {
    "calibration_drift",
    "leaderboard_freshness",
    "mdp_policy_alignment",
    "pooled_training",
    "predicted_vs_realized",
    "projected_lineup",
    # 2026-07-12: the DD-band bucket makes this check the absolute-level
    # monitor for chronic slot miscalibration; a persistent WARN must reach
    # the DM digest, not sit in the journal.
    "realized_calibration",
    "dd_pair_residual_corr",
    "dd_pair_realized_shortfall",
    "same_team_corr",
    "unit_drift",                  # persistent config drift (audit F12)
}

REPEATED_ATTENTION_MIN_STREAK = 2
OOM_EVIDENCE_RE = re.compile(
    r"(?<![a-z0-9])(?:oom(?:[-_\s]?kill(?:ed)?)?|out[-_\s]?of[-_\s]?memory)(?![a-z0-9])",
    re.IGNORECASE,
)


def _read_state(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": STATE_SCHEMA_VERSION, "sources": {}}
    try:
        body = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read WARN attention state at %s: %s", path, exc)
        return {"schema_version": STATE_SCHEMA_VERSION, "sources": {}}
    if not isinstance(body, dict):
        return {"schema_version": STATE_SCHEMA_VERSION, "sources": {}}
    sources = body.get("sources")
    if not isinstance(sources, dict):
        sources = {}
    return {"schema_version": STATE_SCHEMA_VERSION, "sources": sources}


def _write_state(path: Path, state: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        log.warning("could not write WARN attention state at %s: %s", path, exc)


def _source_key(alert: Alert) -> str:
    # Streak identity = incident identity (round-2 review #3 of the DD-band
    # work): realized_calibration's two buckets are distinct incidents — a
    # source-keyed streak would label a DD-band WARN "2nd consecutive day"
    # because the OTHER bucket warned yesterday. Sources that don't set
    # incident_key keep source-keyed streaks (zero state migration; a
    # pre-change source-keyed entry is never read under a bucket key and
    # simply stays inert in the state file).
    return alert.incident_key or alert.source


def _is_consecutive(last_seen: str | None, today: date) -> bool:
    if not last_seen:
        return False
    try:
        return date.fromisoformat(last_seen) == today - timedelta(days=1)
    except ValueError:
        return False


def update_warn_streaks(
    alerts: list[Alert],
    *,
    state_path: Path,
    today: date,
) -> dict[str, dict]:
    """Persist and return WARN streak metadata keyed by alert source."""
    state = _read_state(state_path)
    sources = state["sources"]
    rows: dict[str, dict] = {}

    seen: dict[str, Alert] = {}
    for alert in alerts:
        if alert.level == "WARN":
            seen.setdefault(_source_key(alert), alert)

    for key, alert in seen.items():
        prior = sources.get(key) if isinstance(sources.get(key), dict) else {}
        last_seen = prior.get("last_seen")
        prior_streak = int(prior.get("streak", 0) or 0)
        if last_seen == today.isoformat():
            streak = max(prior_streak, 1)
        elif _is_consecutive(last_seen, today):
            streak = prior_streak + 1
        else:
            streak = 1

        row = {
            "last_seen": today.isoformat(),
            "streak": streak,
            "last_message": alert.message,
        }
        sources[key] = row
        rows[key] = row

    state["schema_version"] = STATE_SCHEMA_VERSION
    state["sources"] = sources
    _write_state(state_path, state)
    return rows


def _streak_label(streak: int) -> str:
    if streak <= 1:
        return "1st observed day"
    if streak == 2:
        return "2nd consecutive day"
    if streak == 3:
        return "3rd consecutive day"
    return f"{streak}th consecutive day"


def _with_streak(alert: Alert, streak: int) -> Alert:
    return Alert(
        level=alert.level,
        source=alert.source,
        message=f"{alert.message} ({_streak_label(streak)})",
        # Preserve the dedup identity through reconstruction (round-2 review
        # #4 follow-up): dropping it degrades distinct incidents sharing a
        # source back to source-level dedup.
        incident_key=alert.incident_key,
    )


def _has_oom_evidence(alerts: list[Alert]) -> Alert | None:
    for alert in alerts:
        text = f"{alert.source} {alert.message}".lower()
        if OOM_EVIDENCE_RE.search(text):
            return alert
    return None


def build_policy_alerts(
    alerts: list[Alert],
    *,
    streaks: dict[str, dict],
) -> tuple[list[Alert], list[Alert]]:
    """Return additional CRITICAL alerts and selected WARN attention alerts."""
    policy_critical: list[Alert] = []
    attention: list[Alert] = []

    oom_alert = _has_oom_evidence(alerts)
    memory_warn = next(
        (a for a in alerts if a.level == "WARN" and a.source == "memory_growth"),
        None,
    )
    critical_oom_already_present = any(
        a.level == "CRITICAL"
        and OOM_EVIDENCE_RE.search(f"{a.source} {a.message}")
        for a in alerts
    )
    if oom_alert is not None and not critical_oom_already_present:
        message = f"OOM evidence detected: {oom_alert.source}: {oom_alert.message}"
        if memory_warn is not None:
            message += f"; memory_growth: {memory_warn.message}"
        policy_critical.append(Alert(
            level="CRITICAL",
            source="analytics_job_oom",
            message=message,
        ))
    elif memory_warn is not None and oom_alert is not None:
        policy_critical.append(Alert(
            level="CRITICAL",
            source="memory_oom_correlation",
            message=(
                "memory_growth WARN paired with same-day OOM evidence: "
                f"{memory_warn.message}; {oom_alert.source}: {oom_alert.message}"
            ),
        ))

    for alert in alerts:
        if alert.level != "WARN":
            continue
        row = streaks.get(_source_key(alert), {})
        streak = int(row.get("streak", 1) or 1)
        if alert.source in ALWAYS_ATTENTION_WARN_SOURCES:
            attention.append(_with_streak(alert, streak))
        elif (
            alert.source in REPEATED_ATTENTION_WARN_SOURCES
            and streak >= REPEATED_ATTENTION_MIN_STREAK
        ):
            attention.append(_with_streak(alert, streak))
        elif alert.source == "memory_growth" and any(
            a.source == "restart_spike" and a.level == "CRITICAL" for a in alerts
        ):
            attention.append(_with_streak(alert, streak))

    return policy_critical, attention


def apply_warn_attention_policy(
    alerts: list[Alert],
    *,
    state_path: Path,
    today: date,
) -> tuple[list[Alert], list[Alert]]:
    streaks = update_warn_streaks(alerts, state_path=state_path, today=today)
    return build_policy_alerts(alerts, streaks=streaks)
