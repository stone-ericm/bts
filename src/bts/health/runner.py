"""Health check runner — calls each check and dispatches alerts.

The runner is the single entrypoint called from scheduler.py at end-of-day.
It runs each check independently (a failure in one doesn't prevent others),
collects all alerts, logs them at appropriate levels, and dispatches a single
Bluesky DM summarizing all CRITICAL alerts (only if dm_recipient is set).
"""

from __future__ import annotations

import logging
import os
from datetime import date
from pathlib import Path

from bts.health import (
    blend_training,
    calibration,
    disk_fill,
    leaderboard_freshness,
    memory_growth,
    pitcher_sparsity,
    pooled_training,
    post_failure,
    predicted_vs_realized,
    projected_lineup,
    realized_calibration,
    restart_spike,
    same_team_corr,
    streak_validation,
)
from bts.health.alert import Alert, dispatch_dm_for_critical, log_alerts

log = logging.getLogger(__name__)


def _safe_run(name: str, fn) -> list[Alert]:
    """Wrap a check call so one check's bug can't break the others."""
    try:
        return fn()
    except Exception as e:
        log.exception(f"health check '{name}' raised: {e}")
        return []


def run_all_checks(
    picks_dir: Path,
    models_dir: Path,
    dm_recipient: str | None,
    scheduler_pid: int | None = None,
    current_nrestarts: int | None = None,
    today: date | None = None,
    thresholds_overrides: dict | None = None,
    pooled_dir: Path | None = None,
    data_dir: Path | None = None,
    leaderboard_dir: Path | None = None,
) -> list[Alert]:
    """Run all enabled health checks. Returns aggregated alerts.

    Each check is independent — a per-check failure logs and is skipped.
    Final dispatch (log + DM) is on the aggregated set.

    `scheduler_pid` and `current_nrestarts` are runtime info the caller
    must supply (the scheduler knows its own pid and can read NRestarts via
    systemctl). They're optional — corresponding checks are skipped if absent.
    """
    overrides = thresholds_overrides or {}
    alerts: list[Alert] = []

    # Calibration drift (existing)
    alerts.extend(_safe_run("calibration", lambda: calibration.check(
        picks_dir, today=today, thresholds=overrides.get("calibration"),
    )))

    # Tier 1 — silent failures with damage
    alerts.extend(_safe_run("blend_training", lambda: blend_training.check(
        models_dir, today=today,
    )))
    if pooled_dir is not None:
        alerts.extend(_safe_run("pooled_training", lambda: pooled_training.check(
            pooled_dir=pooled_dir, today=today,
        )))
    alerts.extend(_safe_run("post_failure", lambda: post_failure.check(
        picks_dir, today=today,
    )))
    if current_nrestarts is not None:
        alerts.extend(_safe_run("restart_spike", lambda: restart_spike.check(
            picks_dir, current_nrestarts=current_nrestarts, today=today,
        )))

    # Tier 2 — quality decay
    alerts.extend(_safe_run("predicted_vs_realized", lambda: predicted_vs_realized.check(
        picks_dir, today=today, thresholds=overrides.get("predicted_vs_realized"),
    )))
    alerts.extend(_safe_run("realized_calibration", lambda: realized_calibration.check(
        picks_dir, today=today, thresholds=overrides.get("realized_calibration"),
        data_dir=data_dir,
    )))
    alerts.extend(_safe_run("same_team_corr", lambda: same_team_corr.check(
        picks_dir, today=today, thresholds=overrides.get("same_team_corr"),
    )))
    alerts.extend(_safe_run("pitcher_sparsity", lambda: pitcher_sparsity.check(
        picks_dir, today=today, thresholds=overrides.get("pitcher_sparsity"),
    )))
    alerts.extend(_safe_run("projected_lineup", lambda: projected_lineup.check(
        picks_dir, today=today, thresholds=overrides.get("projected_lineup"),
    )))
    if leaderboard_dir is not None:
        alerts.extend(_safe_run("leaderboard_freshness", lambda: leaderboard_freshness.check(
            leaderboard_dir, thresholds=overrides.get("leaderboard_freshness"),
        )))

    # Tier 3 — process integrity
    alerts.extend(_safe_run("disk_fill", lambda: disk_fill.check(
        picks_dir, thresholds=overrides.get("disk_fill"),
    )))
    if scheduler_pid is not None:
        # history_path enables daily JSONL append + Tuesday-EOD weekly digest INFO.
        # Defaults to data/health_state/memory_growth_history.jsonl on bts-mlb;
        # callers can override via thresholds_overrides["memory_growth_history"].
        memory_history_path = (overrides.get("memory_growth_history")
                                if "memory_growth_history" in overrides
                                else picks_dir.parent / "health_state" / "memory_growth_history.jsonl")
        alerts.extend(_safe_run("memory_growth", lambda: memory_growth.check(
            pid=scheduler_pid, thresholds=overrides.get("memory_growth"),
            history_path=memory_history_path, today=today,
        )))
    alerts.extend(_safe_run("streak_validation", lambda: streak_validation.check(picks_dir)))

    log_alerts(alerts)
    dispatch_dm_for_critical(alerts, dm_recipient)
    return alerts


def read_systemd_nrestarts(unit: str = "bts-scheduler") -> int | None:
    """Read NRestarts from systemctl. Returns None if unavailable.

    Designed to be called from inside the scheduler service itself.
    Uses subprocess.run with low timeout so a stuck systemctl can't
    hang the end-of-day flow.
    """
    import subprocess
    try:
        r = subprocess.run(
            ["systemctl", "--user", "show", unit, "-p", "NRestarts", "--value"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return None
        val = r.stdout.strip()
        return int(val) if val.isdigit() else None
    except Exception:
        return None


def get_self_pid() -> int:
    """Get the current process PID (for memory check)."""
    return os.getpid()
