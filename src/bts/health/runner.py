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
    analytics_artifacts_missing,
    backup_freshness,
    blend_training,
    calibration,
    contest_state,
    disk_fill,
    fallback_defer,
    leaderboard_freshness,
    park_drag_freshness,
    live_forward_resolution,
    memory_growth,
    mdp_policy_alignment,
    pitcher_sparsity,
    pooled_training,
    pick_entry,
    post_failure,
    postponed_pick,
    predicted_vs_realized,
    projected_lineup,
    realized_calibration,
    restart_spike,
    result_resolution,
    same_team_corr,
    scheduler_state_integrity,
    slate_auc,
    streak_validation,
    unit_drift,
)
from bts.health.alert import Alert, dispatch_dm_for_health_alerts, log_alerts
from bts.health.attention import apply_warn_attention_policy

log = logging.getLogger(__name__)


def _safe_run(name: str, fn) -> list[Alert]:
    """Wrap a check call so one check's bug can't break the others.

    A crashing check is itself surfaced as a CRITICAL (a dead smoke detector is
    worse than a noisy one): without this, a check that raises every run produces
    no alert and no DM, only a journalctl line nobody reads (audit H3).
    """
    try:
        return fn()
    except Exception as e:
        log.exception(f"health check '{name}' raised: {e}")
        return [Alert(
            level="CRITICAL",
            source="health_runner",
            message=f"health check '{name}' crashed: {type(e).__name__}: {e}",
            # Distinct dedup identity per crashed check: two different dead
            # detectors the same day must both reach the operator (review #6).
            incident_key=f"health_runner:{name}",
        )]


def _path(value: Path | str) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _repo_root_from_picks_dir(picks_dir: Path) -> Path:
    if picks_dir.name == "picks" and picks_dir.parent.name == "data":
        return picks_dir.parent.parent
    return picks_dir.parent


def _repo_path(picks_dir: Path, value: Path | str) -> Path:
    path = _path(value)
    return path if path.is_absolute() else _repo_root_from_picks_dir(picks_dir) / path


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
    shadow_model_enabled: bool = False,
    live_forward_capture_enabled: bool = False,
    live_forward_capture_artifact_root: Path | None = None,
    live_forward_resolve_status_root: Path | None = None,
    live_forward_capture_unit: str | None = "bts-live-forward-capture.service",
    shadow_unit: str | None = None,
    contest_state_expected: bool = False,
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
    alerts.extend(_safe_run("contest_state", lambda: contest_state.check(
        picks_dir,
        expected=contest_state_expected,
    )))
    alerts.extend(_safe_run("blend_training", lambda: blend_training.check(
        models_dir, today=today,
    )))
    if pooled_dir is not None:
        alerts.extend(_safe_run("pooled_training", lambda: pooled_training.check(
            pooled_dir=pooled_dir, today=today,
        )))
    alerts.extend(_safe_run("pick_entry", lambda: pick_entry.check(
        picks_dir, today=today,
    )))
    alerts.extend(_safe_run("post_failure", lambda: post_failure.check(
        picks_dir, today=today,
    )))
    alerts.extend(_safe_run("fallback_defer", lambda: fallback_defer.check(
        picks_dir, today=today,
    )))
    alerts.extend(_safe_run("postponed_pick", lambda: postponed_pick.check(
        picks_dir, today=today,
    )))
    if current_nrestarts is not None:
        alerts.extend(_safe_run("restart_spike", lambda: restart_spike.check(
            picks_dir, current_nrestarts=current_nrestarts, today=today,
        )))
    # Runs unconditionally (Codex r2 #4): the check self-silences when no
    # files exist, and gating on shadow_model_enabled would stop monitoring
    # already-written unresolved files the moment generation is turned off.
    alerts.extend(_safe_run("result_resolution", lambda: result_resolution.check(
        picks_dir, today=today, thresholds=overrides.get("result_resolution"),
    )))
    if shadow_model_enabled or live_forward_capture_enabled:
        alerts.extend(_safe_run("analytics_artifacts_missing", lambda: (
            analytics_artifacts_missing.check(
                picks_dir,
                today=today,
                shadow_expected=shadow_model_enabled,
                capture_expected=live_forward_capture_enabled,
                capture_artifact_root=live_forward_capture_artifact_root,
                capture_unit=live_forward_capture_unit,
                shadow_unit=shadow_unit,
                scheduler_unit="bts-scheduler.service",
            )
        )))
    if live_forward_capture_enabled:
        capture_root = _repo_path(
            picks_dir,
            live_forward_capture_artifact_root
            or analytics_artifacts_missing.DEFAULT_CAPTURE_ARTIFACT_ROOT,
        )
        resolve_status_root = _repo_path(
            picks_dir,
            live_forward_resolve_status_root
            or Path("data/validation/decision_weighted_lgbm_v0_live_forward_resolved_status"),
        )
        alerts.extend(_safe_run("live_forward_resolution", lambda: (
            live_forward_resolution.check(
                preoutcome_root=capture_root,
                status_root=resolve_status_root,
                today=today,
                thresholds=overrides.get("live_forward_resolution"),
            )
        )))

    # Tier 2 — quality decay
    alerts.extend(_safe_run("predicted_vs_realized", lambda: predicted_vs_realized.check(
        picks_dir, today=today, thresholds=overrides.get("predicted_vs_realized"),
    )))
    # Compute current deploy timestamp once; passed to checks that need
    # since-deploy filtering to avoid pooling iteration-contaminated picks
    # (see project_bts_production_realized_contaminated.md).
    since_deploy_iso = realized_calibration._current_deploy_iso(Path("."))
    alerts.extend(_safe_run("realized_calibration", lambda: realized_calibration.check(
        picks_dir, today=today, thresholds=overrides.get("realized_calibration"),
        data_dir=data_dir, since_deploy_iso=since_deploy_iso,
    )))
    if data_dir is not None:
        alerts.extend(_safe_run("slate_auc", lambda: slate_auc.check(
            picks_dir, data_dir=data_dir, today=today, thresholds=overrides.get("slate_auc"),
        )))
    alerts.extend(_safe_run("mdp_policy_alignment", lambda: mdp_policy_alignment.check(
        picks_dir,
        policy_path=models_dir / "mdp_policy.npz",
        today=today,
        thresholds=overrides.get("mdp_policy_alignment"),
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
    alerts.extend(_safe_run("park_drag_freshness", lambda: park_drag_freshness.check(
        Path("data/external/park_drag"), today=today,
        thresholds=overrides.get("park_drag_freshness"),
    )))

    # Tier 3 — process integrity
    alerts.extend(_safe_run("disk_fill", lambda: disk_fill.check(
        picks_dir, thresholds=overrides.get("disk_fill"),
    )))
    alerts.extend(_safe_run("backup_freshness", lambda: backup_freshness.check(
        picks_dir.parent / "health_state",
        thresholds=overrides.get("backup_freshness"),
    )))
    alerts.extend(_safe_run("unit_drift", lambda: unit_drift.check(
        installed_dir=Path.home() / ".config" / "systemd" / "user",
        repo_units_dir=_repo_path(picks_dir, "scripts/systemd"),
    )))
    alerts.extend(_safe_run("scheduler_state_integrity", lambda: scheduler_state_integrity.check(
        picks_dir, today=today, thresholds=overrides.get("scheduler_state_integrity"),
    )))
    if scheduler_pid is not None:
        # history_path enables daily JSONL append + Tuesday-EOD weekly digest INFO.
        # Defaults to data/health_state/memory_growth_history.jsonl on bts-mlb;
        # callers can override via thresholds_overrides["memory_growth_history"].
        memory_history_path = (overrides.get("memory_growth_history")
                                if "memory_growth_history" in overrides
                                else picks_dir.parent / "health_state" / "memory_growth_history.jsonl")
        memory_history_path = _path(memory_history_path)
        alerts.extend(_safe_run("memory_growth", lambda: memory_growth.check(
            pid=scheduler_pid, thresholds=overrides.get("memory_growth"),
            history_path=memory_history_path, today=today,
        )))
    alerts.extend(_safe_run("streak_validation", lambda: streak_validation.check(picks_dir)))

    if today is None:
        today = date.today()
    warn_attention_path = (overrides.get("warn_attention_state")
                           if "warn_attention_state" in overrides
                           else picks_dir.parent / "health_state" / "warn_attention_state.json")
    health_dm_status_path = (overrides.get("health_dm_delivery_status")
                             if "health_dm_delivery_status" in overrides
                             else picks_dir.parent / "health_state" / "health_dm_delivery_status.json")
    policy_alerts, warn_attention = apply_warn_attention_policy(
        alerts,
        state_path=_path(warn_attention_path),
        today=today,
    )
    alerts.extend(policy_alerts)

    log_alerts(alerts)
    dispatch_dm_for_health_alerts(
        alerts,
        dm_recipient,
        warn_attention=warn_attention,
        status_path=_path(health_dm_status_path),
        # Dedup by the PROCESSED day, not the wall date: a post-midnight EOD
        # for date D must not consume D+1's "already sent today" budget
        # (round-2 review #5).
        now_et_date=today,
    )
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
