#!/usr/bin/env python3
"""Guarded one-shot live-forward artifact capture.

This runner is intentionally idempotent so it can be called by a frequent
systemd timer. It does not create production picks. It waits for the production
pick JSON to exist, refuses resolved pick files, exports the frozen
candidate-vs-production ranked slates, and immediately verifies the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SCHEMA_VERSION = "live_forward_capture_once_v1"
DEFAULT_CANDIDATE = "decision_weighted_lgbm_v0"
DEFAULT_ARTIFACT_ROOT = Path(
    "data/validation/decision_weighted_lgbm_v0_live_forward"
)
DEFAULT_PRODUCTION_ROOT = Path("/home/bts/projects/bts")
DEFAULT_LIVE_FORWARD_ROOT = Path("/home/bts/projects/bts-live-forward")
DEFAULT_PYTHON = Path("/home/bts/projects/bts/.venv/bin/python")

# D8 (2026-09-14): research-only forecast capture on days with NO production pick
# (MDP/tail skip days). Lives in a SEPARATE artifact root so the official
# decision-weighted live-forward stream (`discover_dates` on DEFAULT_ARTIFACT_ROOT)
# never sees it; the sidecar is the completion contract and is written LAST.
DEFAULT_RESEARCH_ROOT = Path(
    "data/validation/decision_weighted_lgbm_v0_live_forward_research"
)
RESEARCH_SIDECAR_NAME = "research_capture.json"
# Acceptance marker written AFTER the sidecar is in place; its published_at is sampled
# after that rename, so it is the evidence that the content was published pre-first-pitch.
RESEARCH_ACCEPTED_NAME = "research_capture.accepted.json"
RESEARCH_REASON = "no_production_pick_skip_day"
# Contest entry closes 5 min before first pitch (bts.picks.SUBMISSION_CUTOFF_MIN);
# a forecast captured later than that is not a decision-time forecast.
RESEARCH_DEADLINE_BUFFER_MIN = 5


@dataclass(frozen=True)
class CaptureConfig:
    date: str
    production_root: Path
    live_forward_root: Path
    python: Path
    candidate: str
    artifact_root: Path
    data_dir: Path
    picks_dir: Path
    top_n: int
    overwrite: bool
    fail_on_pending: bool
    auto_recapture_on_snapshot_drift: bool
    # D8 research stream (opt-in; behavior is byte-identical when False).
    capture_research_on_skip: bool = False
    research_root: Path = DEFAULT_RESEARCH_ROOT
    # Injectable clock for the pre-first-pitch deadline (tests); None = wall clock.
    now: datetime | None = None


def today_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)


def config_now(config: CaptureConfig) -> datetime:
    return config.now if config.now is not None else utc_now_dt()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolve_under(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def git_head(path: Path) -> str:
    result = run(["git", "rev-parse", "HEAD"], cwd=path)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def pa_rows_for_date(data_dir: Path, date: str) -> tuple[Path, int]:
    import pandas as pd

    year = pd.Timestamp(date).year
    path = data_dir / f"pa_{year}.parquet"
    if not path.exists():
        return path, 0
    frame = pd.read_parquet(path, columns=["date"])
    date_keys = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return path, int(date_keys.eq(date).sum())


def bts_command_env(config: CaptureConfig) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(config.live_forward_root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("BTS_LGBM_DETERMINISTIC", "1")
    env.setdefault("BTS_LGBM_RANDOM_STATE", "42")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("NUMEXPR_NUM_THREADS", "2")
    return env


def run_bts(
    config: CaptureConfig,
    cli_args: list[str],
) -> subprocess.CompletedProcess[str]:
    return run(
        [str(config.python), "-c", "from bts.cli import cli; cli()", *cli_args],
        cwd=config.live_forward_root,
        env=bts_command_env(config),
    )


def pick_is_unresolved(pick: dict[str, Any]) -> bool:
    result = pick.get("result")
    return result is None or result == "" or result == "pending"


def production_pick_snapshot_sha256(manifest: dict[str, Any]) -> str | None:
    snapshot = manifest.get("production_pick_snapshot")
    if not isinstance(snapshot, dict):
        return None
    value = snapshot.get("source_sha256")
    return value if isinstance(value, str) and value else None


def production_pick_snapshot_json(manifest: dict[str, Any]) -> dict[str, Any] | None:
    snapshot = manifest.get("production_pick_snapshot")
    if not isinstance(snapshot, dict):
        return None
    value = snapshot.get("production_pick_json")
    return value if isinstance(value, dict) else None


BENIGN_NULL_DECISION_FIELDS = {
    "feature_env_schema_version",
    "feature_env",
    "feature_env_hash",
}


def decision_snapshot(pick: dict[str, Any]) -> dict[str, Any]:
    """Return the pick JSON fields that define the locked pre-outcome choice."""
    # Result fields are appended after games finish; they must not make an
    # otherwise matching at-lock decision snapshot look stale. Nullable
    # provenance fields can also be backfilled by newer schema code after an
    # older pick resolves; absent and null are equivalent for drift detection.
    snapshot = {
        key: value
        for key, value in pick.items()
        if key not in {"result", "slot_results"}
    }
    for key in BENIGN_NULL_DECISION_FIELDS:
        if snapshot.get(key) is None:
            snapshot.pop(key, None)
    return snapshot


def existing_snapshot_state(
    *,
    manifest_path: Path,
    pick_path: Path,
    expected_date: str,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "production_pick_snapshot_checked": False,
        "current_pick_sha256": None,
        "artifact_pick_snapshot_sha256": None,
        "snapshot_matches_current_pick": None,
        "snapshot_decision_matches_current_pick": None,
        "stale_pick_snapshot": None,
        "current_pick_result": None,
        "current_pick_date": None,
        "current_pick_date_matches": None,
        "snapshot_check_error": None,
    }
    if not pick_path.exists():
        return state

    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        state["snapshot_check_error"] = f"could not read existing manifest: {exc}"
        return state

    try:
        pick = read_json(pick_path)
    except (OSError, json.JSONDecodeError) as exc:
        state["snapshot_check_error"] = f"could not read current production pick: {exc}"
        return state

    state["production_pick_snapshot_checked"] = True
    state["current_pick_result"] = pick.get("result")
    state["current_pick_date"] = pick.get("date")
    state["current_pick_date_matches"] = str(pick.get("date")) == str(expected_date)
    if not state["current_pick_date_matches"]:
        state["snapshot_check_error"] = (
            f"current production pick date {pick.get('date')!r} does not match "
            f"{expected_date!r}"
        )
        return state

    current_sha = file_sha256(pick_path)
    snapshot_sha = production_pick_snapshot_sha256(manifest)
    snapshot_json = production_pick_snapshot_json(manifest)
    decision_matches = None
    if snapshot_json is not None:
        decision_matches = decision_snapshot(snapshot_json) == decision_snapshot(pick)
    state["current_pick_sha256"] = current_sha
    state["artifact_pick_snapshot_sha256"] = snapshot_sha
    state["snapshot_matches_current_pick"] = snapshot_sha == current_sha
    state["snapshot_decision_matches_current_pick"] = decision_matches
    state["stale_pick_snapshot"] = (
        not decision_matches if decision_matches is not None else snapshot_sha != current_sha
    )
    return state


def stale_backup_dir(artifact_dir: Path, snapshot_sha: str | None) -> Path:
    suffix = (snapshot_sha or "missing")[:12]
    base = artifact_dir.with_name(f"{artifact_dir.name}.stale_pick_snapshot.{suffix}")
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = artifact_dir.with_name(f"{base.name}.{counter}")
        counter += 1
    return candidate


def status_payload(
    *,
    config: CaptureConfig,
    status: str,
    message: str,
    production_head: str | None = None,
    live_forward_head: str | None = None,
    artifact_dir: Path | None = None,
    verification_path: Path | None = None,
    pick_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "message": message,
        "date": config.date,
        "candidate": config.candidate,
        "production_root": str(config.production_root),
        "live_forward_root": str(config.live_forward_root),
        "production_head": production_head,
        "live_forward_head": live_forward_head,
        "pick_path": str(pick_path) if pick_path is not None else None,
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
        "verification_path": (
            str(verification_path) if verification_path is not None else None
        ),
    }
    if extra:
        payload.update(extra)
    return payload


def verify_artifact(
    config: CaptureConfig,
    *,
    artifact_dir: Path,
    verification_path: Path,
    live_forward_head: str,
    require_pick_snapshot: bool = True,
) -> subprocess.CompletedProcess[str]:
    args = [
        "experiment",
        "verify-candidate-artifacts",
        "--artifact-dir",
        str(artifact_dir),
        "--expected-run-kind",
        "live_forward_preoutcome",
        "--expected-candidate",
        config.candidate,
        "--expected-date",
        config.date,
        "--expected-git-commit",
        live_forward_head,
        "--expected-top-n",
        str(config.top_n),
        "--require-live-preoutcome",
    ]
    if require_pick_snapshot:
        # Official stream: parity with the production pick is mandatory.
        args.append("--require-production-pick-snapshot")
    args += ["--save", str(verification_path)]
    return run_bts(config, args)


def export_artifact(
    config: CaptureConfig,
    *,
    pick_path: Path | None,
    artifact_dir: Path,
) -> subprocess.CompletedProcess[str]:
    args = [
        "experiment",
        "export-live-candidate-artifacts",
        "--date",
        config.date,
        "--candidate",
        config.candidate,
        "--output-dir",
        str(artifact_dir),
        "--data-dir",
        str(resolve_under(config.production_root, config.data_dir)),
        "--top-n",
        str(config.top_n),
        "--no-refresh-data",
    ]
    if pick_path is not None:
        args += ["--production-pick-file", str(pick_path)]
    return run_bts(config, args)


def refresh_stale_artifact(
    config: CaptureConfig,
    *,
    pick_path: Path,
    artifact_dir: Path,
    verification_path: Path,
    status_path: Path,
    production_head: str,
    live_forward_head: str,
    snapshot_state: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    refresh_dir = artifact_dir.with_name(f".{artifact_dir.name}.refreshing")
    refresh_verification_path = refresh_dir / "verification.json"
    if refresh_dir.exists():
        shutil.rmtree(refresh_dir)

    export = export_artifact(
        config,
        pick_path=pick_path,
        artifact_dir=refresh_dir,
    )
    if export.returncode != 0:
        payload = status_payload(
            config=config,
            status="failed_recapture_export",
            message=(export.stdout + export.stderr).strip(),
            production_head=production_head,
            live_forward_head=live_forward_head,
            artifact_dir=artifact_dir,
            verification_path=verification_path,
            pick_path=pick_path,
            extra=snapshot_state,
        )
        write_json(status_path, payload)
        if refresh_dir.exists():
            shutil.rmtree(refresh_dir)
        return 1, payload

    verify = verify_artifact(
        config,
        artifact_dir=refresh_dir,
        verification_path=refresh_verification_path,
        live_forward_head=live_forward_head,
    )
    if verify.returncode != 0:
        payload = status_payload(
            config=config,
            status="failed_recapture_verify",
            message=(export.stdout + export.stderr + verify.stdout + verify.stderr).strip(),
            production_head=production_head,
            live_forward_head=live_forward_head,
            artifact_dir=artifact_dir,
            verification_path=verification_path,
            pick_path=pick_path,
            extra=snapshot_state,
        )
        write_json(status_path, payload)
        if refresh_dir.exists():
            shutil.rmtree(refresh_dir)
        return 1, payload

    backup_dir = stale_backup_dir(
        artifact_dir,
        snapshot_state.get("artifact_pick_snapshot_sha256"),
    )
    try:
        shutil.move(str(artifact_dir), backup_dir)
        shutil.move(str(refresh_dir), artifact_dir)
    except OSError as exc:
        rollback_error = None
        if not artifact_dir.exists() and backup_dir.exists():
            try:
                shutil.move(str(backup_dir), artifact_dir)
            except OSError as rollback_exc:
                rollback_error = str(rollback_exc)
        payload = status_payload(
            config=config,
            status="failed_recapture_swap",
            message=f"could not swap refreshed artifact into place: {exc}",
            production_head=production_head,
            live_forward_head=live_forward_head,
            artifact_dir=artifact_dir,
            verification_path=verification_path,
            pick_path=pick_path,
            extra={
                **snapshot_state,
                "stale_artifact_backup_dir": str(backup_dir),
                "recapture_swap_rollback_error": rollback_error,
            },
        )
        write_json(status_path, payload)
        return 1, payload

    refreshed_snapshot_state = existing_snapshot_state(
        manifest_path=artifact_dir / "manifest.json",
        pick_path=pick_path,
        expected_date=config.date,
    )
    payload = status_payload(
        config=config,
        status="recaptured_due_to_snapshot_drift",
        message=(export.stdout + export.stderr + verify.stdout + verify.stderr).strip(),
        production_head=production_head,
        live_forward_head=live_forward_head,
        artifact_dir=artifact_dir,
        verification_path=verification_path,
        pick_path=pick_path,
        extra={
            **refreshed_snapshot_state,
            "previous_artifact_pick_snapshot_sha256": snapshot_state.get(
                "artifact_pick_snapshot_sha256"
            ),
            "previous_snapshot_matches_current_pick": snapshot_state.get(
                "snapshot_matches_current_pick"
            ),
            "previous_stale_pick_snapshot": snapshot_state.get("stale_pick_snapshot"),
            "pa_outcome_check_path": snapshot_state.get("pa_outcome_check_path"),
            "pa_outcome_rows_for_date": snapshot_state.get("pa_outcome_rows_for_date"),
            "stale_artifact_backup_dir": str(backup_dir),
        },
    )
    write_json(status_path, payload)
    return 0, payload


def capture_once(config: CaptureConfig) -> tuple[int, dict[str, Any]]:
    pick_path = resolve_under(config.production_root, config.picks_dir) / (
        f"{config.date}.json"
    )
    artifact_dir = resolve_under(config.production_root, config.artifact_root) / config.date
    verification_path = artifact_dir / "verification.json"
    status_path = artifact_dir / "capture_status.json"

    production_head = git_head(config.production_root)
    live_forward_head = git_head(config.live_forward_root)

    manifest_path = artifact_dir / "manifest.json"
    if manifest_path.exists() and not config.overwrite:
        snapshot_state = existing_snapshot_state(
            manifest_path=manifest_path,
            pick_path=pick_path,
            expected_date=config.date,
        )
        if snapshot_state.get("current_pick_date_matches") is False:
            payload = status_payload(
                config=config,
                status="failed_pick_date_mismatch",
                message=str(snapshot_state.get("snapshot_check_error")),
                production_head=production_head,
                live_forward_head=live_forward_head,
                artifact_dir=artifact_dir,
                verification_path=verification_path,
                pick_path=pick_path,
                extra=snapshot_state,
            )
            write_json(status_path, payload)
            return 1, payload
        if snapshot_state.get("stale_pick_snapshot") is True:
            if not config.auto_recapture_on_snapshot_drift:
                payload = status_payload(
                    config=config,
                    status="stale_pick_snapshot",
                    message=(
                        "existing artifact production_pick_snapshot does not match "
                        "the current production pick; recapture is not authorized"
                    ),
                    production_head=production_head,
                    live_forward_head=live_forward_head,
                    artifact_dir=artifact_dir,
                    verification_path=verification_path,
                    pick_path=pick_path,
                    extra=snapshot_state,
                )
                write_json(status_path, payload)
                return 1, payload
            try:
                pick = read_json(pick_path)
            except (OSError, json.JSONDecodeError) as exc:
                snapshot_state["snapshot_check_error"] = (
                    f"could not reread current production pick: {exc}"
                )
                payload = status_payload(
                    config=config,
                    status="failed_recapture_pick_read",
                    message=f"could not reread current production pick: {exc}",
                    production_head=production_head,
                    live_forward_head=live_forward_head,
                    artifact_dir=artifact_dir,
                    verification_path=verification_path,
                    pick_path=pick_path,
                    extra=snapshot_state,
                )
                write_json(status_path, payload)
                return 1, payload
            if not pick_is_unresolved(pick):
                payload = status_payload(
                    config=config,
                    status="failed_recapture_post_resolution",
                    message=(
                        "existing artifact production_pick_snapshot does not match "
                        "the current production pick, but the pick already has "
                        f"result={pick.get('result')!r}; refusing after-the-fact refresh"
                    ),
                    production_head=production_head,
                    live_forward_head=live_forward_head,
                    artifact_dir=artifact_dir,
                    verification_path=verification_path,
                    pick_path=pick_path,
                    extra=snapshot_state,
                )
                write_json(status_path, payload)
                return 1, payload

            try:
                pa_path, n_pa_rows = pa_rows_for_date(
                    resolve_under(config.production_root, config.data_dir),
                    config.date,
                )
            except Exception as exc:
                payload = status_payload(
                    config=config,
                    status="failed_recapture_outcome_check",
                    message=f"could not check processed PA outcomes before recapture: {exc}",
                    production_head=production_head,
                    live_forward_head=live_forward_head,
                    artifact_dir=artifact_dir,
                    verification_path=verification_path,
                    pick_path=pick_path,
                    extra=snapshot_state,
                )
                write_json(status_path, payload)
                return 1, payload
            if n_pa_rows > 0:
                payload = status_payload(
                    config=config,
                    status="failed_recapture_post_outcomes",
                    message=(
                        "existing artifact production_pick_snapshot does not match "
                        "the current production pick, but processed PA outcomes "
                        f"already contain {n_pa_rows} rows for {config.date}; "
                        "refusing after-outcome recapture"
                    ),
                    production_head=production_head,
                    live_forward_head=live_forward_head,
                    artifact_dir=artifact_dir,
                    verification_path=verification_path,
                    pick_path=pick_path,
                    extra={
                        **snapshot_state,
                        "pa_outcome_check_path": str(pa_path),
                        "pa_outcome_rows_for_date": n_pa_rows,
                    },
                )
                write_json(status_path, payload)
                return 1, payload

            return refresh_stale_artifact(
                config,
                pick_path=pick_path,
                artifact_dir=artifact_dir,
                verification_path=verification_path,
                status_path=status_path,
                production_head=production_head,
                live_forward_head=live_forward_head,
                snapshot_state={
                    **snapshot_state,
                    "pa_outcome_check_path": str(pa_path),
                    "pa_outcome_rows_for_date": n_pa_rows,
                },
            )

        verify = verify_artifact(
            config,
            artifact_dir=artifact_dir,
            verification_path=verification_path,
            live_forward_head=live_forward_head,
        )
        status = "existing_verified" if verify.returncode == 0 else "failed_verify_existing"
        payload = status_payload(
            config=config,
            status=status,
            message=(verify.stdout + verify.stderr).strip(),
            production_head=production_head,
            live_forward_head=live_forward_head,
            artifact_dir=artifact_dir,
            verification_path=verification_path,
            pick_path=pick_path,
            extra=snapshot_state,
        )
        write_json(status_path, payload)
        return (0 if verify.returncode == 0 else 1), payload
    if artifact_dir.exists() and any(artifact_dir.iterdir()) and not config.overwrite:
        payload = status_payload(
            config=config,
            status="failed_partial_artifact_dir",
            message="artifact directory exists without manifest; refusing overwrite",
            production_head=production_head,
            live_forward_head=live_forward_head,
            artifact_dir=artifact_dir,
            verification_path=verification_path,
            pick_path=pick_path,
        )
        write_json(status_path, payload)
        return 1, payload

    if not pick_path.exists():
        if config.capture_research_on_skip:
            research = research_capture_once(
                config,
                pick_path=pick_path,
                production_head=production_head,
                live_forward_head=live_forward_head,
            )
            if research is not None:
                return research
        payload = status_payload(
            config=config,
            status="pending_pick",
            message="production pick file does not exist yet",
            production_head=production_head,
            live_forward_head=live_forward_head,
            artifact_dir=artifact_dir,
            verification_path=verification_path,
            pick_path=pick_path,
        )
        return (2 if config.fail_on_pending else 0), payload

    try:
        pick = read_json(pick_path)
    except (OSError, json.JSONDecodeError) as exc:
        payload = status_payload(
            config=config,
            status="transient_pick_read_error",
            message=f"could not read production pick file yet: {exc}",
            production_head=production_head,
            live_forward_head=live_forward_head,
            artifact_dir=artifact_dir,
            verification_path=verification_path,
            pick_path=pick_path,
        )
        return 0, payload
    if str(pick.get("date")) != config.date:
        payload = status_payload(
            config=config,
            status="failed_pick_date_mismatch",
            message=f"pick file date {pick.get('date')!r} does not match {config.date!r}",
            production_head=production_head,
            live_forward_head=live_forward_head,
            artifact_dir=artifact_dir,
            verification_path=verification_path,
            pick_path=pick_path,
        )
        return 1, payload
    if not pick_is_unresolved(pick):
        payload = status_payload(
            config=config,
            status="failed_pick_already_resolved",
            message=f"pick result is already {pick.get('result')!r}",
            production_head=production_head,
            live_forward_head=live_forward_head,
            artifact_dir=artifact_dir,
            verification_path=verification_path,
            pick_path=pick_path,
        )
        return 1, payload

    try:
        pa_path, n_pa_rows = pa_rows_for_date(
            resolve_under(config.production_root, config.data_dir),
            config.date,
        )
    except Exception as exc:
        payload = status_payload(
            config=config,
            status="failed_export_outcome_check",
            message=f"could not check processed PA outcomes before export: {exc}",
            production_head=production_head,
            live_forward_head=live_forward_head,
            artifact_dir=artifact_dir,
            verification_path=verification_path,
            pick_path=pick_path,
        )
        write_json(status_path, payload)
        return 1, payload
    if n_pa_rows > 0:
        payload = status_payload(
            config=config,
            status="failed_export_post_outcomes",
            message=(
                f"processed PA outcomes already contain {n_pa_rows} rows for "
                f"{config.date}; refusing post-outcome capture"
            ),
            production_head=production_head,
            live_forward_head=live_forward_head,
            artifact_dir=artifact_dir,
            verification_path=verification_path,
            pick_path=pick_path,
            extra={
                "pa_outcome_check_path": str(pa_path),
                "pa_outcome_rows_for_date": n_pa_rows,
            },
        )
        write_json(status_path, payload)
        return 1, payload

    export = export_artifact(
        config,
        pick_path=pick_path,
        artifact_dir=artifact_dir,
    )
    if export.returncode != 0:
        payload = status_payload(
            config=config,
            status="failed_export",
            message=(export.stdout + export.stderr).strip(),
            production_head=production_head,
            live_forward_head=live_forward_head,
            artifact_dir=artifact_dir,
            verification_path=verification_path,
            pick_path=pick_path,
        )
        write_json(status_path, payload)
        return 1, payload

    verify = verify_artifact(
        config,
        artifact_dir=artifact_dir,
        verification_path=verification_path,
        live_forward_head=live_forward_head,
    )
    payload = status_payload(
        config=config,
        status="exported_verified" if verify.returncode == 0 else "failed_verify",
        message=(export.stdout + export.stderr + verify.stdout + verify.stderr).strip(),
        production_head=production_head,
        live_forward_head=live_forward_head,
        artifact_dir=artifact_dir,
        verification_path=verification_path,
        pick_path=pick_path,
        extra={
            "pa_outcome_check_path": str(pa_path),
            "pa_outcome_rows_for_date": n_pa_rows,
        },
    )
    write_json(status_path, payload)
    return (0 if verify.returncode == 0 else 1), payload


# ---------------------------------------------------------------------------
# D8 research-only capture (no production pick). Design + Codex review:
# .codex-review/season-wrap/d8-design.md / d8-codex.md (2026-09-14).
# ---------------------------------------------------------------------------


def _read_json_readonly(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Parse JSON without ANY side effect (never quarantine/rename like load_state)."""
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(obj, dict):
        return None, "not a JSON object"
    return obj, None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def earliest_first_pitch_utc(games: Any) -> datetime | None:
    """Earliest `game_time_et` in the scheduler's persisted games list, as aware UTC.

    Returns None when the list is missing/empty or any entry is unparseable —
    callers FAIL CLOSED (no capture) in that case."""
    if not isinstance(games, list) or not games:
        return None
    times: list[datetime] = []
    for game in games:
        raw = game.get("game_time_et") if isinstance(game, dict) else None
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("America/New_York"))
        times.append(parsed.astimezone(timezone.utc))
    return min(times)


RESEARCH_SIDECAR_SCHEMA = "live_forward_research_capture_v1"
RESEARCH_RUN_KIND = "live_forward_preoutcome"


def research_contract_files(date: str) -> list[str]:
    return [
        "manifest.json",
        "verification.json",
        f"profiles/production/live_{date}.parquet",
        f"profiles/candidate/live_{date}.parquet",
    ]


def _parse_iso_utc(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def research_contract_problems(research_dir: Path, *, date: str, candidate: str) -> list[str]:
    """Identity/integrity checks on the exporter+verifier outputs (sidecar excluded)."""
    problems: list[str] = []
    for rel in research_contract_files(date):
        if not (research_dir / rel).exists():
            problems.append(f"contract file missing: {rel}")
    if problems:
        return problems
    manifest, err = _read_json_readonly(research_dir / "manifest.json")
    if manifest is None:
        return [f"manifest unreadable: {err}"]
    if str(manifest.get("date")) != date:
        problems.append(f"manifest date {manifest.get('date')!r} != {date!r}")
    if str(manifest.get("candidate_name")) != candidate:
        problems.append(
            f"manifest candidate {manifest.get('candidate_name')!r} != {candidate!r}"
        )
    if manifest.get("run_kind") != RESEARCH_RUN_KIND:
        problems.append(f"manifest run_kind {manifest.get('run_kind')!r} != {RESEARCH_RUN_KIND!r}")
    if not isinstance(manifest.get("git_commit"), str) or not manifest.get("git_commit"):
        problems.append("manifest has no git_commit")
    if isinstance(manifest.get("production_pick_snapshot"), dict):
        problems.append("manifest carries a production_pick_snapshot (not a research capture)")
    verification, err = _read_json_readonly(research_dir / "verification.json")
    if verification is None:
        problems.append(f"verification unreadable: {err}")
    elif verification.get("ok") is not True:
        problems.append("verification.json ok is not true")
    return problems


def research_sidecar_state(
    research_dir: Path, *, date: str, candidate: str
) -> tuple[str, str]:
    """Classify an existing research dir: ``valid`` (complete, accepted contract),
    ``incompatible`` (a complete capture for ANOTHER candidate — never relabel or
    replace it), or ``invalid`` (partial/corrupt/tampered/unaccepted — recoverable
    only pregame). Restart validation re-checks identity, provenance binding, the
    start cutoff, completion ordering and the post-publication acceptance marker."""
    sidecar_path = research_dir / RESEARCH_SIDECAR_NAME
    if not sidecar_path.exists():
        return "invalid", "sidecar missing"
    sidecar, err = _read_json_readonly(sidecar_path)
    if sidecar is None:
        return "invalid", f"sidecar unreadable: {err}"
    if sidecar.get("schema_version") != RESEARCH_SIDECAR_SCHEMA:
        return "invalid", f"sidecar schema {sidecar.get('schema_version')!r}"
    if sidecar.get("research_only") is not True or sidecar.get("eligible_for_official_read") is not False:
        return "invalid", "sidecar research/eligibility labels are wrong"
    if sidecar.get("reason") != RESEARCH_REASON or str(sidecar.get("date")) != date:
        return "invalid", "sidecar reason/date mismatch"
    for key in ("production_head", "live_forward_head"):
        if not isinstance(sidecar.get(key), str) or not sidecar.get(key):
            return "invalid", f"sidecar provenance field {key} missing"
    hashes = sidecar.get("file_sha256")
    if not isinstance(hashes, dict):
        return "invalid", "sidecar has no file hashes"
    for rel in research_contract_files(date):
        path = research_dir / rel
        if not path.exists():
            return "invalid", f"contract file missing: {rel}"
        if hashes.get(rel) != file_sha256(path):
            return "invalid", f"hash mismatch: {rel}"
    sidecar_candidate = str(sidecar.get("candidate"))
    problems = research_contract_problems(research_dir, date=date, candidate=sidecar_candidate)
    if problems:
        return "invalid", "; ".join(problems)
    # Provenance binding: the frozen live-forward head recorded in the sidecar must be
    # the commit the exporter stamped and the verifier checked. (production_head is
    # recorded, not compared: deploys legitimately move the production checkout.)
    manifest, _ = _read_json_readonly(research_dir / "manifest.json")
    verification, _ = _read_json_readonly(research_dir / "verification.json")
    head = sidecar["live_forward_head"]
    if manifest is None or manifest.get("git_commit") != head:
        return "invalid", "sidecar live_forward_head does not match manifest git_commit"
    ver_manifest = verification.get("manifest") if isinstance(verification, dict) else None
    if isinstance(ver_manifest, dict) and ver_manifest.get("git_commit") not in (None, head):
        return "invalid", "verification report was produced for a different git_commit"
    started = _parse_iso_utc(sidecar.get("started_at"))
    completed = _parse_iso_utc(sidecar.get("completed_at"))
    first_pitch = _parse_iso_utc(sidecar.get("earliest_first_pitch_utc"))
    if started is None or completed is None or first_pitch is None:
        return "invalid", "sidecar timestamps unparseable"
    if started >= first_pitch - timedelta(minutes=RESEARCH_DEADLINE_BUFFER_MIN):
        return "invalid", "sidecar started_at violates the pre-first-pitch start cutoff"
    if not (started <= completed < first_pitch):
        return "invalid", "sidecar timestamps are not started <= completed < first pitch"
    accepted_path = research_dir / RESEARCH_ACCEPTED_NAME
    if not accepted_path.exists():
        return "invalid", "acceptance marker missing (publication never confirmed)"
    accepted, err = _read_json_readonly(accepted_path)
    if accepted is None:
        return "invalid", f"acceptance marker unreadable: {err}"
    if accepted.get("sidecar_sha256") != file_sha256(sidecar_path):
        return "invalid", "acceptance marker does not match the sidecar"
    published = _parse_iso_utc(accepted.get("published_at"))
    if published is None or not (completed <= published < first_pitch):
        return "invalid", "publication was not confirmed before first pitch"
    if sidecar_candidate != candidate:
        return "incompatible", (
            f"complete research capture exists for candidate {sidecar_candidate!r}, "
            f"requested {candidate!r}"
        )
    return "valid", "ok"


def _quarantine_dir(research_dir: Path, label: str) -> Path:
    stamp = utc_now_dt().strftime("%Y%m%dT%H%M%SZ")
    base = research_dir.with_name(f"{research_dir.name}.{label}.{stamp}")
    target = base
    counter = 1
    while target.exists() or target.is_symlink():
        target = research_dir.with_name(f"{base.name}.{counter}")
        counter += 1
    shutil.move(str(research_dir), str(target))
    return target


def _roots_overlap(a: Path, b: Path) -> bool:
    ra, rb = a.resolve(), b.resolve()
    return ra == rb or ra.is_relative_to(rb) or rb.is_relative_to(ra)


def research_root_conflict(config: CaptureConfig) -> str | None:
    """The research root must not touch anything live: the official artifact root,
    the picks/state tree, the processed-data tree, or the production root itself."""
    research_root = resolve_under(config.production_root, config.research_root)
    protected = {
        "official artifact root": resolve_under(config.production_root, config.artifact_root),
        "picks dir": resolve_under(config.production_root, config.picks_dir),
        "data dir": resolve_under(config.production_root, config.data_dir),
    }
    for label, path in protected.items():
        if _roots_overlap(research_root, path):
            return f"research root {research_root} overlaps {label} {path}"
    prod = config.production_root.resolve()
    rr = research_root.resolve()
    if rr == prod or prod.is_relative_to(rr):
        return f"research root {research_root} is the production root or an ancestor of it"
    return None


def research_dir_escapes(research_dir: Path, research_root: Path) -> str | None:
    """Refuse a per-date dir that is a symlink or resolves outside the research root."""
    if research_dir.is_symlink():
        return f"{research_dir} is a symlink"
    if research_dir.exists() and not research_dir.resolve().is_relative_to(research_root.resolve()):
        return f"{research_dir} resolves outside {research_root}"
    return None


def research_capture_once(
    config: CaptureConfig,
    *,
    pick_path: Path,
    production_head: str | None,
    live_forward_head: str | None,
) -> tuple[int, dict[str, Any]] | None:
    """Capture the day's ranked slates WITHOUT a production pick, into the research root.

    Returns None when the day shows no skip signal (caller falls through to the
    unchanged `pending_pick`). Trigger = the scheduler's persisted CURRENT skip intent
    (`final_skip_candidate`; a later pick attempt clears it) with no scoreable
    decision. Everything read here is parsed read-only. The forecast must START
    before first pitch − 5 min and be PUBLISHED before first pitch (the clock is
    re-sampled right before export and again after hashing, immediately before the
    sidecar is written); anything else is refused or discarded. The sidecar (written
    last, atomically, with hashes + identity + timestamps) is the only thing that
    makes a capture count, and restart validation re-checks all of it."""
    research_root = resolve_under(config.production_root, config.research_root)
    research_dir = research_root / config.date
    status_path = research_dir / "capture_status.json"

    def payload_for(status: str, message: str, **extra: Any) -> dict[str, Any]:
        return status_payload(
            config=config,
            status=status,
            message=message,
            production_head=production_head,
            live_forward_head=live_forward_head,
            artifact_dir=research_dir,
            verification_path=research_dir / "verification.json",
            pick_path=pick_path,
            extra={"research_stream": True, "research_root": str(research_root), **extra},
        )

    conflict = research_root_conflict(config)
    if conflict:
        return 1, payload_for("failed_research_root_overlap", conflict)

    picks_dir = resolve_under(config.production_root, config.picks_dir)
    state_path = picks_dir / config.date / "scheduler_state.json"
    if not state_path.exists():
        return None
    state, err = _read_json_readonly(state_path)
    if state is None:
        return 0, payload_for("research_pending_state_unreadable",
                              f"scheduler_state.json unreadable: {err}")
    if str(state.get("date")) != config.date:
        return None
    candidate = state.get("final_skip_candidate")
    if not isinstance(candidate, dict) or not candidate:
        return None  # no CURRENT skip intent (skip_notified_at alone is history)

    decision_path = picks_dir / config.date / "decision.json"
    decision: dict[str, Any] | None = None
    decision_note: str | None = None
    if decision_path.exists():
        decision, err = _read_json_readonly(decision_path)
        if decision is None:
            return 0, payload_for("research_pending_decision_unreadable",
                                  f"decision.json unreadable: {err}")
        if str(decision.get("date")) != config.date:
            decision_note = f"decision.json date {decision.get('date')!r} != {config.date!r}; ignored"
            decision = None
        elif decision.get("scoreable") is True:
            return None  # a pick was committed; the official path owns this day

    first_pitch = earliest_first_pitch_utc(state.get("games"))
    if first_pitch is None:
        return 0, payload_for("research_pending_no_game_times",
                              "scheduler state has no parseable game times; failing closed")
    first_pitch_iso = first_pitch.isoformat()
    deadline = first_pitch - timedelta(minutes=RESEARCH_DEADLINE_BUFFER_MIN)

    escape = research_dir_escapes(research_dir, research_root)
    if escape:
        return 1, payload_for("failed_research_dir_escapes_root", escape,
                              earliest_first_pitch_utc=first_pitch_iso)

    quarantined: Path | None = None
    if research_dir.exists() and any(research_dir.iterdir()):
        kind, why = research_sidecar_state(research_dir, date=config.date, candidate=config.candidate)
        if kind == "valid":
            return 0, payload_for("research_existing_verified",
                                  "complete research capture already present",
                                  earliest_first_pitch_utc=first_pitch_iso)
        if kind == "incompatible":
            return 1, payload_for("research_existing_incompatible", why,
                                  earliest_first_pitch_utc=first_pitch_iso)
        if config_now(config) >= deadline:
            return 1, payload_for("research_partial_after_deadline",
                                  f"incomplete research capture ({why}) and the capture "
                                  "deadline has passed; left for offline reconciliation",
                                  earliest_first_pitch_utc=first_pitch_iso)
        quarantined = _quarantine_dir(research_dir, "partial")

    started_at = config_now(config)  # re-sampled right before the export starts
    if started_at >= deadline:
        return 0, payload_for("research_deadline_passed",
                              f"now {started_at.isoformat()} is at/after first pitch − "
                              f"{RESEARCH_DEADLINE_BUFFER_MIN} min ({deadline.isoformat()})",
                              earliest_first_pitch_utc=first_pitch_iso,
                              quarantined_partial=str(quarantined) if quarantined else None)

    export = export_artifact(config, pick_path=None, artifact_dir=research_dir)
    if export.returncode != 0:
        payload = payload_for("failed_research_export", (export.stdout + export.stderr).strip(),
                              earliest_first_pitch_utc=first_pitch_iso,
                              quarantined_partial=str(quarantined) if quarantined else None)
        write_json(status_path, payload)
        return 1, payload
    verify = verify_artifact(
        config,
        artifact_dir=research_dir,
        verification_path=research_dir / "verification.json",
        live_forward_head=live_forward_head or "",
        require_pick_snapshot=False,
    )
    if verify.returncode != 0:
        payload = payload_for("failed_research_verify", (verify.stdout + verify.stderr).strip(),
                              earliest_first_pitch_utc=first_pitch_iso)
        write_json(status_path, payload)
        return 1, payload

    problems = research_contract_problems(research_dir, date=config.date, candidate=config.candidate)
    if problems:
        payload = payload_for("failed_research_contract_incomplete",
                              "export/verify succeeded but the contract is incomplete: "
                              + "; ".join(problems),
                              earliest_first_pitch_utc=first_pitch_iso)
        write_json(status_path, payload)
        return 1, payload

    hashes = {rel: file_sha256(research_dir / rel) for rel in research_contract_files(config.date)}
    completed_at = config_now(config)  # sampled AFTER hashing, before the sidecar is written

    def discard_late(stage: str, when: datetime) -> tuple[int, dict[str, Any]]:
        late = _quarantine_dir(research_dir, "late")
        payload = payload_for("discarded_research_after_first_pitch",
                              f"{stage} at {when.isoformat()}, at/after first pitch "
                              f"{first_pitch_iso}; not a decision-time forecast",
                              earliest_first_pitch_utc=first_pitch_iso, discarded_to=str(late))
        write_json(late / "capture_status.json", payload)
        return 1, payload

    if completed_at >= first_pitch:
        return discard_late("capture completed", completed_at)

    sidecar = {
        "schema_version": RESEARCH_SIDECAR_SCHEMA,
        "research_only": True,
        "eligible_for_official_read": False,
        "reason": RESEARCH_REASON,
        "date": config.date,
        "candidate": config.candidate,
        "trigger": {
            "kind": "provisional_scheduler_skip_state",
            "final_classification": "reconcile_in_season_ledger",
            "final_skip_candidate": candidate,
            "decision_present": decision is not None,
            "decision_action": decision.get("action") if decision else None,
            "decision_note": decision_note,
        },
        "earliest_first_pitch_utc": first_pitch_iso,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "production_head": production_head,
        "live_forward_head": live_forward_head,
        "quarantined_partial": str(quarantined) if quarantined else None,
        "file_sha256": hashes,
    }
    sidecar_path = research_dir / RESEARCH_SIDECAR_NAME
    _atomic_write_json(sidecar_path, sidecar)
    # Phase 2: the acceptance marker. published_at is sampled AFTER the sidecar rename,
    # so it bounds the moment every piece of forecast content was in place. A crash
    # before this marker leaves an UNACCEPTED capture (restart treats it as partial).
    published_at = config_now(config)
    if published_at >= first_pitch:
        return discard_late("sidecar publication landed", published_at)
    _atomic_write_json(research_dir / RESEARCH_ACCEPTED_NAME, {
        "schema_version": RESEARCH_SIDECAR_SCHEMA,
        "date": config.date,
        "candidate": config.candidate,
        "sidecar_sha256": file_sha256(sidecar_path),
        "published_at": published_at.isoformat(),
        "earliest_first_pitch_utc": first_pitch_iso,
    })
    post_publish = config_now(config)
    if post_publish >= first_pitch:
        # Conservative: the acceptance record itself was not durable before first pitch.
        return discard_late("acceptance marker landed", post_publish)
    payload = payload_for("captured_research_no_production_pick",
                          "research-only ranked slates captured (no production pick)",
                          earliest_first_pitch_utc=first_pitch_iso,
                          sidecar_path=str(sidecar_path),
                          published_at=published_at.isoformat())
    write_json(status_path, payload)
    return 0, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=today_et(), help="YYYY-MM-DD ET")
    parser.add_argument("--production-root", type=Path, default=DEFAULT_PRODUCTION_ROOT)
    parser.add_argument("--live-forward-root", type=Path, default=DEFAULT_LIVE_FORWARD_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--picks-dir", type=Path, default=Path("data/picks"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--auto-recapture-on-snapshot-drift",
        action="store_true",
        help=(
            "When an existing artifact snapshots an older production pick file, "
            "recapture automatically if the current pick is unresolved and no "
            "processed PA outcomes exist for the date."
        ),
    )
    parser.add_argument(
        "--fail-on-pending",
        action="store_true",
        help="Return exit code 2 instead of 0 when the pick file is not present.",
    )
    parser.add_argument(
        "--capture-research-on-skip",
        action="store_true",
        help=(
            "D8: on a day with NO production pick where the scheduler has persisted a "
            "current skip intent, export the ranked slates into --research-root as a "
            "research-only capture (never eligible for the official read)."
        ),
    )
    parser.add_argument("--research-root", type=Path, default=DEFAULT_RESEARCH_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = CaptureConfig(
        date=args.date,
        production_root=args.production_root,
        live_forward_root=args.live_forward_root,
        python=args.python,
        candidate=args.candidate,
        artifact_root=args.artifact_root,
        data_dir=args.data_dir,
        picks_dir=args.picks_dir,
        top_n=args.top_n,
        overwrite=args.overwrite,
        fail_on_pending=args.fail_on_pending,
        auto_recapture_on_snapshot_drift=args.auto_recapture_on_snapshot_drift,
        capture_research_on_skip=args.capture_research_on_skip,
        research_root=args.research_root,
    )
    exit_code, payload = capture_once(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
