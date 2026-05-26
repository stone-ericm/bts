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
from datetime import datetime, timezone
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


def today_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
) -> subprocess.CompletedProcess[str]:
    return run_bts(
        config,
        [
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
            "--require-production-pick-snapshot",
            "--save",
            str(verification_path),
        ],
    )


def export_artifact(
    config: CaptureConfig,
    *,
    pick_path: Path,
    artifact_dir: Path,
) -> subprocess.CompletedProcess[str]:
    return run_bts(
        config,
        [
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
            "--production-pick-file",
            str(pick_path),
        ],
    )


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
    )
    exit_code, payload = capture_once(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
