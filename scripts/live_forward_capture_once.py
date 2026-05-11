#!/usr/bin/env python3
"""Guarded one-shot live-forward artifact capture.

This runner is intentionally idempotent so it can be called by a frequent
systemd timer. It does not create production picks. It waits for the production
pick JSON to exist, refuses resolved pick files, exports the frozen
candidate-vs-production ranked slates, and immediately verifies the artifact.
"""

from __future__ import annotations

import argparse
import json
import os
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


def today_et() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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
) -> dict[str, Any]:
    return {
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


def capture_once(config: CaptureConfig) -> tuple[int, dict[str, Any]]:
    pick_path = resolve_under(config.production_root, config.picks_dir) / (
        f"{config.date}.json"
    )
    artifact_dir = resolve_under(config.production_root, config.artifact_root) / config.date
    verification_path = artifact_dir / "verification.json"
    status_path = artifact_dir / "capture_status.json"

    production_head = git_head(config.production_root)
    live_forward_head = git_head(config.live_forward_root)

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

    manifest_path = artifact_dir / "manifest.json"
    if manifest_path.exists() and not config.overwrite:
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

    export = run_bts(
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
    )
    exit_code, payload = capture_once(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
