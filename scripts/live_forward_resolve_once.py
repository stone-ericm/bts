#!/usr/bin/env python3
"""Guarded one-shot live-forward artifact resolver.

This runner scans captured live-forward preoutcome artifacts, joins realized
outcomes through the existing BTS resolver CLI, and verifies resolved outputs.
It is intentionally idempotent for systemd timers: missing captures and missing
PA outcomes are pending states, existing resolved manifests are verified, and
preoutcome artifact directories are never modified.
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


SCHEMA_VERSION = "live_forward_resolve_once_v1"
DEFAULT_CANDIDATE = "decision_weighted_lgbm_v0"
DEFAULT_PREOUTCOME_ROOT = Path(
    "data/validation/decision_weighted_lgbm_v0_live_forward"
)
DEFAULT_RESOLVED_ROOT = Path(
    "data/validation/decision_weighted_lgbm_v0_live_forward_resolved"
)
DEFAULT_STATUS_ROOT = Path(
    "data/validation/decision_weighted_lgbm_v0_live_forward_resolved_status"
)
DEFAULT_PRODUCTION_ROOT = Path("/home/bts/projects/bts")
DEFAULT_PYTHON = Path("/home/bts/projects/bts/.venv/bin/python")
PENDING_RESOLVE_MARKERS = (
    # Contract with src/bts/experiment/artifacts.py and cli.py: these errors
    # mean outcome evidence is not ready yet, not that resolution is invalid.
    "missing outcomes for ",
    "missing processed PA parquet:",
)


@dataclass(frozen=True)
class ResolveConfig:
    production_root: Path
    python: Path
    candidate: str
    preoutcome_root: Path
    resolved_root: Path
    status_root: Path
    data_dir: Path
    top_n: int
    dates: tuple[str, ...]
    overwrite: bool
    fail_on_pending: bool


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


def bts_command_env(config: ResolveConfig) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(config.production_root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("BTS_LGBM_DETERMINISTIC", "1")
    env.setdefault("BTS_LGBM_RANDOM_STATE", "42")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("NUMEXPR_NUM_THREADS", "2")
    return env


def run_bts(
    config: ResolveConfig,
    cli_args: list[str],
) -> subprocess.CompletedProcess[str]:
    return run(
        [str(config.python), "-c", "from bts.cli import cli; cli()", *cli_args],
        cwd=config.production_root,
        env=bts_command_env(config),
    )


def manifest_dates(manifest: dict[str, Any]) -> list[str]:
    dates = manifest.get("dates")
    if dates is None:
        date = manifest.get("date")
        dates = [date] if date else []
    return [str(date) for date in dates if date is not None]


def discover_dates(preoutcome_root: Path) -> list[str]:
    if not preoutcome_root.exists():
        return []
    return sorted(
        path.name for path in preoutcome_root.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    )


def pa_rows_for_dates(
    data_dir: Path,
    dates: list[str],
) -> tuple[list[Path], list[Path], int]:
    import pandas as pd

    years = sorted({pd.Timestamp(date).year for date in dates})
    paths: list[Path] = []
    missing: list[Path] = []
    rows = 0
    for year in years:
        path = data_dir / f"pa_{year}.parquet"
        if not path.exists():
            missing.append(path)
            continue
        paths.append(path)
        frame = pd.read_parquet(path, columns=["date"])
        date_keys = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        rows += int(date_keys.isin(dates).sum())
    return paths, missing, rows


def status_payload(
    *,
    config: ResolveConfig,
    status: str,
    message: str,
    date: str | None,
    dates: list[str] | None = None,
    production_head: str | None = None,
    source_manifest_path: Path | None = None,
    source_manifest: dict[str, Any] | None = None,
    preoutcome_dir: Path | None = None,
    resolved_dir: Path | None = None,
    resolution_path: Path | None = None,
    verification_path: Path | None = None,
    pa_data_paths: list[Path] | None = None,
    missing_pa_data_paths: list[Path] | None = None,
    n_outcome_rows_for_dates: int | None = None,
    missing_count: int | None = None,
    terminal_void_count: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": status,
        "message": message,
        "date": date,
        "dates": dates,
        "candidate": config.candidate,
        "production_root": str(config.production_root),
        "production_head": production_head,
        "preoutcome_dir": str(preoutcome_dir) if preoutcome_dir is not None else None,
        "resolved_dir": str(resolved_dir) if resolved_dir is not None else None,
        "source_manifest": (
            str(source_manifest_path) if source_manifest_path is not None else None
        ),
        "source_run_kind": (
            source_manifest.get("run_kind") if source_manifest is not None else None
        ),
        "source_git_commit": (
            source_manifest.get("git_commit") if source_manifest is not None else None
        ),
        "live_forward_head": (
            source_manifest.get("git_commit") if source_manifest is not None else None
        ),
        "resolved_manifest": (
            str(resolved_dir / "manifest.json") if resolved_dir is not None else None
        ),
        "resolution_path": (
            str(resolution_path) if resolution_path is not None else None
        ),
        "verification_path": (
            str(verification_path) if verification_path is not None else None
        ),
        "data_dir": str(resolve_under(config.production_root, config.data_dir)),
        "pa_data_paths": [str(path) for path in (pa_data_paths or [])],
        "missing_pa_data_paths": [
            str(path) for path in (missing_pa_data_paths or [])
        ],
        "n_outcome_rows_for_dates": n_outcome_rows_for_dates,
        "missing_count": missing_count,
        "terminal_void_count": terminal_void_count,
        "methodology_constraints": [
            "preoutcome artifact directory is read-only",
            "missing PA outcomes are pending evidence, never coerced to misses",
            "resolution is performed by bts experiment resolve-live-candidate-artifacts",
            "resolved artifacts are verified before success",
        ],
    }


def verify_resolved_artifact(
    config: ResolveConfig,
    *,
    resolved_dir: Path,
    verification_path: Path,
    source_manifest: dict[str, Any],
    date: str,
) -> subprocess.CompletedProcess[str]:
    return run_bts(
        config,
        [
            "experiment",
            "verify-candidate-artifacts",
            "--artifact-dir",
            str(resolved_dir),
            "--expected-run-kind",
            "live_forward_resolved",
            "--expected-candidate",
            str(source_manifest.get("candidate_name") or config.candidate),
            "--expected-date",
            date,
            "--expected-git-commit",
            str(source_manifest.get("git_commit")),
            "--expected-top-n",
            str(source_manifest.get("top_n") or config.top_n),
            "--require-production-pick-snapshot",
            "--save",
            str(verification_path),
        ],
    )


def read_resolution_counts(resolution_path: Path) -> dict[str, int | None]:
    if not resolution_path.exists():
        return {"missing_count": None, "terminal_void_count": None}
    try:
        report = read_json(resolution_path)
    except (OSError, json.JSONDecodeError):
        return {"missing_count": None, "terminal_void_count": None}
    return {
        "missing_count": report.get("missing_count"),
        "terminal_void_count": report.get("terminal_void_count"),
    }


def _is_pending_resolution_error(message: str) -> bool:
    return any(marker in message for marker in PENDING_RESOLVE_MARKERS)


def resolve_artifact_date(
    config: ResolveConfig,
    *,
    date: str,
    production_head: str,
) -> tuple[int, dict[str, Any]]:
    preoutcome_root = resolve_under(config.production_root, config.preoutcome_root)
    resolved_root = resolve_under(config.production_root, config.resolved_root)
    status_root = resolve_under(config.production_root, config.status_root)
    data_dir = resolve_under(config.production_root, config.data_dir)
    preoutcome_dir = preoutcome_root / date
    resolved_dir = resolved_root / date
    status_path = status_root / f"{date}.json"
    source_manifest_path = preoutcome_dir / "manifest.json"
    resolution_path = resolved_dir / "resolution.json"
    verification_path = resolved_dir / "verification.json"

    if preoutcome_dir.resolve() == resolved_dir.resolve():
        payload = status_payload(
            config=config,
            status="failed_same_source_and_output",
            message="resolved output directory must differ from preoutcome directory",
            date=date,
            production_head=production_head,
            preoutcome_dir=preoutcome_dir,
            resolved_dir=resolved_dir,
            resolution_path=resolution_path,
            verification_path=verification_path,
        )
        write_json(status_path, payload)
        return 1, payload

    if not source_manifest_path.exists():
        payload = status_payload(
            config=config,
            status="pending_preoutcome_artifact",
            message="preoutcome manifest does not exist yet",
            date=date,
            production_head=production_head,
            preoutcome_dir=preoutcome_dir,
            resolved_dir=resolved_dir,
            resolution_path=resolution_path,
            verification_path=verification_path,
        )
        write_json(status_path, payload)
        return (2 if config.fail_on_pending else 0), payload

    try:
        source_manifest = read_json(source_manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        payload = status_payload(
            config=config,
            status="failed_source_manifest_read",
            message=f"could not read preoutcome manifest: {exc}",
            date=date,
            production_head=production_head,
            source_manifest_path=source_manifest_path,
            preoutcome_dir=preoutcome_dir,
            resolved_dir=resolved_dir,
            resolution_path=resolution_path,
            verification_path=verification_path,
        )
        write_json(status_path, payload)
        return 1, payload

    dates = manifest_dates(source_manifest)
    if not dates:
        payload = status_payload(
            config=config,
            status="failed_source_manifest_dates",
            message="source manifest has no live-forward dates",
            date=date,
            production_head=production_head,
            source_manifest_path=source_manifest_path,
            source_manifest=source_manifest,
            preoutcome_dir=preoutcome_dir,
            resolved_dir=resolved_dir,
            resolution_path=resolution_path,
            verification_path=verification_path,
        )
        write_json(status_path, payload)
        return 1, payload

    if source_manifest.get("run_kind") != "live_forward_preoutcome":
        payload = status_payload(
            config=config,
            status="failed_source_run_kind",
            message=(
                "expected live_forward_preoutcome, "
                f"found {source_manifest.get('run_kind')!r}"
            ),
            date=date,
            dates=dates,
            production_head=production_head,
            source_manifest_path=source_manifest_path,
            source_manifest=source_manifest,
            preoutcome_dir=preoutcome_dir,
            resolved_dir=resolved_dir,
            resolution_path=resolution_path,
            verification_path=verification_path,
        )
        write_json(status_path, payload)
        return 1, payload

    base_status_kwargs = {
        "config": config,
        "date": date,
        "dates": dates,
        "production_head": production_head,
        "source_manifest_path": source_manifest_path,
        "source_manifest": source_manifest,
        "preoutcome_dir": preoutcome_dir,
        "resolved_dir": resolved_dir,
        "resolution_path": resolution_path,
        "verification_path": verification_path,
    }

    manifest_path = resolved_dir / "manifest.json"
    if manifest_path.exists() and not config.overwrite:
        verify = verify_resolved_artifact(
            config,
            resolved_dir=resolved_dir,
            verification_path=verification_path,
            source_manifest=source_manifest,
            date=date,
        )
        resolution_counts = read_resolution_counts(resolution_path)
        has_terminal_voids = bool(resolution_counts["terminal_void_count"])
        status = (
            "existing_verified_with_voids"
            if verify.returncode == 0 and has_terminal_voids
            else "existing_verified"
            if verify.returncode == 0
            else "failed_verify_existing"
        )
        payload = status_payload(
            **base_status_kwargs,
            status=status,
            message=(verify.stdout + verify.stderr).strip(),
            missing_count=resolution_counts["missing_count"],
            terminal_void_count=resolution_counts["terminal_void_count"],
        )
        write_json(status_path, payload)
        write_json(resolved_dir / "resolve_status.json", payload)
        return (0 if verify.returncode == 0 else 1), payload

    if resolved_dir.exists() and any(resolved_dir.iterdir()) and not config.overwrite:
        payload = status_payload(
            **base_status_kwargs,
            status="failed_partial_resolved_dir",
            message="resolved directory exists without manifest; refusing overwrite",
        )
        write_json(status_path, payload)
        return 1, payload

    try:
        pa_paths, missing_pa_paths, n_pa_rows = pa_rows_for_dates(data_dir, dates)
    except Exception as exc:
        payload = status_payload(
            **base_status_kwargs,
            status="failed_pa_data_read",
            message=f"could not read processed PA outcomes: {exc}",
        )
        write_json(status_path, payload)
        return 1, payload

    common_status_kwargs = {
        **base_status_kwargs,
        "pa_data_paths": pa_paths,
        "missing_pa_data_paths": missing_pa_paths,
        "n_outcome_rows_for_dates": n_pa_rows,
    }

    if missing_pa_paths or n_pa_rows == 0:
        payload = status_payload(
            **common_status_kwargs,
            status="pending_outcomes",
            message="processed PA outcomes are not available for artifact date(s)",
        )
        write_json(status_path, payload)
        return (2 if config.fail_on_pending else 0), payload

    resolve = run_bts(
        config,
        [
            "experiment",
            "resolve-live-candidate-artifacts",
            "--artifact-dir",
            str(preoutcome_dir),
            "--output-dir",
            str(resolved_dir),
            "--data-dir",
            str(data_dir),
            "--treat-void-games-as-terminal",
            "--save",
            str(resolution_path),
        ] + (["--overwrite"] if config.overwrite else []),
    )
    resolve_message = (resolve.stdout + resolve.stderr).strip()
    if resolve.returncode != 0:
        status = (
            "pending_outcomes"
            if _is_pending_resolution_error(resolve_message)
            else "failed_resolve"
        )
        payload = status_payload(
            **common_status_kwargs,
            status=status,
            message=resolve_message,
        )
        write_json(status_path, payload)
        if status == "pending_outcomes":
            return (2 if config.fail_on_pending else 0), payload
        return 1, payload

    resolution_counts = read_resolution_counts(resolution_path)
    verify = verify_resolved_artifact(
        config,
        resolved_dir=resolved_dir,
        verification_path=verification_path,
        source_manifest=source_manifest,
        date=date,
    )
    terminal_void_count = resolution_counts["terminal_void_count"]
    verified_status = (
        "resolved_with_voids"
        if verify.returncode == 0 and terminal_void_count
        else "resolved_verified"
        if verify.returncode == 0
        else "failed_verify"
    )
    payload = status_payload(
        **common_status_kwargs,
        status=verified_status,
        message=(resolve_message + "\n" + verify.stdout + verify.stderr).strip(),
        missing_count=resolution_counts["missing_count"],
        terminal_void_count=terminal_void_count,
    )
    write_json(status_path, payload)
    write_json(resolved_dir / "resolve_status.json", payload)
    return (0 if verify.returncode == 0 else 1), payload


def resolve_once(config: ResolveConfig) -> tuple[int, dict[str, Any]]:
    production_head = git_head(config.production_root)
    preoutcome_root = resolve_under(config.production_root, config.preoutcome_root)
    dates = list(config.dates) if config.dates else discover_dates(preoutcome_root)

    if not dates:
        payload = status_payload(
            config=config,
            status="no_preoutcome_artifacts",
            message="no preoutcome artifact manifests found",
            date=None,
            production_head=production_head,
        )
        return 0, payload

    statuses: list[dict[str, Any]] = []
    exit_code = 0
    for date in dates:
        code, payload = resolve_artifact_date(
            config,
            date=date,
            production_head=production_head,
        )
        statuses.append(payload)
        if code not in (0, 2):
            exit_code = 1
        elif code == 2 and config.fail_on_pending and exit_code == 0:
            exit_code = 2

    aggregate = status_payload(
        config=config,
        status="completed" if exit_code == 0 else "failed",
        message=f"processed {len(statuses)} live-forward artifact date(s)",
        date=None,
        production_head=production_head,
    )
    aggregate["statuses"] = statuses
    aggregate["status_counts"] = {
        status: sum(1 for item in statuses if item["status"] == status)
        for status in sorted({item["status"] for item in statuses})
    }
    return exit_code, aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        action="append",
        default=[],
        help="YYYY-MM-DD artifact date to resolve. Repeatable. Defaults to scanning.",
    )
    parser.add_argument("--production-root", type=Path, default=DEFAULT_PRODUCTION_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--preoutcome-root", type=Path, default=DEFAULT_PREOUTCOME_ROOT)
    parser.add_argument("--resolved-root", type=Path, default=DEFAULT_RESOLVED_ROOT)
    parser.add_argument("--status-root", type=Path, default=DEFAULT_STATUS_ROOT)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fail-on-pending",
        action="store_true",
        help="Return exit code 2 instead of 0 for pending preoutcome/outcome data.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ResolveConfig(
        production_root=args.production_root,
        python=args.python,
        candidate=args.candidate,
        preoutcome_root=args.preoutcome_root,
        resolved_root=args.resolved_root,
        status_root=args.status_root,
        data_dir=args.data_dir,
        top_n=args.top_n,
        dates=tuple(args.date),
        overwrite=args.overwrite,
        fail_on_pending=args.fail_on_pending,
    )
    exit_code, payload = resolve_once(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
