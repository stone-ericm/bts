"""Restic backup of irreplaceable operational state to R2 (audit F5).

data/picks (daily decisions, contest ledger, delivery IDs, skip-shadow
markers, the MANUAL saver flag) and data/health_state exist only on the
production box: gitignored, excluded from the R2 artifact sync, no
provider snapshot verified. Box loss = irrecoverable operational state —
the saver flag is deliberately not inferable (see ARCHITECTURE, saver
section). This module wraps restic (encrypted, versioned, deduplicated)
against the existing R2 bucket under a `restic/` prefix.

Two backup sets:
    ops     — data/picks + data/health_state, small and hot (cron every 3h)
    archive — data/leaderboard + data/hetzner_results + data/external,
              research data that exists nowhere else (cron daily)

Secrets flow exclusively via the subprocess environment (never argv):
R2_* creds are mapped to the AWS_* names restic's S3 backend expects,
RESTIC_PASSWORD comes from .env. Each run writes a per-set status entry
to data/health_state/backup_status.json for the backup_freshness health
source — the scheduler never invokes restic itself.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

DEFAULT_BUCKET = "bts-backup-data"
REPO_PREFIX = "restic"
STATUS_FILENAME = "backup_status.json"
DEFAULT_TIMEOUT_SEC = 3600


@dataclass(frozen=True)
class BackupSet:
    name: str
    paths: tuple[str, ...]
    retention: tuple[str, ...]


BACKUP_SETS: dict[str, BackupSet] = {
    "ops": BackupSet(
        name="ops",
        paths=("data/picks", "data/health_state"),
        retention=(
            "--keep-hourly", "48",
            "--keep-daily", "30",
            "--keep-weekly", "26",
        ),
    ),
    "archive": BackupSet(
        name="archive",
        paths=("data/leaderboard", "data/hetzner_results", "data/external"),
        retention=(
            "--keep-daily", "14",
            "--keep-weekly", "12",
            "--keep-monthly", "12",
        ),
    ),
}


def restic_bin(env: dict | None = None) -> str:
    env = env or {}
    if env.get("BTS_RESTIC_BIN"):
        return env["BTS_RESTIC_BIN"]
    found = shutil.which("restic")
    if found:
        return found
    return str(Path.home() / ".local" / "bin" / "restic")


def restic_env(env: dict) -> dict:
    """Build the restic subprocess environment from BTS env vars.

    Raises RuntimeError naming the first missing variable — a backup that
    silently runs against the wrong repo is worse than one that fails loudly.
    """
    password = env.get("RESTIC_PASSWORD")
    if not password:
        raise RuntimeError(
            "RESTIC_PASSWORD not set — required for restic backup (see .env)"
        )

    repository = env.get("BTS_RESTIC_REPOSITORY")
    if not repository:
        account = env.get("R2_ACCOUNT_ID")
        if not account:
            raise RuntimeError(
                "R2_ACCOUNT_ID not set and no BTS_RESTIC_REPOSITORY override"
            )
        bucket = env.get("R2_BUCKET", DEFAULT_BUCKET)
        repository = (
            f"s3:https://{account}.r2.cloudflarestorage.com/{bucket}/{REPO_PREFIX}"
        )

    access_key = env.get("R2_ACCESS_KEY_ID")
    secret = env.get("R2_SECRET_ACCESS_KEY")
    if repository.startswith("s3:") and not (access_key and secret):
        raise RuntimeError(
            "R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY not set — required for "
            "the S3 (R2) restic backend"
        )

    out = dict(env)
    out["RESTIC_REPOSITORY"] = repository
    out["RESTIC_PASSWORD"] = password
    if access_key:
        out["AWS_ACCESS_KEY_ID"] = access_key
    if secret:
        out["AWS_SECRET_ACCESS_KEY"] = secret
    return out


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def read_status(repo_root: Path) -> dict:
    path = Path(repo_root) / "data" / "health_state" / STATUS_FILENAME
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _write_status_entry(repo_root: Path, set_name: str, entry: dict) -> None:
    from bts.util import atomic_write_text

    health_dir = Path(repo_root) / "data" / "health_state"
    health_dir.mkdir(parents=True, exist_ok=True)
    status = read_status(repo_root)
    status[set_name] = entry
    atomic_write_text(health_dir / STATUS_FILENAME, json.dumps(status, indent=2))


def _parse_summary(stdout: str) -> dict:
    """Extract restic's --json summary line (last message_type=summary)."""
    summary = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if obj.get("message_type") == "summary":
            summary = obj
    return summary


def run_backup(
    set_name: str,
    repo_root: Path,
    *,
    env: dict,
    runner: Callable = subprocess.run,
    now_fn: Callable[[], datetime] = _now_utc,
) -> dict:
    """Run one restic backup set + retention forget. Returns the status entry.

    Never raises on restic failure — writes ok=False status (preserving the
    prior last_success_at so the freshness check measures real data-loss
    exposure) and lets the caller decide the exit code.
    """
    bset = BACKUP_SETS[set_name]
    repo_root = Path(repo_root)
    started = now_fn()
    prior = read_status(repo_root).get(set_name, {})

    existing = [p for p in bset.paths if (repo_root / p).exists()]
    skipped = sorted(set(bset.paths) - set(existing))

    def finish(entry: dict) -> dict:
        entry = {
            "set": set_name,
            "started_at": started.isoformat(),
            "finished_at": now_fn().isoformat(),
            **entry,
        }
        if "last_success_at" not in entry:
            entry["last_success_at"] = prior.get("last_success_at")
        _write_status_entry(repo_root, set_name, entry)
        return entry

    if not existing:
        return finish({
            "ok": False,
            "error": f"all paths missing under {repo_root}: {sorted(bset.paths)}",
        })

    try:
        renv = restic_env(env)
    except RuntimeError as e:
        return finish({"ok": False, "error": str(e)})

    bin_ = restic_bin(env)
    backup_cmd = [
        bin_, "backup",
        *[str(repo_root / p) for p in existing],
        "--tag", set_name,
        "--exclude", "*.lock",
        "--json",
    ]
    result = runner(
        backup_cmd, env=renv, capture_output=True, text=True,
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    if result.returncode != 0:
        return finish({
            "ok": False,
            "rc": result.returncode,
            "error": (result.stderr or result.stdout or "").strip()[-500:],
        })

    summary = _parse_summary(result.stdout or "")

    forget_cmd = [bin_, "forget", "--tag", set_name, *bset.retention]
    forget = runner(
        forget_cmd, env=renv, capture_output=True, text=True,
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    finished = now_fn()

    entry = {
        "ok": True,
        "rc": 0,
        "last_success_at": finished.isoformat(),
        "snapshot_id": summary.get("snapshot_id"),
        "data_added": summary.get("data_added"),
        "total_files_processed": summary.get("total_files_processed"),
        "duration_sec": (finished - started).total_seconds(),
        "paths": existing,
    }
    if skipped:
        entry["skipped_paths"] = skipped
    if forget.returncode != 0:
        # Retention failure doesn't invalidate the snapshot; note it.
        entry["forget_error"] = (forget.stderr or "").strip()[-300:]
    return finish(entry)


def run_prune(
    *,
    env: dict,
    runner: Callable = subprocess.run,
) -> dict:
    """Reclaim space from forgotten snapshots (weekly cron; IO-heavy)."""
    try:
        renv = restic_env(env)
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    result = runner(
        [restic_bin(env), "prune"],
        env=renv, capture_output=True, text=True, timeout=DEFAULT_TIMEOUT_SEC,
    )
    out = {"ok": result.returncode == 0, "rc": result.returncode}
    if result.returncode != 0:
        out["error"] = (result.stderr or result.stdout or "").strip()[-300:]
    return out


# ------------------------------------------------------------- restore drill

def verify_restored_ops_tree(root: Path) -> list[str]:
    """Verify a restored ops tree proves the F5 recovery claims.

    restic restores with original absolute paths under --target, so search
    recursively rather than assuming the box's directory layout.
    Returns a list of problems (empty = drill passed).
    """
    root = Path(root)
    problems: list[str] = []

    savers = list(root.rglob("saver_state.json"))
    if not savers:
        problems.append("saver_state.json not found in restore")
    else:
        try:
            saver = json.loads(savers[0].read_text())
            state = saver.get("state")
            if not isinstance(state, str) or not state:
                problems.append(f"saver_state.json has invalid state: {state!r}")
        except (OSError, ValueError) as e:
            problems.append(f"saver_state.json unreadable/corrupt: {e}")

    ledgers = list(root.rglob("contest_ledger.jsonl"))
    if not ledgers:
        problems.append("contest_ledger.jsonl not found in restore")
    else:
        try:
            lines = [ln for ln in ledgers[0].read_text().splitlines() if ln.strip()]
            if not lines:
                problems.append("contest_ledger.jsonl is empty")
            else:
                json.loads(lines[-1])
        except (OSError, ValueError) as e:
            problems.append(f"contest_ledger.jsonl unreadable/corrupt: {e}")

    decisions = sorted(root.rglob("decision.json"))
    if not decisions:
        problems.append("no decision.json files found in restore")
    else:
        try:
            latest = json.loads(decisions[-1].read_text())
            if "action" not in latest or "date" not in latest:
                problems.append(f"latest decision.json missing action/date: {decisions[-1]}")
        except (OSError, ValueError) as e:
            problems.append(f"latest decision.json unreadable/corrupt: {e}")

    return problems


def restore_drill(
    *,
    repo_root: Path,
    target: Path,
    env: dict,
    runner: Callable = subprocess.run,
) -> dict:
    """Restore the latest ops snapshot to `target` and verify recoverability.

    Proves the two claims the F5 audit finding demands: the manual saver
    flag and current-day decision provenance actually come back from R2.
    """
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    try:
        renv = restic_env(env)
    except RuntimeError as e:
        return {"ok": False, "problems": [str(e)]}

    cmd = [
        restic_bin(env), "restore", "latest",
        "--tag", "ops",
        "--target", str(target),
    ]
    result = runner(
        cmd, env=renv, capture_output=True, text=True, timeout=DEFAULT_TIMEOUT_SEC,
    )
    if result.returncode != 0:
        return {
            "ok": False,
            "problems": [f"restic restore failed: {(result.stderr or '').strip()[-300:]}"],
        }

    problems = verify_restored_ops_tree(target)
    return {"ok": not problems, "problems": problems, "target": str(target)}
