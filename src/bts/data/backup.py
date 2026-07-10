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
    import fcntl

    from bts.util import atomic_write_text

    health_dir = Path(repo_root) / "data" / "health_state"
    health_dir.mkdir(parents=True, exist_ok=True)
    # flock serializes the read-modify-write: a slow ops run finishing
    # alongside an archive run must not clobber the other's entry
    # (Codex review I8). The final write stays atomic (tmp+rename).
    lock_path = health_dir / (STATUS_FILENAME + ".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
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

    missing = sorted(p for p in bset.paths if not (repo_root / p).exists())

    def finish(entry: dict) -> dict:
        entry = {
            "set": set_name,
            "started_at": started.isoformat(),
            "finished_at": now_fn().isoformat(),
            **entry,
        }
        if "last_success_at" not in entry:
            entry["last_success_at"] = prior.get("last_success_at")
        # Carry the last SUCCESSFUL snapshot id across failures (Codex round-2
        # R1): without it, one failed run made the drill fall back to
        # 'latest --tag ops' — which can be the failed run's partial snapshot.
        if "last_success_snapshot_id" not in entry:
            carried = prior.get("last_success_snapshot_id")
            if carried is None and prior.get("ok") and prior.get("snapshot_id"):
                carried = prior.get("snapshot_id")
            entry["last_success_snapshot_id"] = carried
        _write_status_entry(repo_root, set_name, entry)
        return entry

    if missing:
        # Every path in a set is REQUIRED. A "success" that silently omitted
        # data/picks would advance last_success_at and keep health green
        # while the irreplaceable payload went unprotected (Codex review I1).
        return finish({
            "ok": False,
            "error": f"required backup paths missing under {repo_root}: {missing}",
        })

    try:
        renv = restic_env(env)
    except RuntimeError as e:
        return finish({"ok": False, "error": str(e)})

    bin_ = restic_bin(env)
    backup_cmd = [
        bin_, "backup",
        *[str(repo_root / p) for p in bset.paths],
        "--tag", set_name,
        "--exclude", "*.lock",
        "--json",
    ]
    try:
        result = runner(
            backup_cmd, env=renv, capture_output=True, text=True,
            timeout=DEFAULT_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        # A hung restic (stale repo lock, network stall) must still leave a
        # failed status entry — the freshness check measures data-loss
        # exposure from this file, not from cron tracebacks.
        return finish({
            "ok": False,
            "error": f"restic backup timed out after {DEFAULT_TIMEOUT_SEC}s",
        })
    except OSError as e:
        # Missing/unexecutable restic binary (cron installed before the
        # install script ran). Without a status entry the absent-file
        # convention keeps health silent forever (Codex review I3).
        return finish({
            "ok": False,
            "error": f"restic could not be executed ({bin_}): {e}",
        })
    if result.returncode != 0:
        return finish({
            "ok": False,
            "rc": result.returncode,
            "error": (result.stderr or result.stdout or "").strip()[-500:],
        })

    summary = _parse_summary(result.stdout or "")

    forget_cmd = [bin_, "forget", "--tag", set_name, *bset.retention]
    try:
        forget = runner(
            forget_cmd, env=renv, capture_output=True, text=True,
            timeout=DEFAULT_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        forget = subprocess.CompletedProcess(
            forget_cmd, returncode=-1, stdout="", stderr="forget timed out",
        )
    finished = now_fn()

    entry = {
        "ok": True,
        "rc": 0,
        "last_success_at": finished.isoformat(),
        "snapshot_id": summary.get("snapshot_id"),
        "last_success_snapshot_id": summary.get("snapshot_id"),
        "data_added": summary.get("data_added"),
        "total_files_processed": summary.get("total_files_processed"),
        "duration_sec": (finished - started).total_seconds(),
        "paths": list(bset.paths),
    }
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

    streaks = list(root.rglob("streak.json"))
    if not streaks:
        problems.append("streak.json not found in restore")
    else:
        try:
            json.loads(streaks[0].read_text())
        except (OSError, ValueError) as e:
            problems.append(f"streak.json unreadable/corrupt: {e}")

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
    """Restore the last SUCCESSFUL ops snapshot to `target` and verify it.

    Proves the claims the F5 audit finding demands: streak, the manual saver
    flag, the contest ledger, and decision provenance actually come back
    from R2. Two hardenings from the Codex review (I4): the target must be
    empty/new (stale files from an earlier drill could pass verification a
    partial snapshot never earned), and when backup_status.json records a
    successful snapshot_id we restore THAT — `latest --tag ops` could select
    a partial (rc=3) snapshot restic created during a run we recorded as
    failed.
    """
    target = Path(target)
    if target.exists() and any(target.iterdir()):
        return {
            "ok": False,
            "problems": [f"target {target} is not empty — a reused drill target "
                         f"can mask omissions in the restored snapshot"],
        }
    target.mkdir(parents=True, exist_ok=True)
    try:
        renv = restic_env(env)
    except RuntimeError as e:
        return {"ok": False, "problems": [str(e)]}

    ops_status = read_status(repo_root).get("ops", {})
    # last_success_snapshot_id survives failed runs (round-2 R1); legacy
    # fallback: an ok entry's snapshot_id.
    snapshot = ops_status.get("last_success_snapshot_id")
    if not snapshot and ops_status.get("ok"):
        snapshot = ops_status.get("snapshot_id")
    cmd = [restic_bin(env), "restore", snapshot or "latest"]
    if snapshot is None:
        cmd += ["--tag", "ops"]
    cmd += ["--target", str(target)]
    result = runner(
        cmd, env=renv, capture_output=True, text=True, timeout=DEFAULT_TIMEOUT_SEC,
    )
    if result.returncode != 0:
        return {
            "ok": False,
            "problems": [f"restic restore failed: {(result.stderr or '').strip()[-300:]}"],
        }

    problems = verify_restored_ops_tree(target)
    return {
        "ok": not problems,
        "problems": problems,
        "target": str(target),
        "snapshot": snapshot or "latest",
    }
