"""Live in-flight audit progress scanner.

Powers /api/audit-progress. The local mirror at ``data/<provider>_results/<dir>/``
is EMPTY during a run — ``retrieve_one`` in ``scripts/audit_driver.py`` rsyncs only
at final teardown. So live progress has to come from SSH to each box, reading
completion markers in ``/root/audit.log``. This reuses the bts-user SSH key at
``~bts/.ssh/id_ed25519`` that was distributed to every audit box's
``authorized_keys`` during provisioning (see ``audit_driver.py:81`` cloud-init).
"""
from __future__ import annotations

import concurrent.futures
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",
]


PROGRESS_QUERY = r"""
STATE=RUN
[ -f /root/audit.done ] && STATE=DONE
COUNT=$(grep -c '=== seed=.* done' /root/audit.log 2>/dev/null || echo 0)
printf 'STATE:%s\n' "$STATE"
printf 'COUNT:%s\n' "$COUNT"
printf 'LAST:%s\n' "$(tail -1 /root/audit.log 2>/dev/null | head -c 140)"
"""


SshRunner = Callable[[str, str, int], subprocess.CompletedProcess]


def _default_ssh_runner(ip: str, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *SSH_OPTS, f"root@{ip}", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def load_boxes(audit_dir: Path) -> list[dict]:
    """Read ``boxes.json`` from an audit output directory."""
    path = audit_dir / "boxes.json"
    if not path.exists():
        raise FileNotFoundError(f"boxes.json not found in {audit_dir}")
    return json.loads(path.read_text())


def _parse_seeds_file(seeds_file: Path) -> list[int]:
    """Parse comma- or newline-separated seeds. Matches the format of
    ``scripts/audit_seeds_*.txt`` where the default48 file is newline-separated
    and the extension_n100 file is a single comma-separated line."""
    raw = seeds_file.read_text().strip()
    tokens = re.split(r"[,\s]+", raw)
    return [int(t) for t in tokens if t]


def _distribute_seeds(box_names: list[str], seeds: list[int]) -> dict[str, list[int]]:
    """Round-robin distribution, matches ``scripts/audit_driver.distribute_seeds``
    so reconstruction is faithful to what an already-running audit scheduled."""
    queues: dict[str, list[int]] = {name: [] for name in box_names}
    for i, s in enumerate(seeds):
        queues[box_names[i % len(box_names)]].append(s)
    return queues


def _parse_progress_output(stdout: str) -> dict:
    """Parse the three-line STATE/COUNT/LAST output from PROGRESS_QUERY."""
    fields: dict[str, str] = {}
    for line in stdout.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fields[k.strip()] = v
    state_raw = fields.get("STATE", "").strip()
    count_raw = fields.get("COUNT", "0").strip()
    last = fields.get("LAST", "")
    try:
        count = int(count_raw)
    except ValueError:
        count = 0
    state = "done" if state_raw == "DONE" else "running"
    return {"state": state, "completed_seeds": count, "last_seed_event": last}


def _scan_one(
    box: dict,
    ssh_runner: SshRunner,
    timeout: int,
) -> dict:
    """SSH to one box and parse progress. Never raises — errors are captured
    in the returned dict so one bad box can't poison the whole scan."""
    name = box["name"]
    ip = box["ipv4"]
    try:
        r = ssh_runner(ip, PROGRESS_QUERY, timeout)
    except subprocess.TimeoutExpired as e:
        return {
            "name": name,
            "ip": ip,
            "state": "error",
            "completed_seeds": None,
            "last_seed_event": "",
            "error": f"ssh timeout after {e.timeout}s",
        }
    except Exception as e:  # pragma: no cover - defense-in-depth
        return {
            "name": name,
            "ip": ip,
            "state": "error",
            "completed_seeds": None,
            "last_seed_event": "",
            "error": f"{type(e).__name__}: {e}",
        }

    if r.returncode != 0:
        return {
            "name": name,
            "ip": ip,
            "state": "error",
            "completed_seeds": None,
            "last_seed_event": "",
            "error": (r.stderr or "").strip()[:200] or f"ssh exit {r.returncode}",
        }

    parsed = _parse_progress_output(r.stdout)
    return {
        "name": name,
        "ip": ip,
        "state": parsed["state"],
        "completed_seeds": parsed["completed_seeds"],
        "last_seed_event": parsed["last_seed_event"],
        "error": None,
    }


def scan_audit_progress(
    audit_dir: Path,
    seeds_file: Optional[Path] = None,
    ssh_runner: Optional[SshRunner] = None,
    max_workers: int = 8,
    timeout: int = 20,
) -> dict:
    """Scan an in-flight audit by SSHing to every box in its ``boxes.json``.

    Parameters
    ----------
    audit_dir : Path
        Directory containing ``boxes.json`` (e.g. ``data/vultr_results/audit_ext_n100_v4``).
    seeds_file : Path | None
        If supplied, parsed and round-robin-distributed to derive per-box
        ``expected_seeds``. Without it, expected counts are ``None`` and the
        overall percentage is omitted.
    ssh_runner : callable | None
        Injectable for tests. Defaults to real ``ssh`` subprocess.
    max_workers : int
        ThreadPool width. 8 covers a 26-box fleet in ~3 rounds worst case.
    timeout : int
        Per-box SSH timeout in seconds.
    """
    if ssh_runner is None:
        ssh_runner = _default_ssh_runner

    boxes = load_boxes(audit_dir)
    box_names = [b["name"] for b in boxes]

    expected_per_box: dict[str, Optional[int]] = {n: None for n in box_names}
    total_expected: Optional[int] = None
    if seeds_file is not None:
        seeds = _parse_seeds_file(seeds_file)
        queues = _distribute_seeds(box_names, seeds)
        expected_per_box = {n: len(queues[n]) for n in box_names}
        total_expected = len(seeds)

    scanned: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_scan_one, b, ssh_runner, timeout): b["name"] for b in boxes}
        for fut in concurrent.futures.as_completed(futs):
            scanned[futs[fut]] = fut.result()

    # Rebuild in boxes.json order + attach expected_seeds
    box_results: list[dict] = []
    for name in box_names:
        row = scanned[name]
        row["expected_seeds"] = expected_per_box[name]
        box_results.append(row)

    completed = sum(
        (b["completed_seeds"] or 0)
        for b in box_results
        if b["completed_seeds"] is not None
    )
    boxes_done = sum(1 for b in box_results if b["state"] == "done")
    pct: Optional[float] = None
    if total_expected:
        pct = round(100.0 * completed / total_expected, 1)

    return {
        "audit_dir": audit_dir.name,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "boxes": box_results,
        "overall": {
            "completed": completed,
            "expected": total_expected,
            "boxes_done": boxes_done,
            "boxes_total": len(box_results),
            "pct_seeds": pct,
        },
    }
