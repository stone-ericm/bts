#!/usr/bin/env python3
"""C1 job guard: the CPU cap holds across a job's whole cgroup (Eric 2026-10-05, register row C1-4b-deferral; code
review r3 B4).

Every C1 box job runs as `systemd-run ... <python> guard.py --cpu-seconds N --pause-dir <c1> --unit <unit> -- <cmd>`.
`LimitCPU` is a per-process rlimit: a job that spawns children can exceed it in total. The guard closes that gap:
- **What it measures:** it runs the command and polls its own cgroup v2 `cpu.stat` (`usage_usec`) every POLL_S
  seconds. That is the cumulative CPU of every process in the unit: the guard, the job and all of its children.
- **When it acts:** at `budget - cores * POLL_S`. Usage can grow by at most cores x POLL_S between polls, so the
  total stays under the budget.
- **What it does on an overrun:** it first writes a durable `OVERRUN_<unit>.json` into the C1 pause directory; the
  launcher then refuses every C1 job until Eric's recorded decision releases it. Then it kills the whole cgroup
  (`cgroup.kill`, the guard included); without `cgroup.kill` it kills the job's process group.
- **Without an overrun:** it exits with the job's own exit code.

Stdlib only and run by absolute path, so it works from any job's working directory (for example a generator
worktree that has no `scripts/audit/c1`).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OVERRUN_EXIT = 86
POLL_S = 2.0


def own_cgroup(proc_cgroup_text: str, root: Path = Path("/sys/fs/cgroup")) -> Path:
    """The cgroup v2 directory of this process (the `0::<path>` line of /proc/self/cgroup)."""
    for line in proc_cgroup_text.splitlines():
        if line.startswith("0::"):
            return root / line[3:].lstrip("/")
    raise RuntimeError("no cgroup v2 entry in /proc/self/cgroup: the cumulative CPU cap cannot be enforced")


def cgroup_cpu_seconds(cg: Path) -> float:
    for line in (cg / "cpu.stat").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "usage_usec":
            return int(parts[1]) / 1e6
    raise RuntimeError(f"{cg}/cpu.stat has no usage_usec")


def write_marker(path: Path, rec: dict) -> None:
    """Durable: temp file fsynced, renamed, directory fsynced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(rec, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def watch(proc, read_usage, *, budget: float, ncpu: int, poll: float, on_overrun, kill, sleep=time.sleep) -> int:
    """Poll until the job exits (its exit code) or its cumulative CPU reaches the action threshold (OVERRUN_EXIT,
    after the durable marker and the kill). A total found over budget at exit is still recorded as an overrun."""
    act_at = budget - ncpu * poll
    while True:
        used = read_usage()
        if used >= act_at:
            on_overrun(used)
            kill()
            return OVERRUN_EXIT
        rc = proc.poll()
        if rc is not None:
            used = read_usage()
            if used > budget:
                on_overrun(used)
                return OVERRUN_EXIT
            return rc
        sleep(poll)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-seconds", type=float, required=True)
    ap.add_argument("--pause-dir", type=Path, required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not (args.cpu_seconds > 0):
        print("guard: a command and a positive --cpu-seconds are required", file=sys.stderr)
        return 2
    cg = own_cgroup(Path("/proc/self/cgroup").read_text())
    read = lambda: cgroup_cpu_seconds(cg)  # noqa: E731
    read()                                         # fail before starting the job if the total cannot be read
    proc = subprocess.Popen(command, start_new_session=True)

    def kill() -> None:
        killer = cg / "cgroup.kill"
        if killer.exists():
            killer.write_text("1")                 # every process in the unit, this guard included
        os.killpg(proc.pid, signal.SIGKILL)

    def forward(signum, _frame):                  # systemd stop: take the job's process group down with us
        try:
            os.killpg(proc.pid, signum)
        finally:
            sys.exit(128 + signum)
    signal.signal(signal.SIGTERM, forward)

    def on_overrun(used: float) -> None:
        write_marker(args.pause_dir / f"OVERRUN_{args.unit}.json",
                     {"unit": args.unit, "source": "guard", "cpu_seconds": round(used, 3),
                      "budget_seconds": args.cpu_seconds, "written_utc": datetime.now(timezone.utc).isoformat()})
        print(f"guard: CPU overrun ({used:.1f} s of {args.cpu_seconds:.0f} s): C1 paused", file=sys.stderr, flush=True)

    return watch(proc, read, budget=args.cpu_seconds, ncpu=os.cpu_count() or 1, poll=POLL_S,
                 on_overrun=on_overrun, kill=kill)


if __name__ == "__main__":
    sys.exit(main())
