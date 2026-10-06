#!/usr/bin/env python3
"""C1 job guard: the CPU cap holds across a job's whole cgroup (Eric 2026-10-05, register row C1-4b-deferral; C1
infrastructure review r1 B1/B4/B7).

Every C1 box job runs as `systemd-run --user -p Delegate=yes ... <python> guard.py --cpu-seconds N --pause-dir <c1>
--unit <unit> -- <cmd>`. With the delegated cgroup the guard splits the unit in two leaves before the job starts:
- **`guard/`** holds this process. It runs at normal priority.
- **`payload/`** holds the job. The guard applies nice 10 to it at exec. Every descendant stays in this cgroup,
  including those that start a new session.

It checks that it can read the unit's `cpu.stat` and that `payload/cgroup.kill` exists with write access (no test
write is made); if not, the job never starts.

**Confinement precondition (stated, not enforced):** the job's work must stay in the `payload/` subtree. It must not
move processes elsewhere in the delegated unit, nor ask the user manager for another unit. Nested cgroups inside
`payload/` are covered: `cgroup.kill` and `cgroup.events` are recursive.

**The cap.** The guard polls the unit's cumulative CPU (`cpu.stat` usage_usec: guard plus job plus all children)
every POLL_S seconds, and acts at `budget - cores * (POLL_S + SLACK_S)`. On a trip it first kills the payload cgroup
(`payload/cgroup.kill`), with no I/O before the kill. Only then does it record anything.

**The bound** assumes the guard is scheduled within SLACK_S of its sleep. It runs at nice 0 while the payload runs at
nice 10. That assumption is not proved here. Any overshoot past the budget is still recorded: a final total over
budget is reported as an overrun, which pauses C1.

**Emptiness before the measurement** (infra review r2 N5): after the job leader exits, or after a kill, the guard
reads `payload/cgroup.events` (`populated`, recursive). A populated payload is killed. If it is not confirmed empty
within CONFIRM_S, the result is `guard-error` (an enforcement failure that pauses C1), not a clean exit.

**Terminal receipt.** On every exit path the guard can run, it writes a durable `TERMINAL_<unit>.json`: the result
(`exit` with the job's code / `overrun` / `oom` / `terminated` / `guard-error`), the unit's final CPU total, and the
budget. The launcher reconciles every launched unit's `PENDING_` record against this receipt:
- every result other than a clean `exit` pauses all of C1;
- a missing receipt does too (the guard was killed or failed before writing it).

If the guard itself dies, systemd stops the unit and kills the payload with it (KillMode=control-group).

Stdlib only and run by absolute path, so it works from any job's working directory.
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

POLL_S = 2.0
SLACK_S = 3.0
CONFIRM_S = 10.0
OVERRUN_EXIT = 86
GUARD_ERROR_EXIT = 87
PAUSE_RESULTS = ("overrun", "oom", "terminated", "guard-error")


class GuardError(RuntimeError):
    pass


def _proc_cgroup_text() -> str:
    return Path("/proc/self/cgroup").read_text()


def own_cgroup(proc_cgroup_text: str, root: Path = Path("/sys/fs/cgroup")) -> Path:
    """The cgroup v2 directory of this process (the `0::<path>` line of /proc/self/cgroup)."""
    for line in proc_cgroup_text.splitlines():
        if line.startswith("0::"):
            return root / line[3:].lstrip("/")
    raise GuardError("no cgroup v2 entry in /proc/self/cgroup: the cumulative CPU cap cannot be enforced")


def cgroup_cpu_seconds(cg: Path) -> float:
    for line in (cg / "cpu.stat").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "usage_usec":
            return int(parts[1]) / 1e6
    raise GuardError(f"{cg}/cpu.stat has no usage_usec")


def oom_kills(cg: Path) -> int:
    f = cg / "memory.events"
    if not f.exists():
        return 0
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "oom_kill":
            return int(parts[1])
    return 0


def populated(cg: Path) -> bool:
    """cgroup v2 `cgroup.events` populated: 1 while any process lives anywhere in the subtree."""
    for line in (cg / "cgroup.events").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "populated":
            return parts[1] == "1"
    raise GuardError(f"{cg}/cgroup.events has no populated field")


def confirm_empty(payload: Path, *, timeout: float | None = None, sleep=None) -> bool:
    """Kill the payload subtree if anything is left, then wait for it to be empty. False = not confirmed."""
    timeout = CONFIRM_S if timeout is None else timeout
    sleep = sleep or time.sleep
    if not populated(payload):
        return True
    (payload / "cgroup.kill").write_text("1")
    waited = 0.0
    while waited < timeout:
        if not populated(payload):
            return True
        sleep(0.2)
        waited += 0.2
    return not populated(payload)


def act_at(budget: float, ncpu: int) -> float:
    return budget - ncpu * (POLL_S + SLACK_S)


def min_budget(ncpu: int) -> float:
    """The smallest budget the guard can enforce with headroom: twice its action margin."""
    return 2 * ncpu * (POLL_S + SLACK_S)


def prepare(cg: Path, pid: int) -> Path:
    """Split the unit into guard/ and payload/ leaves, move the guard, and prove the kill capability before the job
    starts. Returns the payload cgroup."""
    try:
        cgroup_cpu_seconds(cg)
        for leaf in ("guard", "payload"):
            (cg / leaf).mkdir(exist_ok=True)
        (cg / "guard" / "cgroup.procs").write_text(str(pid))
    except OSError as e:
        raise GuardError(f"cannot split the unit cgroup (is Delegate=yes set?): {e}") from e
    killer = cg / "payload" / "cgroup.kill"
    if not (killer.exists() and os.access(killer, os.W_OK)):
        raise GuardError(f"{killer} is missing or not writable: whole-cgroup termination is unavailable")
    return cg / "payload"


def write_receipt(path: Path, rec: dict) -> None:
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


def watch(proc, read_usage, *, threshold: float, poll: float, kill, sleep=time.sleep) -> tuple[str, int | None]:
    """Poll until the job exits (("exit", code)) or the cumulative CPU reaches the threshold (kill first, then
    ("overrun", None)). Nothing is written before the kill."""
    while True:
        if read_usage() >= threshold:
            kill()
            return "overrun", None
        rc = proc.poll()
        if rc is not None:
            return "exit", rc
        sleep(poll)


def classify(kind: str, rc: int | None, *, final_cpu: float, budget: float, oom_before: int, oom_after: int) -> str:
    """The terminal result: an OOM kill or a total over budget is a pause even when the job itself exited."""
    if kind == "overrun" or final_cpu > budget:
        return "overrun"
    if oom_after > oom_before:
        return "oom"
    return "exit"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-seconds", type=float, required=True)
    ap.add_argument("--pause-dir", type=Path, required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    ncpu = os.cpu_count() or 1
    started = datetime.now(timezone.utc).isoformat()
    receipt = args.pause_dir / f"TERMINAL_{args.unit}.json"
    base = {"unit": args.unit, "budget_seconds": args.cpu_seconds, "act_at_seconds": act_at(args.cpu_seconds, ncpu),
            "poll_s": POLL_S, "slack_s": SLACK_S, "ncpu": ncpu, "started_utc": started}

    def finish(result: str, rc, cpu, **extra) -> int:
        write_receipt(receipt, {**base, "result": result, "rc": rc, "cpu_seconds": cpu,
                                "ended_utc": datetime.now(timezone.utc).isoformat(), **extra})
        return {"exit": rc if isinstance(rc, int) and rc >= 0 else 1, "overrun": OVERRUN_EXIT}.get(result, GUARD_ERROR_EXIT)

    if not command or not (args.cpu_seconds >= min_budget(ncpu)):
        return finish("guard-error", None, None, reason=f"a command and --cpu-seconds >= {min_budget(ncpu)} are required")
    try:
        cg = own_cgroup(_proc_cgroup_text())
        payload = prepare(cg, os.getpid())
    except (GuardError, OSError) as e:
        return finish("guard-error", None, None, reason=str(e))
    oom_before = oom_kills(cg)

    def enter_payload() -> None:                   # runs in the child before exec
        try:
            (payload / "cgroup.procs").write_text(str(os.getpid()))
            os.nice(10)
        except BaseException as e:                 # make the cause visible: Popen reports only "preexec_fn"
            os.write(2, f"guard preexec failed: {type(e).__name__}: {e}\n".encode())
            raise

    def kill() -> None:
        (payload / "cgroup.kill").write_text("1")
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass

    try:
        proc = subprocess.Popen(command, preexec_fn=enter_payload, start_new_session=True)
    except (OSError, subprocess.SubprocessError) as e:
        return finish("guard-error", None, cgroup_cpu_seconds(cg), reason=f"the job could not start: {e}")

    def on_term(signum, _frame):                   # systemd stop or RuntimeMaxSec: kill the job, then record
        kill()
        empty = confirm_empty(payload)
        sys.exit(finish("terminated", None, cgroup_cpu_seconds(cg), signal=signum, payload_empty=empty))
    signal.signal(signal.SIGTERM, on_term)

    try:
        kind, rc = watch(proc, lambda: cgroup_cpu_seconds(cg), threshold=act_at(args.cpu_seconds, ncpu), poll=POLL_S,
                         kill=kill)
    except Exception as e:  # noqa: BLE001 - any monitor failure stops the job and pauses C1
        kill()
        return finish("guard-error", None, cgroup_cpu_seconds(cg), reason=f"monitor failed: {type(e).__name__}: {e}")
    leftover = populated(payload)                  # recursive: nested payload cgroups included
    if not confirm_empty(payload):
        return finish("guard-error", rc, cgroup_cpu_seconds(cg), reason="the payload was not confirmed empty after kill",
                      leftover=leftover)
    final = cgroup_cpu_seconds(cg)
    result = classify(kind, rc, final_cpu=final, budget=args.cpu_seconds, oom_before=oom_before, oom_after=oom_kills(cg))
    if result == "overrun":
        print(f"guard: CPU overrun ({final:.1f} s; budget {args.cpu_seconds:.0f} s): job killed, C1 paused",
              file=sys.stderr, flush=True)
    return finish(result, rc, final, leftover=leftover)


if __name__ == "__main__":
    sys.exit(main())
