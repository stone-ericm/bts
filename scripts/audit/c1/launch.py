"""C1 launcher: the only way a C1 job starts on the box (docs/audit/2026-10-04-d4-cycle-proposal.md §§3, 5).

Under one exclusive launch lock (code review r3 B5: two launches cannot race) it sweeps the journal into the compute
ledger, records overruns, and refuses unless:
- the name is valid, and the declared budgets are finite and positive;
- no other C1 unit is active (one job at a time);
- no 403/429 stop or unresolved request receipt is in force;
- **no overrun pause is in force:**
  - an `OVERRUN_<unit>.json` comes from the guard (a CPU overrun) or from a systemd `timeout`/`oom-kill` failure;
  - it pauses every C1 job, not just that one;
  - it is released only by `RESUME_<unit>.json` naming the unit and citing a RULED register row that names it,
    approved by Eric (Eric 2026-10-05, register row C1-4b-deferral);
- a failed earlier unit of the same job has been acknowledged;
- the ledger gate is ok (50 = stop and report until Eric's acknowledgement file exists; 100 = stop; a declared budget
  may not cross 100);
- from 2027-03-01, the scheduler is asleep for longer than the job's declared wall time (fails closed when no sleep
  line is found).

**The unit's limits:** nice 10, MemoryMax=12G, OOMScoreAdjust=1000, RuntimeMaxSec = the declared wall time, and a
LimitCPU per-process backstop. The command runs under `guard.py`, which caps the unit's **cumulative** CPU (all
processes) at min(declared budget, remaining cycle budget).

    python -m scripts.audit.c1.launch status
    python -m scripts.audit.c1.launch run --name r2-build --cpu-hours 2 --max-hours 3 -- <command ...>
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.audit.c1 import ledger

REPO = Path(__file__).resolve().parents[3]          # the C1 worktree the job's code runs from
GUARD = Path(__file__).resolve().with_name("guard.py")   # run by absolute path: works from any job's cwd
PYTHON = sys.executable
REGISTER = REPO / "docs" / "audit" / "2026-09-22-exposure-register.md"
OVERRUN_RESULTS = ("timeout", "oom-kill")             # systemd's own overrun kills (CPU overruns come from the guard)
RESULT_RE = re.compile(r"Failed with result '([^']+)'")
DATA_ROOT = Path.home() / "projects" / "bts" / "data"  # production tree's data: hetzner_results is restic-backed
PROD_ENV = DATA_ROOT.parent / ".env"
CHECKPOINT_ACK_NAME = "CHECKPOINT_50_ACK.json"
JOURNAL_SINCE = "2026-10-04"
GUARD_FROM = datetime(2027, 3, 1, tzinfo=timezone.utc)   # no MLB opener has been earlier; fail closed from here
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
SLEEP_RE = re.compile(r"(?:Sleeping|Idle) until .*\((?:(\d+) min|([\d.]+)h)\)")


def c1_paths(data_root: Path) -> dict:
    d = data_root / "hetzner_results" / "c1"
    return {"ledger": d / "compute_ledger.tsv", "ack": d / CHECKPOINT_ACK_NAME}


def rate_limit_stops(data_root: Path) -> list[str]:
    """Everything that pauses C1: any 403/429 stop marker under the C1 results tree or the static capture, and any
    acquisition receipts directory with an intent that has no completion (an unresolved request: its outcome, which
    may have been a rate limit, is unknown)."""
    c1 = data_root / "hetzner_results" / "c1"
    found = [str(p.relative_to(data_root)) for p in sorted(c1.rglob("STOP_403_429.json"))]
    capture = data_root / "leaderboard" / "static_snapshots" / "_receipts" / "STOP_403_429.json"
    found += [str(capture.relative_to(data_root))] if capture.exists() else []
    for rdir in sorted(c1.rglob("receipts")):
        if not rdir.is_dir():
            continue
        intents, done = [], set()
        for f in sorted(rdir.glob("*.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("kind") == "intent":
                    intents.append(r.get("attempt_id"))
                elif r.get("kind") == "completion":
                    done.add(r.get("attempt_id"))
        open_ = [a for a in intents if a not in done]
        if open_:
            found.append(f"unresolved request receipts: {rdir.parent.relative_to(data_root)} ({len(open_)})")
    return found


def _ts(d: dict) -> datetime:
    return datetime.fromtimestamp(int(d["__REALTIME_TIMESTAMP"]) / 1_000_000, tz=timezone.utc)


def scheduler_wake(lines: list[str]) -> datetime | None:
    """The scheduler's next wake time, only if its latest journal message is a sleep/idle line."""
    entries = [json.loads(l) for l in lines if l.strip()]
    entries = [e for e in entries if str(e.get("MESSAGE") or "").strip()]
    if not entries:
        return None
    last = max(entries, key=_ts)
    m = SLEEP_RE.search(str(last["MESSAGE"]))
    if not m:
        return None
    delta = timedelta(minutes=int(m.group(1))) if m.group(1) else timedelta(hours=float(m.group(2)))
    return _ts(last) + delta


def season_guard(now: datetime, sched_lines: list[str], max_hours: float) -> tuple[bool, str]:
    if now < GUARD_FROM:
        return True, "off-season: no sleep-window requirement"
    wake = scheduler_wake(sched_lines)
    if wake is None:
        return False, "outside the scheduler sleep window: no current sleep/idle line"
    if now + timedelta(hours=max_hours) > wake:
        return False, f"outside the scheduler sleep window: wakes {wake.isoformat()}, job may run {max_hours} h"
    return True, f"scheduler asleep until {wake.isoformat()}"


def record_overruns(c1_dir: Path, lines: list[str]) -> list[str]:
    """Materialize systemd overrun kills (timeout, oom-kill) as durable pause markers, so a pause outlives journal
    retention. Returns the units newly recorded."""
    out = []
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        unit = (d.get("USER_UNIT") or "").removesuffix(".service")
        m = RESULT_RE.search(str(d.get("MESSAGE") or ""))
        if unit.startswith("c1-") and m and m.group(1) in OVERRUN_RESULTS:
            marker = c1_dir / f"OVERRUN_{unit}.json"
            if not marker.exists():
                _write_marker(marker, {"unit": unit, "source": "journal", "result": m.group(1),
                                       "written_utc": datetime.now(timezone.utc).isoformat()})
                out.append(unit)
    return out


def _write_marker(path: Path, rec: dict) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("c1_guard", GUARD)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    g.write_marker(path, rec)


def _register_row(text: str, row_id: str) -> str | None:
    return next((l for l in text.splitlines() if l.startswith(f"| {row_id} |")), None)


def overrun_stops(c1_dir: Path, register_text: str) -> list[str]:
    """Overrun markers without a valid release. A release (`RESUME_<unit>.json`) must name the unit, give a reason,
    say approved_by Eric, and cite a RULED register row whose text names that unit (a generic ruling does not)."""
    stops = []
    for marker in sorted(c1_dir.glob("OVERRUN_*.json")):
        unit = marker.stem.removeprefix("OVERRUN_")
        res = c1_dir / f"RESUME_{unit}.json"
        problems = ["no release"]
        if res.exists():
            try:
                rec = json.loads(res.read_text())
            except ValueError:
                rec = None
            problems = []
            if not isinstance(rec, dict):
                problems.append("unreadable release")
            else:
                if rec.get("unit") != unit:
                    problems.append("does not name the unit")
                if not (isinstance(rec.get("reason"), str) and rec["reason"].strip()):
                    problems.append("no reason")
                if rec.get("approved_by") != "Eric":
                    problems.append("not approved by Eric")
                row = _register_row(register_text, rec.get("register_row")) if isinstance(rec.get("register_row"), str) else None
                if row is None or "**RULED" not in row or unit not in row:
                    problems.append("no RULED register row naming the unit")
        if problems:
            stops.append(f"{unit} ({', '.join(problems)})")
    return stops


@contextlib.contextmanager
def launch_lock(c1_dir: Path):
    """One launcher at a time across sweep, admission and start (code review r3 B5)."""
    c1_dir.mkdir(parents=True, exist_ok=True)
    with open(c1_dir / ".launch.lock", "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("refusing: another C1 launcher holds the launch lock") from None
        yield


def plan_launch(*, name: str, cpu_hours: float, max_hours: float, rows: list[dict], acked: bool,
                active_units: list[str], sched_lines: list[str], now: datetime, cwd: str, command: list[str],
                env_file: str = str(PROD_ENV), log_dir: Path | None = None,
                rate_limit_stops: list[str] | None = None, failed_units: list[str] | None = None,
                acked_failures: list[str] | None = None, overrun_stops: list[str] | None = None,
                pause_dir: Path | None = None) -> dict:
    total = ledger.total_hours(rows)
    reasons = []
    if not NAME_RE.match(name):
        reasons.append(f"invalid name {name!r}: lowercase letters, digits and '-' only")
    for label, v in (("cpu-hours", cpu_hours), ("max-hours", max_hours)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            reasons.append(f"invalid {label} {v!r}: a finite positive number is required")
    if overrun_stops:
        reasons.append("overrun pause in force (all C1 is paused until Eric's recorded decision releases it): "
                       + ", ".join(overrun_stops))
    if not command:
        reasons.append("no command given")
    if active_units:
        reasons.append(f"another C1 job is active: {', '.join(active_units)}")
    if rate_limit_stops:
        reasons.append("403/429 stop in force (C1 is paused until Eric records a decision): "
                       + ", ".join(rate_limit_stops))
    prior = [u for u in (failed_units or []) if u.startswith(f"c1-{name}-") and u not in (acked_failures or [])]
    if prior:
        reasons.append(f"a previous {name} unit failed ({', '.join(prior)}); no automatic retry without an "
                       "acknowledged failure")
    g = ledger.gate(total, cpu_hours, acked) if not any(r.startswith("invalid ") for r in reasons) else "ok"
    if g != "ok":
        reasons.append({"checkpoint": f"checkpoint: {total:.2f} CPU-h >= {ledger.CHECKPOINT_H}; stop and report "
                                      f"until Eric's acknowledgement ({CHECKPOINT_ACK_NAME}) exists",
                        "over_cap": f"over_cap: {total:.2f} + declared {cpu_hours} > {ledger.CAP_H} CPU-h",
                        "stop": f"stop: cycle cap reached ({total:.2f} CPU-h)"}[g])
    ok_season, season_note = season_guard(now, sched_lines, max_hours)
    if not ok_season:
        reasons.append(season_note)
    out = {"ok": not reasons, "reasons": reasons, "total_cpu_hours": round(total, 4), "season": season_note}
    if reasons:
        return out
    unit = f"c1-{name}-{now.strftime('%Y%m%dT%H%M%SZ')}"
    log = (log_dir or Path.home() / "logs") / f"c1-{name}.log"
    limit = int(min(cpu_hours, ledger.CAP_H - total) * 3600)   # the job's own declared budget, within the cycle
    pdir = pause_dir or c1_paths(DATA_ROOT)["ledger"].parent
    out.update(unit=unit, limit_cpu_seconds=limit, log=str(log), argv=[
        "systemd-run", "--user", f"--unit={unit}", "--collect", "--nice=10",
        "-p", "MemoryMax=12G", "-p", "OOMScoreAdjust=1000", "-p", f"LimitCPU={limit}",
        "-p", f"RuntimeMaxSec={int(max_hours * 3600)}",
        "-p", f"EnvironmentFile={env_file}",
        "-p", f"StandardOutput=append:{log}", "-p", f"StandardError=append:{log}",
        f"--working-directory={cwd}", PYTHON, str(GUARD), "--cpu-seconds", str(limit), "--pause-dir", str(pdir),
        "--unit", unit, "--", *command])
    return out


def _run(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=True).stdout


def failed_units(lines: list[str]) -> list[str]:
    """C1 units that systemd reports as failed (a limit kill, timeout, signal or non-zero exit)."""
    out = []
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        unit = d.get("USER_UNIT") or ""
        if unit.startswith("c1-") and "Failed with result" in str(d.get("MESSAGE") or ""):
            out.append(unit.removesuffix(".service"))
    return sorted(set(out))


def _journal() -> list[str]:
    return _run(["journalctl", "--user", "-o", "json", "--since", JOURNAL_SINCE,
                 "--output-fields=USER_UNIT,USER_INVOCATION_ID,CPU_USAGE_NSEC,MESSAGE"]).splitlines()


def sweep(path: Path, lines: list[str]) -> list[dict]:
    """Merge every stopped C1 unit's CPU record from the user journal into the ledger file; return the ledger."""
    rows = ledger.merge(ledger.read_tsv(path), ledger.parse_journal(lines))
    ledger.write_tsv(path, rows)
    return rows


def main(argv=None) -> int:
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--cpu-hours", type=float, required=True, help="declared CPU budget for this job")
    r.add_argument("--max-hours", type=float, required=True, help="declared wall-time budget for this job")
    r.add_argument("--cwd", default=str(REPO))
    r.add_argument("--ack-failure", action="append", default=[],
                   help="a failed earlier unit of this job, acknowledged so it may be launched again")
    r.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    paths = c1_paths(args.data_root)
    c1_dir = paths["ledger"].parent
    try:
        with launch_lock(c1_dir):
            return _locked(args, paths, c1_dir)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2


def _locked(args, paths: dict, c1_dir: Path) -> int:
    journal = _journal()
    rows = sweep(paths["ledger"], journal)
    record_overruns(c1_dir, journal)
    register_text = REGISTER.read_text() if REGISTER.exists() else ""
    overruns = overrun_stops(c1_dir, register_text)
    acked = paths["ack"].exists()
    if args.cmd == "status":
        total = ledger.total_hours(rows)
        print(json.dumps({"total_cpu_hours": round(total, 4), "jobs": len(rows), "checkpoint_acked": acked,
                          "rate_limit_stops": rate_limit_stops(args.data_root), "overrun_stops": overruns,
                          "failed_units": failed_units(journal),
                          "gate": ledger.gate(total, 0.0, acked), "ledger": str(paths["ledger"])}, indent=1))
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    active = [l.split()[0] for l in _run(["systemctl", "--user", "list-units", "c1-*", "--all", "--plain",
                                          "--no-legend", "--state=active,activating,deactivating"]).splitlines()
              if l.strip()]
    sched = _run(["journalctl", "--user", "-u", "bts-scheduler", "-o", "json", "-n", "50",
                  "--output-fields=MESSAGE"]).splitlines()
    p = plan_launch(name=args.name, cpu_hours=args.cpu_hours, max_hours=args.max_hours, rows=rows, acked=acked,
                    active_units=active, sched_lines=sched, now=datetime.now(timezone.utc), cwd=args.cwd,
                    env_file=str(args.data_root.parent / ".env"), command=command,
                    rate_limit_stops=rate_limit_stops(args.data_root), failed_units=failed_units(journal),
                    acked_failures=args.ack_failure, overrun_stops=overruns, pause_dir=c1_dir)
    print(json.dumps(p, indent=1))
    if not p["ok"]:
        return 2
    subprocess.run(p["argv"], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
