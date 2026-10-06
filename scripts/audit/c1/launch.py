"""C1 launcher: the only way a C1 job starts on the box (docs/audit/2026-10-04-d4-cycle-proposal.md §§3, 5).

One canonical cycle root (`~/projects/bts/data/hetzner_results/c1`) holds the ledger, the lock and every stop. There is
no alternate-root option, so no launch can use a different pause namespace (C1 infrastructure review r1 B3). Under
one exclusive launch lock (r3 B5):
1. **An active C1 unit refuses the launch before anything else** (r1 B2).
2. **Accounting:** sweep the journal into the ledger, then reconcile every launched unit's `PENDING_<unit>.json`
   against the guard's durable `TERMINAL_<unit>.json` (r1 B4):
   - the guard's CPU is counted when the journal has no row for the unit;
   - a missing or invalid receipt reserves the unit's full budget in the ledger;
   - a missing or invalid receipt, or any result other than a clean `exit`, materializes `OVERRUN_<unit>.json`;
   - nothing is moved: pause markers, then the ledger, then a durable `RECONCILED_<unit>.json` (crash-safe; r2 N1);
   - unit names carry a random suffix and are never reused (r2 N4);
   - an ordinary failure (exit rc != 0) still blocks a same-name relaunch after its journal entry is gone (r2 N3);
   - a failed `systemd-run` keeps the PENDING record (r2 N2);
   - systemd `timeout` / `oom-kill` failures are materialized the same way.
3. **The launch is refused unless:**
   - the name is valid; the declared budgets are finite and positive; and the effective CPU budget is at least the
     guard's enforceable minimum;
   - no other C1 unit is active (checked again just before the start);
   - no 403/429 stop or unresolved request receipt is in force;
   - **no overrun pause is in force:** every `OVERRUN_<unit>.json` pauses all of C1. It is released only by
     `RESUME_<unit>.json` that carries the marker's sha256 and cites register row `C1-resume-<unit>`. That row must
     record exactly "**RULED <date>: RESUME `<unit>` overrun `<sha prefix>`**", with Eric as its source (r1 B5). A
     mention, a denial or another ruling does not release it;
   - a failed earlier unit of the same job has been acknowledged (an ordinary job exit; it does not pause the cycle);
   - the ledger gate is ok (50 = stop and report until Eric's acknowledgement file exists; 100 = stop; a declared budget
     may not cross 100);
   - from 2027-03-01, the scheduler is asleep for longer than the job's declared wall time (fails closed when no sleep
     line is found). Known limit: the season guard reads the latest *realtime* journal timestamp, so a clock rollback
     could select an older sleep line (r1, preexisting; not addressed here).
4. **Start:** write a durable `PENDING_<unit>.json`, then `systemd-run`.

**The unit's limits:** Delegate=yes (the guard's cgroup split), MemoryMax=12G, OOMScoreAdjust=1000, RuntimeMaxSec = the
declared wall time, and a LimitCPU per-process backstop. The command runs under `guard.py`, which applies nice 10 to the
job and caps the unit's **cumulative** CPU at min(declared budget, remaining cycle budget). See that module for what
the bound assumes.

    python -m scripts.audit.c1.launch status
    python -m scripts.audit.c1.launch run --name r2-build --cpu-hours 2 --max-hours 3 -- <command ...>
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.audit.c1 import ledger

REPO = Path(__file__).resolve().parents[3]          # the C1 worktree the job's code runs from
GUARD = Path(__file__).resolve().with_name("guard.py")   # run by absolute path: works from any job's cwd
PYTHON = sys.executable
REGISTER = REPO / "docs" / "audit" / "2026-09-22-exposure-register.md"
OVERRUN_RESULTS = ("timeout", "oom-kill")             # systemd's own overrun kills
RESULT_RE = re.compile(r"Failed with result '([^']+)'")
DATA_ROOT = Path.home() / "projects" / "bts" / "data"  # the one canonical root (hetzner_results is restic-backed)
PROD_ENV = DATA_ROOT.parent / ".env"
CHECKPOINT_ACK_NAME = "CHECKPOINT_50_ACK.json"
JOURNAL_SINCE = "2026-10-04"
GUARD_FROM = datetime(2027, 3, 1, tzinfo=timezone.utc)   # no MLB opener has been earlier; fail closed from here
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
SLEEP_RE = re.compile(r"(?:Sleeping|Idle) until .*\((?:(\d+) min|([\d.]+)h)\)")


def _guard():
    spec = importlib.util.spec_from_file_location("c1_guard", GUARD)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


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


def _write(path: Path, rec: dict) -> None:
    _guard().write_receipt(path, rec)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_overruns(c1_dir: Path, lines: list[str]) -> list[str]:
    """Materialize systemd overrun kills (timeout, oom-kill) as durable pause markers. Returns the units newly
    recorded."""
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
                _write(marker, {"unit": unit, "source": "journal", "result": m.group(1),
                                "written_utc": datetime.now(timezone.utc).isoformat()})
                out.append(unit)
    return out


TERMINAL_RESULTS = ("exit", "overrun", "oom", "terminated", "guard-error")


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0


def terminal_problems(t, pending: dict, unit: str) -> list[str]:
    """A guard receipt is trusted only if it is this invocation's and well formed (infra review r2 N4). A clean
    exit must carry a valid rc and a measured CPU total within the budget."""
    if not isinstance(t, dict):
        return ["not a JSON object"]
    p = []
    if t.get("unit") != unit:
        p.append("names another unit")
    if not (_num(t.get("budget_seconds")) and float(t["budget_seconds"]) == float(pending.get("limit_cpu_seconds", -1))):
        p.append("budget does not match the PENDING record")
    if t.get("result") not in TERMINAL_RESULTS:
        p.append(f"unknown result {t.get('result')!r}")
    if t.get("cpu_seconds") is not None and not _num(t.get("cpu_seconds")):
        p.append("invalid CPU value")
    if t.get("result") == "exit":
        if not (isinstance(t.get("rc"), int) and not isinstance(t.get("rc"), bool)):
            p.append("an exit without an integer rc")
        if not _num(t.get("cpu_seconds")):
            p.append("an exit without a measured CPU total")
        elif _num(t.get("budget_seconds")) and t["cpu_seconds"] > t["budget_seconds"]:
            p.append("an exit whose CPU total exceeds its budget")
    return p


def reconcile(c1_dir: Path, ledger_path: Path, rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Every launched invocation's PENDING record against its guard TERMINAL receipt (call only with no C1 unit
    active). Crash-safe (infra review r2 N1): nothing is moved. For each PENDING without a RECONCILED record:
    1. write the pause marker if one is due;
    2. commit the ledger (the guard's measured CPU, or a reservation of the full budget);
    3. write a durable RECONCILED_<unit>.json.

    A crash anywhere repeats the same idempotent steps on the next sweep.
    - **A valid receipt:** its CPU enters the ledger when the journal has no row for the unit; a result other than a
      clean exit pauses C1.
    - **A missing or invalid receipt:** the full budget is reserved and C1 pauses (unreconciled).

    Returns (ledger rows, units reconciled now)."""
    done = []
    for pend in sorted(c1_dir.glob("PENDING_*.json")):
        unit = pend.stem.removeprefix("PENDING_")
        if (c1_dir / f"RECONCILED_{unit}.json").exists():
            continue
        pending = json.loads(pend.read_text())
        term, marker = c1_dir / f"TERMINAL_{unit}.json", c1_dir / f"OVERRUN_{unit}.json"
        try:
            t = json.loads(term.read_text()) if term.exists() else None
        except ValueError:
            t = "unreadable"
        problems = ["no guard TERMINAL receipt (the guard was killed or failed)"] if t is None else \
            terminal_problems(t, pending, unit)
        if problems:
            result, cpu = "unreconciled", None
            rows = ledger.add_guard_row(rows, unit, float(pending["limit_cpu_seconds"]), "reserved", reserve=True)
        else:
            result, cpu = t["result"], t.get("cpu_seconds")
            rows = ledger.add_guard_row(rows, unit, float(cpu), t.get("ended_utc") or "") if _num(cpu) else \
                ledger.add_guard_row(rows, unit, float(pending["limit_cpu_seconds"]), "reserved", reserve=True)
        if result != "exit" and not marker.exists():
            _write(marker, {"unit": unit, "source": "terminal" if not problems else "unreconciled",
                            "result": result, "problems": problems,
                            "written_utc": datetime.now(timezone.utc).isoformat()})
        ledger.write_tsv(ledger_path, rows)
        _write(c1_dir / f"RECONCILED_{unit}.json", {"unit": unit, "result": result, "cpu_seconds": cpu,
                                                    "rc": t.get("rc") if isinstance(t, dict) else None,
                                                    "problems": problems,
                                                    "written_utc": datetime.now(timezone.utc).isoformat()})
        done.append(unit)
    return rows, done


def receipt_failures(c1_dir: Path) -> list[str]:
    """Units whose reconciled receipt shows an ordinary job failure (exit with a non-zero rc): the same-name retry
    gate holds even after the journal entry is gone (infra review r2 N3)."""
    out = []
    for f in sorted(c1_dir.glob("RECONCILED_*.json")):
        r = json.loads(f.read_text())
        if r.get("result") == "exit" and r.get("rc") not in (0, None):
            out.append(r["unit"])
    return out


def _register_row(text: str, row_id: str) -> list[str] | None:
    line = next((l for l in text.splitlines() if l.startswith(f"| {row_id} |")), None)
    return [c.strip() for c in line.strip().strip("|").split("|")] if line else None


def overrun_stops(c1_dir: Path, register_text: str) -> list[str]:
    """Overrun markers without a valid release (r1 B5). `RESUME_<unit>.json` must carry the marker's sha256 and cite
    row `C1-resume-<unit>`. That row's ruling cell must be exactly "**RULED <date>: RESUME `<unit>` overrun
    `<prefix of that sha>`**", and its source cell must name Eric."""
    stops = []
    for marker in sorted(c1_dir.glob("OVERRUN_*.json")):
        unit = marker.stem.removeprefix("OVERRUN_")
        res = c1_dir / f"RESUME_{unit}.json"
        problems = ["no release"]
        if res.exists():
            problems = []
            try:
                rec = json.loads(res.read_text())
            except ValueError:
                rec = None
            if not isinstance(rec, dict) or rec.get("unit") != unit:
                problems.append("the release does not name this unit")
            elif rec.get("overrun_sha256") != _sha(marker):
                problems.append("the release is not bound to this overrun")
            else:
                row = _register_row(register_text, f"C1-resume-{unit}")
                want = re.compile(r"^\*\*RULED \d{4}-\d{2}-\d{2}: RESUME `(?P<u>[^`]+)` overrun `(?P<s>[0-9a-f]{12,64})`\*\*$")
                m = want.match(row[2]) if row and len(row) >= 4 else None
                if not (m and m.group("u") == unit and _sha(marker).startswith(m.group("s"))
                        and re.match(r"^Eric(?:$|\s)", row[-1])):
                    problems.append(f"no register row C1-resume-{unit} recording Eric's RESUME of this overrun")
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
                pause_dir: Path | None = None, ncpu: int | None = None) -> dict:
    total = ledger.total_hours(rows)
    reasons = []
    if not NAME_RE.match(name):
        reasons.append(f"invalid name {name!r}: lowercase letters, digits and '-' only")
    valid = True
    for label, v in (("cpu-hours", cpu_hours), ("max-hours", max_hours)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            reasons.append(f"invalid {label} {v!r}: a finite positive number is required")
            valid = False
    if not command:
        reasons.append("no command given")
    if active_units:
        reasons.append(f"another C1 job is active: {', '.join(active_units)}")
    if rate_limit_stops:
        reasons.append("403/429 stop in force (C1 is paused until Eric records a decision): "
                       + ", ".join(rate_limit_stops))
    if overrun_stops:
        reasons.append("overrun pause in force (all C1 is paused until Eric's recorded decision releases it): "
                       + ", ".join(overrun_stops))
    prior = [u for u in (failed_units or []) if u.startswith(f"c1-{name}-") and u not in (acked_failures or [])]
    if prior:
        reasons.append(f"a previous {name} unit failed ({', '.join(prior)}); no automatic retry without an "
                       "acknowledged failure")
    limit = 0
    if valid:
        g = ledger.gate(total, cpu_hours, acked)
        if g != "ok":
            reasons.append({"checkpoint": f"checkpoint: {total:.2f} CPU-h >= {ledger.CHECKPOINT_H}; stop and report "
                                          f"until Eric's acknowledgement ({CHECKPOINT_ACK_NAME}) exists",
                            "over_cap": f"over_cap: {total:.2f} + declared {cpu_hours} > {ledger.CAP_H} CPU-h",
                            "stop": f"stop: cycle cap reached or the ledger is invalid ({total:.2f} CPU-h)"}[g])
        limit = int(min(cpu_hours, ledger.CAP_H - total) * 3600) if math.isfinite(total) else 0
        floor = _guard().min_budget(ncpu or os.cpu_count() or 1)
        if limit < floor:
            reasons.append(f"invalid effective CPU budget {limit} s: below the guard's enforceable minimum {floor:.0f} s")
    ok_season, season_note = season_guard(now, sched_lines, max_hours if valid else 0)
    if not ok_season:
        reasons.append(season_note)
    out = {"ok": not reasons, "reasons": reasons, "total_cpu_hours": round(total, 4), "season": season_note}
    if reasons:
        return out
    unit = f"c1-{name}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"   # never reused (infra r2 N4)
    log = (log_dir or Path.home() / "logs") / f"c1-{name}.log"
    pdir = pause_dir or c1_paths(DATA_ROOT)["ledger"].parent
    out.update(unit=unit, limit_cpu_seconds=limit, log=str(log), argv=[
        "systemd-run", "--user", f"--unit={unit}", "--collect",
        "-p", "Delegate=yes", "-p", "MemoryMax=12G", "-p", "OOMScoreAdjust=1000", "-p", f"LimitCPU={limit}",
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


def _active() -> list[str]:
    return [l.split()[0] for l in _run(["systemctl", "--user", "list-units", "c1-*", "--all", "--plain",
                                        "--no-legend", "--state=active,activating,deactivating"]).splitlines()
            if l.strip()]


def sweep(path: Path, lines: list[str]) -> list[dict]:
    """Merge every stopped C1 unit's CPU record from the user journal into the ledger file; return the ledger."""
    rows = ledger.merge(ledger.read_tsv(path), ledger.parse_journal(lines))
    ledger.write_tsv(path, rows)
    return rows


def main(argv=None) -> int:
    os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    ap = argparse.ArgumentParser()
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
    paths = c1_paths(DATA_ROOT)
    c1_dir = paths["ledger"].parent
    try:
        with launch_lock(c1_dir):
            return _locked(args, paths, c1_dir)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2


def _locked(args, paths: dict, c1_dir: Path) -> int:
    active = _active()
    if active and args.cmd == "run":
        raise SystemExit(f"refusing: another C1 job is active ({', '.join(active)}); accounting waits for it to end")
    journal = _journal()
    rows = sweep(paths["ledger"], journal)
    if not active:
        rows, _ = reconcile(c1_dir, paths["ledger"], rows)
    record_overruns(c1_dir, journal)
    register_text = REGISTER.read_text() if REGISTER.exists() else ""
    overruns = overrun_stops(c1_dir, register_text)
    acked = paths["ack"].exists()
    if args.cmd == "status":
        total = ledger.total_hours(rows)
        print(json.dumps({"total_cpu_hours": round(total, 4), "jobs": len(rows), "checkpoint_acked": acked,
                          "active": active, "pending": sorted(p.name for p in c1_dir.glob("PENDING_*.json")),
                          "rate_limit_stops": rate_limit_stops(DATA_ROOT), "overrun_stops": overruns,
                          "failed_units": sorted(set(failed_units(journal)) | set(receipt_failures(c1_dir))),
                          "gate": ledger.gate(total, 0.0, acked), "ledger": str(paths["ledger"])}, indent=1))
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    sched = _run(["journalctl", "--user", "-u", "bts-scheduler", "-o", "json", "-n", "50",
                  "--output-fields=MESSAGE"]).splitlines()
    p = plan_launch(name=args.name, cpu_hours=args.cpu_hours, max_hours=args.max_hours, rows=rows, acked=acked,
                    active_units=_active(), sched_lines=sched, now=datetime.now(timezone.utc), cwd=args.cwd,
                    env_file=str(PROD_ENV), command=command,
                    rate_limit_stops=rate_limit_stops(DATA_ROOT),
                    failed_units=sorted(set(failed_units(journal)) | set(receipt_failures(c1_dir))),
                    acked_failures=args.ack_failure, overrun_stops=overruns, pause_dir=c1_dir)
    print(json.dumps(p, indent=1))
    if not p["ok"]:
        return 2
    _write(c1_dir / f"PENDING_{p['unit']}.json", {"unit": p["unit"], "limit_cpu_seconds": p["limit_cpu_seconds"],
                                                   "declared_cpu_hours": args.cpu_hours, "max_hours": args.max_hours,
                                                   "written_utc": datetime.now(timezone.utc).isoformat()})
    try:
        subprocess.run(p["argv"], check=True)
    except subprocess.CalledProcessError:
        raise SystemExit(
            f"systemd-run failed for {p['unit']}; launch state is unresolved, PENDING retained"
        ) from None
    return 0


if __name__ == "__main__":
    sys.exit(main())
