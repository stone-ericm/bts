"""C1 launcher: the only way a C1 job starts on the box (docs/audit/2026-10-04-d4-cycle-proposal.md §§3, 5).

Before launching it sweeps the journal into the compute ledger, then refuses unless: the name is valid; no other C1
unit is active (one job at a time); the ledger gate is ok (50 = stop and report until Eric's acknowledgement file
exists; 100 = stop; a declared budget may not cross 100); and, from 2027-03-01, the scheduler is asleep for longer
than the job's declared wall time (fails closed when no sleep line is found). The unit carries nice 10,
MemoryMax=12G, OOMScoreAdjust=1000 and LimitCPU = the remaining cycle budget, so no job can run past the cap.

    python -m scripts.audit.c1.launch status
    python -m scripts.audit.c1.launch run --name r2-build --cpu-hours 2 --max-hours 3 -- <command ...>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.audit.c1 import ledger

REPO = Path(__file__).resolve().parents[3]          # the C1 worktree the job's code runs from
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


def plan_launch(*, name: str, cpu_hours: float, max_hours: float, rows: list[dict], acked: bool,
                active_units: list[str], sched_lines: list[str], now: datetime, cwd: str, command: list[str],
                env_file: str = str(PROD_ENV), log_dir: Path | None = None) -> dict:
    total = ledger.total_hours(rows)
    reasons = []
    if not NAME_RE.match(name):
        reasons.append(f"invalid name {name!r}: lowercase letters, digits and '-' only")
    if not command:
        reasons.append("no command given")
    if active_units:
        reasons.append(f"another C1 job is active: {', '.join(active_units)}")
    g = ledger.gate(total, cpu_hours, acked)
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
    limit = int((ledger.CAP_H - total) * 3600)
    out.update(unit=unit, limit_cpu_seconds=limit, log=str(log), argv=[
        "systemd-run", "--user", f"--unit={unit}", "--collect", "--nice=10",
        "-p", "MemoryMax=12G", "-p", "OOMScoreAdjust=1000", "-p", f"LimitCPU={limit}",
        "-p", f"EnvironmentFile={env_file}",
        "-p", f"StandardOutput=append:{log}", "-p", f"StandardError=append:{log}",
        f"--working-directory={cwd}", *command])
    return out


def _run(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=True).stdout


def sweep(path: Path) -> list[dict]:
    """Merge every stopped C1 unit's CPU record from the user journal into the ledger file; return the ledger."""
    new = ledger.parse_journal(_run(["journalctl", "--user", "-o", "json", "--since", JOURNAL_SINCE,
                                     "--output-fields=USER_UNIT,USER_INVOCATION_ID,CPU_USAGE_NSEC"]).splitlines())
    rows = ledger.merge(ledger.read_tsv(path), new)
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
    r.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    paths = c1_paths(args.data_root)
    rows = sweep(paths["ledger"])
    acked = paths["ack"].exists()
    if args.cmd == "status":
        total = ledger.total_hours(rows)
        print(json.dumps({"total_cpu_hours": round(total, 4), "jobs": len(rows), "checkpoint_acked": acked,
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
                    env_file=str(args.data_root.parent / ".env"), command=command)
    print(json.dumps(p, indent=1))
    if not p["ok"]:
        return 2
    subprocess.run(p["argv"], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
