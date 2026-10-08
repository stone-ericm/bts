"""C1 compute ledger (cycle approved by Eric 2026-10-04; docs/audit/2026-10-04-d4-cycle-proposal.md §3).

Every C1 box job runs as a transient ``c1-*`` user unit. When a unit stops, systemd journals its cgroup CPU time
(``CPU_USAGE_NSEC``) with a unique ``USER_INVOCATION_ID``; the ledger is the deduplicated set of those records.
The cycle cap is 100 CPU-hours with a stop-and-report at 50 (resumed only by Eric's written acknowledgement); Eric raised
the cap to 165 for the C2 framing screen's stage two (register row C2-framing-stage-two-cap, 2026-10-08).

Known undercount (measured on the box 2026-10-04, systemd 252, user-manager LogLevel=info): systemd journals the
"Consumed ... CPU time" record at info level only for a unit that used a mentionworthy amount (a 0.6 s job left no
record; a 1.4 s job did). Each job below that threshold goes uncounted, by under 1.4 CPU-seconds per job."""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PREFIX = "c1-"
CHECKPOINT_H = 50.0
CAP_H = 165.0                  # Eric, register row C2-framing-stage-two-cap (2026-10-08): raised from 100
COLUMNS = ("invocation", "unit", "stopped_at", "cpu_seconds")


class LedgerConflict(ValueError):
    """One invocation id carrying two different records."""


def parse_journal(lines: Iterable[str]) -> list[dict]:
    """C1 unit-stop CPU records from ``journalctl -o json`` lines. A C1 record without an invocation id is refused:
    it could not be deduplicated, so counting it could double-count or drop CPU time."""
    rows = []
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        unit = d.get("USER_UNIT") or ""
        if "CPU_USAGE_NSEC" not in d or not (unit.startswith(PREFIX) and unit.endswith(".service")):
            continue
        inv = d.get("USER_INVOCATION_ID")
        if not inv:
            raise ValueError(f"C1 CPU record without an invocation id: {unit}")
        ts = datetime.fromtimestamp(int(d["__REALTIME_TIMESTAMP"]) / 1_000_000, tz=timezone.utc)
        rows.append({"invocation": inv, "unit": unit, "stopped_at": ts.isoformat(),
                     "cpu_seconds": int(d["CPU_USAGE_NSEC"]) / 1e9})
    return rows


GUARD_KEYS = ("guard:", "reserve:")


def _journal_units(rows: list[dict]) -> set:
    return {r["unit"] for r in rows if not str(r["invocation"]).startswith(GUARD_KEYS)}


def merge(rows: list[dict], new: list[dict]) -> list[dict]:
    """Union by invocation id. A journal record for a unit replaces that unit's guard/reservation row (the journal's
    cgroup CPU is authoritative once it exists)."""
    by_inv = {r["invocation"]: r for r in rows}
    for r in new:
        old = by_inv.get(r["invocation"])
        if old is None:
            by_inv[r["invocation"]] = r
        elif old != r:
            raise LedgerConflict(f"invocation {r['invocation']}: {old} != {r}")
    journaled = _journal_units(list(by_inv.values()))
    kept = [r for r in by_inv.values() if not (str(r["invocation"]).startswith(GUARD_KEYS) and r["unit"] in journaled)]
    return sorted(kept, key=lambda r: (r["stopped_at"], r["invocation"]))


def add_guard_row(rows: list[dict], unit: str, cpu_seconds: float, stopped_at: str, *, reserve: bool = False) -> list[dict]:
    """Account a launched unit the journal has not recorded (code review r1 B4): the guard's measured CPU, or, with
    no guard receipt, a reservation of the unit's full budget. A later journal record replaces it (merge)."""
    svc = f"{unit}.service"
    if svc in _journal_units(rows):
        return rows
    if not (math.isfinite(cpu_seconds) and cpu_seconds >= 0):
        raise ValueError(f"invalid CPU value for {unit}: {cpu_seconds}")
    key = f"{'reserve' if reserve else 'guard'}:{unit}"
    rest = [r for r in rows if r["invocation"] not in (f"guard:{unit}", f"reserve:{unit}")]
    return sorted(rest + [{"invocation": key, "unit": svc, "stopped_at": stopped_at, "cpu_seconds": cpu_seconds}],
                  key=lambda r: (r["stopped_at"], r["invocation"]))


def total_hours(rows: list[dict]) -> float:
    return sum(r["cpu_seconds"] for r in rows) / 3600


def read_tsv(path: Path) -> list[dict]:
    """The ledger rows; a non-finite or negative CPU value is refused (code review r3 B5), never summed."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = [{**r, "cpu_seconds": float(r["cpu_seconds"])} for r in csv.DictReader(f, delimiter="\t")]
    bad = [r["invocation"] for r in rows if not (math.isfinite(r["cpu_seconds"]) and r["cpu_seconds"] >= 0)]
    if bad:
        raise ValueError(f"invalid CPU values in the ledger: {bad[:3]}")
    return rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    """Atomic and durable, through a unique temporary file (two writers never share one)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)          # the rename itself is durable (r1 B4)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def gate(total_h: float, declared_h: float, checkpoint_acked: bool) -> str:
    """``stop`` at the cap (or on a non-finite or negative total); ``checkpoint`` at 50 until Eric's acknowledgement
    exists; ``over_cap`` when the job's declared CPU budget would cross 100; else ``ok``. Crossing 50 during a job is
    allowed: the next launch stops."""
    if not (math.isfinite(total_h) and total_h >= 0):
        return "stop"
    if total_h >= CAP_H:
        return "stop"
    if total_h >= CHECKPOINT_H and not checkpoint_acked:
        return "checkpoint"
    if total_h + declared_h > CAP_H:
        return "over_cap"
    return "ok"
