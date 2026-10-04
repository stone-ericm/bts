"""C1 compute ledger (cycle approved by Eric 2026-10-04; docs/audit/2026-10-04-d4-cycle-proposal.md §3).

Every C1 box job runs as a transient ``c1-*`` user unit. When a unit stops, systemd journals its cgroup CPU time
(``CPU_USAGE_NSEC``) with a unique ``USER_INVOCATION_ID``; the ledger is the deduplicated set of those records.
The cycle cap is 100 CPU-hours with a stop-and-report at 50 (resumed only by Eric's written acknowledgement)."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PREFIX = "c1-"
CHECKPOINT_H = 50.0
CAP_H = 100.0
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


def merge(rows: list[dict], new: list[dict]) -> list[dict]:
    by_inv = {r["invocation"]: r for r in rows}
    for r in new:
        old = by_inv.get(r["invocation"])
        if old is None:
            by_inv[r["invocation"]] = r
        elif old != r:
            raise LedgerConflict(f"invocation {r['invocation']}: {old} != {r}")
    return sorted(by_inv.values(), key=lambda r: (r["stopped_at"], r["invocation"]))


def total_hours(rows: list[dict]) -> float:
    return sum(r["cpu_seconds"] for r in rows) / 3600


def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [{**r, "cpu_seconds": float(r["cpu_seconds"])} for r in csv.DictReader(f, delimiter="\t")]


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def gate(total_h: float, declared_h: float, checkpoint_acked: bool) -> str:
    """``stop`` at the cap; ``checkpoint`` at 50 until Eric's acknowledgement exists; ``over_cap`` when the job's
    declared CPU budget would cross 100; else ``ok``. Crossing 50 during a job is allowed: the next launch stops."""
    if total_h >= CAP_H:
        return "stop"
    if total_h >= CHECKPOINT_H and not checkpoint_acked:
        return "checkpoint"
    if total_h + declared_h > CAP_H:
        return "over_cap"
    return "ok"
