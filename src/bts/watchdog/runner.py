"""The watchdog job runner (registration §2, R3, R5).

A job runs its checks under a job singleton lock.
- **Every check runs**, whatever the others do. A check that raises, or that returns no result, is a
  `checker_failure`; it never becomes silence.
- **Results** are written beneath the owned root: `results/<ET date>/<stamp>-<job>.json`.
- **Notifications:** faults and checker failures are queued and flushed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bts.watchdog.result import CheckResult, Status
from bts.watchdog.root import JobBusy, OwnedRoot  # noqa: F401  (JobBusy re-exported for callers)


@dataclass(frozen=True)
class Context:
    et_date: str
    now: datetime
    root: OwnedRoot


@dataclass(frozen=True)
class JobOutcome:
    results: list
    path: Path


def run_job(job: str, checks, *, root: OwnedRoot, clock, notifier) -> JobOutcome:
    with root.job_lock(job):
        now = clock.now()
        ctx = Context(now.date().isoformat(), now, root)
        results: list[CheckResult] = []
        for check in checks:
            name = getattr(check, "__name__", repr(check))
            try:
                out = list(check(ctx) or [])
            except Exception as exc:  # noqa: BLE001 - one failing check never suppresses the others
                results.append(CheckResult(name, ctx.et_date, Status.CHECKER_FAILURE,
                                           f"{type(exc).__name__}: {exc}"[:300], incident=f"checker:{name}"))
                continue
            if not out:
                results.append(CheckResult(name, ctx.et_date, Status.CHECKER_FAILURE, "the check returned no result",
                                           incident=f"checker:{name}"))
            results.extend(out)
        path = root.child("results", ctx.et_date, f"{now:%Y%m%dT%H%M%S%f}-{job}.json")
        root.write_atomic(path, (json.dumps({"job": job, "et_date": ctx.et_date, "at": now.isoformat(),
                                             "results": [r.record() for r in results]},
                                            indent=1, sort_keys=True) + "\n").encode())
        if notifier is not None:
            notifier.enqueue(results)
            notifier.flush()
        return JobOutcome(results, path)
