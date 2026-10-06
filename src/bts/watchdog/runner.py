"""The watchdog job runner (registration §2, R3, R5; W0 review r1 B6).

A job runs its checks under a job singleton lock. Each check is isolated:
- **Inside its boundary:** its name lookup, its call, the consumption of its output, and the validation of every
  result.
- **Validation:** each result must be a `CheckResult` with a `Status`, an ISO `et_date` and JSON-serializable
  content.
- **Transactional:** a check's results count only if all of them are valid and the check returned normally. A check
  that raises part-way (a generator, say) has its partial results discarded and replaced by one `checker_failure`
  naming the number discarded.
- **Never silence:** a raise, an empty return or invalid output is a `checker_failure`, targeted at
  (name, `checker:<name>`, date). A later successful execution of the same name closes that episode, whatever the
  business result (W0 r2 N1).

**Outputs:** the results file and the notifications are attempted independently, and each failure is captured
(`persist_error`, `notify_error`) rather than aborting the other. The outcome always carries the collected statuses.
A caller must treat any error as an infrastructure failure, never as completion.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from bts.watchdog.result import CheckResult, Status
from bts.watchdog.root import JobBusy, OwnedRoot  # noqa: F401  (JobBusy re-exported for callers)


@dataclass(frozen=True)
class Context:
    et_date: str
    now: datetime
    root: OwnedRoot


@dataclass
class JobOutcome:
    results: list
    path: Path | None
    persist_error: str | None = None
    notify_error: str | None = None
    notify_report: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.persist_error is None and self.notify_error is None


def _safe_name(check, i: int) -> str:
    try:
        name = getattr(check, "__name__", None)
        if isinstance(name, str) and name:
            return name
    except Exception:  # noqa: BLE001
        pass
    return f"check#{i}"


def _safe_exc(exc: BaseException) -> str:
    try:
        kind = type(exc).__name__
    except Exception:  # noqa: BLE001
        kind = "Exception"
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001
        text = "<unprintable>"
    return f"{kind}: {text}"[:300]


def _valid(r) -> CheckResult:
    if not isinstance(r, CheckResult) or not isinstance(r.status, Status) or not isinstance(r.check, str) or not r.check:
        raise TypeError(f"not a CheckResult with a Status: {type(r).__name__}")
    date.fromisoformat(r.et_date)                       # raises on a missing or malformed date
    json.dumps(r.record())                              # raises on non-serializable content
    return r


def _run_check(check, i: int, ctx: Context) -> tuple[list, str, bool]:
    """(results, the check's registered name, whether it executed successfully)."""
    name = _safe_name(check, i)
    got = 0
    try:
        out = []
        for r in check(ctx) or []:
            got += 1
            out.append(_valid(r))
        if not out:
            raise RuntimeError("the check returned no result")
        return out, name, True
    except Exception as exc:  # noqa: BLE001 - one failing check never suppresses the others
        detail = _safe_exc(exc) + (f" ({got} partial result(s) discarded)" if got else "")
        return [CheckResult(name, ctx.et_date, Status.CHECKER_FAILURE, detail, incident=f"checker:{name}")], name, False


def run_job(job: str, checks, *, root: OwnedRoot, clock, notifier) -> JobOutcome:
    with root.job_lock(job):
        now = clock.now()
        ctx = Context(now.date().isoformat(), now, root)
        results: list[CheckResult] = []
        executed_ok: list[str] = []
        for i, check in enumerate(checks):
            out, name, ok = _run_check(check, i, ctx)
            results.extend(out)
            if ok:
                executed_ok.append(name)
        outcome = JobOutcome(results, None)
        try:
            outcome.path = root.write_atomic(
                ("results", ctx.et_date, f"{now:%Y%m%dT%H%M%S%f}-{job}.json"),
                (json.dumps({"job": job, "et_date": ctx.et_date, "at": now.isoformat(),
                             "results": [r.record() for r in results]}, indent=1, sort_keys=True) + "\n").encode())
        except Exception as exc:  # noqa: BLE001 - reported, and notifications are still attempted
            outcome.persist_error = _safe_exc(exc)
        if notifier is not None:
            try:
                notifier.enqueue(results, executed_ok=executed_ok)
                outcome.notify_report = notifier.flush()
            except Exception as exc:  # noqa: BLE001 - reported, never success
                outcome.notify_error = _safe_exc(exc)
        return outcome
