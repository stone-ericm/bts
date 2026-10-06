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
  (`<job>/<check_id>`, `checker:<job>/<check_id>`, date). A later successful execution of the same registered
  invocation closes that episode, whatever the business result (W0 r2 N1).

**Registration (W0 r3 R3-1):** a job is a sequence of `(check_id, callable)` pairs. The job name and every id match
`[a-z0-9][a-z0-9_-]*`, ids are unique within the job, and every callable is callable; anything else raises
`RegistrationError` before any check runs. The invocation identity `<job>/<check_id>` never depends on a callable's
display name, so a different callable cannot close another one's checker episode, in its job or any other.
**The `checker:` incident namespace is reserved** for the runner: a business result that uses it is invalid.
**A result must be storable:** its incident and selection are None or a non-empty string, so one check's output can
never make the notification state invalid and drop the other checks' alerts.

**Outputs:** the results file and the notifications are attempted independently, and each failure is captured
(`persist_error`, `notify_error`) rather than aborting the other. The outcome always carries the collected statuses.
A caller must treat any error as an infrastructure failure, never as completion.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from bts.watchdog.result import CHECKER_PREFIX, CheckResult, Status
from bts.watchdog.root import JobBusy, OwnedRoot  # noqa: F401  (JobBusy re-exported for callers)


ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")


class RegistrationError(ValueError):
    """A job or check registration that cannot give every invocation a stable, unique identity."""


def validate_checks(job, checks) -> tuple:
    """The job's checks as a tuple of `(check_id, callable)` pairs; raises RegistrationError on anything else."""
    if not isinstance(job, str) or not ID_RE.fullmatch(job):
        raise RegistrationError(f"invalid job name {job!r}")
    try:
        entries = list(checks)
    except TypeError:
        raise RegistrationError(f"job {job!r}: checks are not a sequence") from None
    out, seen = [], set()
    for entry in entries:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            raise RegistrationError(f"job {job!r}: each check must be a (check_id, callable) pair")
        cid, fn = entry
        if not isinstance(cid, str) or not ID_RE.fullmatch(cid):
            raise RegistrationError(f"job {job!r}: invalid check id {cid!r}")
        if cid in seen:
            raise RegistrationError(f"job {job!r}: duplicate check id {cid!r}")
        if not callable(fn):
            raise RegistrationError(f"job {job!r}: check {cid!r} is not callable")
        seen.add(cid)
        out.append((cid, fn))
    if not out:
        raise RegistrationError(f"job {job!r} has no checks")
    return tuple(out)


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


def _opt_str(v) -> bool:
    return v is None or (isinstance(v, str) and v != "")


def _valid(r) -> CheckResult:
    if not isinstance(r, CheckResult) or not isinstance(r.status, Status) or not isinstance(r.check, str) or not r.check:
        raise TypeError(f"not a CheckResult with a Status: {type(r).__name__}")
    if not (_opt_str(r.incident) and _opt_str(r.selection)):
        raise TypeError("incident and selection must each be None or a non-empty string")
    if r.incident is not None and r.incident.startswith(CHECKER_PREFIX):
        raise ValueError(f"the incident namespace {CHECKER_PREFIX!r} is reserved for the runner")
    date.fromisoformat(r.et_date)                       # raises on a missing or malformed date
    json.dumps(r.record())                              # raises on non-serializable content
    return r


def _run_check(job: str, cid: str, check, ctx: Context) -> tuple[list, str, bool]:
    """(results, the invocation identity `<job>/<check_id>`, whether it executed successfully)."""
    inv = f"{job}/{cid}"
    got = 0
    try:
        out = []
        for r in check(ctx) or []:
            got += 1
            out.append(_valid(r))
        if not out:
            raise RuntimeError("the check returned no result")
        return out, inv, True
    except Exception as exc:  # noqa: BLE001 - one failing check never suppresses the others
        detail = _safe_exc(exc) + (f" ({got} partial result(s) discarded)" if got else "")
        failure = CheckResult(inv, ctx.et_date, Status.CHECKER_FAILURE, detail, incident=f"{CHECKER_PREFIX}{inv}")
        return [failure], inv, False


def run_job(job: str, checks, *, root: OwnedRoot, clock, notifier) -> JobOutcome:
    registered = validate_checks(job, checks)            # refuses before the lock, any check or any write
    with root.job_lock(job):
        now = clock.now()
        ctx = Context(now.date().isoformat(), now, root)
        results: list[CheckResult] = []
        executed_ok: list[str] = []
        for cid, check in registered:
            out, inv, ok = _run_check(job, cid, check, ctx)
            results.extend(out)
            if ok:
                executed_ok.append(inv)
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
