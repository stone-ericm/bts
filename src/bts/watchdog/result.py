"""Check results (registration §2): every check emits a dated status. Missing input never becomes a no-fault result,
and an unexpected exception is a checker failure that cannot suppress other checks."""
from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field


class Status(str, enum.Enum):
    VERIFIED = "verified"                 # verified agreement on qualified evidence
    FAULT = "fault"                       # a detected fault
    PENDING = "pending"                   # waiting on producer work
    UNVERIFIABLE = "unverifiable"         # missing, stale, ambiguous or unbound evidence
    CHECKER_FAILURE = "checker_failure"   # the check itself raised or returned nothing


ALERTING = frozenset({Status.FAULT, Status.CHECKER_FAILURE})
CHECKER_PREFIX = "checker:"                # incidents reserved for the runner's checker failures (W0 r3 R3-1)


@dataclass(frozen=True)
class CheckResult:
    check: str
    et_date: str
    status: Status
    detail: str
    incident: str | None = None           # incident id for dedup and the alert text
    selection: str | None = None          # the affected selection (dedup), e.g. "batter@game"
    evidence: dict = field(default_factory=dict)

    def record(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d
