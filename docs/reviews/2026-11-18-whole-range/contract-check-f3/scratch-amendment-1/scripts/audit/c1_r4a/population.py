"""C1 rank 4a, T2: the eligible forecast population of one fixed contest date (registration §2).

**Source:** the last retained served-slate write of the date (`bts_slate_v2`; the caller reads the stored file once,
plain or gzipped, and passes the decoded bytes). Slates are last-write-wins, so the file is that write. Its exact
bytes are hashed and parsed once, before any outcome join.

**Unsupported slates refuse** (`PopulationError`): unreadable JSON, a schema other than `bts_slate_v2`, another date,
no timezone-aware `written_at`, or `rows` that is not a list. The caller refuses acceptance on these (X-E1); they are
never turned into missing support.

**Row eligibility,** with every exclusion counted by its first failing reason in this order:
1. `invalid_identity`: the row is not an object, or batter_id / game_pk are not positive integers (booleans and
   floats included). Identities are validated before any duplicate key is built, so an invalid value never aliases
   a valid one (`True == 1`) and an unhashable one never raises;
2. `duplicate_identity`: the same valid (date, batter_id, game_pk) appears more than once (every copy is excluded);
3. `invalid_probability`: `p_game_hit` is not an int or float in [0, 1] that is finite (booleans and strings
   included; the type and range are checked before any float conversion, so a huge integer is counted, not raised);
4. `missing_start`: no run-known `game_time`;
5. `missing_status`: no archived schedule status;
6. `status_not_pregame`: the status at the write is not pregame. Pregame is "Scheduled", "Pre-Game", "Warmup" or
   "Delayed Start..."; started, postponed and any unknown status are excluded;
7. `write_not_before_cutoff`: `written_at` is not strictly before `game_time` minus `SUBMISSION_CUTOFF_MIN`.

**States:** each row's lineup state is `projected` (`projected` is true), `confirmed` (false) or `unknown`. The
eligible rows and every exclusion are also counted by state (registration §3).

**Rank 1** is chosen from the eligible pool by the original probability, before outcomes. Ties keep the stored order.
It is never replaced later.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

SCHEMA = "bts_slate_v2"
SUBMISSION_CUTOFF_MIN = 5          # = bts.picks.SUBMISSION_CUTOFF_MIN, restated so the reader pins it
PREGAME = {"Scheduled", "Pre-Game", "Warmup"}


class PopulationError(RuntimeError):
    pass


@dataclass
class Population:
    d: date
    slate_sha256: str
    written_at: str
    eligible: list
    rank1: dict | None
    excluded: dict = field(default_factory=dict)
    projected: dict = field(default_factory=dict)
    excluded_by_state: dict = field(default_factory=dict)
    envelope: dict = field(default_factory=dict)

    @property
    def rank1_index(self) -> int | None:
        return None if self.rank1 is None else next(i for i, r in enumerate(self.eligible) if r is self.rank1)


def _aware(s) -> datetime | None:
    if not isinstance(s, str):
        return None
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def _pid(v) -> bool:
    return type(v) is int and v > 0


def _prob(v) -> bool:
    if type(v) is int:
        return 0 <= v <= 1
    return type(v) is float and math.isfinite(v) and 0.0 <= v <= 1.0


def _pregame(status: str) -> bool:
    return status in PREGAME or status.startswith("Delayed Start")


def state(r) -> str:
    p = r.get("projected") if isinstance(r, dict) else None
    return "projected" if p is True else "confirmed" if p is False else "unknown"


def population(raw: bytes, *, d: date) -> Population:
    sha = hashlib.sha256(raw).hexdigest()
    try:
        obj = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise PopulationError(f"unreadable slate: {exc}") from None
    if not isinstance(obj, dict) or obj.get("schema_version") != SCHEMA or obj.get("date") != d.isoformat():
        raise PopulationError(f"not a {SCHEMA} slate for this date")
    written = _aware(obj.get("written_at"))
    if written is None or not isinstance(obj.get("rows"), list):
        raise PopulationError("the slate has no timezone-aware written_at or no rows list")
    rows = obj["rows"]
    valid_id = [isinstance(r, dict) and _pid(r.get("batter_id")) and _pid(r.get("game_pk")) for r in rows]
    ident = Counter((r["batter_id"], r["game_pk"]) for r, ok in zip(rows, valid_id) if ok)
    excluded, by_state, eligible, projected = Counter(), Counter(), [], Counter()
    for r, ok in zip(rows, valid_id):
        reason = None
        if not ok:
            reason = "invalid_identity"
        elif ident[(r["batter_id"], r["game_pk"])] > 1:
            reason = "duplicate_identity"
        elif not _prob(r.get("p_game_hit")):
            reason = "invalid_probability"
        else:
            start = _aware(r.get("game_time"))
            status = r.get("status")
            if start is None:
                reason = "missing_start"
            elif not isinstance(status, str) or not status:
                reason = "missing_status"
            elif not _pregame(status):
                reason = "status_not_pregame"
            elif not written < start - timedelta(minutes=SUBMISSION_CUTOFF_MIN):
                reason = "write_not_before_cutoff"
        if reason:
            excluded[reason] += 1
            by_state[f"{reason}|{state(r)}"] += 1
            continue
        eligible.append(r)
        projected[state(r)] += 1
    rank1 = None
    for r in eligible:                                   # first maximum in stored order
        if rank1 is None or r["p_game_hit"] > rank1["p_game_hit"]:
            rank1 = r
    envelope = {k: v for k, v in obj.items() if k != "rows"}
    return Population(d, sha, obj["written_at"], eligible, rank1, dict(excluded), dict(projected), dict(by_state),
                      envelope)
