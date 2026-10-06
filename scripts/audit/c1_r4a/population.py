"""C1 rank 4a, T2: the eligible forecast population of one fixed contest date (registration §2).

**Source:** the last retained served-slate write of the date, `data/picks/slates/<date>.json` (`bts_slate_v2`).
Slates are last-write-wins, so the file is that write. Its exact bytes are hashed and parsed once, before any outcome
join.

**Row eligibility,** with every exclusion counted by its first failing reason in this order:
1. `duplicate_identity`: the same (date, batter_id, game_pk) appears more than once (every copy is excluded);
2. `invalid_identity`: batter_id / game_pk are not positive integers;
3. `invalid_probability`: `p_game_hit` is not a finite number in [0, 1] (booleans and strings included);
4. `missing_start`: no run-known `game_time`;
5. `missing_status`: no archived schedule status;
6. `status_not_pregame`: the status at the write is not pregame. Pregame is "Scheduled", "Pre-Game", "Warmup" or
   "Delayed Start..."; started, postponed and any unknown status are excluded;
7. `write_not_before_cutoff`: `written_at` is not strictly before `game_time` minus `SUBMISSION_CUTOFF_MIN`.

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
    return type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1


def _pregame(status: str) -> bool:
    return status in PREGAME or status.startswith("Delayed Start")


def population(raw: bytes, *, d: date) -> Population:
    sha = hashlib.sha256(raw).hexdigest()
    try:
        obj = json.loads(raw)
    except ValueError as exc:
        raise PopulationError(f"unreadable slate: {exc}") from None
    if not isinstance(obj, dict) or obj.get("schema_version") != SCHEMA or obj.get("date") != d.isoformat():
        raise PopulationError("not a bts_slate_v2 slate for this date")
    written = _aware(obj.get("written_at"))
    if written is None or not isinstance(obj.get("rows"), list):
        raise PopulationError("the slate has no timezone-aware written_at or no rows")
    rows = obj["rows"]
    ident = Counter((r.get("batter_id"), r.get("game_pk")) for r in rows if isinstance(r, dict))
    excluded, eligible, projected = Counter(), [], Counter()
    for r in rows:
        if not isinstance(r, dict):
            excluded["invalid_identity"] += 1
            continue
        reason = None
        if ident[(r.get("batter_id"), r.get("game_pk"))] > 1:
            reason = "duplicate_identity"
        elif not (_pid(r.get("batter_id")) and _pid(r.get("game_pk"))):
            reason = "invalid_identity"
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
            continue
        eligible.append(r)
        projected["projected" if r.get("projected") else "confirmed"] += 1
    rank1 = None
    for r in eligible:                                   # first maximum in stored order
        if rank1 is None or r["p_game_hit"] > rank1["p_game_hit"]:
            rank1 = r
    return Population(d, sha, obj["written_at"], eligible, rank1, dict(excluded), dict(projected))
