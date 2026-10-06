"""C1 rank 4a, T3: known hit / no_hit per (batter, game) from an archived MLB live feed (registration §2).

**PA events:** the W1.2 definition, production's `PA_ENDING_EVENTS`; a hit is any of `HIT_EVENTS`.

**The resumed portion is excluded,** as in production's scoring helper: with a `resumeDateTime`, a play at or after
it, or one with no start time, is the resumed portion.

**A complete settled scoring source:**
- the game's status is a completed one ("Final", "Game Over", or "Completed Early" with an optional reason);
- the play list is non-empty, its `atBatIndex` runs 0..n-1 without a gap, and every play is complete;
- every PA-ending play names a positive integer batter id (an unattributable PA makes the game unknown).

Otherwise every outcome of the game is **unknown**, including a partial game with zero hits.

**The archive's limit (stated, not hidden):** production fetches each feed once, the first time its game is Final
(`bts.data.pull.download_game_feed`), with no fetch receipt. A scoring correction made after that fetch but before
the next-day 08:00 ET BTS cutoff is therefore not reflected. "Settled" here means a completed final game with a
complete play list, not a post-cutoff snapshot.

**Outcomes:**
- `hit`: a pre-resume PA ended in a hit;
- `no_hit`: at least one pre-resume PA and no hit;
- `no_pa`: none, in a complete game;
- `unknown`: anything incomplete.

`no_pa` and `unknown` are excluded and counted by the caller.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from bts.data.schema import HIT_EVENTS, PA_ENDING_EVENTS

COMPLETED = re.compile(r"(Final|Game Over|Completed Early)(: .+)?")


class OutcomeError(RuntimeError):
    pass


def _ts(s) -> datetime | None:
    if not isinstance(s, str):
        return None
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


@dataclass
class GameOutcomes:
    game_pk: int
    complete: bool
    reason: str | None
    pre_resume: dict = field(default_factory=dict)   # batter -> [event, ...] (pre-resume PA events only)

    def outcome(self, batter_id: int) -> str:
        if not self.complete:
            return "unknown"
        events = self.pre_resume.get(batter_id, [])
        if not events:
            return "no_pa"
        return "hit" if any(e in HIT_EVENTS for e in events) else "no_hit"


def game_outcomes(raw: bytes, *, game_pk: int) -> GameOutcomes:
    try:
        feed = json.loads(raw)
        gd, ld = feed["gameData"], feed["liveData"]
        pk = gd["game"]["pk"]
        status = (gd.get("status") or {}).get("detailedState")
        plays = ld["plays"]["allPlays"]
    except (ValueError, KeyError, TypeError) as exc:
        raise OutcomeError(f"unreadable feed: {exc}") from None
    if type(pk) is not int or pk != game_pk or feed.get("gamePk") != game_pk:
        raise OutcomeError(f"the feed is not game {game_pk}")
    reason = None
    if not (isinstance(status, str) and COMPLETED.fullmatch(status)):
        reason = f"status {status!r} is not completed"
    elif not isinstance(plays, list) or not plays:
        reason = "no plays"
    elif any((p.get("about") or {}).get("atBatIndex") != i for i, p in enumerate(plays)):
        reason = "the play list is not contiguous from 0"
    elif any((p.get("about") or {}).get("isComplete") is not True for p in plays):
        reason = "an incomplete play"
    resume = (gd.get("datetime") or {}).get("resumeDateTime")
    resume_dt = _ts(resume) if resume else None
    if resume and resume_dt is None:
        reason = reason or "unparseable resumeDateTime"
    pre: dict = {}
    if reason is None:
        for p in plays:
            event = (p.get("result") or {}).get("eventType")
            batter = ((p.get("matchup") or {}).get("batter") or {}).get("id")
            if event not in PA_ENDING_EVENTS:
                continue
            if type(batter) is not int or batter <= 0:
                return GameOutcomes(game_pk, False, "a PA without a positive integer batter id", {})
            if resume_dt is not None:
                start = _ts((p.get("about") or {}).get("startTime"))
                if start is None or start >= resume_dt:
                    continue                             # the resumed portion is never evaluated
            pre.setdefault(batter, []).append(event)
    return GameOutcomes(game_pk, reason is None, reason, pre)
