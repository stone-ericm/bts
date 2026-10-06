"""C1 rank 4a, T3: known hit / no_hit per (batter, game) from an archived MLB live feed (registration §2).

**PA events:** the W1.2 definition, production's `PA_ENDING_EVENTS`; a hit is any of `HIT_EVENTS`.

**The resumed portion is excluded,** as in production's scoring helper: with a `resumeDateTime`, a play at or after
it, or one with no start time, is the resumed portion.

**An independently complete scoring source (review r1 R1).** A completed status and a gap-free play list do not show
that the list is the whole game: a list with its suffix removed passes both. The witness is the same archived feed's
final boxscore and linescore, the sources production grades normal games from (`bts.picks._boxscore_hit`). The game
is complete only if all of these hold; otherwise every outcome of the game is **unknown**:
1. the status is a completed one ("Final", "Game Over", or "Completed Early" with an optional reason);
2. the play list is a non-empty list, its `atBatIndex` runs 0..n-1 without a gap, and every play is complete;
3. every PA-ending play names a positive integer batter and a boolean `isTopInning` (the batting side);
4. the boxscore gives each side's players (keyed `ID<id>`, matching the person's id, no player on both sides) and a
   team hit total, and the linescore gives each side's hits;
5. **the hits reconcile:** for each side, the hit plays in the list equal the boxscore's team hits and the
   linescore's hits; for every batter, the hit plays naming him equal his boxscore hits; and every batter with a
   PA-ending play has a boxscore batting line.

**The event accounting (why hits, not PA totals, are reconciled):**
- **A label depends on two facts:** whether the batter had a pre-resume PA, and whether any of his pre-resume PAs was
  a hit.
  - **Truncation cannot invent a PA:** a retained play is real, so its existence needs no total.
  - **A missing hit is caught:** a suffix holding a hit breaks rule 5. With every hit present, the pre-resume subset
    is exact.
- **Why plate-appearance totals are not compared:**
  - The boxscore's `plateAppearances` counts events outside the W1.2 set; an intentional walk (`intent_walk`) is not
    in `PA_ENDING_EVENTS`.
  - Official scoring charges some PAs to a replaced batter (a pinch hitter entering with two strikes).
  - Requiring equal PA totals would make games unknown for reasons unrelated to the label.
- **no_pa** is the one label that needs an absence, so it needs the boxscore's agreement:
  - no PA-ending play names the batter;
  - his boxscore line shows `plateAppearances` = 0 (or he has no batting line, or he is absent from the boxscore).
  - Any other batter without a pre-resume PA is **unknown**, for example one seen only after the resume, or one
    with only an intentional walk.

**The archive's limit (stated, not hidden):**
- **No later correction:** production fetches each feed once, the first time its game is Final
  (`bts.data.pull.download_game_feed`), with no fetch receipt. A scoring correction made after that fetch is not
  reflected.
- **What the labels are:** the reconciled event labels of that archived snapshot, not verified cutoff-corrected BTS
  settlement. Registration §7 limits the inputs to what production already archives (no new acquisition), so
  nothing stronger is available.

**Outcomes:**
- `hit`: a pre-resume PA ended in a hit;
- `no_hit`: at least one pre-resume PA and no hit;
- `no_pa`: as above;
- `unknown`: anything else.

`no_pa` and `unknown` are excluded and counted by the caller.

**Malformed input:**
- **A feed for another game** raises `OutcomeError`.
- **Any other malformed or unsupported structure,** of any JSON type, makes the game unknown; it never raises
  anything else.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from bts.data.schema import HIT_EVENTS, PA_ENDING_EVENTS

COMPLETED = re.compile(r"(Final|Game Over|Completed Early)(: .+)?")
SIDES = ("away", "home")
PLAYER_KEY = re.compile(r"ID([1-9][0-9]*)")


class OutcomeError(RuntimeError):
    pass


class _Unsupported(Exception):
    pass


def _ts(s) -> datetime | None:
    if not isinstance(s, str):
        return None
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def _obj(x, what: str) -> dict:
    if not isinstance(x, dict):
        raise _Unsupported(f"{what} is not an object")
    return x


def _count(x, what: str) -> int:
    if type(x) is not int or x < 0:
        raise _Unsupported(f"{what} is not a non-negative integer")
    return x


@dataclass
class GameOutcomes:
    game_pk: int
    complete: bool
    reason: str | None
    pre_resume: dict = field(default_factory=dict)   # batter -> [event, ...] (pre-resume PA events only)
    whole_pa: dict = field(default_factory=dict)     # batter -> PA-ending plays in the whole list
    box_pa: dict = field(default_factory=dict)       # batter -> boxscore plateAppearances (None if not given)

    def outcome(self, batter_id) -> str:
        if not self.complete or type(batter_id) is not int:
            return "unknown"
        events = self.pre_resume.get(batter_id, [])
        if events:
            return "hit" if any(e in HIT_EVENTS for e in events) else "no_hit"
        if self.whole_pa.get(batter_id, 0) == 0 and self.box_pa.get(batter_id, 0) == 0:
            return "no_pa"
        return "unknown"


def _boxscore(ld: dict) -> tuple[dict, dict]:
    """({batter: (hits, plateAppearances or None)}, {side: team hits}); a batter without a batting line is absent."""
    teams = _obj(_obj(ld.get("boxscore"), "boxscore").get("teams"), "boxscore.teams")
    lines, team_hits, seen = {}, {}, set()
    for side in SIDES:
        t = _obj(teams.get(side), f"boxscore {side}")
        batting = _obj(_obj(t.get("teamStats"), f"{side} teamStats").get("batting"), f"{side} team batting")
        team_hits[side] = _count(batting.get("hits"), f"{side} team hits")
        for key, entry in _obj(t.get("players"), f"{side} players").items():
            m = PLAYER_KEY.fullmatch(key) if isinstance(key, str) else None
            entry = _obj(entry, f"player {key}")
            pid = _obj(entry.get("person"), f"player {key} person").get("id")
            if m is None or type(pid) is not int or pid != int(m.group(1)):
                raise _Unsupported(f"player key {key!r} does not match its person id")
            if pid in seen:
                raise _Unsupported(f"player {pid} is on both sides")
            seen.add(pid)
            line = _obj(_obj(entry.get("stats"), f"player {pid} stats").get("batting", {}), f"player {pid} batting")
            if line:
                pa = line.get("plateAppearances")
                lines[pid] = (_count(line.get("hits"), f"player {pid} hits"),
                              None if pa is None else _count(pa, f"player {pid} plateAppearances"))
    return lines, team_hits


def _linescore_hits(ld: dict) -> dict:
    teams = _obj(_obj(ld.get("linescore"), "linescore").get("teams"), "linescore.teams")
    return {side: _count(_obj(teams.get(side), f"linescore {side}").get("hits"), f"linescore {side} hits")
            for side in SIDES}


def game_outcomes(raw: bytes, *, game_pk: int) -> GameOutcomes:
    try:
        feed = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        return GameOutcomes(game_pk, False, f"unreadable feed: {exc}"[:200])
    try:
        feed = _obj(feed, "the feed")
        gd, ld = _obj(feed.get("gameData"), "gameData"), _obj(feed.get("liveData"), "liveData")
        pk = _obj(gd.get("game"), "gameData.game").get("pk")
    except _Unsupported as exc:
        return GameOutcomes(game_pk, False, str(exc))
    if type(pk) is not int or pk != game_pk or type(feed.get("gamePk")) is not int or feed["gamePk"] != game_pk:
        raise OutcomeError(f"the feed is not game {game_pk}")
    try:
        return _complete(game_pk, gd, ld)
    except _Unsupported as exc:
        return GameOutcomes(game_pk, False, str(exc))


def _complete(game_pk: int, gd: dict, ld: dict) -> GameOutcomes:
    status = gd.get("status").get("detailedState") if isinstance(gd.get("status"), dict) else None
    if not (isinstance(status, str) and COMPLETED.fullmatch(status)):
        raise _Unsupported(f"status {status!r} is not completed"[:200])
    plays = _obj(ld.get("plays"), "plays").get("allPlays")
    if not isinstance(plays, list) or not plays:
        raise _Unsupported("no plays")
    for i, p in enumerate(plays):
        about = _obj(_obj(p, f"play {i}").get("about"), f"play {i} about")
        if type(about.get("atBatIndex")) is not int or about["atBatIndex"] != i:
            raise _Unsupported("the play list is not contiguous from 0")
        if about.get("isComplete") is not True:
            raise _Unsupported("an incomplete play")
    resume = _obj(gd.get("datetime", {}), "gameData.datetime").get("resumeDateTime")
    resume_dt = _ts(resume) if resume else None
    if resume and resume_dt is None:
        raise _Unsupported("unparseable resumeDateTime")
    pre, whole_pa, whole_hits, side_hits = {}, Counter(), Counter(), Counter()
    for i, p in enumerate(plays):
        event = _obj(p.get("result"), f"play {i} result").get("eventType")
        if not isinstance(event, str) or event not in PA_ENDING_EVENTS:
            continue
        batter = _obj(_obj(p.get("matchup"), f"play {i} matchup").get("batter"), f"play {i} batter").get("id")
        top = p["about"].get("isTopInning")
        if type(batter) is not int or batter <= 0:
            raise _Unsupported("a PA without a positive integer batter id")
        if type(top) is not bool:
            raise _Unsupported("a PA without a batting side")
        hit = event in HIT_EVENTS
        whole_pa[batter] += 1
        whole_hits[batter] += hit
        side_hits["away" if top else "home"] += hit
        if resume_dt is not None:
            start = _ts(p["about"].get("startTime"))
            if start is None or start >= resume_dt:
                continue                             # the resumed portion is never evaluated
        pre.setdefault(batter, []).append(event)
    lines, team_hits = _boxscore(ld)
    line_hits = _linescore_hits(ld)
    for side in SIDES:
        if not side_hits[side] == team_hits[side] == line_hits[side]:
            raise _Unsupported(f"{side} hits do not reconcile: plays {side_hits[side]}, boxscore {team_hits[side]}, "
                               f"linescore {line_hits[side]}")
    for batter in set(whole_pa) | set(lines):
        if batter in whole_pa and batter not in lines:
            raise _Unsupported(f"batter {batter} has a PA but no boxscore batting line")
        box_hits = lines[batter][0] if batter in lines else 0
        if whole_hits[batter] != box_hits:
            raise _Unsupported(f"batter {batter} hits do not reconcile: plays {whole_hits[batter]}, boxscore {box_hits}")
    box_pa = {b: line[1] for b, line in lines.items()}
    return GameOutcomes(game_pk, True, None, pre, dict(whole_pa), box_pa)
