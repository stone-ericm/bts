"""C1 rank 3, T1: certified starting identities and the chronological PA list from one MLB v1.1 feed (registration
`docs/sota_audit/2026-10-04-prereg-c1-pa-count.md` §4; plan `docs/superpowers/plans/2026-10-05-c1-r3-count-build.md`).

Identities are validated before use: positive builtin integers, never coerced (review r1 F4).
- **Starting batters:**
  - A boxscore player whose `battingOrder` is a 3-digit string "X00" (X = 1..9) started in slot X. "Xnn" (nn ≥ 1)
    is that slot's nn-th substitute. Any other code is a problem.
  - The production parser's `lineup_position` (`battingOrder // 100`) mixes starters and substitutes, which is why
    the build needs this metadata.
  - Each side must have nine distinct starters, and no person may appear twice or on both sides.
- **Starting pitcher:** `boxscore.teams[side].pitchers[0]`, cross-checked against the pitcher in the first play of
  the opposing half: any play, PA-ending or not, so a starter removed before completing a PA is still seen.
- **Plays:**
  - every play's `atBatIndex` must run 0, 1, 2, … without a gap (a completeness witness independent of the PA
    parquet);
  - half-innings must be "top" or "bottom";
  - start times, when present, must not go backwards.
- **PAs:**
  - plays whose event is in production's `PA_ENDING_EVENTS` (which, like production's scoring definition, does not
    include `intent_walk`);
  - the resumed-portion flag follows `bts.data.build.parse_game_feed`: with a `resumeDateTime`, a play at or after
    it, or with no start time, is the resumed portion;
  - an unparseable timestamp is a problem here (production's helper raises on it).
- **Completion:** the feed's `gameData.status.detailedState` must be a completed state (Final / Game Over /
  Completed Early).
- **Date:** `officialDate`'s year must be the feed's season.

Problems are listed, not repaired; T2 (`count_verify`) turns them into quarantine decisions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from bts.data.build import _parse_feed_timestamp
from bts.data.schema import PA_ENDING_EVENTS

SIDES = ("away", "home")
CODE_RE = re.compile(r"^([1-9])(\d\d)$")
COMPLETED = ("Final", "Game Over", "Completed Early")


@dataclass(frozen=True)
class PA:
    index: int
    side: str          # the batting side
    inning: int
    batter: int
    pitcher: int
    event: str
    resumed: bool


@dataclass(frozen=True)
class Play:
    """Every play with a result (PA-ending or not, e.g. an intentional walk or an inning-ending caught stealing): the
    batting-order, substitution and zero-PA checks need the full sequence, not just production's PA events."""
    index: int
    side: str
    inning: int
    batter: int
    event: str


@dataclass(frozen=True)
class GameMeta:
    game_pk: int
    top_level_pk: object
    official_date: str
    season: int
    status: str
    completed: bool
    starters: dict      # side -> {slot: batter_id}
    substitutes: dict   # side -> {batter_id: (slot, rank)}
    starting_pitcher: dict  # side -> pitcher_id (the side's own starter)
    box_batters_faced: dict  # side -> the starter's boxscore battersFaced (descriptive; includes intentional walks)
    pas: tuple
    problems: tuple
    plays: tuple = ()
    feed_timestamp: str | None = None   # metaData.timeStamp: the producer version of this feed (provenance)


def _pid(v) -> int | None:
    return v if type(v) is int and v > 0 else None


def _lineups(players: dict, side: str, problems: list) -> tuple[dict, dict]:
    starters, subs = {}, {}
    for p in players.values():
        code = p.get("battingOrder")
        if code is None:
            continue
        pid = _pid((p.get("person") or {}).get("id"))
        m = CODE_RE.match(code) if isinstance(code, str) else None
        if pid is None:
            problems.append(f"{side}: a lineup player without a positive integer id")
            continue
        if m is None:
            problems.append(f"{side}: battingOrder {code!r} for {pid} is not a 3-digit slot code")
            continue
        slot, rank = int(m.group(1)), int(m.group(2))
        if rank == 0:
            if slot in starters:
                problems.append(f"{side}: duplicate starter in slot {slot} ({starters[slot]}, {pid})")
            else:
                starters[slot] = pid
        else:
            subs[pid] = (slot, rank)
    missing = sorted(set(range(1, 10)) - set(starters))
    if missing:
        problems.append(f"{side}: starting slots {missing} are missing")
    if len(set(starters.values())) != len(starters):
        problems.append(f"{side}: the same person starts in two slots")
    if set(starters.values()) & set(subs):
        problems.append(f"{side}: a person is both a starter and a substitute")
    return starters, subs


def _box_bf(players: dict, pid) -> int | None:
    p = players.get(f"ID{pid}") or {}
    v = ((p.get("stats") or {}).get("pitching") or {}).get("battersFaced")
    return v if type(v) is int and v >= 0 else None


def extract(feed: dict) -> GameMeta:
    gd, ld = feed["gameData"], feed["liveData"]
    box = ld["boxscore"]["teams"]
    problems: list[str] = []
    starters, subs, sp, box_bf = {}, {}, {}, {}
    for side in SIDES:
        players = box[side].get("players", {})
        starters[side], subs[side] = _lineups(players, side, problems)
        pitchers = box[side].get("pitchers") or []
        sp[side] = _pid(pitchers[0]) if pitchers else None
        if sp[side] is None:
            problems.append(f"{side}: no valid starting pitcher in the boxscore")
        box_bf[side] = _box_bf(players, sp[side]) if sp[side] else None
    if set(starters["away"].values()) & set(starters["home"].values()):
        problems.append("a person starts for both sides")
    status = (gd.get("status") or {}).get("detailedState") or ""
    completed = status.startswith(COMPLETED)
    if not completed:
        problems.append(f"the feed is not a completed game (status {status!r})")
    official_date = gd["datetime"]["officialDate"]
    season = gd["game"].get("season")
    try:
        season = int(season)
    except (TypeError, ValueError):
        problems.append(f"season {season!r} is not an integer")
        season = -1
    if not (isinstance(official_date, str) and official_date[:4] == str(season)):
        problems.append(f"officialDate {official_date!r} is not in season {season}")
    try:
        resume_dt = _parse_feed_timestamp(gd["datetime"].get("resumeDateTime"))
    except ValueError as e:
        problems.append(f"unparseable resumeDateTime: {e}")
        resume_dt = None
    pas, plays, first_pitcher = [], [], {}
    last_start: datetime | None = None
    for n, play in enumerate(ld["plays"]["allPlays"]):
        about = play.get("about") or {}
        idx = about.get("atBatIndex")
        if idx != n or type(idx) is not int:
            problems.append(f"atBatIndex {idx!r} at play {n}: the play list is not contiguous from 0")
        half = about.get("halfInning")
        if half not in ("top", "bottom"):
            problems.append(f"play {n}: halfInning {half!r} is not top/bottom")
            continue
        side = "home" if half == "bottom" else "away"
        try:
            start = _parse_feed_timestamp(about.get("startTime"))
        except ValueError:
            problems.append(f"play {n}: unparseable startTime")
            start = None
        if start is not None and last_start is not None and start < last_start:
            problems.append(f"play {n}: startTime goes backwards")
        last_start = start or last_start
        matchup = play.get("matchup") or {}
        batter = _pid((matchup.get("batter") or {}).get("id"))
        pitcher = _pid((matchup.get("pitcher") or {}).get("id"))
        if pitcher is not None:
            first_pitcher.setdefault(side, pitcher)
        event = (play.get("result") or {}).get("eventType", "")
        inning = about.get("inning")
        inning = inning if type(inning) is int else -1
        if event and batter is not None:
            plays.append(Play(index=idx, side=side, inning=inning, batter=batter, event=event))
        if event not in PA_ENDING_EVENTS:
            continue
        if batter is None or pitcher is None:
            problems.append(f"play {n}: a PA without valid batter/pitcher ids")
            continue
        resumed = False
        if resume_dt is not None:
            resumed = start is None or start >= resume_dt
        pas.append(PA(index=idx, side=side, inning=inning, batter=batter,
                      pitcher=pitcher, event=event, resumed=resumed))
    for side in SIDES:                      # the side's starter must be the first pitcher the other side faced
        other = "home" if side == "away" else "away"
        if other not in first_pitcher:
            problems.append(f"{side}: no play of the opposing half to confirm the starting pitcher")
        elif sp[side] is not None and first_pitcher[other] != sp[side]:
            problems.append(f"{side}: starting pitcher {sp[side]} was not the first to face the opposing side "
                            f"({first_pitcher[other]})")
    return GameMeta(game_pk=gd["game"]["pk"], top_level_pk=feed.get("gamePk"), official_date=official_date,
                    season=season, status=status, completed=completed, starters=starters, substitutes=subs,
                    starting_pitcher=sp, box_batters_faced=box_bf, pas=tuple(pas), problems=tuple(problems),
                    plays=tuple(plays),
                    feed_timestamp=(feed.get("metaData") or {}).get("timeStamp"))
