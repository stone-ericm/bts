"""C1 rank 3, T1: certified starting identities and the chronological PA list from one MLB v1.1 feed (registration
`docs/sota_audit/2026-10-04-prereg-c1-pa-count.md` §4; plan `docs/superpowers/plans/2026-10-05-c1-r3-count-build.md`).

- **Starting batters:** a boxscore player whose `battingOrder` is an exact multiple of 100 ("100".."900") started in
  slot `battingOrder / 100`; "X01"+ codes are substitutes in slot X. The production parser's `lineup_position`
  (`battingOrder // 100`) mixes the two, which is why the build needs this metadata.
- **Starting pitcher:** `boxscore.teams[side].pitchers[0]`, cross-checked against the pitcher who faced the opposing
  side's first PA.
- **PAs:** `allPlays` in feed order, kept when the event is in production's `PA_ENDING_EVENTS`. A top half means the
  away side is batting. The resumed-portion flag follows `bts.data.build.parse_game_feed` exactly: with a
  `resumeDateTime`, a play at or after it (or with no parseable start time) is the resumed portion.

Structural problems are listed, not repaired. T2 (`count_verify`) turns them into quarantine decisions.
"""
from __future__ import annotations

from dataclasses import dataclass

from bts.data.build import _parse_feed_timestamp
from bts.data.schema import PA_ENDING_EVENTS

SIDES = ("away", "home")


@dataclass(frozen=True)
class PA:
    index: int
    side: str          # the batting side
    batter: int
    pitcher: int
    event: str
    resumed: bool


@dataclass(frozen=True)
class GameMeta:
    game_pk: int
    official_date: str
    season: int
    starters: dict      # side -> {slot: batter_id}
    substitutes: dict   # side -> {batter_id: slot}
    starting_pitcher: dict  # side -> pitcher_id (the side's own starter)
    pas: tuple
    problems: tuple


def _lineups(players: dict, side: str, problems: list) -> tuple[dict, dict]:
    starters, subs = {}, {}
    for p in players.values():
        code = p.get("battingOrder")
        if code is None:
            continue
        pid = p["person"]["id"]
        try:
            n = int(code)
        except (TypeError, ValueError):
            problems.append(f"{side}: battingOrder {code!r} for {pid} is not a number")
            continue
        slot, rank = divmod(n, 100)
        if not 1 <= slot <= 9:
            problems.append(f"{side}: battingOrder {code!r} for {pid} is outside slots 1-9")
        elif rank == 0:
            if slot in starters:
                problems.append(f"{side}: duplicate starter in slot {slot} ({starters[slot]}, {pid})")
            else:
                starters[slot] = pid
        else:
            subs[pid] = slot
    missing = sorted(set(range(1, 10)) - set(starters))
    if missing:
        problems.append(f"{side}: starting slots {missing} are missing")
    return starters, subs


def extract(feed: dict) -> GameMeta:
    gd, ld = feed["gameData"], feed["liveData"]
    box = ld["boxscore"]["teams"]
    problems: list[str] = []
    starters, subs, sp = {}, {}, {}
    for side in SIDES:
        starters[side], subs[side] = _lineups(box[side].get("players", {}), side, problems)
        pitchers = box[side].get("pitchers") or []
        sp[side] = pitchers[0] if pitchers else None
        if sp[side] is None:
            problems.append(f"{side}: no boxscore pitchers")
    resume_dt = _parse_feed_timestamp(gd["datetime"].get("resumeDateTime"))
    pas, last = [], None
    for play in ld["plays"]["allPlays"]:
        about = play["about"]
        idx = about.get("atBatIndex")
        if not isinstance(idx, int) or (last is not None and idx <= last):
            problems.append(f"atBatIndex {idx!r} is not strictly increasing after {last!r}")
        last = idx if isinstance(idx, int) else last
        event = play["result"].get("eventType", "")
        if event not in PA_ENDING_EVENTS:
            continue
        resumed = False
        if resume_dt is not None:
            start = _parse_feed_timestamp(about.get("startTime"))
            resumed = start is None or start >= resume_dt
        side = "home" if about["halfInning"] == "bottom" else "away"
        pas.append(PA(index=idx, side=side, batter=play["matchup"]["batter"]["id"],
                      pitcher=play["matchup"]["pitcher"]["id"], event=event, resumed=resumed))
    for side in SIDES:                      # side's starter faced the other side first
        first = next((p for p in pas if p.side != side), None)
        if first is not None and sp[side] is not None and first.pitcher != sp[side]:
            problems.append(f"{side}: starting pitcher {sp[side]} did not face the first opposing PA ({first.pitcher})")
    return GameMeta(game_pk=gd["game"]["pk"], official_date=gd["datetime"]["officialDate"],
                    season=int(gd["game"]["season"]), starters=starters, substitutes=subs, starting_pitcher=sp,
                    pas=tuple(pas), problems=tuple(problems))
