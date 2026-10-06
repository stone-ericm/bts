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
- **Plays (r3 R3-3):** every raw play is checked before anything is built from it, and none is dropped silently.
  - `about.isComplete` must be exactly true;
  - the result must be a supported completed-turn result (`COMPLETED_TURN`) or open-turn result (`OPEN_TURN`).
    Anything else (an unknown code, a pending ruling, a plate-appearance code production does not count) is a
    problem;
  - the batter and pitcher must be valid ids;
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
- **Completion:**
  - the feed's `gameData.status.detailedState` must be exactly "Final", "Game Over" or "Completed Early", optionally
    followed by ": <reason>";
  - the last play must be complete, in `linescore.currentInning`.
- **Official totals (r2 N2):** each side's team `plateAppearances` and each starter's `battersFaced` are read for T2
  to reconcile against the completed batting turns.
- **Identity (r2 N4, r3 R3-4):**
  - every id is a positive builtin int; the season is a 4-digit string or int;
  - `officialDate` is a real ISO date in that season;
  - innings are positive ints, and the half-innings move forward;
  - no lineup person appears on both sides;
  - a boxscore `players` entry used for a lineup or a starter's battersFaced is keyed `ID<its person.id>`;
  - a substitute is declared once, and each (slot, rank) once;
  - the two sides' starting pitchers differ, and neither is in the other side's lineup. A pitcher in his own
    side's lineup is legal.

Problems are listed, not repaired; T2 (`count_verify`) turns them into quarantine decisions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from bts.data.build import _parse_feed_timestamp
from bts.data.schema import PA_ENDING_EVENTS

SIDES = ("away", "home")
CODE_RE = re.compile(r"([1-9])(\d\d)")
COMPLETED = ("Final", "Game Over", "Completed Early")
STATUS_RE = re.compile(r"(Final|Game Over|Completed Early)(: .+)?")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
SEASON_RE = re.compile(r"(19|20)\d\d")
# A batting turn completes on production's PA events plus the official PA events production does not count
# (intentional walks, batter interference). Any other result leaves the turn open (e.g. an inning-ending
# caught stealing): the same slot must lead off the side's next half-inning.
COMPLETED_TURN = frozenset(PA_ENDING_EVENTS) | {"intent_walk", "batter_interference"}
# The supported open-turn results (r3 R3-3): MLB's own base-running, non-plate-appearance codes (statsapi
# /api/v1/eventTypes, snapshot `mlb_event_types_20261005.json`: baseRunningEvent and not plateAppearance), less the
# pending ruling `os_ruling_pending_prior`. They end a play record without the batter completing his turn, e.g. an
# inning-ending caught stealing or a game-ending wild pitch.
OPEN_TURN = frozenset({
    "balk", "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home", "cs_double_play", "defensive_indiff",
    "error", "forced_balk", "other_advance", "other_out", "passed_ball", "pickoff_1b", "pickoff_2b", "pickoff_3b",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b", "pickoff_caught_stealing_home", "pickoff_error_1b",
    "pickoff_error_2b", "pickoff_error_3b", "runner_double_play", "stolen_base_2b", "stolen_base_3b",
    "stolen_base_home", "wild_pitch"})


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
    batting-turn, substitution and zero-PA checks need the full sequence, not just production's PA events."""
    index: int
    side: str
    inning: int
    batter: int
    pitcher: int
    event: str

    @property
    def completes_turn(self) -> bool:
        return self.event in COMPLETED_TURN


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
    box_batters_faced: dict  # side -> the starter's official battersFaced (reconciled in T2)
    team_pa: dict            # side -> the side's official plateAppearances (reconciled in T2)
    pas: tuple
    problems: tuple
    plays: tuple = ()
    feed_timestamp: str | None = None   # metaData.timeStamp: the producer version of this feed (provenance)


def _pid(v) -> int | None:
    return v if type(v) is int and v > 0 else None


def _lineups(players: dict, side: str, problems: list) -> tuple[dict, dict]:
    starters, subs, ranks = {}, {}, {}
    for key, p in players.items():
        code = p.get("battingOrder")
        if code is None:
            continue
        pid = _pid((p.get("person") or {}).get("id"))
        m = CODE_RE.fullmatch(code) if isinstance(code, str) else None
        if pid is None:
            problems.append(f"{side}: a lineup player without a positive integer id")
            continue
        if key != f"ID{pid}":
            problems.append(f"{side}: players entry {key!r} holds person {pid}")
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
        elif pid in subs:
            problems.append(f"{side}: substitute {pid} is declared twice")
        elif (slot, rank) in ranks:
            problems.append(f"{side}: slot {slot} substitute rank {rank} is declared for {ranks[(slot, rank)]} and {pid}")
        else:
            ranks[(slot, rank)] = pid
            subs[pid] = (slot, rank)
    missing = sorted(set(range(1, 10)) - set(starters))
    if missing:
        problems.append(f"{side}: starting slots {missing} are missing")
    if len(set(starters.values())) != len(starters):
        problems.append(f"{side}: the same person starts in two slots")
    if set(starters.values()) & set(subs):
        problems.append(f"{side}: a person is both a starter and a substitute")
    return starters, subs


def _box_bf(players: dict, pid: int, side: str, problems: list) -> int | None:
    p = players.get(f"ID{pid}")
    if p is None:
        return None
    if _pid((p.get("person") or {}).get("id")) != pid:
        problems.append(f"{side}: players entry ID{pid} holds person {(p.get('person') or {}).get('id')!r}")
        return None
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
        box_bf[side] = _box_bf(players, sp[side], side, problems) if sp[side] else None
    ids = {s: set(starters[s].values()) | set(subs[s]) for s in SIDES}
    if ids["away"] & ids["home"]:
        problems.append("a lineup person appears for both sides")
    if sp["away"] is not None and sp["away"] == sp["home"]:
        problems.append(f"both sides declare starting pitcher {sp['away']}")
    for side, other in (("away", "home"), ("home", "away")):
        if sp[side] is not None and sp[side] in ids[other]:
            problems.append(f"{side}: starting pitcher {sp[side]} is in the {other} lineup")
    status = (gd.get("status") or {}).get("detailedState") or ""
    completed = isinstance(status, str) and STATUS_RE.fullmatch(status) is not None
    if not completed:
        problems.append(f"the feed is not a completed game (status {status!r})")
    official_date = gd["datetime"].get("officialDate")
    raw_season = gd["game"].get("season")
    if type(raw_season) is int and not isinstance(raw_season, bool) and 1900 <= raw_season <= 2099:
        season = raw_season
    elif isinstance(raw_season, str) and SEASON_RE.fullmatch(raw_season):
        season = int(raw_season)
    else:
        problems.append(f"season {raw_season!r} is not a 4-digit year")
        season = -1
    try:
        valid_date = isinstance(official_date, str) and bool(DATE_RE.fullmatch(official_date)) and \
            datetime.fromisoformat(official_date).year == season
    except ValueError:
        valid_date = False
    if not valid_date:
        problems.append(f"officialDate {official_date!r} is not an ISO date in season {season}")
    team_pa = {}
    for side in SIDES:
        v = ((box[side].get("teamStats") or {}).get("batting") or {}).get("plateAppearances")
        team_pa[side] = v if type(v) is int and v >= 0 else None
    try:
        resume_dt = _parse_feed_timestamp(gd["datetime"].get("resumeDateTime"))
    except ValueError as e:
        problems.append(f"unparseable resumeDateTime: {e}")
        resume_dt = None
    pas, plays, first_pitcher = [], [], {}
    last_start: datetime | None = None
    last_half: tuple | None = None
    all_plays = ld["plays"]["allPlays"]
    for n, play in enumerate(all_plays):
        about = play.get("about") or {}
        idx = about.get("atBatIndex")
        if idx != n or type(idx) is not int:
            problems.append(f"atBatIndex {idx!r} at play {n}: the play list is not contiguous from 0")
        half = about.get("halfInning")
        if half not in ("top", "bottom"):
            problems.append(f"play {n}: halfInning {half!r} is not top/bottom")
            continue
        side = "home" if half == "bottom" else "away"
        inning = about.get("inning")
        if not (type(inning) is int and inning >= 1):
            problems.append(f"play {n}: inning {inning!r} is not a positive integer")
            inning = -1
        this_half = (inning, 0 if half == "top" else 1)
        if last_half is not None and this_half < last_half:
            problems.append(f"play {n}: the half-innings go backwards")
        last_half = this_half
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
        complete = about.get("isComplete")
        if complete is not True:
            problems.append(f"play {n}: isComplete is {complete!r}, not true")
        event = (play.get("result") or {}).get("eventType")
        if not (isinstance(event, str) and (event in COMPLETED_TURN or event in OPEN_TURN)):
            problems.append(f"play {n}: result {event!r} is not a supported completed-turn or open-turn result")
            continue
        if batter is None or pitcher is None:
            problems.append(f"play {n}: a result without valid batter/pitcher ids")
            continue
        plays.append(Play(index=idx, side=side, inning=inning, batter=batter, pitcher=pitcher, event=event))
        if event not in PA_ENDING_EVENTS:
            continue
        resumed = False
        if resume_dt is not None:
            resumed = start is None or start >= resume_dt
        pas.append(PA(index=idx, side=side, inning=inning, batter=batter,
                      pitcher=pitcher, event=event, resumed=resumed))
    if not all_plays:
        problems.append("the feed has no plays")
    else:
        last = all_plays[-1].get("about") or {}
        if last.get("isComplete") is not True:
            problems.append("the last play is not complete")
        cur = (ld.get("linescore") or {}).get("currentInning")
        if not (type(cur) is int and cur == last.get("inning")):
            problems.append(f"the last play's inning {last.get('inning')!r} is not linescore.currentInning {cur!r}")
    for side in SIDES:                      # the side's starter must be the first pitcher the other side faced
        other = "home" if side == "away" else "away"
        if other not in first_pitcher:
            problems.append(f"{side}: no play of the opposing half to confirm the starting pitcher")
        elif sp[side] is not None and first_pitcher[other] != sp[side]:
            problems.append(f"{side}: starting pitcher {sp[side]} was not the first to face the opposing side "
                            f"({first_pitcher[other]})")
    game_pk = _pid(gd["game"].get("pk"))
    if game_pk is None:
        problems.append(f"gameData.game.pk {gd['game'].get('pk')!r} is not a positive integer")
    return GameMeta(game_pk=game_pk, top_level_pk=_pid(feed.get("gamePk")), official_date=official_date,
                    season=season, status=status, completed=completed, starters=starters, substitutes=subs,
                    starting_pitcher=sp, box_batters_faced=box_bf, team_pa=team_pa, pas=tuple(pas),
                    problems=tuple(problems),
                    plays=tuple(plays),
                    feed_timestamp=(feed.get("metaData") or {}).get("timeStamp"))
