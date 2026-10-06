"""C1 rank 3, T2: certify or quarantine each game before any aggregation (registration §4; review r1 F4, F5).

**A game is certified only when:**
- the file's gamePk, the feed's top-level gamePk and `gameData.game.pk` agree; the season matches; and the official
  date matches the PA parquet's date for the game;
- T1 found no structural problem: codes, identities, slots, completion, contiguous plays, chronology, the
  starting-pitcher check, timestamps;
- every batter belongs to that side's starters or substitutes;
- **batting turns hold** over every play with a result, not just production's PA events (r2 N3; see
  `_turn_problems`):
  - the first turn is slot 1;
  - a completed turn advances one slot. An intentional walk completes a turn but is not a production PA;
  - an open turn (e.g. an inning-ending caught stealing) must end its half-inning, and its slot must lead off the
    side's next one;
- **official totals reconcile (r2 N2):** each side's `plateAppearances` and its starter's `battersFaced` equal the
  feed's completed turns. With the completed status, the contiguous plays, a complete last play and
  `linescore.currentInning`, this is the completeness evidence independent of the parquet;
- **substitutions hold:** once a slot's substitute has batted, that slot's starter never bats again, and a slot's
  substitutes bat in rank order;
- **a starter with no play has evidence:** a substitute in that slot batted, or the side's turns never reached the slot;
- **completeness against the parquet:** its PA-ending events match the PA parquet's rows for the game, per
  (side, batter), counting all PAs. The 2021–2025 parquets predate `is_resumed_portion`, so
  `read_pa_for_bts_scoring` excludes nothing there; the resumed flag comes from the feed.

  This parquet agreement is in addition to the parquet-independent completion evidence: a completed status and a
  contiguous play list.

**Eligibility:** eligible games are those in the production scoring parquets (production's regular-season set; the
7-inning COVID doubleheaders are already dropped). An eligible game without a usable feed is quarantined. A feed with
no parquet rows is ineligible: counted, not certified.

**Census:** every quarantined game is counted with its reasons. The build stops if more than 1% of eligible games are
quarantined; a bad game is never kept because the total is small.
"""
from __future__ import annotations

from collections import Counter

from scripts.audit.c1_r3.count_meta import SIDES, GameMeta

STOP_RATE = 0.01


def _slot_of(meta: GameMeta, side: str) -> dict:
    return {pid: sl for sl, pid in meta.starters[side].items()} | {pid: sl for pid, (sl, _) in meta.substitutes[side].items()}


def _half_ends(meta: GameMeta) -> set:
    """Indexes of plays that end their half-inning (the next play is in another half, or there is none)."""
    out = set()
    for a, b in zip(meta.plays, meta.plays[1:] + (None,)):
        if b is None or (b.inning, b.side) != (a.inning, a.side):
            out.add(a.index)
    return out


def _turn_problems(meta: GameMeta, side: str) -> list[str]:
    """Batting turns (r2 N3):
    - the side's first turn is slot 1;
    - after a completed turn the next play is the following slot (9 wraps to 1);
    - after an open turn (any other result) the same slot leads off the side's next half-inning, and the open play
      must have ended its half-inning.

    Anything else is a problem."""
    slot_of, ends = _slot_of(meta, side), _half_ends(meta)
    plays = [p for p in meta.plays if p.side == side and p.batter in slot_of]
    out = []
    if plays and slot_of[plays[0].batter] != 1:
        out.append(f"{side}: the first batting turn is slot {slot_of[plays[0].batter]}, not 1")
    for prev, cur in zip(plays, plays[1:]):
        ps, cs = slot_of[prev.batter], slot_of[cur.batter]
        if prev.completes_turn:
            if cs != ps % 9 + 1:
                out.append(f"{side}: after slot {ps} completed its turn, slot {cs} batted (play {cur.index})")
        elif not (prev.index in ends and cs == ps and cur.inning > prev.inning):
            out.append(f"{side}: an open turn in slot {ps} (play {prev.index}, {prev.event}) was not resumed by "
                       f"slot {ps} leading off the next half-inning")
    return out


def _substitution_problems(meta: GameMeta, side: str) -> list[str]:
    out = []
    for slot, starter in meta.starters[side].items():
        subs = {pid: rank for pid, (s, rank) in meta.substitutes[side].items() if s == slot}
        seen_sub, last_rank = False, 0
        for p in (x for x in meta.plays if x.side == side):
            if p.batter == starter and seen_sub:
                out.append(f"{side}: slot {slot} starter {starter} bats after a substitute replaced him")
            elif p.batter in subs:
                seen_sub = True
                if subs[p.batter] < last_rank:
                    out.append(f"{side}: slot {slot} substitutes bat out of rank order")
                last_rank = max(last_rank, subs[p.batter])
    return out


def _zero_pa_problems(meta: GameMeta, side: str) -> list[str]:
    """A starter with no play needs evidence: his slot was never reached, or a substitute of that slot batted."""
    slot_of = _slot_of(meta, side)
    reached = {slot_of[p.batter] for p in meta.plays if p.side == side and p.batter in slot_of}
    batted = {p.batter for p in meta.plays if p.side == side}
    out = []
    for slot, starter in meta.starters[side].items():
        if starter in batted or slot not in reached:
            continue
        if not any(sl == slot and pid in batted for pid, (sl, _) in meta.substitutes[side].items()):
            out.append(f"{side}: slot {slot} starter {starter} has no play and no replacement evidence")
    return out


def _total_problems(meta: GameMeta, side: str) -> list[str]:
    """Official totals against the feed's completed turns (r2 N2): the side's plateAppearances, and its starter's
    battersFaced. A completed turn is production's PA events plus intent_walk and batter_interference. A missing
    total or an unaccounted difference is a problem; the frozen N and BF definitions are unchanged."""
    out = []
    turns = sum(1 for p in meta.plays if p.side == side and p.completes_turn)
    if meta.team_pa[side] is None:
        out.append(f"{side}: no official plateAppearances total")
    elif meta.team_pa[side] != turns:
        out.append(f"{side}: official plateAppearances {meta.team_pa[side]} != {turns} completed turns in the feed")
    sp = meta.starting_pitcher[side]
    faced = sum(1 for p in meta.plays if p.side != side and p.pitcher == sp and p.completes_turn)
    if meta.box_batters_faced[side] is None:
        out.append(f"{side}: no official battersFaced for starting pitcher {sp}")
    elif meta.box_batters_faced[side] != faced:
        out.append(f"{side}: starting pitcher {sp} official battersFaced {meta.box_batters_faced[side]} != {faced} "
                   "completed turns faced in the feed")
    return out


def verify_game(meta: GameMeta, *, pk: int, season: int, parquet: Counter | None,
                parquet_date: str | None = None) -> list[str]:
    reasons = list(meta.problems)
    if meta.game_pk != pk or meta.top_level_pk != pk:
        reasons.append(f"identity: file {pk}, top-level {meta.top_level_pk!r}, gameData {meta.game_pk!r} disagree")
    if meta.season != season:
        reasons.append(f"feed season {meta.season} is not {season}")
    if parquet_date is not None and meta.official_date != parquet_date:
        reasons.append(f"officialDate {meta.official_date} is not the parquet's {parquet_date}")
    for side in SIDES:
        known = set(meta.starters[side].values()) | set(meta.substitutes[side])
        strangers = sorted({p.batter for p in meta.plays if p.side == side and p.batter not in known})
        if strangers:
            reasons.append(f"batters {strangers[:3]} are not in the {side} lineup")
        reasons += (_turn_problems(meta, side) + _substitution_problems(meta, side) + _zero_pa_problems(meta, side)
                    + _total_problems(meta, side))
    feed_counts = Counter((p.side == "home", p.batter) for p in meta.pas)
    if parquet is None:
        reasons.append("no production scoring PA rows for this game")
    elif feed_counts != parquet:
        reasons.append(f"PA completeness: the feed has {sum(feed_counts.values())} PAs, the parquet "
                       f"{sum(parquet.values())}, or they differ by batter")
    return reasons


def census(results: dict, *, eligible: set, feeds_without_parquet: set) -> dict:
    """results: {gamePk: reasons} for every eligible game processed. Eligible games without a usable feed are
    quarantined."""
    quarantined = {pk: r for pk, r in results.items() if r and pk in eligible}
    for pk in sorted(eligible - set(results)):
        quarantined[pk] = ["no re-acquired feed for an eligible game"]
    rate = len(quarantined) / len(eligible) if eligible else 1.0
    return {"eligible": len(eligible), "certified": len(eligible) - len(quarantined),
            "quarantined": dict(sorted(quarantined.items())), "rate": rate, "stop": rate > STOP_RATE,
            "ineligible_feeds": len(feeds_without_parquet),
            "reasons": dict(Counter(r.split(":")[0] for rs in quarantined.values() for r in rs))}
