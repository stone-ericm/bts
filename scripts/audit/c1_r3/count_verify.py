"""C1 rank 3, T2: certify or quarantine each game before any aggregation (registration §4; review r1 F4, F5).

**A game is certified only when:**
- the file's gamePk, the feed's top-level gamePk and `gameData.game.pk` agree; the season matches; and the official
  date matches the PA parquet's date for the game;
- T1 found no structural problem: codes, identities, slots, completion, contiguous plays, chronology, the
  starting-pitcher check, timestamps;
- every batter belongs to that side's starters or substitutes;
- **the batting order holds** over every play with a result, not just production's PA events (an intentional walk
  advances the order but is not a production PA): each side's successive plays advance one slot (9 wraps to 1). The
  same slot repeats only across a half-inning boundary, when an inning ended on a runner out with the batter still
  at the plate;
- **substitutions hold:** once a slot's substitute has batted, that slot's starter never bats again, and a slot's
  substitutes bat in rank order;
- **a starter with no PA has evidence:** a substitute in that slot batted, or that side's PAs never reached the slot;
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


def _order_problems(meta: GameMeta, side: str) -> list[str]:
    slot_of = {pid: s for s, pid in meta.starters[side].items()} | {pid: s for pid, (s, _) in meta.substitutes[side].items()}
    out, prev = [], None
    for p in (x for x in meta.plays if x.side == side):
        slot = slot_of.get(p.batter)
        if slot is None:
            continue                                   # reported as a stranger below
        if prev is not None:
            prev_slot, prev_inning = prev
            step = (slot - prev_slot) % 9
            if not (step == 1 or (step == 0 and p.inning != prev_inning)):
                out.append(f"{side}: batting order jumps from slot {prev_slot} to {slot} at play {p.index}")
        prev = (slot, p.inning)
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
    out = []
    side_pas = [p for p in meta.plays if p.side == side]
    batted = {p.batter for p in side_pas}
    for slot, starter in meta.starters[side].items():
        if starter in batted:
            continue
        replaced = any(s == slot and pid in batted for pid, (s, _) in meta.substitutes[side].items())
        if not (replaced or len(side_pas) < slot):
            out.append(f"{side}: slot {slot} starter {starter} has no PA and no replacement evidence")
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
        reasons += _order_problems(meta, side) + _substitution_problems(meta, side) + _zero_pa_problems(meta, side)
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
