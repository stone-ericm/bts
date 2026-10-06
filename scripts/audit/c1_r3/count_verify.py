"""C1 rank 3, T2: certify or quarantine each game before any aggregation (registration §4).

A game is certified only when:
- its feed identity matches the requested file and season;
- T1 found no structural problem (slots, duplicates, chronology, the starting-pitcher cross-check);
- every PA batter belongs to that side's starters or substitutes;
- its PA-ending events match the production scoring parquet's rows for the game, per (side, batter), counting all
  PAs. The 2021-2025 parquets predate `is_resumed_portion`, so `read_pa_for_bts_scoring` excludes nothing there;
  the resumed flag comes from the feed (T1).

**Eligibility:** eligible games are the games in the production scoring parquets (production's regular-season set,
7-inning COVID doubleheaders already dropped). An eligible game without a re-acquired feed is quarantined. A feed with
no parquet rows is counted as ineligible, not quarantined.

**Census:** every quarantined game is counted with its reasons; the build stops if more than 1% of eligible games are
quarantined. A bad game is never kept because the total is small.
"""
from __future__ import annotations

from collections import Counter

from scripts.audit.c1_r3.count_meta import SIDES, GameMeta

STOP_RATE = 0.01


def verify_game(meta: GameMeta, *, pk: int, season: int, parquet: Counter | None) -> list[str]:
    reasons = list(meta.problems)
    if meta.game_pk != pk:
        reasons.append(f"feed gamePk {meta.game_pk} is not the file's {pk}")
    if meta.season != season:
        reasons.append(f"feed season {meta.season} is not {season}")
    for side in SIDES:
        known = set(meta.starters[side].values()) | set(meta.substitutes[side])
        strangers = sorted({p.batter for p in meta.pas if p.side == side and p.batter not in known})
        if strangers:
            reasons.append(f"batters {strangers[:3]} are not in the {side} lineup")
    feed_counts = Counter((p.side == "home", p.batter) for p in meta.pas)
    if parquet is None:
        reasons.append("no production scoring PA rows for this game")
    elif feed_counts != parquet:
        reasons.append(f"PA completeness: the feed has {sum(feed_counts.values())} PAs, the parquet "
                       f"{sum(parquet.values())}, or they differ by batter")
    return reasons


def census(results: dict, *, eligible: set, feeds_without_parquet: set) -> dict:
    """results: {gamePk: reasons} for every feed parsed. Eligible games without a feed are quarantined."""
    quarantined = {pk: r for pk, r in results.items() if r and pk in eligible}
    for pk in sorted(eligible - set(results)):
        quarantined[pk] = ["no re-acquired feed for an eligible game"]
    rate = len(quarantined) / len(eligible) if eligible else 1.0
    return {"eligible": len(eligible), "certified": len(eligible) - len(quarantined),
            "quarantined": dict(sorted(quarantined.items())), "rate": rate, "stop": rate > STOP_RATE,
            "ineligible_feeds": len(feeds_without_parquet),
            "reasons": dict(Counter(r.split(":")[0] for rs in quarantined.values() for r in rs))}
