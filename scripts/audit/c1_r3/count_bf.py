"""C1 rank 3, T4: each certified starting pitcher's per-start workload (registration §2, §4).

BF is every PA-ending event the side's starting pitcher faced in the game, including the resumed portion: workload
is a baseball fact, unlike the contest-scoring count target. Starts are keyed by official date, so the forecast-time
lag (strictly earlier official dates, the latest five available starts) can be applied without row shifts. The fixed
2021-2025 league median per-start BF is the fallback when a pitcher has no prior start.
"""
from __future__ import annotations

import statistics

from scripts.audit.c1_r3.count_meta import SIDES, GameMeta


def starts(meta: GameMeta) -> list[dict]:
    out = []
    for side in SIDES:
        sp = meta.starting_pitcher[side]
        bf = sum(1 for p in meta.pas if p.side != side and p.pitcher == sp)
        out.append({"pitcher": sp, "game_pk": meta.game_pk, "official_date": meta.official_date, "side": side,
                    "bf": bf, "bf_resumed": sum(1 for p in meta.pas if p.side != side and p.pitcher == sp and p.resumed)})
    return out


def league_median(all_starts) -> float:
    bfs = [s["bf"] for s in all_starts]
    if not bfs:
        raise ValueError("no starts")
    return float(statistics.median(bfs))
