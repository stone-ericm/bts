#!/usr/bin/env python
"""F9 Stage 1: how PREVALENT are the two eval-basis contaminations? (read-only)

The estimated_pa basis (backtest_blend._starter_matchup_representative_rows):
  (a) eligibility = batters with >=1 PA vs the REALIZED first pitcher
  (b) starter identity = realized first pitcher (hindsight)
Production ranks pregame lineups vs the PROBABLE pitcher.

Measures, per season 2021-2025:
  1. dropped batter-games (the audit's 11.33%) split by STARTING-LINEUP
     membership (first 9 distinct batters per side in PA order) — bench/PH
     drops are irrelevant (production excludes them too); LINEUP drops are
     the real eligibility gap.
  2. probable-vs-realized starter mismatch rate per (game, batter-side),
     from the pregame probablePitchers lookup.
"""
import json

import pandas as pd

LOOKUP = {int(k): v for k, v in json.load(
    open("data/models/probable_pitcher_lookup.json")).items()}

print(f"{'season':>6} {'batter_games':>12} {'dropped':>8} {'drop%':>6} "
      f"{'lineup_dropped':>14} {'lineupdrop%':>11} {'st_mismatch%':>12} {'no_prob%':>8}")

tot = {"bg": 0, "dr": 0, "lu": 0, "ludr": 0, "sides": 0, "mm": 0, "np": 0}
for season in (2021, 2022, 2023, 2024, 2025):
    pa = pd.read_parquet(
        f"data/processed/pa_{season}.parquet",
        columns=["game_pk", "is_home", "pitcher_id", "batter_id", "is_hit"],
    )
    work = pa.copy()
    work["_row"] = range(len(work))

    starters = (work.groupby(["game_pk", "is_home"], sort=False)["pitcher_id"]
                .first().rename("st").reset_index())
    work = work.merge(starters, on=["game_pk", "is_home"], how="left", sort=False)

    bg = work[["batter_id", "game_pk", "is_home"]].drop_duplicates()
    faced = (work[work["pitcher_id"] == work["st"]]
             [["batter_id", "game_pk"]].drop_duplicates().assign(kept=True))
    bg = bg.merge(faced, on=["batter_id", "game_pk"], how="left")
    bg["kept"] = bg["kept"].fillna(False).astype(bool)

    first9 = (work.sort_values("_row")
              .drop_duplicates(["game_pk", "is_home", "batter_id"])
              .groupby(["game_pk", "is_home"], sort=False).head(9)
              [["batter_id", "game_pk"]].assign(in_lineup=True))
    bg = bg.merge(first9, on=["batter_id", "game_pk"], how="left")
    bg["in_lineup"] = bg["in_lineup"].fillna(False).astype(bool)

    dropped = bg[~bg["kept"]]
    lineup_dropped = dropped[dropped["in_lineup"]]
    n_lineup = int(bg["in_lineup"].sum())

    # starter identity: batter-side is_home faces the AWAY team's pitcher
    sides = starters.copy()
    sides["probable"] = sides.apply(
        lambda r: (LOOKUP.get(int(r["game_pk"])) or {}).get(
            "away" if r["is_home"] else "home"), axis=1)
    have = sides[sides["probable"].notna()]
    mm = (have["st"] != have["probable"]).mean() if len(have) else float("nan")
    nop = 1 - len(have) / len(sides)

    print(f"{season:>6} {len(bg):>12} {len(dropped):>8} "
          f"{100*len(dropped)/len(bg):>5.1f}% {len(lineup_dropped):>14} "
          f"{100*len(lineup_dropped)/max(n_lineup,1):>10.2f}% "
          f"{100*mm:>11.2f}% {100*nop:>7.2f}%")
    tot["bg"] += len(bg); tot["dr"] += len(dropped)
    tot["lu"] += n_lineup; tot["ludr"] += len(lineup_dropped)
    tot["sides"] += len(have); tot["mm"] += int((have["st"] != have["probable"]).sum())
    tot["np"] += len(sides) - len(have)

print(f"\nALL: batter-games {tot['bg']}, dropped {100*tot['dr']/tot['bg']:.2f}% "
      f"| LINEUP members dropped {100*tot['ludr']/tot['lu']:.2f}% of lineup slots "
      f"({tot['ludr']} of {tot['lu']}) "
      f"| starter mismatch {100*tot['mm']/tot['sides']:.2f}% of sides "
      f"| probable missing {100*tot['np']/(tot['sides']+tot['np']):.2f}%")
print("\nInterpretation: the audit's drop% includes bench/PH production never "
      "ranks. 'lineupdrop%' and 'st_mismatch%' are the production-relevant "
      "contamination rates.")


# ---------------------------------------------------------------------------
# Part 2 (run separately on the box): live cross-check on captured production
# slates — at serving time, does the slate's pitcher_id match a realized
# starter? Result 2026-07-10: 29 days, 286 top-10 candidate rows, 0 mismatches
# (plus 4 rows in games missing from the PA frame: postponed/suspended).
# See docs/audit/2026-07-09-gpt56-sol-audit.md, F9 disposition.
# ---------------------------------------------------------------------------
