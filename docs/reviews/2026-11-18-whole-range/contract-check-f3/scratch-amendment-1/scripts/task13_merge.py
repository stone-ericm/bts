"""Task 13: merge per-seed backtest results into harness-ready format.

Reads:
  data/simulation_seed{N}/backtest_{season}.parquet  (long: date, rank, batter_id, p_game_hit, actual_hit, n_pas)
  data/simulation_seed{N}/pa_predictions_{season}.parquet  (date, game_pk, batter_id, pa_index, p_hit_blend, is_hit)

Writes:
  data/simulation/profiles_seed{N}_season{S}.parquet  (top1/top2 pivoted, with seed+season columns)
  data/simulation/pa_predictions_seed{N}_season{S}.parquet  (renamed to harness conventions)
"""
from __future__ import annotations

import glob
import re
from pathlib import Path
import pandas as pd

OUT_DIR = Path("data/simulation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Profiles: pivot rank to top1/top2 ----
profile_paths = sorted(glob.glob("data/simulation_seed*/backtest_*.parquet"))
print(f"Found {len(profile_paths)} profile parquets")

n_pivoted = 0
for path in profile_paths:
    m = re.search(r"simulation_seed(\d+)/backtest_(\d+)\.parquet", path)
    if not m:
        print(f"  skipped (regex): {path}")
        continue
    seed = int(m.group(1))
    season = int(m.group(2))
    df = pd.read_parquet(path)
    if df.empty:
        print(f"  empty: {path}")
        continue
    rank1 = df[df["rank"] == 1].set_index("date")[["p_game_hit", "actual_hit", "batter_id"]]
    rank2 = df[df["rank"] == 2].set_index("date")[["p_game_hit", "actual_hit", "batter_id"]]
    pivoted = (
        rank1.rename(columns={
            "p_game_hit": "top1_p", "actual_hit": "top1_hit", "batter_id": "top1_batter_id"
        }).join(
            rank2.rename(columns={
                "p_game_hit": "top2_p", "actual_hit": "top2_hit", "batter_id": "top2_batter_id"
            }), how="inner",
        )
    )
    pivoted = pivoted.reset_index()
    pivoted["season"] = season
    pivoted["seed"] = seed
    out = OUT_DIR / f"profiles_seed{seed}_season{season}.parquet"
    pivoted.to_parquet(out, index=False)
    n_pivoted += 1
print(f"Wrote {n_pivoted} pivoted profile parquets to {OUT_DIR}")

# ---- PA predictions: rename to harness conventions ----
pa_paths = sorted(glob.glob("data/simulation_seed*/pa_predictions_*.parquet"))
print(f"\nFound {len(pa_paths)} PA prediction parquets")

n_pa = 0
for path in pa_paths:
    m = re.search(r"simulation_seed(\d+)/pa_predictions_(\d+)\.parquet", path)
    if not m:
        continue
    seed = int(m.group(1))
    season = int(m.group(2))
    df = pd.read_parquet(path)
    if df.empty:
        continue
    df = df.rename(columns={"p_hit_blend": "p_pa", "is_hit": "actual_hit"})
    df["season"] = season
    df["seed"] = seed
    # Build batter_game_id to match harness expectation (groupby unit for within-game PA correlation).
    df["batter_game_id"] = (
        df["batter_id"].astype(str) + "_"
        + df["game_pk"].astype(str) + "_"
        + df["date"].astype(str) + "_seed"
        + str(seed)
    )
    out = OUT_DIR / f"pa_predictions_seed{seed}_season{season}.parquet"
    df[["season", "seed", "date", "batter_game_id", "pa_index", "p_pa", "actual_hit"]].to_parquet(
        out, index=False
    )
    n_pa += 1
print(f"Wrote {n_pa} renamed PA prediction parquets to {OUT_DIR}")

# Summary
profiles_glob = sorted(glob.glob(str(OUT_DIR / "profiles_seed*_season*.parquet")))
pa_glob = sorted(glob.glob(str(OUT_DIR / "pa_predictions_seed*_season*.parquet")))
print(f"\nReady for harness:")
print(f"  --profiles-glob '{OUT_DIR}/profiles_seed*_season*.parquet'  ({len(profiles_glob)} files)")
print(f"  --pa-glob       '{OUT_DIR}/pa_predictions_seed*_season*.parquet'  ({len(pa_glob)} files)")
