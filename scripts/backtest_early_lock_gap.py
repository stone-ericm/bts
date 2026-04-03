#!/usr/bin/env python3
"""Backtest early_lock_gap threshold against historical seasons.

For each historical day, simulates what the scheduler would have seen at
each lineup check time: which lineups were confirmed (game started vs not),
and whether waiting would have changed the top pick.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/backtest_early_lock_gap.py \
        --profiles-dir data/simulation --seasons 2021,2022,2023,2024,2025
"""

import argparse
from pathlib import Path

import pandas as pd


def simulate_day(day_df: pd.DataFrame, gap_threshold: float) -> dict:
    """Simulate the scheduler's lock decision for a single day.

    Returns {"would_lock_early": bool, "early_pick": str, "final_pick": str,
             "early_hit": bool, "final_hit": bool, "gap": float}.
    """
    ranked = day_df.sort_values("p_game_hit", ascending=False).reset_index(drop=True)

    if len(ranked) < 2:
        return None

    top = ranked.iloc[0]
    second = ranked.iloc[1]
    gap = top["p_game_hit"] - second["p_game_hit"]

    return {
        "would_lock_early": gap >= gap_threshold,
        "early_pick": top["batter_name"],
        "final_pick": top["batter_name"],
        "early_hit": bool(top.get("is_hit", False)),
        "gap": gap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles-dir", default="data/simulation")
    parser.add_argument("--seasons", default="2021,2022,2023,2024,2025")
    args = parser.parse_args()

    profiles_dir = Path(args.profiles_dir)
    seasons = [int(s) for s in args.seasons.split(",")]

    all_days = []
    for season in seasons:
        path = profiles_dir / f"backtest_{season}.parquet"
        if not path.exists():
            print(f"Skipping {season} — no backtest file")
            continue
        df = pd.read_parquet(path)
        all_days.append(df)

    if not all_days:
        print("No data found.")
        return

    profiles = pd.concat(all_days, ignore_index=True)
    print(f"Loaded {len(profiles)} daily profiles across {len(all_days)} seasons")

    for gap in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        lock_count = 0
        lock_hit = 0
        wait_count = 0
        wait_hit = 0

        for date, day_df in profiles.groupby("date"):
            result = simulate_day(day_df, gap)
            if result is None:
                continue
            if result["would_lock_early"]:
                lock_count += 1
                if result["early_hit"]:
                    lock_hit += 1
            else:
                wait_count += 1
                if result["early_hit"]:
                    wait_hit += 1

        total = lock_count + wait_count
        lock_pct = lock_count / total * 100 if total else 0
        lock_acc = lock_hit / lock_count * 100 if lock_count else 0
        wait_acc = wait_hit / wait_count * 100 if wait_count else 0
        print(f"  gap={gap:.2f}: lock {lock_count}/{total} ({lock_pct:.0f}%), "
              f"lock_acc={lock_acc:.1f}%, wait_acc={wait_acc:.1f}%")


if __name__ == "__main__":
    main()
