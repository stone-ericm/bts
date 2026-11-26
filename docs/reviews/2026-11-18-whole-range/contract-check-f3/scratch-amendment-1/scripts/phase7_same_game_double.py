"""Phase 7: Test whether double-downs should require different games.

Uses existing backtest profiles to compare:
1. Any double (current): rank-1 + rank-2 regardless of game
2. Different-game double: rank-1 + highest-ranked batter in a DIFFERENT game

Measures P(57) under both rules using the MDP.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase7_same_game_double.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from arch_eval import compute_metrics, print_comparison

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]


def build_game_pk_lookup(pa_data: pd.DataFrame) -> dict:
    """Build (date, batter_id) → game_pk mapping from PA data."""
    pa_games = pa_data.groupby(["date", "batter_id"])["game_pk"].first().reset_index()
    pa_games["date"] = pd.to_datetime(pa_games["date"]).dt.date
    lookup = {}
    for _, row in pa_games.iterrows():
        lookup[(row["date"], row["batter_id"])] = int(row["game_pk"])
    return lookup


def add_game_pk_to_profiles(profiles: pd.DataFrame, gp_lookup: dict) -> pd.DataFrame:
    """Add game_pk to profiles via PA data lookup."""
    profiles = profiles.copy()
    profiles["_date_key"] = pd.to_datetime(profiles["date"]).dt.date
    profiles["game_pk"] = profiles.apply(
        lambda r: gp_lookup.get((r["_date_key"], r["batter_id"])), axis=1
    )
    profiles.drop(columns=["_date_key"], inplace=True)
    return profiles


def apply_different_game_rule(profiles: pd.DataFrame) -> pd.DataFrame:
    """Re-rank so rank-2 is always from a different game than rank-1.

    For each day, rank-1 stays the same. Rank-2 becomes the highest-ranked
    batter in a different game_pk. Other ranks shift accordingly.
    """
    result = []
    for date, day_df in profiles.groupby("date"):
        day_df = day_df.sort_values("rank").copy()

        if len(day_df) < 2:
            result.append(day_df)
            continue

        rank1 = day_df.iloc[0]
        rank1_gpk = rank1["game_pk"]

        # Find best batter in a different game
        diff_game = day_df[(day_df["game_pk"] != rank1_gpk) & (day_df["rank"] > 1)]

        if len(diff_game) == 0:
            # All top batters in same game — keep as is
            result.append(day_df)
            continue

        new_rank2 = diff_game.iloc[0]

        # Rebuild: rank1 stays, new_rank2 becomes rank 2, rest shifts
        new_day = pd.DataFrame([rank1.to_dict()])
        new_day = pd.concat([new_day, pd.DataFrame([new_rank2.to_dict()])], ignore_index=True)

        # Add remaining batters (excluding new_rank2) in original order
        remaining = day_df[
            (day_df["batter_id"] != rank1["batter_id"]) &
            (day_df["batter_id"] != new_rank2["batter_id"])
        ]
        new_day = pd.concat([new_day, remaining], ignore_index=True)
        new_day["rank"] = range(1, len(new_day) + 1)
        new_day["date"] = date

        result.append(new_day)

    return pd.concat(result, ignore_index=True)


def count_same_game_doubles(profiles: pd.DataFrame) -> dict:
    """Count how often rank-1 and rank-2 are in the same game."""
    same = 0
    total = 0
    for date, day_df in profiles.groupby("date"):
        day_df = day_df.sort_values("rank")
        if len(day_df) < 2:
            continue
        r1_gpk = day_df.iloc[0]["game_pk"]
        r2_gpk = day_df.iloc[1]["game_pk"]
        if pd.notna(r1_gpk) and pd.notna(r2_gpk):
            total += 1
            if r1_gpk == r2_gpk:
                same += 1
    return {"same_game": same, "diff_game": total - same, "total": total,
            "same_pct": same / total if total > 0 else 0}


def main():
    # Load profiles
    print("Loading backtest profiles...", file=sys.stderr)
    profiles_dfs = []
    sim_dir = Path("data/simulation")
    for season in TEST_SEASONS:
        path = sim_dir / f"backtest_{season}.parquet"
        if path.exists():
            profiles_dfs.append(pd.read_parquet(path))
    profiles = pd.concat(profiles_dfs, ignore_index=True)
    profiles["date"] = pd.to_datetime(profiles["date"])

    # Load PA data for game_pk
    print("Loading PA data for game_pk mapping...", file=sys.stderr)
    proc = Path("data/processed")
    pa_dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
    pa_data = pd.concat(pa_dfs, ignore_index=True)

    gp_lookup = build_game_pk_lookup(pa_data)
    profiles = add_game_pk_to_profiles(profiles, gp_lookup)

    valid = profiles["game_pk"].notna().sum()
    print(f"Profiles with game_pk: {valid}/{len(profiles)}", file=sys.stderr)

    # How often does this matter?
    stats = count_same_game_doubles(profiles)
    print(f"\nSame-game doubles: {stats['same_game']}/{stats['total']} "
          f"({stats['same_pct']:.1%})", file=sys.stderr)

    # Compute metrics for both rules
    results = {}

    print("\nComputing metrics: any-game doubles...", file=sys.stderr)
    results["any_game"] = compute_metrics(profiles)

    print("Computing metrics: different-game doubles...", file=sys.stderr)
    diff_profiles = apply_different_game_rule(profiles)
    results["diff_game"] = compute_metrics(diff_profiles)

    print_comparison(results, "Phase 7: Same-Game vs Different-Game Doubles")

    # Show what changes
    print(f"\nSame-game doubles occur {stats['same_pct']:.1%} of days "
          f"({stats['same_game']}/{stats['total']})")
    print(f"On those days, rank-2 shifts to a different game.")


if __name__ == "__main__":
    main()
