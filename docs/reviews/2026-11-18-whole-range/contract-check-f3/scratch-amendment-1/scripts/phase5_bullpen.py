"""Phase 5: Team bullpen composite feature.

Computes opp_bullpen_hr_30g — rolling 30-day hit rate against each team's
relievers (identified via probable pitcher from raw game feeds).

Tests it as a new feature in FEATURE_COLS (all blend models see it).

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/phase5_bullpen.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from arch_eval import (
    load_data, walk_forward_backtest, compute_metrics, print_comparison,
)
from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
from bts.features.compute import FEATURE_COLS, STATCAST_COLS

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]


def build_game_info_lookup(raw_dir: str = "data/raw") -> dict:
    """Build game_pk → {away_team_id, home_team_id, away_probable_id, home_probable_id}.

    Uses raw game feed JSON files.
    """
    lookup = {}
    raw = Path(raw_dir)
    for season_dir in sorted(raw.iterdir()):
        if not season_dir.is_dir():
            continue
        for f in season_dir.glob("*.json"):
            try:
                d = json.loads(f.read_text())
                pk = int(f.stem)
                gd = d.get("gameData", {})
                teams = gd.get("teams", {})
                pp = gd.get("probablePitchers", {})
                lookup[pk] = {
                    "away_team_id": teams.get("away", {}).get("id"),
                    "home_team_id": teams.get("home", {}).get("id"),
                    "away_probable_id": pp.get("away", {}).get("id"),
                    "home_probable_id": pp.get("home", {}).get("id"),
                }
            except Exception:
                continue
    print(f"Built game info lookup: {len(lookup)} games", file=sys.stderr)
    return lookup


def compute_bullpen_feature(df: pd.DataFrame, game_info: dict) -> pd.DataFrame:
    """Add opp_bullpen_hr_30g to the DataFrame.

    For each PA, determines if the pitcher is a reliever (not the probable
    starter for their team). Then computes a rolling 30-day average hit rate
    against each team's relievers. The feature for each batter-game is the
    opposing team's bullpen quality.
    """
    df = df.copy()

    # Map game_pk to team IDs and probable pitcher IDs
    df["_away_team_id"] = df["game_pk"].map(
        lambda pk: game_info.get(pk, {}).get("away_team_id")
    )
    df["_home_team_id"] = df["game_pk"].map(
        lambda pk: game_info.get(pk, {}).get("home_team_id")
    )
    df["_away_probable"] = df["game_pk"].map(
        lambda pk: game_info.get(pk, {}).get("away_probable_id")
    )
    df["_home_probable"] = df["game_pk"].map(
        lambda pk: game_info.get(pk, {}).get("home_probable_id")
    )

    # Determine pitcher's team and whether they're a reliever
    # is_home = True means batter is home, so pitcher is away team
    df["_pitcher_team_id"] = np.where(
        df["is_home"], df["_away_team_id"], df["_home_team_id"]
    )
    df["_opp_team_id"] = np.where(
        df["is_home"], df["_home_team_id"], df["_away_team_id"]
    )

    # Starter = probable pitcher for the pitching team
    df["_probable_for_pitcher_team"] = np.where(
        df["is_home"], df["_away_probable"], df["_home_probable"]
    )
    df["_is_reliever_pa"] = df["pitcher_id"] != df["_probable_for_pitcher_team"]

    # Count PAs with game_info coverage
    has_info = df["_pitcher_team_id"].notna().sum()
    is_reliever = df["_is_reliever_pa"].sum()
    print(f"  Game info coverage: {has_info:,}/{len(df):,} PAs", file=sys.stderr)
    print(f"  Reliever PAs: {is_reliever:,} ({is_reliever/len(df):.1%})", file=sys.stderr)

    # Compute daily reliever hit rate per pitching team
    reliever_pas = df[df["_is_reliever_pa"]].copy()
    daily_bullpen = reliever_pas.groupby(["_pitcher_team_id", "date"]).agg(
        bp_hits=("is_hit", "sum"),
        bp_pas=("is_hit", "count"),
    ).reset_index().sort_values(["_pitcher_team_id", "date"])
    daily_bullpen["bp_hr"] = daily_bullpen["bp_hits"] / daily_bullpen["bp_pas"]

    # Rolling 30-day average per team (shift by 1 to prevent leakage)
    daily_bullpen["bullpen_hr_30g"] = (
        daily_bullpen.groupby("_pitcher_team_id")["bp_hr"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=10).mean())
    )

    # Map back to each PA: the opposing team's bullpen quality
    # For each batter, the opposing team is _pitcher_team_id
    # (the team doing the pitching = the team whose bullpen matters)
    bp_lookup = daily_bullpen.set_index(["_pitcher_team_id", "date"])["bullpen_hr_30g"]

    df["opp_bullpen_hr_30g"] = df.apply(
        lambda r: bp_lookup.get((r["_pitcher_team_id"], r["date"]))
        if pd.notna(r["_pitcher_team_id"]) else np.nan,
        axis=1,
    )

    coverage = df["opp_bullpen_hr_30g"].notna().sum()
    print(f"  Bullpen feature coverage: {coverage:,}/{len(df):,} "
          f"({coverage/len(df):.1%})", file=sys.stderr)

    # Clean up temp columns
    for col in ["_away_team_id", "_home_team_id", "_away_probable",
                "_home_probable", "_pitcher_team_id", "_opp_team_id",
                "_probable_for_pitcher_team", "_is_reliever_pa"]:
        df.drop(columns=[col], inplace=True)

    return df


def build_bullpen_blend_configs():
    """Build blend configs with opp_bullpen_hr_30g added to all models."""
    new_configs = []
    for name, cols in BLEND_CONFIGS:
        new_configs.append((name, cols + ["opp_bullpen_hr_30g"]))
    return new_configs


def main():
    df = load_data()

    print("\nBuilding game info lookup from raw feeds...", file=sys.stderr)
    game_info = build_game_info_lookup()

    print("Computing bullpen feature...", file=sys.stderr)
    df = compute_bullpen_feature(df, game_info)

    # Quick diagnostic: what does the feature look like?
    bp = df[df["opp_bullpen_hr_30g"].notna()]["opp_bullpen_hr_30g"]
    print(f"\n  opp_bullpen_hr_30g stats:", file=sys.stderr)
    print(f"    mean={bp.mean():.4f}, std={bp.std():.4f}, "
          f"min={bp.min():.4f}, max={bp.max():.4f}", file=sys.stderr)

    configs_without = list(BLEND_CONFIGS)
    configs_with = build_bullpen_blend_configs()

    results = {}

    # Baseline (no bullpen feature)
    print(f"\n{'='*60}", file=sys.stderr)
    print("BASELINE (no bullpen feature)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    profiles_base = walk_forward_backtest(
        df, TEST_SEASONS, configs_without, LGB_PARAMS, game_level=False,
    )
    results["no_bullpen"] = compute_metrics(profiles_base)

    # With bullpen feature in all models
    print(f"\n{'='*60}", file=sys.stderr)
    print("WITH opp_bullpen_hr_30g (all 12 models)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    profiles_bp = walk_forward_backtest(
        df, TEST_SEASONS, configs_with, LGB_PARAMS, game_level=False,
    )
    results["with_bullpen"] = compute_metrics(profiles_bp)

    print_comparison(results, "Phase 5: Team Bullpen Composite")

    # Quality bin detail
    print("\nQuality bin detail (early phase):")
    base_bins = results["no_bullpen"]["early_bins"].bins
    bp_bins = results["with_bullpen"]["early_bins"].bins
    print(f"  {'Bin':>4} | {'Base P(hit)':>12} {'BP P(hit)':>12} | {'Base freq':>10} {'BP freq':>10}")
    print(f"  {'-'*60}")
    for bb, bpb in zip(base_bins, bp_bins):
        print(f"  Q{bb.index+1:>3} | {bb.p_hit:>12.4f} {bpb.p_hit:>12.4f} "
              f"| {bb.frequency:>10.3f} {bpb.frequency:>10.3f}")


if __name__ == "__main__":
    main()
