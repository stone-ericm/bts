"""Analyze starter vs reliever PA distribution and debut pitchers."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

dfs = []
for year in [2023, 2024, 2025]:
    dfs.append(pd.read_parquet(f"data/processed/pa_{year}.parquet"))
df = pd.concat(dfs, ignore_index=True)
df["date"] = pd.to_datetime(df["date"])

print("Analyzing starter vs reliever PAs...")

# For each batter in each game, identify if they face the starter or a reliever
# Starter = first pitcher that team's batters face
game_starters = {}
for gpk, game in df.groupby("game_pk"):
    for is_home in [True, False]:
        side = game[game["is_home"] == is_home].sort_values("date")
        if len(side) > 0:
            game_starters[(gpk, is_home)] = side.iloc[0]["pitcher_id"]

df["starter_id"] = df.apply(
    lambda r: game_starters.get((r["game_pk"], r["is_home"])), axis=1
)
df["vs_starter"] = df["pitcher_id"] == df["starter_id"]

print(f"  vs Starter: {df['vs_starter'].sum():,} ({df['vs_starter'].mean():.1%})")
print(f"  vs Reliever: {(~df['vs_starter']).sum():,} ({(~df['vs_starter']).mean():.1%})")

# By PA number
df_sorted = df.sort_values(["game_pk", "batter_id", "date"])
df_sorted["pa_num"] = df_sorted.groupby(["game_pk", "batter_id"]).cumcount() + 1

print(f"\n  By PA number (% facing starter):")
for pa in range(1, 6):
    subset = df_sorted[df_sorted["pa_num"] == pa]
    pct = subset["vs_starter"].mean()
    n = len(subset)
    print(f"    PA #{pa}: {pct:.1%} face starter (n={n:,})")

# Debut pitchers
print(f"\n  Debut pitchers per season:")
all_data = []
for year in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
    try:
        all_data.append(pd.read_parquet(f"data/processed/pa_{year}.parquet"))
    except FileNotFoundError:
        pass
all_df = pd.concat(all_data, ignore_index=True)

for year in [2023, 2024, 2025]:
    prior_pitchers = set(all_df[all_df["season"] < year]["pitcher_id"])
    season = all_df[all_df["season"] == year]
    season_pitchers = set(season["pitcher_id"])
    debuts = season_pitchers - prior_pitchers
    debut_pas = season[season["pitcher_id"].isin(debuts)]
    print(f"    {year}: {len(debuts)} debut pitchers, {len(debut_pas):,} PAs ({len(debut_pas)/len(season):.1%})")

# What do we know about debut pitchers? Do batters hit better or worse against them?
print(f"\n  Hit rate vs debut pitchers vs experienced:")
for year in [2023, 2024, 2025]:
    prior_pitchers = set(all_df[all_df["season"] < year]["pitcher_id"])
    season = all_df[all_df["season"] == year]
    debut_pas = season[season["pitcher_id"].isin(season["pitcher_id"].unique()) &
                       ~season["pitcher_id"].isin(prior_pitchers)]
    exp_pas = season[season["pitcher_id"].isin(prior_pitchers)]
    if len(debut_pas) > 0:
        print(f"    {year}: debut={debut_pas['is_hit'].mean():.3f} exp={exp_pas['is_hit'].mean():.3f}")
