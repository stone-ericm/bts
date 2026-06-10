"""Systematic leakage audit for BTS v2 model."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

dfs = []
for year in [2023, 2024, 2025]:
    dfs.append(pd.read_parquet(f"data/processed/pa_{year}.parquet"))
df = pd.concat(dfs, ignore_index=True)
df["date"] = pd.to_datetime(df["date"])

print("=" * 60)
print("SYSTEMATIC LEAKAGE AUDIT")
print("=" * 60)

# AUDIT 1: Doubleheader leakage
print("\n--- AUDIT 1: Doubleheader leakage ---")
games_per_day = df.groupby(["batter_id", "date"])["game_pk"].nunique()
doubleheaders = games_per_day[games_per_day > 1]
total_batter_dates = len(games_per_day)
pct = len(doubleheaders) / total_batter_dates
print(f"Batter-dates with 2+ games: {len(doubleheaders)} / {total_batter_dates:,} ({pct:.2%})")
print(f"VERDICT: {'MINOR' if pct < 0.01 else 'SIGNIFICANT'}")

# (Former AUDIT 2 & 4 — pitcher-archetype K-Means cluster stability — REMOVED.
#  The K-Means archetype features (`_compute_pitcher_archetypes`) were dropped
#  from the model as 90.8% unstable across train/test splits, so there is no
#  longer anything to audit here. The import previously crashed this whole
#  script on line 1. See ARCHITECTURE.md "Dropped features".)

# AUDIT 3: Expanding feature spot check
print("\n--- AUDIT 3: Expanding feature spot check ---")
from bts.features.compute import compute_all_features
df_feat = compute_all_features(df)

np.random.seed(42)
test_2025 = df_feat[df_feat["season"] == 2025].dropna(subset=["platoon_hr"])
samples = test_2025.sample(10)

mismatches = 0
for idx, row in samples.iterrows():
    bid = row["batter_id"]
    dt = row["date"]
    ph = row["pitch_hand"]
    feat_val = row["platoon_hr"]

    prior = df[(df["batter_id"] == bid) & (df["date"] < dt) & (df["pitch_hand"] == ph)]
    if len(prior) >= 30:
        manual = prior["is_hit"].mean()
        if abs(feat_val - manual) >= 0.005:
            mismatches += 1
            print(f"  MISMATCH: batter={bid} date={dt.date()} hand={ph}")
            print(f"    feature={feat_val:.4f} manual={manual:.4f} diff={abs(feat_val-manual):.4f}")

if mismatches == 0:
    print(f"  10/10 spot checks PASS — platoon_hr clean")
else:
    print(f"  {mismatches}/10 MISMATCHES — LEAKAGE")

# Spot check batter_gb_hit_rate
print("\n  Spot checking batter_gb_hit_rate...")
samples2 = df_feat[(df_feat["season"] == 2025) & df_feat["batter_gb_hit_rate"].notna()].sample(10, random_state=99)
gb_mismatches = 0
for idx, row in samples2.iterrows():
    bid = row["batter_id"]
    dt = row["date"]
    feat_val = row["batter_gb_hit_rate"]

    prior_gb = df[(df["batter_id"] == bid) & (df["date"] < dt) &
                  df["launch_angle"].notna() & (df["launch_angle"] < 10)]
    if len(prior_gb) >= 20:
        manual = prior_gb["is_hit"].mean()
        if abs(feat_val - manual) >= 0.01:
            gb_mismatches += 1
            print(f"  MISMATCH: batter={bid} date={dt.date()}")
            print(f"    feature={feat_val:.4f} manual={manual:.4f}")

if gb_mismatches == 0:
    print(f"  10/10 spot checks PASS — batter_gb_hit_rate clean")
else:
    print(f"  {gb_mismatches}/10 MISMATCHES")

# Spot check park_factor
print("\n  Spot checking park_factor...")
pf_samples = df_feat[(df_feat["season"] == 2025) & df_feat["park_factor"].notna()].sample(5, random_state=77)
pf_mismatches = 0
overall_hr = df["is_hit"].mean()
for idx, row in pf_samples.iterrows():
    vid = row["venue_id"]
    dt = row["date"]
    feat_val = row["park_factor"]
    prior_venue = df[(df["venue_id"] == vid) & (df["date"] < dt)]
    if len(prior_venue) >= 100:
        manual = prior_venue["is_hit"].mean() / overall_hr
        if abs(feat_val - manual) >= 0.02:
            pf_mismatches += 1
            print(f"  MISMATCH: venue={vid} date={dt.date()} feat={feat_val:.3f} manual={manual:.3f}")

if pf_mismatches == 0:
    print(f"  5/5 spot checks PASS — park_factor clean")
else:
    print(f"  {pf_mismatches}/5 MISMATCHES")

# AUDIT 4: The nuclear test — compare features computed on
# train-only vs full dataset for the SAME test PA
print("\n--- AUDIT 4: Nuclear test — train-only vs full features ---")
# Compute features using ONLY 2023-2024 data
train_only = df[df["season"] < 2025].copy()
df_train_feat = compute_all_features(train_only)

# For batter_hr_7g on the last day of 2024, values should be identical
# regardless of whether 2025 data exists
last_2024_date = df_train_feat["date"].max()
last_day_train = df_train_feat[df_train_feat["date"] == last_2024_date]
last_day_full = df_feat[(df_feat["date"] == last_2024_date)]

common_batters = set(last_day_train["batter_id"]) & set(last_day_full["batter_id"])
if common_batters:
    bid = list(common_batters)[0]
    # Includes the promoted bpm feature (whose inference path broke once,
    # 2026-04-29) and the bullpen composite — both production FEATURE_COLS.
    for col in ["batter_hr_7g", "batter_hr_30g", "platoon_hr", "park_factor",
                "batter_pitcher_shrunk_hr", "opp_bullpen_hr_30g"]:
        val_train = last_day_train[last_day_train["batter_id"] == bid][col].iloc[0] if col in last_day_train.columns else "N/A"
        val_full = last_day_full[last_day_full["batter_id"] == bid][col].iloc[0] if col in last_day_full.columns else "N/A"
        match = "MATCH" if (pd.isna(val_train) and pd.isna(val_full)) or (not pd.isna(val_train) and abs(val_train - val_full) < 0.001) else "DIFFERS"
        print(f"  {col:<25}: train-only={val_train if isinstance(val_train, str) else f'{val_train:.4f}'} full={val_full if isinstance(val_full, str) else f'{val_full:.4f}'} → {match}")

print("\n" + "=" * 60)
print("FINAL LEAKAGE SUMMARY")
print("=" * 60)
