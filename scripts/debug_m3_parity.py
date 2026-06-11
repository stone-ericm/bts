"""Debug the replay parity-check mismatch: which feature/entity/date diverges
between the STALE arm construction and production _build_feature_lookups."""

import numpy as np
import pandas as pd
from pathlib import Path

from bts.features.compute import compute_all_features
from bts.model.predict import _build_feature_lookups

import importlib.util
spec = importlib.util.spec_from_file_location("replay", "scripts/replay_m3_serving_parity.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)

proc = Path("data/processed")
seasons = range(2017, 2025)
df = pd.concat([pd.read_parquet(proc / f"pa_{s}.parquet") for s in seasons], ignore_index=True)
df_feat = compute_all_features(df)
df_feat["date"] = pd.to_datetime(df_feat["date"])
print("features computed", flush=True)

season_df = df_feat[df_feat["season"] == 2024]
starters = season_df[season_df["lineup_position"].between(1, 9)]
slate = starters.groupby(["game_pk", "batter_id"], as_index=False).first()
side_starter = (
    starters.groupby(["game_pk", "is_home"])["pitcher_id"]
    .agg(lambda s: s.mode().iat[0]).rename("starter_pid").reset_index()
)
slate = slate.merge(side_starter, on=["game_pk", "is_home"], how="left")
slate["pitcher_id"] = slate["starter_pid"].fillna(slate["pitcher_id"])
hand_map = df_feat.dropna(subset=["pitch_hand"]).groupby("pitcher_id")["pitch_hand"].last()
slate["pitch_hand"] = slate["pitcher_id"].map(hand_map)

fresh, stale = replay.build_arm_values(df_feat, slate)
print("arms built", flush=True)

dates = sorted(slate["date"].unique())
d = dates[len(dates) // 2]
lk = _build_feature_lookups(df_feat[df_feat["date"] < d])
rows = stale[stale["date"] == d]
print(f"checking {len(rows)} slate rows on {d}", flush=True)

diffs = []
for _, r in rows.iterrows():
    for col in replay.KEY_GROUPS[("batter_id",)]:
        v_lk = lk["batter"].get(col, {}).get(r["batter_id"])
        if v_lk is None:
            v_lk = lk.get("batter_statcast", {}).get(col, {}).get(r["batter_id"])
        v_arm = r.get(col)
        if v_lk is not None and pd.notna(v_arm) and abs(v_lk - v_arm) > 1e-9:
            diffs.append((col, int(r["batter_id"]), v_lk, v_arm, abs(v_lk - v_arm)))
    v_lk = lk.get("batter_pitcher_hr", {}).get((r["batter_id"], r["pitcher_id"]))
    v_arm = r.get("batter_pitcher_shrunk_hr")
    if v_lk is not None and pd.notna(v_arm) and abs(v_lk - v_arm) > 1e-9:
        diffs.append(("bpm", (int(r["batter_id"]), int(r["pitcher_id"])), v_lk, v_arm, abs(v_lk - v_arm)))
    v_lk = lk.get("pitcher_hr", {}).get(r["pitcher_id"])
    v_arm = r.get("pitcher_hr_30g")
    if v_lk is not None and pd.notna(v_arm) and abs(v_lk - v_arm) > 1e-9:
        diffs.append(("pitcher_hr_30g", int(r["pitcher_id"]), v_lk, v_arm, abs(v_lk - v_arm)))

dd = pd.DataFrame(diffs, columns=["col", "key", "v_lookup", "v_arm", "absdiff"])
print(f"\n{len(dd)} mismatches > 1e-9")
if len(dd):
    print(dd.groupby("col")["absdiff"].agg(["count", "mean", "max"]).sort_values("max", ascending=False))
    print("\nworst 15:")
    print(dd.sort_values("absdiff", ascending=False).head(15).to_string())

    # drill into the single worst case
    col, key, v_lk, v_arm, _ = dd.sort_values("absdiff", ascending=False).iloc[0][["col", "key", "v_lookup", "v_arm", "absdiff"]].tolist()
    print(f"\nDRILL: col={col} key={key}")
    if col == "bpm":
        b, p = key
        hist = df_feat[(df_feat["batter_id"] == b) & (df_feat["pitcher_id"] == p)][
            ["date", "is_hit", "batter_pitcher_shrunk_hr"]].sort_values("date")
        print(hist.tail(10).to_string())
    elif col == "pitcher_hr_30g":
        hist = df_feat[df_feat["pitcher_id"] == key][["date", "pitcher_hr_30g"]].drop_duplicates().sort_values("date")
        print(hist.tail(6).to_string())
    else:
        hist = df_feat[df_feat["batter_id"] == key][["date", col]].drop_duplicates().sort_values("date")
        print(hist.tail(6).to_string())
