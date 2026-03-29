"""Feature computation for PA-level hit prediction.

All rolling features use shift(1) to prevent temporal leakage —
they only use data available before the game in question.
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PITCH_TYPE_COLS = ["FF", "SI", "SL", "CH", "CU", "FC", "FS", "KC", "ST", "SV"]


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for every PA in the DataFrame.

    Args:
        df: PA-level DataFrame with columns from schema.PA_COLUMNS.
            Must be sorted by date.

    Returns:
        DataFrame with original columns plus computed feature columns.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)

    # --- Game-level aggregates (needed for rolling computations) ---
    batter_games = df.groupby(["batter_id", "date", "game_pk"]).agg(
        game_hits=("is_hit", "sum"),
        game_pas=("is_hit", "count"),
        game_hit_rate=("is_hit", "mean"),
    ).reset_index().sort_values(["batter_id", "date"])

    # --- Batter rolling hit rates ---
    for w in [7, 14, 30, 60, 120]:
        col = f"batter_hr_{w}g"
        batter_games[col] = (
            batter_games.groupby("batter_id")["game_hit_rate"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=max(3, w // 4)).mean())
        )

    # --- Batter rolling whiff rate ---
    def _whiff_rate(calls):
        if not isinstance(calls, (list, np.ndarray)) or len(calls) == 0:
            return np.nan
        swings = sum(1 for c in calls if c in ("S", "F", "X", "D", "E"))
        whiffs = sum(1 for c in calls if c == "S")
        return whiffs / swings if swings > 0 else np.nan

    df["pa_whiff_rate"] = df["pitch_calls"].apply(_whiff_rate)
    game_whiff = df.groupby(["batter_id", "date", "game_pk"])["pa_whiff_rate"].mean().reset_index()
    game_whiff = game_whiff.sort_values(["batter_id", "date"])
    for w in [30, 60]:
        col = f"batter_whiff_{w}g"
        game_whiff[col] = (
            game_whiff.groupby("batter_id")["pa_whiff_rate"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=10).mean())
        )
    batter_games = batter_games.merge(
        game_whiff[["batter_id", "date", "game_pk", "batter_whiff_30g", "batter_whiff_60g"]],
        on=["batter_id", "date", "game_pk"], how="left",
    )

    # --- Batter count tendency ---
    df["count_diff"] = df["final_count_balls"] - df["final_count_strikes"]
    game_count = df.groupby(["batter_id", "date", "game_pk"])["count_diff"].mean().reset_index()
    game_count.columns = ["batter_id", "date", "game_pk", "avg_count_diff"]
    game_count = game_count.sort_values(["batter_id", "date"])
    game_count["batter_count_tendency_30g"] = (
        game_count.groupby("batter_id")["avg_count_diff"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=10).mean())
    )
    batter_games = batter_games.merge(
        game_count[["batter_id", "date", "game_pk", "batter_count_tendency_30g"]],
        on=["batter_id", "date", "game_pk"], how="left",
    )

    # --- Batter GB hit rate (speed proxy) ---
    df["is_groundball"] = df["launch_angle"].notna() & (df["launch_angle"] < 10)
    gb_only = df[df["is_groundball"]].copy()
    batter_gb = gb_only.groupby("batter_id").agg(
        gb_hits=("is_hit", "sum"), gb_total=("is_hit", "count"),
    ).reset_index()
    batter_gb["batter_gb_hit_rate"] = np.where(
        batter_gb["gb_total"] >= 20,
        batter_gb["gb_hits"] / batter_gb["gb_total"],
        np.nan,
    )
    gb_map = dict(zip(batter_gb["batter_id"], batter_gb["batter_gb_hit_rate"]))

    # --- Platoon historical H/PA ---
    platoon = df.groupby(["batter_id", "pitch_hand"]).agg(
        ph_hits=("is_hit", "sum"), ph_pas=("is_hit", "count"),
    ).reset_index()
    platoon["platoon_hr"] = np.where(
        platoon["ph_pas"] >= 30,
        platoon["ph_hits"] / platoon["ph_pas"],
        np.nan,
    )
    platoon_map = {}
    for _, row in platoon.iterrows():
        platoon_map[(row["batter_id"], row["pitch_hand"])] = row["platoon_hr"]

    # --- Pitcher archetypes ---
    pitcher_cluster_map, n_clusters = _compute_pitcher_archetypes(df)

    # --- Batter x archetype hit rate ---
    df["pitcher_cluster"] = df["pitcher_id"].map(pitcher_cluster_map)
    batter_arch = df.dropna(subset=["pitcher_cluster"]).groupby(
        ["batter_id", "pitcher_cluster"]
    ).agg(arch_hits=("is_hit", "sum"), arch_pas=("is_hit", "count")).reset_index()
    batter_arch["batter_vs_arch_hr"] = np.where(
        batter_arch["arch_pas"] >= 15,
        batter_arch["arch_hits"] / batter_arch["arch_pas"],
        np.nan,
    )
    arch_map = {}
    for _, row in batter_arch.iterrows():
        arch_map[(row["batter_id"], row["pitcher_cluster"])] = row["batter_vs_arch_hr"]

    # --- Pitcher rolling H/9 ---
    pitcher_games = df.groupby(["pitcher_id", "date", "game_pk"]).agg(
        p_hits=("is_hit", "sum"), p_pas=("is_hit", "count"),
    ).reset_index().sort_values(["pitcher_id", "date"])
    pitcher_games["p_hit_rate"] = pitcher_games["p_hits"] / pitcher_games["p_pas"]
    pitcher_games["pitcher_hr_30g"] = (
        pitcher_games.groupby("pitcher_id")["p_hit_rate"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=10).mean())
    )

    # --- Park factor ---
    venue_hr = df.groupby("venue_id")["is_hit"].mean()
    overall_hr = df["is_hit"].mean()
    park_factor = (venue_hr / overall_hr).to_dict()

    # --- Rest days ---
    batter_dates = df.groupby(["batter_id", "date"]).size().reset_index()[["batter_id", "date"]]
    batter_dates = batter_dates.drop_duplicates().sort_values(["batter_id", "date"])
    batter_dates["days_rest"] = batter_dates.groupby("batter_id")["date"].diff().dt.days
    rest_map = {}
    for _, row in batter_dates.iterrows():
        rest_map[(row["batter_id"], row["date"])] = row["days_rest"]

    # === Merge everything back to PA level ===

    # Combine all game-level batter features into one DataFrame
    merge_keys = ["batter_id", "date", "game_pk"]
    rolling_cols = [c for c in batter_games.columns if c not in merge_keys
                    and c not in ("game_hits", "game_pas", "game_hit_rate")]
    batter_features = batter_games[merge_keys + rolling_cols].drop_duplicates(subset=merge_keys)
    df = df.merge(batter_features, on=merge_keys, how="left")

    # Pitcher game-level rolling features
    pitcher_merge = pitcher_games[["pitcher_id", "date", "game_pk", "pitcher_hr_30g"]].drop_duplicates(
        subset=["pitcher_id", "date", "game_pk"]
    )
    df = df.merge(pitcher_merge, on=["pitcher_id", "date", "game_pk"], how="left")

    # Static batter features
    df["batter_gb_hit_rate"] = df["batter_id"].map(gb_map)
    df["platoon_hr"] = df.apply(
        lambda r: platoon_map.get((r["batter_id"], r["pitch_hand"])), axis=1
    )
    df["batter_vs_arch_hr"] = df.apply(
        lambda r: arch_map.get((r["batter_id"], r.get("pitcher_cluster"))), axis=1
    )

    # --- Pitcher arsenal entropy ---
    def _pitch_entropy(types):
        if not isinstance(types, (list, np.ndarray)) or len(types) == 0:
            return np.nan
        counts = Counter(types)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    df["pa_pitch_entropy"] = df["pitch_types"].apply(_pitch_entropy)
    pitcher_entropy = df.groupby(["pitcher_id", "date", "game_pk"])["pa_pitch_entropy"].mean().reset_index()
    pitcher_entropy = pitcher_entropy.sort_values(["pitcher_id", "date"])
    pitcher_entropy["pitcher_entropy_30g"] = (
        pitcher_entropy.groupby("pitcher_id")["pa_pitch_entropy"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=10).mean())
    )
    df = df.merge(
        pitcher_entropy[["pitcher_id", "date", "game_pk", "pitcher_entropy_30g"]].drop_duplicates(
            subset=["pitcher_id", "date", "game_pk"]
        ),
        on=["pitcher_id", "date", "game_pk"], how="left",
    )

    # Context features
    df["park_factor"] = df["venue_id"].map(park_factor)
    df["days_rest"] = df.apply(
        lambda r: rest_map.get((r["batter_id"], r["date"])), axis=1
    )

    return df


def _compute_pitcher_archetypes(df: pd.DataFrame, n_clusters: int = 8) -> tuple[dict, int]:
    """Cluster pitchers by arsenal profile.

    Returns:
        (pitcher_id -> cluster_id mapping, number of clusters)
    """
    pitcher_arsenals = {}
    for pid, group in df.groupby("pitcher_id"):
        all_pitches = []
        for types in group["pitch_types"]:
            if isinstance(types, (list, np.ndarray)):
                all_pitches.extend(types)
        if len(all_pitches) >= 100:
            counts = Counter(all_pitches)
            total = sum(counts.values())
            pitcher_arsenals[pid] = {k: v / total for k, v in counts.items()}

    if not pitcher_arsenals:
        return {}, 0

    matrix = []
    pids = []
    for pid, arsenal in pitcher_arsenals.items():
        row = [arsenal.get(pt, 0.0) for pt in PITCH_TYPE_COLS]
        matrix.append(row)
        pids.append(pid)

    scaled = StandardScaler().fit_transform(matrix)
    clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaled)

    return dict(zip(pids, clusters)), n_clusters


# Feature columns used for model training.
# lineup_position, is_home, and batter_whiff_30g were dropped after
# ablation study showed they hurt P@1 by +4.3%, +2.7%, +2.7% respectively.
# lineup_position effect is handled by the game-level aggregation step.
FEATURE_COLS = [
    "batter_hr_7g",
    "batter_hr_30g",
    "batter_hr_60g",
    "batter_hr_120g",
    "batter_count_tendency_30g",
    "batter_gb_hit_rate",
    "platoon_hr",
    "batter_vs_arch_hr",
    "pitcher_hr_30g",
    "pitcher_cluster",
    "pitcher_entropy_30g",
    "weather_temp",
    "park_factor",
    "days_rest",
]
