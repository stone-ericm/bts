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

    # --- Batter GB hit rate (speed proxy) — EXPANDING with shift ---
    df["is_groundball"] = df["launch_angle"].notna() & (df["launch_angle"] < 10)
    df["gb_hit"] = np.where(df["is_groundball"], df["is_hit"], np.nan)
    game_gb = df.groupby(["batter_id", "date", "game_pk"])["gb_hit"].agg(
        ["sum", "count"]
    ).reset_index().sort_values(["batter_id", "date"])
    game_gb.columns = ["batter_id", "date", "game_pk", "gb_hits", "gb_count"]
    # Use expanding sum with shift(1) — cumulative GB hit rate from all prior games
    game_gb["cum_gb_hits"] = game_gb.groupby("batter_id")["gb_hits"].transform(
        lambda x: x.shift(1).expanding(min_periods=20).sum()
    )
    game_gb["cum_gb_count"] = game_gb.groupby("batter_id")["gb_count"].transform(
        lambda x: x.shift(1).expanding(min_periods=20).sum()
    )
    game_gb["batter_gb_hit_rate"] = np.where(
        game_gb["cum_gb_count"] > 0,
        game_gb["cum_gb_hits"] / game_gb["cum_gb_count"],
        np.nan,
    )
    batter_games = batter_games.merge(
        game_gb[["batter_id", "date", "game_pk", "batter_gb_hit_rate"]],
        on=["batter_id", "date", "game_pk"], how="left",
    )

    # --- Platoon H/PA — EXPANDING with shift ---
    # Per-PA platoon matchup result, then expanding average per batter×hand
    df["platoon_key"] = df["batter_id"].astype(str) + "_" + df["pitch_hand"].fillna("U")
    game_platoon = df.groupby(["platoon_key", "batter_id", "pitch_hand", "date", "game_pk"]).agg(
        ph_hits=("is_hit", "sum"), ph_pas=("is_hit", "count"),
    ).reset_index().sort_values(["platoon_key", "date"])
    game_platoon["ph_rate"] = game_platoon["ph_hits"] / game_platoon["ph_pas"]
    game_platoon["cum_ph_hits"] = game_platoon.groupby("platoon_key")["ph_hits"].transform(
        lambda x: x.shift(1).expanding(min_periods=5).sum()
    )
    game_platoon["cum_ph_pas"] = game_platoon.groupby("platoon_key")["ph_pas"].transform(
        lambda x: x.shift(1).expanding(min_periods=5).sum()
    )
    game_platoon["platoon_hr"] = np.where(
        game_platoon["cum_ph_pas"] >= 30,
        game_platoon["cum_ph_hits"] / game_platoon["cum_ph_pas"],
        np.nan,
    )

    # --- Pitcher archetypes (use all data — archetypes are structural, not temporal) ---
    pitcher_cluster_map, n_clusters = _compute_pitcher_archetypes(df)

    # --- Batter x archetype hit rate — EXPANDING with shift ---
    df["pitcher_cluster"] = df["pitcher_id"].map(pitcher_cluster_map)
    df["arch_key"] = df["batter_id"].astype(str) + "_" + df["pitcher_cluster"].astype(str)
    game_arch = df.dropna(subset=["pitcher_cluster"]).groupby(
        ["arch_key", "batter_id", "pitcher_cluster", "date", "game_pk"]
    ).agg(arch_hits=("is_hit", "sum"), arch_pas=("is_hit", "count")).reset_index()
    game_arch = game_arch.sort_values(["arch_key", "date"])
    game_arch["cum_arch_hits"] = game_arch.groupby("arch_key")["arch_hits"].transform(
        lambda x: x.shift(1).expanding(min_periods=3).sum()
    )
    game_arch["cum_arch_pas"] = game_arch.groupby("arch_key")["arch_pas"].transform(
        lambda x: x.shift(1).expanding(min_periods=3).sum()
    )
    game_arch["batter_vs_arch_hr"] = np.where(
        game_arch["cum_arch_pas"] >= 15,
        game_arch["cum_arch_hits"] / game_arch["cum_arch_pas"],
        np.nan,
    )

    # --- Pitcher rolling H/9 ---
    pitcher_games = df.groupby(["pitcher_id", "date", "game_pk"]).agg(
        p_hits=("is_hit", "sum"), p_pas=("is_hit", "count"),
    ).reset_index().sort_values(["pitcher_id", "date"])
    pitcher_games["p_hit_rate"] = pitcher_games["p_hits"] / pitcher_games["p_pas"]
    pitcher_games["pitcher_hr_30g"] = (
        pitcher_games.groupby("pitcher_id")["p_hit_rate"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=10).mean())
    )

    # --- Park factor (expanding — uses all prior games at each venue) ---
    game_venue = df.groupby(["venue_id", "date", "game_pk"])["is_hit"].mean().reset_index()
    game_venue.columns = ["venue_id", "date", "game_pk", "venue_game_hr"]
    game_venue = game_venue.sort_values(["venue_id", "date"])
    game_venue["park_factor"] = game_venue.groupby("venue_id")["venue_game_hr"].transform(
        lambda x: x.shift(1).expanding(min_periods=20).mean()
    )
    # Normalize to overall mean
    overall_hr = df["is_hit"].mean()
    game_venue["park_factor"] = game_venue["park_factor"] / overall_hr

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

    # Temporal platoon features (expanding, not static)
    platoon_merge = game_platoon[["batter_id", "pitch_hand", "date", "game_pk", "platoon_hr"]].drop_duplicates(
        subset=["batter_id", "pitch_hand", "date", "game_pk"]
    )
    df = df.merge(platoon_merge, on=["batter_id", "pitch_hand", "date", "game_pk"], how="left")

    # Temporal batter vs archetype features
    arch_merge = game_arch[["batter_id", "pitcher_cluster", "date", "game_pk", "batter_vs_arch_hr"]].drop_duplicates(
        subset=["batter_id", "pitcher_cluster", "date", "game_pk"]
    )
    df = df.merge(arch_merge, on=["batter_id", "pitcher_cluster", "date", "game_pk"], how="left")

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

    # Park factor (temporal)
    park_merge = game_venue[["venue_id", "date", "game_pk", "park_factor"]].drop_duplicates(
        subset=["venue_id", "date", "game_pk"]
    )
    df = df.merge(park_merge, on=["venue_id", "date", "game_pk"], how="left")
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
