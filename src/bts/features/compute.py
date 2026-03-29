"""Feature computation for PA-level hit prediction.

TEMPORAL GUARANTEE: Every feature for a PA on date D uses ONLY data from
dates strictly before D. This is enforced by:
1. All rolling/expanding features group by (entity, date) and use shift(1)
   on date-level aggregates — so doubleheader games on the same date are
   merged before shifting.
2. No clustering or global aggregation that uses future data.
3. Park factor normalization uses expanding league-wide mean.
"""

import numpy as np
import pandas as pd
from collections import Counter

# Optimal training window: 2019 onward. Adding 2017-2018 hurts P@1 by 1.1%
# because the game changed enough that old training examples add noise.
# Features are still computed from ALL available history (expanding features
# benefit from more data), but the model should be trained on 2019+ PAs.
TRAIN_START_YEAR = 2019


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for every PA in the DataFrame.

    Temporal guarantee: every feature value for a PA on date D
    uses only data from dates strictly before D.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)

    # --- Date-level batter aggregates (merge doubleheaders) ---
    batter_dates = df.groupby(["batter_id", "date"]).agg(
        date_hits=("is_hit", "sum"),
        date_pas=("is_hit", "count"),
        date_hit_rate=("is_hit", "mean"),
    ).reset_index().sort_values(["batter_id", "date"])

    # --- Batter rolling hit rates (date-level, shift by date) ---
    for w in [7, 30, 60, 120]:
        col = f"batter_hr_{w}g"
        batter_dates[col] = (
            batter_dates.groupby("batter_id")["date_hit_rate"]
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
    date_whiff = df.groupby(["batter_id", "date"])["pa_whiff_rate"].mean().reset_index()
    date_whiff = date_whiff.sort_values(["batter_id", "date"])
    date_whiff["batter_whiff_60g"] = (
        date_whiff.groupby("batter_id")["pa_whiff_rate"]
        .transform(lambda x: x.shift(1).rolling(60, min_periods=10).mean())
    )
    batter_dates = batter_dates.merge(
        date_whiff[["batter_id", "date", "batter_whiff_60g"]],
        on=["batter_id", "date"], how="left",
    )

    # --- Batter count tendency ---
    df["count_diff"] = df["final_count_balls"] - df["final_count_strikes"]
    date_count = df.groupby(["batter_id", "date"])["count_diff"].mean().reset_index()
    date_count.columns = ["batter_id", "date", "avg_count_diff"]
    date_count = date_count.sort_values(["batter_id", "date"])
    date_count["batter_count_tendency_30g"] = (
        date_count.groupby("batter_id")["avg_count_diff"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=10).mean())
    )
    batter_dates = batter_dates.merge(
        date_count[["batter_id", "date", "batter_count_tendency_30g"]],
        on=["batter_id", "date"], how="left",
    )

    # --- Batter GB hit rate (speed proxy) — expanding, date-level ---
    df["is_groundball"] = df["launch_angle"].notna() & (df["launch_angle"] < 10)
    df["gb_hit"] = np.where(df["is_groundball"], df["is_hit"], np.nan)
    date_gb = df.groupby(["batter_id", "date"])["gb_hit"].agg(
        ["sum", "count"]
    ).reset_index().sort_values(["batter_id", "date"])
    date_gb.columns = ["batter_id", "date", "gb_hits", "gb_count"]
    date_gb["cum_gb_hits"] = date_gb.groupby("batter_id")["gb_hits"].transform(
        lambda x: x.shift(1).expanding(min_periods=20).sum()
    )
    date_gb["cum_gb_count"] = date_gb.groupby("batter_id")["gb_count"].transform(
        lambda x: x.shift(1).expanding(min_periods=20).sum()
    )
    date_gb["batter_gb_hit_rate"] = np.where(
        date_gb["cum_gb_count"] > 0,
        date_gb["cum_gb_hits"] / date_gb["cum_gb_count"],
        np.nan,
    )
    batter_dates = batter_dates.merge(
        date_gb[["batter_id", "date", "batter_gb_hit_rate"]],
        on=["batter_id", "date"], how="left",
    )

    # --- Platoon H/PA — expanding, date-level ---
    df["platoon_key"] = df["batter_id"].astype(str) + "_" + df["pitch_hand"].fillna("U")
    date_platoon = df.groupby(["platoon_key", "batter_id", "pitch_hand", "date"]).agg(
        ph_hits=("is_hit", "sum"), ph_pas=("is_hit", "count"),
    ).reset_index().sort_values(["platoon_key", "date"])
    date_platoon["cum_ph_hits"] = date_platoon.groupby("platoon_key")["ph_hits"].transform(
        lambda x: x.shift(1).expanding(min_periods=5).sum()
    )
    date_platoon["cum_ph_pas"] = date_platoon.groupby("platoon_key")["ph_pas"].transform(
        lambda x: x.shift(1).expanding(min_periods=5).sum()
    )
    date_platoon["platoon_hr"] = np.where(
        date_platoon["cum_ph_pas"] >= 30,
        date_platoon["cum_ph_hits"] / date_platoon["cum_ph_pas"],
        np.nan,
    )

    # --- Pitcher rolling H allowed (date-level) ---
    pitcher_dates = df.groupby(["pitcher_id", "date"]).agg(
        p_hits=("is_hit", "sum"), p_pas=("is_hit", "count"),
    ).reset_index().sort_values(["pitcher_id", "date"])
    pitcher_dates["p_hit_rate"] = pitcher_dates["p_hits"] / pitcher_dates["p_pas"]
    pitcher_dates["pitcher_hr_30g"] = (
        pitcher_dates.groupby("pitcher_id")["p_hit_rate"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=10).mean())
    )

    # --- Pitcher arsenal entropy (date-level) ---
    def _pitch_entropy(types):
        if not isinstance(types, (list, np.ndarray)) or len(types) == 0:
            return np.nan
        counts = Counter(types)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    df["pa_pitch_entropy"] = df["pitch_types"].apply(_pitch_entropy)
    date_entropy = df.groupby(["pitcher_id", "date"])["pa_pitch_entropy"].mean().reset_index()
    date_entropy = date_entropy.sort_values(["pitcher_id", "date"])
    date_entropy["pitcher_entropy_30g"] = (
        date_entropy.groupby("pitcher_id")["pa_pitch_entropy"]
        .transform(lambda x: x.shift(1).rolling(30, min_periods=10).mean())
    )

    # --- Park factor (expanding, date-level, expanding normalization) ---
    date_venue = df.groupby(["venue_id", "date"])["is_hit"].mean().reset_index()
    date_venue.columns = ["venue_id", "date", "venue_date_hr"]
    date_venue = date_venue.sort_values(["venue_id", "date"])
    date_venue["venue_expanding_hr"] = date_venue.groupby("venue_id")["venue_date_hr"].transform(
        lambda x: x.shift(1).expanding(min_periods=20).mean()
    )
    # Normalize by expanding league-wide mean (not full-dataset mean)
    league_daily = df.groupby("date")["is_hit"].mean().reset_index()
    league_daily.columns = ["date", "league_hr"]
    league_daily = league_daily.sort_values("date")
    league_daily["league_expanding_hr"] = league_daily["league_hr"].shift(1).expanding(min_periods=30).mean()
    date_venue = date_venue.merge(league_daily[["date", "league_expanding_hr"]], on="date", how="left")
    date_venue["park_factor"] = np.where(
        date_venue["league_expanding_hr"] > 0,
        date_venue["venue_expanding_hr"] / date_venue["league_expanding_hr"],
        np.nan,
    )

    # --- Rest days ---
    rest_dates = df.groupby(["batter_id", "date"]).size().reset_index()[["batter_id", "date"]]
    rest_dates = rest_dates.drop_duplicates().sort_values(["batter_id", "date"])
    rest_dates["days_rest"] = rest_dates.groupby("batter_id")["date"].diff().dt.days

    # === Merge everything back to PA level ===

    # Batter date-level features → PA level (one-to-many: date → PAs on that date)
    merge_keys = ["batter_id", "date"]
    rolling_cols = [c for c in batter_dates.columns if c not in merge_keys
                    and c not in ("date_hits", "date_pas", "date_hit_rate")]
    df = df.merge(
        batter_dates[merge_keys + rolling_cols].drop_duplicates(subset=merge_keys),
        on=merge_keys, how="left",
    )

    # Pitcher date-level features
    df = df.merge(
        pitcher_dates[["pitcher_id", "date", "pitcher_hr_30g"]].drop_duplicates(subset=["pitcher_id", "date"]),
        on=["pitcher_id", "date"], how="left",
    )

    # Pitcher entropy
    df = df.merge(
        date_entropy[["pitcher_id", "date", "pitcher_entropy_30g"]].drop_duplicates(subset=["pitcher_id", "date"]),
        on=["pitcher_id", "date"], how="left",
    )

    # Platoon (batter × pitch_hand × date)
    df = df.merge(
        date_platoon[["batter_id", "pitch_hand", "date", "platoon_hr"]].drop_duplicates(
            subset=["batter_id", "pitch_hand", "date"]
        ),
        on=["batter_id", "pitch_hand", "date"], how="left",
    )

    # Park factor (venue × date)
    df = df.merge(
        date_venue[["venue_id", "date", "park_factor"]].drop_duplicates(subset=["venue_id", "date"]),
        on=["venue_id", "date"], how="left",
    )

    # Rest days (batter × date)
    df = df.merge(
        rest_dates[["batter_id", "date", "days_rest"]],
        on=["batter_id", "date"], how="left",
    )

    return df


# Feature columns — provably leak-free.
# No clustering features (pitcher_cluster, batter_vs_arch_hr) —
# K-Means centroids are unstable and use full-dataset pitch data.
FEATURE_COLS = [
    "batter_hr_7g",
    "batter_hr_30g",
    "batter_hr_60g",
    "batter_hr_120g",
    "batter_whiff_60g",
    "batter_count_tendency_30g",
    "batter_gb_hit_rate",
    "platoon_hr",
    "pitcher_hr_30g",
    "pitcher_entropy_30g",
    "weather_temp",
    "park_factor",
    "days_rest",
]
