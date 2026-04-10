"""Feature computation for PA-level hit prediction.

TEMPORAL GUARANTEE: Every feature for a PA on date D uses ONLY data from
dates strictly before D. This is enforced by:
1. All rolling/expanding features group by (entity, date) and use shift(1)
   on date-level aggregates — so doubleheader games on the same date are
   merged before shifting.
2. No clustering or global aggregation that uses future data.
3. Park factor normalization uses expanding league-wide mean.
"""

import json

import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

# Optimal training window: 2019 onward. Adding 2017-2018 hurts P@1 by 1.1%
# because the game changed enough that old training examples add noise.
# Features are still computed from ALL available history (expanding features
# benefit from more data), but the model should be trained on 2019+ PAs.
TRAIN_START_YEAR = 2019


_probable_pitcher_cache_path = Path("data/models/probable_pitcher_lookup.json")


def _build_probable_pitcher_lookup(raw_dir: str = "data/raw") -> dict:
    """Build game_pk → probable pitcher + team ID lookup from raw feeds.

    Caches to disk and only scans new game feeds on subsequent calls.
    Returns {game_pk: {"away": pitcher_id, "home": pitcher_id,
                        "away_tid": team_id, "home_tid": team_id}}.
    """
    # Load existing cache
    lookup = {}
    if _probable_pitcher_cache_path.exists():
        try:
            cached = json.loads(_probable_pitcher_cache_path.read_text())
            lookup = {int(k): v for k, v in cached.items()}
        except Exception:
            pass

    raw = Path(raw_dir)
    if not raw.exists():
        return lookup

    # Scan only files not already in cache
    new_count = 0
    for season_dir in sorted(raw.iterdir()):
        if not season_dir.is_dir():
            continue
        for f in season_dir.glob("*.json"):
            pk = int(f.stem)
            if pk in lookup:
                continue
            try:
                d = json.loads(f.read_text())
                gd = d.get("gameData", {})
                teams = gd.get("teams", {})
                pp = gd.get("probablePitchers", {})
                lookup[pk] = {
                    "away": pp.get("away", {}).get("id"),
                    "home": pp.get("home", {}).get("id"),
                    "away_tid": teams.get("away", {}).get("id"),
                    "home_tid": teams.get("home", {}).get("id"),
                }
                new_count += 1
            except Exception:
                continue

    # Save updated cache
    if new_count > 0:
        _probable_pitcher_cache_path.parent.mkdir(parents=True, exist_ok=True)
        _probable_pitcher_cache_path.write_text(
            json.dumps({str(k): v for k, v in lookup.items()})
        )

    return lookup


def _is_barrel(ev, la):
    """MLB barrel classification from exit velocity + launch angle."""
    if pd.isna(ev) or pd.isna(la) or ev < 98:
        return False
    bonus = (min(ev, 116) - 98) * 2
    la_min = max(8, 26 - bonus)
    la_max = min(50, 30 + bonus)
    return la_min <= la <= la_max


def _mean_of_list(lst):
    """Mean of a list, ignoring None values. Returns NaN if empty."""
    if not isinstance(lst, (list, np.ndarray)):
        return np.nan
    vals = [v for v in lst if v is not None]
    return np.mean(vals) if vals else np.nan


def _total_break(vert_list, horiz_list):
    """Mean total break magnitude from per-pitch vertical and horizontal break lists."""
    if not isinstance(vert_list, (list, np.ndarray)):
        return np.nan
    total = []
    for v, h in zip(vert_list, horiz_list):
        if v is not None and h is not None:
            total.append(np.sqrt(v**2 + h**2))
    return np.mean(total) if total else np.nan


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

    # --- Batter Statcast batted ball features (date-level) ---
    df["is_barrel"] = df.apply(lambda r: _is_barrel(r["launch_speed"], r["launch_angle"]), axis=1)
    df["is_hard_hit"] = df["launch_speed"].notna() & (df["launch_speed"] >= 95)
    df["is_sweet_spot"] = df["launch_angle"].notna() & (df["launch_angle"] >= 8) & (df["launch_angle"] <= 32)
    df["has_batted_ball"] = df["launch_speed"].notna()

    date_batted = df.groupby(["batter_id", "date"]).agg(
        barrels=("is_barrel", "sum"),
        hard_hits=("is_hard_hit", "sum"),
        sweet_spots=("is_sweet_spot", "sum"),
        batted_balls=("has_batted_ball", "sum"),
        avg_ev=("launch_speed", lambda x: x.dropna().mean() if x.notna().any() else np.nan),
    ).reset_index().sort_values(["batter_id", "date"])

    date_batted["barrel_rate"] = np.where(
        date_batted["batted_balls"] > 0,
        date_batted["barrels"] / date_batted["batted_balls"],
        np.nan,
    )
    date_batted["hard_hit_rate"] = np.where(
        date_batted["batted_balls"] > 0,
        date_batted["hard_hits"] / date_batted["batted_balls"],
        np.nan,
    )
    date_batted["sweet_spot_rate"] = np.where(
        date_batted["batted_balls"] > 0,
        date_batted["sweet_spots"] / date_batted["batted_balls"],
        np.nan,
    )

    date_batted["batter_barrel_rate_30g"] = date_batted.groupby("batter_id")["barrel_rate"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_batted["batter_hard_hit_rate_30g"] = date_batted.groupby("batter_id")["hard_hit_rate"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_batted["batter_sweet_spot_rate_30g"] = date_batted.groupby("batter_id")["sweet_spot_rate"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_batted["batter_avg_ev_30g"] = date_batted.groupby("batter_id")["avg_ev"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )

    batter_dates = batter_dates.merge(
        date_batted[["batter_id", "date", "batter_barrel_rate_30g", "batter_hard_hit_rate_30g",
                      "batter_sweet_spot_rate_30g", "batter_avg_ev_30g"]],
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

    # --- Pitcher Statcast features (date-level) ---
    # Force float64 dtype on computed per-PA stats. When pyarrow reads nested
    # list columns (pitch_speeds, etc.) with mixed None/array values, pandas
    # may infer object dtype from .apply() even though all results are numeric.
    # LightGBM rejects object-dtype columns, so we coerce here defensively.
    df["pa_avg_velo"] = pd.to_numeric(df["pitch_speeds"].apply(_mean_of_list), errors="coerce")
    df["pa_avg_spin"] = pd.to_numeric(df["pitch_spin_rates"].apply(_mean_of_list), errors="coerce")
    df["pa_avg_extension"] = pd.to_numeric(df["pitch_extensions"].apply(_mean_of_list), errors="coerce")
    df["pa_total_break"] = pd.to_numeric(df.apply(
        lambda r: _total_break(r.get("pitch_break_vertical", []), r.get("pitch_break_horizontal", [])), axis=1
    ), errors="coerce")

    date_pitch_stats = df.groupby(["pitcher_id", "date"]).agg(
        avg_velo=("pa_avg_velo", "mean"),
        avg_spin=("pa_avg_spin", "mean"),
        avg_extension=("pa_avg_extension", "mean"),
        avg_break=("pa_total_break", "mean"),
    ).reset_index().sort_values(["pitcher_id", "date"])

    date_pitch_stats["pitcher_avg_velo_30g"] = date_pitch_stats.groupby("pitcher_id")["avg_velo"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_pitch_stats["pitcher_avg_spin_30g"] = date_pitch_stats.groupby("pitcher_id")["avg_spin"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_pitch_stats["pitcher_avg_extension_30g"] = date_pitch_stats.groupby("pitcher_id")["avg_extension"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_pitch_stats["pitcher_break_total_30g"] = date_pitch_stats.groupby("pitcher_id")["avg_break"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )

    # --- Batter: average pitch velocity faced (date-level) ---
    date_velo_faced = df.groupby(["batter_id", "date"])["pa_avg_velo"].mean().reset_index()
    date_velo_faced.columns = ["batter_id", "date", "avg_velo_faced"]
    date_velo_faced = date_velo_faced.sort_values(["batter_id", "date"])
    date_velo_faced["batter_avg_velo_faced_30g"] = date_velo_faced.groupby("batter_id")["avg_velo_faced"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
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

    # --- Catcher framing proxy (pitcher-level, expanding) ---
    # Borderline pitch = near horizontal edge (|pX| 0.5-1.2) or near vertical edge
    # of strike zone. Higher called-strike rate on these = better catcher framing.
    # Uses pitcher_id as proxy (captures team-level effect). Expanding with shift(1).
    # NOTE: Savant prior-season framing (static) tested worse than this expanding proxy.
    def _borderline_csr(row):
        calls = row.get("pitch_calls")
        px = row.get("pitch_px")
        pz = row.get("pitch_pz")
        sz_top = row.get("sz_top")
        sz_bottom = row.get("sz_bottom")
        if not isinstance(calls, (list, np.ndarray)) or len(calls) == 0:
            return np.nan
        if px is None or pz is None:
            return np.nan
        borderline = 0
        called_strikes = 0
        for c, x, z in zip(calls, px, pz):
            if x is None or z is None or sz_top is None or sz_bottom is None:
                continue
            if 0.5 < abs(x) < 1.2 or abs(z - sz_top) < 0.3 or abs(z - sz_bottom) < 0.3:
                borderline += 1
                if c == "C":
                    called_strikes += 1
        return called_strikes / borderline if borderline > 0 else np.nan

    df["pa_borderline_csr"] = df.apply(_borderline_csr, axis=1)
    date_framing = df.groupby(["pitcher_id", "date"])["pa_borderline_csr"].mean().reset_index()
    date_framing.columns = ["pitcher_id", "date", "date_csr"]
    date_framing = date_framing.sort_values(["pitcher_id", "date"])
    date_framing["pitcher_catcher_framing"] = date_framing.groupby("pitcher_id")["date_csr"].transform(
        lambda x: x.shift(1).expanding(min_periods=5).mean()
    )

    # --- Rest days ---
    rest_dates = df.groupby(["batter_id", "date"]).size().reset_index()[["batter_id", "date"]]
    rest_dates = rest_dates.drop_duplicates().sort_values(["batter_id", "date"])
    rest_dates["days_rest"] = rest_dates.groupby("batter_id")["date"].diff().dt.days

    # --- Team bullpen composite (rolling reliever quality per team) ---
    # Identify relievers: any pitcher who is NOT the probable starter for their team.
    # Probable pitcher data comes from raw game feeds.
    probable_pitchers = _build_probable_pitcher_lookup()
    if probable_pitchers:
        # Determine pitcher's team: if batter is_home, pitcher is away team
        df["_probable_pid"] = df["game_pk"].map(
            lambda pk: probable_pitchers.get(pk, {}).get("away")
            if not df.loc[df["game_pk"] == pk, "is_home"].iloc[0]
            else probable_pitchers.get(pk, {}).get("home")
            if pk in probable_pitchers else None
        )
        # Vectorized: build per-row probable pitcher ID
        away_prob = df["game_pk"].map(lambda pk: probable_pitchers.get(pk, {}).get("away"))
        home_prob = df["game_pk"].map(lambda pk: probable_pitchers.get(pk, {}).get("home"))
        # If batter is_home, pitcher is on away team → use away probable
        # If batter is away, pitcher is on home team → use home probable
        df["_probable_pid"] = np.where(df["is_home"], away_prob, home_prob)
        df["_is_reliever_pa"] = (df["pitcher_id"] != df["_probable_pid"]) & df["_probable_pid"].notna()

        # Identify pitching team: opposite of batter's side
        # Use game_pk-based team IDs from the lookup
        away_tid = df["game_pk"].map(lambda pk: probable_pitchers.get(pk, {}).get("away_tid"))
        home_tid = df["game_pk"].map(lambda pk: probable_pitchers.get(pk, {}).get("home_tid"))
        df["_pitcher_team_id"] = np.where(df["is_home"], away_tid, home_tid)

        # Compute daily reliever hit rate per pitching team
        reliever_pas = df[df["_is_reliever_pa"]].copy()
        daily_bullpen = reliever_pas.groupby(["_pitcher_team_id", "date"]).agg(
            bp_hits=("is_hit", "sum"), bp_pas=("is_hit", "count"),
        ).reset_index().sort_values(["_pitcher_team_id", "date"])
        daily_bullpen["bp_hr"] = daily_bullpen["bp_hits"] / daily_bullpen["bp_pas"]

        # Rolling 30-day average per team (shift by 1 for leakage prevention)
        daily_bullpen["opp_bullpen_hr_30g"] = (
            daily_bullpen.groupby("_pitcher_team_id")["bp_hr"]
            .transform(lambda x: x.shift(1).rolling(30, min_periods=10).mean())
        )

        # Merge back: each PA gets the opposing team's bullpen quality
        bp_merge = daily_bullpen[["_pitcher_team_id", "date", "opp_bullpen_hr_30g"]].drop_duplicates(
            subset=["_pitcher_team_id", "date"]
        )
        df = df.merge(bp_merge, on=["_pitcher_team_id", "date"], how="left")

        df.rename(columns={"_pitcher_team_id": "opp_pitching_team_id"}, inplace=True)
        df.drop(columns=["_probable_pid", "_is_reliever_pa"], inplace=True)
    else:
        df["opp_bullpen_hr_30g"] = np.nan

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

    # Pitcher Statcast
    df = df.merge(
        date_pitch_stats[["pitcher_id", "date", "pitcher_avg_velo_30g", "pitcher_avg_spin_30g",
                           "pitcher_avg_extension_30g", "pitcher_break_total_30g"]]
        .drop_duplicates(subset=["pitcher_id", "date"]),
        on=["pitcher_id", "date"], how="left",
    )

    # Batter velo faced
    df = df.merge(
        date_velo_faced[["batter_id", "date", "batter_avg_velo_faced_30g"]]
        .drop_duplicates(subset=["batter_id", "date"]),
        on=["batter_id", "date"], how="left",
    )

    # Catcher framing (pitcher × date)
    df = df.merge(
        date_framing[["pitcher_id", "date", "pitcher_catcher_framing"]]
        .drop_duplicates(subset=["pitcher_id", "date"]),
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


# Feature columns (15) — provably leak-free. 14 baseline features plus
# team bullpen composite (rolling 30-day reliever hit rate for the opposing team).
# Bullpen feature improved MDP P(57) from 4.85% to 5.73% by polarizing quality bins.
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
    "pitcher_catcher_framing",
    "opp_bullpen_hr_30g",
    "weather_temp",
    "park_factor",
    "days_rest",
]

# Statcast features (9) — computed from game feed pitchData/hitData.
# These don't improve the single model but add diversity to the 12-model blend.
# Each blend variant uses FEATURE_COLS + one Statcast feature.
STATCAST_COLS = [
    "batter_barrel_rate_30g",
    "batter_hard_hit_rate_30g",
    "batter_sweet_spot_rate_30g",
    "batter_avg_ev_30g",
    "pitcher_avg_velo_30g",
    "pitcher_avg_spin_30g",
    "pitcher_avg_extension_30g",
    "pitcher_break_total_30g",
    "batter_avg_velo_faced_30g",
]
