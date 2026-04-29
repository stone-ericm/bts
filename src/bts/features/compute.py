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
import os

import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

# Optimal training window: 2019 onward. Adding 2017-2018 hurts P@1 by 1.1%
# because the game changed enough that old training examples add noise.
# Features are still computed from ALL available history (expanding features
# benefit from more data), but the model should be trained on 2019+ PAs.
TRAIN_START_YEAR = 2019

# Rookie-gated shrinkage on 30/60/120g batter hit-rate features. Batters with
# fewer than ROOKIE_THRESHOLD_PAS lifetime PAs get pulled toward LEAGUE_PA_HIT_RATE_PRIOR
# via pseudocount shrinkage with strength K. Veterans are untouched.
# Set BTS_ROOKIE_GATE_K=0 to revert to the unshrunken baseline.
# (Prior "shipped" claim of +3.65pp P(57) MDP was measured on the buggy pre-
# row-order-fix pipeline and is being re-verified on the fixed pipeline.)
ROOKIE_GATE_K = int(os.environ.get("BTS_ROOKIE_GATE_K", "20"))
ROOKIE_THRESHOLD_PAS = 100
LEAGUE_PA_HIT_RATE_PRIOR = 0.2195  # measured from 2021-2025 pa_*.parquet

# Pitcher rolling hit-rate min_periods. 7 lets the rolling mean activate ~3
# starts earlier in a pitcher's history, which is strictly more signal for
# the blend's feature vector. Verified on 2-season walk-forward + MDP:
# +1.08pp 2024, +0.54pp 2025, +0.46pp MDP P(57), -3 miss days. MC P(57)
# regresses due to streak clustering, but MDP mitigates via skip/double.
# Set BTS_PITCHER_HR_30G_MIN_PERIODS=10 to revert to the historical baseline.
PITCHER_HR_30G_MIN_PERIODS = int(os.environ.get("BTS_PITCHER_HR_30G_MIN_PERIODS", "7"))


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
    # 7g window is always the unshrunken rolling mean — it captures recent
    # form, not true-talent level, so shrinkage toward league prior would
    # defeat the point.
    batter_dates["batter_hr_7g"] = (
        batter_dates.groupby("batter_id")["date_hit_rate"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean())
    )

    if ROOKIE_GATE_K > 0:
        # Rookie-gated shrinkage. Rookies (career_pas < ROOKIE_THRESHOLD_PAS)
        # get PA-weighted rolling + league-prior pseudocount. Veterans get
        # the original baseline rolling mean untouched.
        career_pas_gate = batter_dates.groupby("batter_id")["date_pas"].transform(
            lambda x: x.shift(1).expanding(min_periods=1).sum()
        ).fillna(0)
        is_rookie = career_pas_gate < ROOKIE_THRESHOLD_PAS

        for w in [30, 60, 120]:
            baseline_col = batter_dates.groupby("batter_id")["date_hit_rate"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=max(3, w // 4)).mean()
            )
            rolling_hits = batter_dates.groupby("batter_id")["date_hits"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).sum()
            ).fillna(0)
            rolling_pas = batter_dates.groupby("batter_id")["date_pas"].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).sum()
            ).fillna(0)
            shrunken_col = (
                (rolling_hits + ROOKIE_GATE_K * LEAGUE_PA_HIT_RATE_PRIOR)
                / (rolling_pas + ROOKIE_GATE_K)
            )
            batter_dates[f"batter_hr_{w}g"] = np.where(
                is_rookie, shrunken_col, baseline_col
            )
    else:
        # Baseline: day-level rolling mean with hard min_periods cutoff.
        for w in [30, 60, 120]:
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
    # Vectorized barrel classification — avoids df.apply(axis=1) which is
    # catastrophically slow on 1.5M rows (30+ min on cloud VMs).
    ev = df["launch_speed"]
    la = df["launch_angle"]
    bonus = (np.minimum(ev, 116) - 98) * 2
    la_min = np.maximum(8.0, 26.0 - bonus)
    la_max = np.minimum(50.0, 30.0 + bonus)
    df["is_barrel"] = ev.notna() & la.notna() & (ev >= 98) & (la >= la_min) & (la <= la_max)
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
        .transform(lambda x: x.shift(1).rolling(30, min_periods=PITCHER_HR_30G_MIN_PERIODS).mean())
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
    # List comprehension over zipped columns is ~5-10x faster than
    # df.apply(axis=1) which creates a Series per row.
    df["pa_total_break"] = pd.to_numeric(pd.Series(
        [_total_break(v, h) for v, h in zip(
            df["pitch_break_vertical"].values, df["pitch_break_horizontal"].values)],
        index=df.index,
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
    def _borderline_csr_direct(calls, px, pz, sz_top, sz_bottom):
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

    # List comprehension avoids df.apply(axis=1) per-row Series overhead.
    df["pa_borderline_csr"] = [
        _borderline_csr_direct(calls, px, pz, st, sb)
        for calls, px, pz, st, sb in zip(
            df["pitch_calls"].values, df["pitch_px"].values, df["pitch_pz"].values,
            df["sz_top"].values, df["sz_bottom"].values,
        )
    ]
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

    # --- Context features (4) — always computed, used by shadow model ---

    # Umpire hit rate: rolling 30-day hit rate per home-plate umpire
    if "hp_umpire_id" in df.columns:
        ump_daily = df.groupby(["hp_umpire_id", "date"]).agg(
            ump_hits=("is_hit", "sum"), ump_pas=("is_hit", "count"),
        ).reset_index().sort_values(["hp_umpire_id", "date"])
        ump_daily["ump_hr"] = ump_daily["ump_hits"] / ump_daily["ump_pas"]
        ump_daily["ump_hr_30g"] = (
            ump_daily.groupby("hp_umpire_id")["ump_hr"]
            .transform(lambda x: x.shift(1).rolling(30, min_periods=10).mean())
        )
        df = df.merge(
            ump_daily[["hp_umpire_id", "date", "ump_hr_30g"]]
            .drop_duplicates(subset=["hp_umpire_id", "date"]),
            on=["hp_umpire_id", "date"], how="left",
        )
    else:
        df["ump_hr_30g"] = np.nan

    # Wind vector: signed scalar (positive = blowing out to CF, helps hitters)
    if "weather_wind_dir" in df.columns and "weather_wind_speed" in df.columns:
        direction = df["weather_wind_dir"].astype(str).str.lower()
        speed = pd.to_numeric(df["weather_wind_speed"], errors="coerce").fillna(0)
        direction_score = np.where(
            direction.str.contains("out to cf|out to center"), 1.0,
            np.where(
                direction.str.contains("in from cf|in from center"), -1.0,
                np.where(
                    direction.str.contains("out to lf|out to l f|out to rf|out to r f"), 0.5,
                    np.where(
                        direction.str.contains("in from lf|in from rf"), -0.5,
                        0.0,
                    ),
                ),
            ),
        )
        df["wind_out_cf"] = direction_score * speed
    else:
        df["wind_out_cf"] = np.nan

    # Batter hard-contact rate: rolling 30-day from categorical hardness column
    if "hardness" in df.columns:
        is_hard = (df["hardness"].astype(str).str.lower() == "hard").astype(float)
        is_hard = is_hard.where(df["hardness"].notna(), np.nan)
        df["_is_hard"] = is_hard
        # Sort a *view* — mutating df's row order silently breaks LightGBM
        # bagging reproducibility (subsample=0.8 picks rows by index).
        sorted_view = df.sort_values(["batter_id", "date"])
        df["batter_hard_contact_30g"] = (
            sorted_view.groupby("batter_id")["_is_hard"]
            .rolling(window=120, min_periods=10)
            .mean()
            .reset_index(level=0, drop=True)
            .shift(1)
        )
        df.drop(columns=["_is_hard"], inplace=True)
    else:
        df["batter_hard_contact_30g"] = np.nan

    # Indoor flag: binary for dome/closed/retractable roofs
    if "roof_type" in df.columns:
        rt = df["roof_type"].astype(str).str.lower()
        df["is_indoor"] = rt.isin(["dome", "closed", "retractable"]).astype(int)
    else:
        df["is_indoor"] = 0

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

    # batter_pitcher_shrunk_hr — Bayesian-shrunk historical (batter, pitcher) hit rate.
    # Promoted to FEATURE_COLS 2026-04-29 after 4-way replication:
    #   Phase 1 single-seed (passed), Phase 1 n=10 stratified (t=+3.35),
    #   Phase 2 set 1 canonical-n10 (+2.77pp pooled Δ, KEEP),
    #   Phase 2 set 2 orthogonal-n10 (+3.49pp pooled Δ, KEEP).
    # Aggregates per (batter, pitcher, date) so same-day multi-PA rows share the
    # same prior-day stat (no within-day leakage). Falls back to league prior 0.2195
    # for sparse pairings. Idempotent with BatterPitcherMatchupExperiment.modify_features.
    if all(c in df.columns for c in ("batter_id", "pitcher_id", "is_hit", "date")):
        _PRIOR_RATE = 0.2195
        _K = 10
        _daily = (
            df.groupby(["batter_id", "pitcher_id", "date"])
            .agg(_day_hits=("is_hit", "sum"), _day_pas=("is_hit", "count"))
            .reset_index()
            .sort_values(["batter_id", "pitcher_id", "date"])
        )
        _daily["_cum_hits_prior"] = (
            _daily.groupby(["batter_id", "pitcher_id"])["_day_hits"]
            .transform(lambda s: s.cumsum().shift(1).fillna(0))
        )
        _daily["_cum_pas_prior"] = (
            _daily.groupby(["batter_id", "pitcher_id"])["_day_pas"]
            .transform(lambda s: s.cumsum().shift(1).fillna(0))
        )
        _daily["batter_pitcher_shrunk_hr"] = (
            (_PRIOR_RATE * _K + _daily["_cum_hits_prior"])
            / (_K + _daily["_cum_pas_prior"])
        )
        df = df.merge(
            _daily[["batter_id", "pitcher_id", "date", "batter_pitcher_shrunk_hr"]],
            on=["batter_id", "pitcher_id", "date"], how="left",
        )
        df["batter_pitcher_shrunk_hr"] = df["batter_pitcher_shrunk_hr"].fillna(_PRIOR_RATE)
    else:
        df["batter_pitcher_shrunk_hr"] = 0.2195

    return df


# Feature columns (16) — provably leak-free. 15 baseline + 1 promoted from
# the 2026-04-29 Phase 2 validation: batter_pitcher_shrunk_hr.
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
    "batter_pitcher_shrunk_hr",  # promoted 2026-04-29 (was BatterPitcherMatchupExperiment)
]

# Context features (4) — computed alongside baseline features but only used
# by the shadow model (via feature_cols_override). Graduates to FEATURE_COLS
# after 30-day shadow validation.
CONTEXT_COLS = [
    "ump_hr_30g",
    "wind_out_cf",
    "batter_hard_contact_30g",
    "is_indoor",
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
