"""Phase 1 feature experiments."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from bts.experiment.base import ExperimentDef
from bts.experiment.registry import register
from bts.features.compute import FEATURE_COLS


class EBShrinkageExperiment(ExperimentDef):
    """Replace raw rolling averages with beta-binomial EB shrunken estimates."""

    def __init__(self):
        super().__init__(
            name="eb_shrinkage",
            phase=1,
            category="feature",
            description="Beta-binomial empirical Bayes shrinkage on rolling hit rates",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_cols = ["batter_hr_7g", "batter_hr_30g", "batter_hr_60g", "batter_hr_120g"]
        for col in rolling_cols:
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            if len(vals) < 50:
                continue
            pop_mean = vals.mean()
            pop_var = vals.var()
            if pop_var <= 0 or pop_mean <= 0 or pop_mean >= 1:
                continue
            # Method of moments for Beta(alpha, beta)
            alpha = pop_mean * (pop_mean * (1 - pop_mean) / pop_var - 1)
            beta = (1 - pop_mean) * (pop_mean * (1 - pop_mean) / pop_var - 1)
            if alpha <= 0 or beta <= 0:
                continue
            n_eff = alpha + beta
            # Approximate n from window name (~3.5 PA per game)
            window_map = {"7g": 7, "30g": 30, "60g": 60, "120g": 120}
            suffix = col.split("_")[-1]
            n_approx = window_map.get(suffix, 30) * 3.5
            weight = n_approx / (n_approx + n_eff)
            df[col] = weight * df[col] + (1 - weight) * pop_mean
        return df


class KLDivergenceExperiment(ExperimentDef):
    """Replace pitcher entropy with Fisher-Rao distance to batter comfort zone."""

    def __init__(self):
        super().__init__(
            name="kl_divergence",
            phase=1,
            category="feature",
            description="Fisher-Rao distance between pitcher mix and batter comfort zone",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # PA data uses 'pitch_types' (list per PA) — explode to per-pitch rows
        if "pitch_types" not in df.columns:
            df["pitcher_batter_fr_distance"] = np.nan
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Explode pitch_types lists to per-pitch long format
        exp = df[["pitcher_id", "batter_id", "date", "pitch_types"]].copy()
        exp["pitch_types"] = exp["pitch_types"].apply(
            lambda x: list(x) if isinstance(x, (list, np.ndarray)) else []
        )
        exp = exp.explode("pitch_types").dropna(subset=["pitch_types"])
        exp = exp[exp["pitch_types"] != ""]
        exp = exp.rename(columns={"pitch_types": "pt"})

        if len(exp) < 1000:
            df["pitcher_batter_fr_distance"] = np.nan
            return df

        # Per-pitcher cumulative pitch-type distribution (shifted to avoid leakage)
        pitcher_daily = (
            exp.groupby(["pitcher_id", "date", "pt"]).size().unstack(fill_value=0)
        )
        pitcher_cum = pitcher_daily.groupby(level=0).cumsum()
        # Shift within pitcher to use only PRIOR dates
        pitcher_cum_shifted = pitcher_cum.groupby(level=0).shift(1).fillna(0)
        pitcher_dist = pitcher_cum_shifted.div(
            pitcher_cum_shifted.sum(axis=1).replace(0, np.nan), axis=0
        )

        # Per-batter cumulative pitch-type distribution (comfort zone)
        batter_daily = (
            exp.groupby(["batter_id", "date", "pt"]).size().unstack(fill_value=0)
        )
        batter_cum = batter_daily.groupby(level=0).cumsum()
        batter_cum_shifted = batter_cum.groupby(level=0).shift(1).fillna(0)
        batter_dist = batter_cum_shifted.div(
            batter_cum_shifted.sum(axis=1).replace(0, np.nan), axis=0
        )

        # Align on union of pitch types
        all_types = sorted(set(pitcher_dist.columns) | set(batter_dist.columns))
        for t in all_types:
            if t not in pitcher_dist.columns:
                pitcher_dist[t] = np.nan
            if t not in batter_dist.columns:
                batter_dist[t] = np.nan
        pitcher_dist = pitcher_dist[all_types].fillna(1e-6)
        batter_dist = batter_dist[all_types].fillna(1e-6)

        # Renormalize after fill
        pitcher_dist = pitcher_dist.div(pitcher_dist.sum(axis=1), axis=0)
        batter_dist = batter_dist.div(batter_dist.sum(axis=1), axis=0)

        # Build per-PA distance via merge
        df_keys = df[["pitcher_id", "batter_id", "date"]].copy()
        pitcher_lookup = pitcher_dist.reset_index()
        batter_lookup = batter_dist.reset_index()

        df_p = df_keys.merge(
            pitcher_lookup, on=["pitcher_id", "date"], how="left", suffixes=("", "_p"),
        )
        df_full = df_p.merge(
            batter_lookup, on=["batter_id", "date"], how="left", suffixes=("_p", "_b"),
        )

        p_cols = [f"{t}_p" for t in all_types]
        b_cols = [f"{t}_b" for t in all_types]
        # If suffix didn't apply because no name conflict, columns are just `t`
        # Detect actual column naming
        actual_p_cols = [c for c in df_full.columns if c.endswith("_p") and c != "pitcher_id"]
        actual_b_cols = [c for c in df_full.columns if c.endswith("_b") and c != "bat_side"]

        # Fall back: just use the all_types columns directly
        if not actual_p_cols:
            # Merge didn't suffix because no conflict — distinguish manually
            df["pitcher_batter_fr_distance"] = np.nan
            return df

        try:
            P = df_full[actual_p_cols].values
            Q = df_full[actual_b_cols].values
            # Fisher-Rao distance via Bhattacharyya
            bhatt = np.sum(np.sqrt(np.maximum(P * Q, 0)), axis=1)
            fr_dist = 2.0 * np.arccos(np.clip(bhatt, -1.0, 1.0))
            df["pitcher_batter_fr_distance"] = fr_dist
        except Exception:
            df["pitcher_batter_fr_distance"] = np.nan

        return df

    def feature_cols(self) -> list[str]:
        cols = [c for c in FEATURE_COLS if c != "pitcher_entropy_30g"]
        cols.append("pitcher_batter_fr_distance")
        return cols


class BattingHeatQExperiment(ExperimentDef):
    """Add Batting Heat Index (Q) — consecutive-game weighted streakiness."""

    def __init__(self):
        super().__init__(
            name="batting_heat_q",
            phase=1,
            category="feature",
            description="Batting Heat Index: consecutive-game hit streaks weighted by BA",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Per-batter, per-date hit indicator and BA
        per_day = df.groupby(["batter_id", "date"]).agg(
            day_hit=("is_hit", "max"),
            day_ba=("is_hit", "mean"),
        ).reset_index().sort_values(["batter_id", "date"])

        # Compute streak and rolling BA-during-streak per batter
        q_by_batter_date: dict = {}
        for batter_id, group in per_day.groupby("batter_id"):
            streak = 0
            streak_ba_sum = 0.0
            for _, row in group.iterrows():
                d = row["date"]
                # Q reflects state BEFORE this date (no leakage)
                q_by_batter_date[(batter_id, d)] = (
                    streak * (streak_ba_sum / streak) if streak > 0 else 0.0
                )
                if row["day_hit"] > 0:
                    streak += 1
                    streak_ba_sum += row["day_ba"]
                else:
                    streak = 0
                    streak_ba_sum = 0.0

        # Map back to PA-level via merge
        df["batting_heat_q"] = df.set_index(["batter_id", "date"]).index.map(
            q_by_batter_date.get
        )
        df["batting_heat_q"] = df["batting_heat_q"].fillna(0.0)
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["batting_heat_q"]


class GBPlatoonExperiment(ExperimentDef):
    """Add groundball-rate platoon interaction feature."""

    def __init__(self):
        super().__init__(
            name="gb_platoon",
            phase=1,
            category="feature",
            description="Groundball rate x same/opposite handedness interaction",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "bat_side" not in df.columns or "pitch_hand" not in df.columns:
            df["gb_platoon_rate"] = np.nan
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Derive batted_ball_type from launch_angle (MLB standard ranges):
        #   GB: la <= 10
        #   LD: 10 < la <= 25
        #   FB: 25 < la <= 50
        #   PU: la > 50
        # Only valid when there was contact (launch_speed > 0)
        if "launch_angle" in df.columns:
            la = df["launch_angle"]
            ls = df.get("launch_speed", pd.Series(0, index=df.index))
            has_contact = (ls.fillna(0) > 0) & la.notna()
            df["_is_gb"] = ((la <= 10) & has_contact).astype(float)
            df.loc[~has_contact, "_is_gb"] = np.nan
        else:
            df["gb_platoon_rate"] = np.nan
            return df

        # Same-handedness flag
        df["_same_hand"] = (df["bat_side"] == df["pitch_hand"]).astype(float)

        # Expanding GB rate per batter × handedness matchup
        df = df.sort_values(["batter_id", "date"])
        df["gb_platoon_rate"] = (
            df.groupby(["batter_id", "_same_hand"])["_is_gb"]
            .expanding()
            .mean()
            .shift(1)
            .reset_index(level=[0, 1], drop=True)
        )

        df.drop(columns=["_is_gb", "_same_hand"], inplace=True, errors="ignore")
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["gb_platoon_rate"]


class HitTypeParkFactorsExperiment(ExperimentDef):
    """Replace single park_factor with hit-type-specific factors."""

    def __init__(self):
        super().__init__(
            name="hit_type_park",
            phase=1,
            category="feature",
            description="Separate park factors for singles, doubles, triples",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "event_type" not in df.columns or "venue_id" not in df.columns:
            for col in ["park_factor_1b", "park_factor_2b", "park_factor_3b"]:
                df[col] = df.get("park_factor", np.nan)
            return df

        for hit_type, col_name in [
            ("single", "park_factor_1b"),
            ("double", "park_factor_2b"),
            ("triple", "park_factor_3b"),
        ]:
            venue_rates = df[df["event_type"] == hit_type].groupby("venue_id").size()
            venue_totals = df.groupby("venue_id").size()
            venue_factor = (venue_rates / venue_totals).fillna(0)
            league_avg = venue_factor.mean()
            if league_avg > 0:
                venue_factor = venue_factor / league_avg
            else:
                venue_factor[:] = 1.0
            df[col_name] = df["venue_id"].map(venue_factor).fillna(1.0)

        return df

    def feature_cols(self) -> list[str]:
        cols = [c for c in FEATURE_COLS if c != "park_factor"]
        return cols + ["park_factor_1b", "park_factor_2b", "park_factor_3b"]


class StreakLengthFeatureExperiment(ExperimentDef):
    """Add streak length as a direct model feature (conditional on Phase 0)."""

    def __init__(self):
        super().__init__(
            name="streak_length_feature",
            phase=1,
            category="feature",
            description="Current streak length as model feature",
            dependencies=["streak_length_dependence"],
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["batter_id", "date"])
        streak_col = np.zeros(len(df), dtype=int)

        for batter_id, group in df.groupby("batter_id"):
            date_hits = group.groupby("date")["is_hit"].max()
            streak = 0
            streak_by_date = {}
            for d in sorted(date_hits.index):
                streak_by_date[d] = streak
                streak = streak + 1 if date_hits[d] > 0 else 0
            for idx in group.index:
                streak_col[idx] = streak_by_date.get(group.loc[idx, "date"], 0)

        df["batter_streak_length"] = streak_col
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["batter_streak_length"]


# ============================================================================
# Dormant column experiments — features from PA columns we already collect
# but never turned into model features.
# ============================================================================


def _rolling_rate_by_group(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    window: int,
    feature_name: str,
) -> pd.DataFrame:
    """Compute per-group rolling mean of target, shifted by 1 to prevent leakage.

    Helper for "rolling hit rate per umpire/catcher" type features.
    Aggregates per-(group, date) first, then rolls.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Daily aggregate per group
    daily = df.groupby([group_col, "date"]).agg(
        rate=(target_col, "mean"),
        count=(target_col, "count"),
    ).reset_index().sort_values([group_col, "date"])

    # Weighted rolling mean: sum of hits / sum of PAs over window days
    # Using expanding with shift(1) and window limit
    result_rows = []
    for g, group in daily.groupby(group_col):
        group = group.sort_values("date")
        # Rolling window by calendar days — approximate with N rows
        group["_rolling_hits"] = (group["rate"] * group["count"]).rolling(
            window, min_periods=1
        ).sum()
        group["_rolling_count"] = group["count"].rolling(window, min_periods=1).sum()
        group[feature_name] = (group["_rolling_hits"] / group["_rolling_count"]).shift(1)
        result_rows.append(group[[group_col, "date", feature_name]])

    lookup = pd.concat(result_rows, ignore_index=True)
    if feature_name in df.columns:
        df = df.drop(columns=[feature_name])
    df = df.merge(lookup, on=[group_col, "date"], how="left")
    return df


class UmpireHitRateExperiment(ExperimentDef):
    """Rolling 30-day hit rate allowed per home-plate umpire.

    92 unique umpires. Research suggests umpire strike zone varies by 45%+
    in K rate between strictest and most lenient. This hasn't been tested
    as a direct model feature.
    """

    def __init__(self):
        super().__init__(
            name="umpire_hit_rate",
            phase=1,
            category="feature",
            description="Rolling 30-day hit rate per HP umpire",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "hp_umpire_id" not in df.columns:
            df["ump_hr_30g"] = np.nan
            return df
        return _rolling_rate_by_group(
            df, "hp_umpire_id", "is_hit", window=30, feature_name="ump_hr_30g"
        )

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["ump_hr_30g"]


class CatcherHitRateExperiment(ExperimentDef):
    """Rolling 30-day hit rate allowed per fielding catcher.

    Goes beyond the existing pitcher_catcher_framing proxy by capturing the
    catcher's direct impact on hit rate allowed (includes framing + blocking
    + pitch calling). 107 unique catchers.
    """

    def __init__(self):
        super().__init__(
            name="catcher_hit_rate",
            phase=1,
            category="feature",
            description="Rolling 30-day hit rate per fielding catcher",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "fielding_catcher_id" not in df.columns:
            df["catcher_hr_30g"] = np.nan
            return df
        return _rolling_rate_by_group(
            df, "fielding_catcher_id", "is_hit", window=30, feature_name="catcher_hr_30g"
        )

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["catcher_hr_30g"]


class WindVectorExperiment(ExperimentDef):
    """Wind direction × speed as a signed scalar (positive = blowing out to CF).

    Research: wind blowing out to CF at 10+ mph adds +22 BA points; wind in
    at 10+ mph subtracts 17 points. Current model has raw weather_temp but
    not wind, and weather_wind_dir is a text field that needs parsing.
    """

    def __init__(self):
        super().__init__(
            name="wind_vector",
            phase=1,
            category="feature",
            description="Wind direction * speed (positive = blowing out to CF)",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "weather_wind_dir" not in df.columns or "weather_wind_speed" not in df.columns:
            df["wind_out_cf"] = np.nan
            return df

        direction = df["weather_wind_dir"].astype(str).str.lower()
        speed = pd.to_numeric(df["weather_wind_speed"], errors="coerce").fillna(0)

        # Score direction from -1 (blowing in to CF) to +1 (blowing out to CF)
        # CF-out wind boosts hitters; CF-in wind suppresses.
        direction_score = np.where(
            direction.str.contains("out to cf|out to center"), 1.0,
            np.where(
                direction.str.contains("in from cf|in from center"), -1.0,
                np.where(
                    direction.str.contains("out to lf|out to l f|out to rf|out to r f"), 0.5,
                    np.where(
                        direction.str.contains("in from lf|in from rf"), -0.5,
                        0.0,  # L-to-R, R-to-L, calm, none, etc.
                    ),
                ),
            ),
        )
        df["wind_out_cf"] = direction_score * speed
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["wind_out_cf"]


class BatterTrajectoryMixExperiment(ExperimentDef):
    """Rolling batter line drive rate (trajectory == 'line_drive') over 30 days.

    Line drive rate is one of the strongest predictors of BABIP at the batter
    level. 68% of PAs have trajectory data (missing on walks, strikeouts, etc).
    Different from existing batter_gb_hit_rate which is a GB-conditional rate.
    """

    def __init__(self):
        super().__init__(
            name="batter_ld_rate",
            phase=1,
            category="feature",
            description="Rolling 30-day line drive rate per batter",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "trajectory" not in df.columns:
            df["batter_ld_rate_30g"] = np.nan
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        # Line drive indicator: 1 if trajectory is line_drive, 0 if any other contact,
        # NaN if no contact (walk, strikeout, HBP)
        traj = df["trajectory"].astype(str).str.lower()
        is_ld = traj.str.contains("line_drive").astype(float)
        is_ld = is_ld.where(df["trajectory"].notna(), np.nan)
        df["_is_ld"] = is_ld

        # Per-batter rolling rate with shift(1)
        df = df.sort_values(["batter_id", "date"])
        df["batter_ld_rate_30g"] = (
            df.groupby("batter_id")["_is_ld"]
            .rolling(window=30 * 4, min_periods=10)  # ~4 PA/game × 30 games
            .mean()
            .reset_index(level=0, drop=True)
            .shift(1)
        )
        df.drop(columns=["_is_ld"], inplace=True)
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["batter_ld_rate_30g"]


class BatterHardnessRateExperiment(ExperimentDef):
    """Rolling batter hard-hit rate from the categorical `hardness` column.

    Different from Statcast barrel_rate (which uses EV+LA): hardness is a
    scouting-derived 3-tier classification (hard/medium/soft) that's more
    stable and has better pre-2015 coverage. 68% populated.
    """

    def __init__(self):
        super().__init__(
            name="batter_hardness",
            phase=1,
            category="feature",
            description="Rolling 30-day hard-contact rate per batter",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "hardness" not in df.columns:
            df["batter_hard_contact_30g"] = np.nan
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        is_hard = (df["hardness"].astype(str).str.lower() == "hard").astype(float)
        is_hard = is_hard.where(df["hardness"].notna(), np.nan)
        df["_is_hard"] = is_hard

        df = df.sort_values(["batter_id", "date"])
        df["batter_hard_contact_30g"] = (
            df.groupby("batter_id")["_is_hard"]
            .rolling(window=30 * 4, min_periods=10)
            .mean()
            .reset_index(level=0, drop=True)
            .shift(1)
        )
        df.drop(columns=["_is_hard"], inplace=True)
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["batter_hard_contact_30g"]


class RoofTypeExperiment(ExperimentDef):
    """Binary indicator for indoor (dome or closed retractable) vs outdoor play.

    Indoor games have no wind/temperature effects and smaller variance in
    ball carry. Only a few parks qualify (Tampa, Toronto, Arizona, Houston,
    Milwaukee, Minnesota, Seattle).
    """

    def __init__(self):
        super().__init__(
            name="roof_indoor",
            phase=1,
            category="feature",
            description="Binary flag: indoor (dome or closed) vs outdoor",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "roof_type" not in df.columns:
            df["is_indoor"] = 0
            return df
        rt = df["roof_type"].astype(str).str.lower()
        df = df.copy()
        df["is_indoor"] = rt.isin(["dome", "closed", "retractable"]).astype(int)
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["is_indoor"]


class HeatDomeBinaryExperiment(ExperimentDef):
    """Binary heat_dome flag at temp >= 90°F."""

    def __init__(self):
        super().__init__(
            name="heat_dome",
            phase=1,
            category="feature",
            description="Binary weather_temp >= 90 heat indicator",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "weather_temp" not in df.columns:
            df["heat_dome"] = 0.0
            return df
        df["heat_dome"] = (df["weather_temp"].astype(float) >= 90).astype(float)
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["heat_dome"]


class HeatDome95Experiment(ExperimentDef):
    """Tighter heat_dome at temp >= 95°F (matches the observed raw-miss-rate spike)."""

    def __init__(self):
        super().__init__(
            name="heat_dome_95",
            phase=1,
            category="feature",
            description="Binary weather_temp >= 95 heat indicator",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "weather_temp" not in df.columns:
            df["heat_dome_95"] = 0.0
            return df
        df["heat_dome_95"] = (df["weather_temp"].astype(float) >= 95).astype(float)
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["heat_dome_95"]


class HeatIndexLinearExperiment(ExperimentDef):
    """Continuous heat activation above 85°F: max(0, temp - 85)."""

    def __init__(self):
        super().__init__(
            name="heat_index_linear",
            phase=1,
            category="feature",
            description="Continuous max(0, weather_temp - 85)",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "weather_temp" not in df.columns:
            df["heat_index_linear"] = 0.0
            return df
        df["heat_index_linear"] = np.maximum(0.0, df["weather_temp"].astype(float) - 85.0)
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["heat_index_linear"]


class HeatIndexSquaredExperiment(ExperimentDef):
    """Convex heat penalty above 85°F: max(0, temp - 85)^2."""

    def __init__(self):
        super().__init__(
            name="heat_index_squared",
            phase=1,
            category="feature",
            description="Convex max(0, weather_temp - 85)^2",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "weather_temp" not in df.columns:
            df["heat_index_squared"] = 0.0
            return df
        df["heat_index_squared"] = np.maximum(0.0, df["weather_temp"].astype(float) - 85.0) ** 2
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["heat_index_squared"]


class BatterPitcherMatchupExperiment(ExperimentDef):
    """Historical (batter, pitcher) hit rate with Bayesian shrinkage.

    For each PA, compute the empirical hit rate for that specific
    (batter_id, pitcher_id) pair across all PRIOR encounters, shrunk toward
    the league prior using pseudocount K. Sparse pairings fall back to the
    prior; frequent pairings express their actual history.

    Aggregates per (batter, pitcher, date) first so same-day multi-PA rows
    share the same prior-day stat (no within-day leakage).
    """

    PRIOR_RATE = 0.2195
    K = 10

    def __init__(self):
        super().__init__(
            name="batter_pitcher_matchup",
            phase=1,
            category="feature",
            description="Bayesian-shrunk historical hit rate per (batter, pitcher) pair",
        )

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Idempotent guard: as of 2026-04-29, batter_pitcher_shrunk_hr is in
        # FEATURE_COLS and computed by compute_all_features. If the column
        # already exists, this experiment is a no-op (re-running the merge
        # would create _x/_y suffixes and break downstream access).
        if "batter_pitcher_shrunk_hr" in df.columns:
            return df

        required = ("batter_id", "pitcher_id", "is_hit", "date")
        if not all(c in df.columns for c in required):
            df = df.copy()
            df["batter_pitcher_shrunk_hr"] = self.PRIOR_RATE
            return df

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        daily = (
            df.groupby(["batter_id", "pitcher_id", "date"])
            .agg(day_hits=("is_hit", "sum"), day_pas=("is_hit", "count"))
            .reset_index()
            .sort_values(["batter_id", "pitcher_id", "date"])
        )

        daily["cum_hits_prior"] = (
            daily.groupby(["batter_id", "pitcher_id"])["day_hits"]
            .transform(lambda s: s.cumsum().shift(1).fillna(0))
        )
        daily["cum_pas_prior"] = (
            daily.groupby(["batter_id", "pitcher_id"])["day_pas"]
            .transform(lambda s: s.cumsum().shift(1).fillna(0))
        )
        daily["batter_pitcher_shrunk_hr"] = (
            (self.PRIOR_RATE * self.K + daily["cum_hits_prior"])
            / (self.K + daily["cum_pas_prior"])
        )

        df = df.merge(
            daily[["batter_id", "pitcher_id", "date", "batter_pitcher_shrunk_hr"]],
            on=["batter_id", "pitcher_id", "date"],
            how="left",
        )
        df["batter_pitcher_shrunk_hr"] = df["batter_pitcher_shrunk_hr"].fillna(self.PRIOR_RATE)
        return df

    def feature_cols(self) -> list[str]:
        return FEATURE_COLS + ["batter_pitcher_shrunk_hr"]


# --- Statcast "fixed addition" experiments ---
#
# The 9 Statcast features are already computed by compute_all_features as part
# of STATCAST_COLS, but only used in the 12-model blend as per-variant additions
# (one Statcast feature per model). The historical "add all 9 as fixed features"
# test was rejected. Here we re-test each feature INDIVIDUALLY as a fixed
# addition to FEATURE_COLS, which isolates each one's effect under multi-seed
# evaluation in Phase 2b.
_STATCAST_FEATURES_TO_TEST = [
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


def _make_statcast_add_experiment(feature: str) -> type:
    """Factory: produce an ExperimentDef subclass that adds one Statcast feature to FEATURE_COLS."""
    short = feature.replace("_30g", "")
    exp_name = f"statcast_add_{short}"

    class _StatcastAddExperiment(ExperimentDef):
        def __init__(self):
            super().__init__(
                name=exp_name,
                phase=1,
                category="feature",
                description=f"Promote {feature} from blend-variant to fixed FEATURE_COLS baseline",
            )

        def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
            # Feature is already computed by compute_all_features; no-op here.
            return df

        def feature_cols(self) -> list[str]:
            return FEATURE_COLS + [feature]

    class_name = "StatcastAdd_" + short
    _StatcastAddExperiment.__name__ = class_name
    _StatcastAddExperiment.__qualname__ = class_name
    return _StatcastAddExperiment


_STATCAST_EXPERIMENT_CLASSES = [
    _make_statcast_add_experiment(f) for f in _STATCAST_FEATURES_TO_TEST
]


register(EBShrinkageExperiment())
register(KLDivergenceExperiment())
register(BattingHeatQExperiment())
register(GBPlatoonExperiment())
register(HitTypeParkFactorsExperiment())
register(StreakLengthFeatureExperiment())
# Dormant column experiments
register(UmpireHitRateExperiment())
register(CatcherHitRateExperiment())
register(WindVectorExperiment())
register(BatterTrajectoryMixExperiment())
register(BatterHardnessRateExperiment())
register(RoofTypeExperiment())
# Phase 2b (post-2026-04-14 audit) — multi-seed retest of historical rejects
register(HeatDomeBinaryExperiment())
register(HeatDome95Experiment())
register(HeatIndexLinearExperiment())
register(HeatIndexSquaredExperiment())
register(BatterPitcherMatchupExperiment())
for _cls in _STATCAST_EXPERIMENT_CLASSES:
    register(_cls())
