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


class VennABERSExperiment(ExperimentDef):
    """Use Venn-ABERS interval width as uncertainty signal for skip decisions.

    Fits two isotonic regressions on the rank-1 calibration set (imputing
    hit=0 and hit=1). The width [p0, p1] measures epistemic uncertainty.
    Wide intervals → less confident → skip.
    """

    def __init__(self):
        super().__init__(
            name="venn_abers_width",
            phase=1,
            category="strategy",
            description="Venn-ABERS isotonic calibration interval width as skip signal",
        )

    def modify_strategy(self, profiles_df, quality_bins):
        from sklearn.isotonic import IsotonicRegression

        df = profiles_df.copy().reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        rank1 = df[df["rank"] == 1].copy()
        if len(rank1) < 50:
            return df, quality_bins

        # Use earliest 30% of dates for calibration training (no leakage)
        rank1 = rank1.sort_values("date")
        split = len(rank1) // 3
        cal = rank1.iloc[:split]

        if len(cal) < 20:
            return df, quality_bins

        ir0 = IsotonicRegression(out_of_bounds="clip")
        ir1 = IsotonicRegression(out_of_bounds="clip")

        cal_y = cal["actual_hit"].values
        cal_p = cal["p_game_hit"].values

        ir0.fit(cal_p, cal_y)
        cal_y_optimistic = np.minimum(cal_y + 0.1, 1.0)
        ir1.fit(cal_p, cal_y_optimistic)

        # Compute width per profile vectorized
        all_p = df["p_game_hit"].values.astype(float)
        p0 = ir0.predict(all_p)
        p1 = ir1.predict(all_p)
        width = np.abs(p1 - p0)

        # Narrow intervals get full credit; wide intervals get downscored
        max_width = max(float(width.max()), 1e-6)
        confidence = 1.0 - (width / max_width) * 0.3  # max 30% downweight
        new_p = all_p * confidence
        new_p = np.where(np.isnan(new_p), all_p, new_p)
        df["p_game_hit"] = new_p

        # Re-rank within each day
        df = df.sort_values(["date", "p_game_hit"], ascending=[True, False]).reset_index(drop=True)
        df["rank"] = df.groupby("date").cumcount() + 1

        try:
            from bts.simulate.quality_bins import compute_bins
            new_bins = compute_bins(df)
        except Exception:
            new_bins = quality_bins

        return df, new_bins


class QuantileQ10Experiment(ExperimentDef):
    """Train quantile regression model at alpha=0.10 for conservative skip signal."""

    def __init__(self):
        super().__init__(
            name="quantile_q10",
            phase=1,
            category="strategy",
            description="LightGBM quantile q10 as additional skip signal",
        )

    def modify_blend_configs(self, configs):
        # Add a quantile model alongside the blend.
        # When used in modify_strategy, this model's outputs will be available
        # via per-model capture (column m_quantile_q10).
        return configs + [
            ("quantile_q10", FEATURE_COLS, {"objective": "quantile", "alpha": 0.10}),
        ]

    def requires_per_model_capture(self) -> bool:
        return True

    def modify_strategy(self, profiles_df, quality_bins):
        # Use q10 as a skip signal: when q10 is below threshold, downweight pick
        df = profiles_df.copy().reset_index(drop=True)
        if "m_quantile_q10" not in df.columns:
            return df, quality_bins

        # Vectorized: where q10 missing, leave p_game_hit unchanged
        q10 = df["m_quantile_q10"].values.astype(float)
        original_p = df["p_game_hit"].values.astype(float)

        confidence = np.where(
            np.isnan(q10),
            1.0,
            0.5 + 0.5 * np.clip(q10 / 0.7, 0.0, 1.0),
        )
        new_p = original_p * confidence
        # Guard against any NaN propagation
        new_p = np.where(np.isnan(new_p), original_p, new_p)
        df["p_game_hit"] = new_p

        # Re-rank within each day
        df = df.sort_values(["date", "p_game_hit"], ascending=[True, False]).reset_index(drop=True)
        df["rank"] = df.groupby("date").cumcount() + 1

        try:
            from bts.simulate.quality_bins import compute_bins
            new_bins = compute_bins(df)
        except Exception:
            new_bins = quality_bins

        return df, new_bins


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


register(EBShrinkageExperiment())
register(KLDivergenceExperiment())
register(BattingHeatQExperiment())
register(GBPlatoonExperiment())
register(HitTypeParkFactorsExperiment())
register(VennABERSExperiment())
register(QuantileQ10Experiment())
register(StreakLengthFeatureExperiment())
# Dormant column experiments
register(UmpireHitRateExperiment())
register(CatcherHitRateExperiment())
register(WindVectorExperiment())
register(BatterTrajectoryMixExperiment())
register(BatterHardnessRateExperiment())
register(RoofTypeExperiment())
