"""Phase 0 diagnostic experiments."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from bts.experiment.base import ExperimentDef
from bts.experiment.registry import register
from bts.features.compute import FEATURE_COLS


class StabilitySelectionDiagnostic(ExperimentDef):
    """Run LightGBM on bootstrap samples per season, compute feature stability."""

    def __init__(self):
        super().__init__(
            name="stability_selection",
            phase=0,
            category="diagnostic",
            description="Feature stability across bootstrap samples and seasons",
        )

    def run_diagnostic(self, df: pd.DataFrame, profiles: dict) -> dict:
        import lightgbm as lgb
        from bts.model.predict import LGB_PARAMS

        n_bootstrap = 100
        seasons = sorted(df["season"].unique())
        feature_cols = [c for c in FEATURE_COLS if c in df.columns]

        season_stability: dict[int, dict[str, float]] = {}
        for season in seasons:
            if len(df[df["season"] == season]) < 200:
                continue
            train = df[df["season"] < season]
            if len(train) < 500:
                continue

            selection_counts: dict[str, int] = {f: 0 for f in feature_cols}
            for b in range(n_bootstrap):
                sample = train.sample(frac=0.6, replace=True, random_state=b)
                X = sample[feature_cols]
                y = sample["is_hit"]
                mask = X.notna().any(axis=1)
                if mask.sum() < 50:
                    continue
                model = lgb.LGBMClassifier(
                    **{**LGB_PARAMS, "n_estimators": 50}, random_state=b,
                )
                model.fit(X[mask], y[mask])
                importances = dict(zip(feature_cols, model.feature_importances_))
                for feat, imp in importances.items():
                    if imp > 0:
                        selection_counts[feat] += 1

            season_stability[season] = {
                f: count / n_bootstrap for f, count in selection_counts.items()
            }
            print(f"  Season {season}: {n_bootstrap} bootstraps done", file=sys.stderr)

        all_features_stability: dict[str, float] = {}
        for feat in feature_cols:
            scores = [ss.get(feat, 0) for ss in season_stability.values()]
            all_features_stability[feat] = float(min(scores)) if scores else 0.0

        return {
            "feature_stability": all_features_stability,
            "per_season": {int(k): v for k, v in season_stability.items()},
            "n_bootstrap": n_bootstrap,
            "n_seasons": len(season_stability),
        }


class WassersteinDriftDiagnostic(ExperimentDef):
    """Compute per-feature Wasserstein distances between season pairs."""

    def __init__(self):
        super().__init__(
            name="wasserstein_drift",
            phase=0,
            category="diagnostic",
            description="Per-feature distributional drift across seasons",
        )

    def run_diagnostic(self, df: pd.DataFrame, profiles: dict) -> dict:
        from scipy.stats import wasserstein_distance

        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        seasons = sorted(df["season"].unique())

        drift: dict[str, dict[str, float]] = {}
        for feat in feature_cols:
            pair_dists: dict[str, float] = {}
            for i, s1 in enumerate(seasons):
                for s2 in seasons[i + 1:]:
                    v1 = df.loc[df["season"] == s1, feat].dropna().values
                    v2 = df.loc[df["season"] == s2, feat].dropna().values
                    if len(v1) > 10 and len(v2) > 10:
                        pair_dists[f"{s1}-{s2}"] = float(wasserstein_distance(v1, v2))
            drift[feat] = pair_dists

        mean_drift = {}
        for feat, pairs in drift.items():
            vals = list(pairs.values())
            mean_drift[feat] = float(np.mean(vals)) if vals else 0.0

        return {
            "feature_drift": mean_drift,
            "pairwise_drift": drift,
            "n_seasons": len(seasons),
        }


class StreakLengthDependenceDiagnostic(ExperimentDef):
    """Check if P@1 degrades as streak length increases."""

    def __init__(self):
        super().__init__(
            name="streak_length_dependence",
            phase=0,
            category="diagnostic",
            description="P@1 stratified by simulated streak length",
        )

    def run_diagnostic(self, df: pd.DataFrame, profiles: dict) -> dict:
        all_profiles = pd.concat(profiles.values(), ignore_index=True) if profiles else pd.DataFrame()
        if all_profiles.empty or "rank" not in all_profiles.columns:
            return {"p1_by_streak_bucket": {}, "note": "No profiles available"}

        rank1 = all_profiles[all_profiles["rank"] == 1].sort_values("date").copy()

        streak = 0
        streak_at_pick: list[int] = []
        for _, row in rank1.iterrows():
            streak_at_pick.append(streak)
            streak = streak + 1 if row["actual_hit"] > 0 else 0

        rank1 = rank1.iloc[:len(streak_at_pick)].copy()
        rank1["streak_at_pick"] = streak_at_pick

        bins = [0, 5, 10, 20, 30, 50, 200]
        labels = ["0-4", "5-9", "10-19", "20-29", "30-49", "50+"]
        rank1["streak_bucket"] = pd.cut(
            rank1["streak_at_pick"], bins=bins, labels=labels, right=False,
        )

        p1_by_bucket = rank1.groupby("streak_bucket", observed=True)["actual_hit"].agg(["mean", "count"])
        result = {}
        for bucket, row in p1_by_bucket.iterrows():
            result[str(bucket)] = {"p_at_1": float(row["mean"]), "n_days": int(row["count"])}

        return {"p1_by_streak_bucket": result}


class AFTShapeDiagnostic(ExperimentDef):
    """Fit Weibull AFT model to streak termination data."""

    def __init__(self):
        super().__init__(
            name="aft_shape",
            phase=0,
            category="diagnostic",
            description="Weibull shape parameter for streak hazard",
        )

    def run_diagnostic(self, df: pd.DataFrame, profiles: dict) -> dict:
        all_profiles = pd.concat(profiles.values(), ignore_index=True) if profiles else pd.DataFrame()
        if all_profiles.empty:
            return {"shape": None, "note": "No profiles available"}

        rank1 = all_profiles[all_profiles["rank"] == 1].sort_values("date")

        streaks: list[int] = []
        current = 0
        for hit in rank1["actual_hit"]:
            if hit:
                current += 1
            else:
                streaks.append(max(current, 1))
                current = 0
        if current > 0:
            streaks.append(current)

        if len(streaks) < 10:
            return {"shape": None, "note": "Too few streaks"}

        from scipy.stats import weibull_min
        shape, _, scale = weibull_min.fit(streaks, floc=0)

        interpretation = (
            "increasing hazard (streaks get harder)" if shape > 1 else
            "decreasing hazard (hot-hand stabilization)" if shape < 1 else
            "constant hazard (geometric/independence)"
        )

        return {
            "shape": float(shape),
            "scale": float(scale),
            "n_streaks": len(streaks),
            "mean_streak": float(np.mean(streaks)),
            "interpretation": interpretation,
        }


class ADWINChangepointDiagnostic(ExperimentDef):
    """Detect within-season calibration drift via ADWIN."""

    def __init__(self):
        super().__init__(
            name="adwin_changepoint",
            phase=0,
            category="diagnostic",
            description="Within-season Brier score changepoints",
        )

    def run_diagnostic(self, df: pd.DataFrame, profiles: dict) -> dict:
        try:
            from river.drift import ADWIN
        except ImportError:
            return {"error": "river library not installed. pip install river"}

        all_profiles = pd.concat(profiles.values(), ignore_index=True) if profiles else pd.DataFrame()
        if all_profiles.empty:
            return {"changepoints": [], "note": "No profiles available"}

        rank1 = all_profiles[all_profiles["rank"] == 1].sort_values("date")
        brier_scores = (rank1["actual_hit"] - rank1["p_game_hit"]) ** 2

        detector = ADWIN(delta=0.002)
        changepoints = []
        for i, (date, bs) in enumerate(zip(rank1["date"], brier_scores)):
            detector.update(float(bs))
            if detector.drift_detected:
                changepoints.append({"index": i, "date": str(date)})

        return {
            "changepoints": changepoints,
            "n_changepoints": len(changepoints),
            "n_days": len(rank1),
        }


register(StabilitySelectionDiagnostic())
register(WassersteinDriftDiagnostic())
register(StreakLengthDependenceDiagnostic())
register(AFTShapeDiagnostic())
register(ADWINChangepointDiagnostic())
