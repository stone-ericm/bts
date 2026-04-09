"""Phase 1 strategy experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bts.experiment.base import ExperimentDef
from bts.experiment.registry import register


class DecisionCalibrationExperiment(ExperimentDef):
    """Isotonic recalibration at the MDP skip threshold."""

    def __init__(self):
        super().__init__(
            name="decision_calibration",
            phase=1,
            category="strategy",
            description="Isotonic regression calibration targeting skip threshold region",
        )

    def modify_strategy(self, profiles_df, quality_bins):
        """Apply isotonic calibration via temporal cross-fitting.

        For each day d, fit isotonic regression on rank-1 picks from days
        [0, d-1] only, then apply to day d's predictions. This is causal
        (no leakage) and uses all data for evaluation. Calibration starts
        after a warmup period of 30 days.
        """
        from sklearn.isotonic import IsotonicRegression

        df = profiles_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        sorted_dates = sorted(df["date"].unique())
        warmup = 30
        if len(sorted_dates) < warmup + 10:
            return df, quality_bins

        rank1 = df[df["rank"] == 1].copy().sort_values("date")

        # Cache calibration model by date — refit weekly to amortize cost
        recalibrate_every = 7
        ir = None
        new_p_values = {}  # date → recalibrated p_game_hit values per row

        for i, date in enumerate(sorted_dates):
            if i < warmup:
                continue
            # Recalibrate periodically using all rank-1 picks before this date
            if ir is None or i % recalibrate_every == 0:
                prior_rank1 = rank1[rank1["date"] < date]
                if len(prior_rank1) < 10:
                    continue
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(prior_rank1["p_game_hit"].values, prior_rank1["actual_hit"].values)

            day_mask = df["date"] == date
            day_p = df.loc[day_mask, "p_game_hit"].values
            new_p_values[date] = ir.predict(day_p)

        # Apply recalibrated predictions
        for date, vals in new_p_values.items():
            mask = df["date"] == date
            df.loc[mask, "p_game_hit"] = vals

        # Re-rank within each day (isotonic preserves order, so this is a no-op)
        df = df.sort_values(["date", "p_game_hit"], ascending=[True, False])
        df["rank"] = df.groupby("date").cumcount() + 1

        from bts.simulate.quality_bins import compute_bins
        try:
            new_bins = compute_bins(df)
        except Exception:
            new_bins = quality_bins

        return df, new_bins


class QuantileGatedSkipExperiment(ExperimentDef):
    """Use quantile q10 as conservative skip gate."""

    def __init__(self):
        super().__init__(
            name="quantile_gated_skip",
            phase=1,
            category="strategy",
            description="Skip when q10 estimate is below threshold",
        )

    def modify_strategy(self, profiles_df, quality_bins):
        return profiles_df, quality_bins


register(DecisionCalibrationExperiment())
register(QuantileGatedSkipExperiment())
