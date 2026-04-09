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
        from sklearn.isotonic import IsotonicRegression

        df = profiles_df.copy()
        rank1 = df[df["rank"] == 1].copy()
        if len(rank1) < 20:
            return df, quality_bins

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(rank1["p_game_hit"].values, rank1["actual_hit"].values)

        df["p_game_hit"] = ir.predict(df["p_game_hit"].values)

        from bts.simulate.quality_bins import compute_bins
        new_bins = compute_bins(df)

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
