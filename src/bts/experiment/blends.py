"""Phase 1 blend experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bts.experiment.base import ExperimentDef
from bts.experiment.registry import register


class FWLSExperiment(ExperimentDef):
    """Feature-Weighted Linear Stacking: context-dependent blend weights."""

    def __init__(self):
        super().__init__(
            name="fwls",
            phase=1,
            category="blend",
            description="Ridge-penalized linear meta-learner with context features",
        )

    def modify_strategy(self, profiles_df, quality_bins):
        # FWLS requires capturing per-model predictions during walk-forward.
        # Full integration handled in runner when this experiment is active.
        return profiles_df, quality_bins


class FixedShareHedgeExperiment(ExperimentDef):
    """Fixed-Share Hedge: online-adaptive blend weights."""

    def __init__(self):
        super().__init__(
            name="fixed_share_hedge",
            phase=1,
            category="blend",
            description="Hedge algorithm with Fixed-Share mixing (alpha=0.05)",
        )

    def modify_strategy(self, profiles_df, quality_bins):
        return profiles_df, quality_bins


class CopulaDoublesExperiment(ExperimentDef):
    """Gaussian copula for double-down joint probability."""

    def __init__(self):
        super().__init__(
            name="copula_doubles",
            phase=1,
            category="blend",
            description="Gaussian copula P(both hit) instead of P(A)*P(B)",
        )

    def modify_strategy(self, profiles_df, quality_bins):
        return profiles_df, quality_bins


register(FWLSExperiment())
register(FixedShareHedgeExperiment())
register(CopulaDoublesExperiment())
