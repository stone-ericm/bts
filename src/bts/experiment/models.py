"""Phase 1 model experiments."""

from __future__ import annotations

from bts.experiment.base import ExperimentDef
from bts.experiment.registry import register
from bts.features.compute import FEATURE_COLS


class LambdaRankExperiment(ExperimentDef):
    """Add a LambdaRank model as 13th blend member."""

    def __init__(self):
        super().__init__(
            name="lambdarank",
            phase=1,
            category="model",
            description="LambdaRank blend member optimizing NDCG@1",
        )

    def modify_blend_configs(self, configs):
        return configs + [
            ("lambdarank", FEATURE_COLS, {"objective": "lambdarank", "lambdarank_truncation_level": 1})
        ]


class CatBoostExperiment(ExperimentDef):
    """Add a CatBoost model with has_time=True as blend member."""

    def __init__(self):
        super().__init__(
            name="catboost",
            phase=1,
            category="model",
            description="CatBoost with ordered boosting (temporal gradient safety)",
        )

    def modify_blend_configs(self, configs):
        return configs + [("catboost", FEATURE_COLS, {"engine": "catboost", "has_time": True})]


class XENDCGExperiment(ExperimentDef):
    """Add XE-NDCG model as blend member."""

    def __init__(self):
        super().__init__(
            name="xendcg",
            phase=1,
            category="model",
            description="XE-NDCG ranking objective (convex NDCG bound)",
        )

    def modify_blend_configs(self, configs):
        return configs + [("xendcg", FEATURE_COLS, {"objective": "rank_xendcg"})]


class VRExExperiment(ExperimentDef):
    """V-REx: penalize cross-season loss variance via iterative reweighting.

    Registers a 13th blend member whose training path checks for ``vrex_beta``
    in the blend config's ``extra_params`` (see ``backtest_blend.py``). The
    training hook is keyed off extras, not global lgb_params, so this must go
    through ``modify_blend_configs`` rather than ``modify_training_params``.
    """

    def __init__(self):
        super().__init__(
            name="vrex",
            phase=1,
            category="model",
            description="V-REx season reweighting to reduce year-to-year instability",
        )

    def modify_blend_configs(self, configs):
        return configs + [
            ("vrex", FEATURE_COLS, {"vrex_beta": 10.0, "vrex_rounds": 5})
        ]


register(LambdaRankExperiment())
register(CatBoostExperiment())
register(XENDCGExperiment())
register(VRExExperiment())
