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


class LambdaRankTop1Experiment(ExperimentDef):
    """LambdaRank tuned specifically for top-1 ranking (Tier 5 lit-import).

    The existing LambdaRankExperiment uses ``lambdarank_truncation_level=1`` to
    truncate pairwise comparisons at position 1, but otherwise leaves
    LightGBM's defaults in place (``eval_at`` defaults to multiple positions,
    ``label_gain`` to a multi-grade gain table).

    For the BTS task — picking ONE batter per day — every NDCG signal except
    @1 is wasted training pressure. This variant pins ``eval_at=[1]`` and
    ``ndcg_eval_at=[1]`` so internal NDCG calculations only consider the top
    of the list, configures ``label_gain=[0, 1]`` for binary relevance
    (no multi-grade), and sets explicit boost hyperparams to make the
    comparison reproducible against the existing lambdarank run.

    Eligible for the model-swap fast path: appends a 13th blend config
    without touching features or LGB_PARAMS.
    """

    def __init__(self):
        super().__init__(
            name="lambdarank_top1",
            phase=1,
            category="model",
            description="LambdaRank tuned for top-1 ranking (NDCG@1 + binary label_gain)",
        )

    def modify_blend_configs(self, configs):
        return configs + [
            (
                "lambdarank_top1",
                FEATURE_COLS,
                {
                    "objective": "lambdarank",
                    "eval_at": [1],
                    "ndcg_eval_at": [1],
                    "label_gain": [0, 1],
                    "lambdarank_truncation_level": 1,
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "num_leaves": 31,
                },
            ),
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
register(LambdaRankTop1Experiment())
register(CatBoostExperiment())
register(XENDCGExperiment())
register(VRExExperiment())
