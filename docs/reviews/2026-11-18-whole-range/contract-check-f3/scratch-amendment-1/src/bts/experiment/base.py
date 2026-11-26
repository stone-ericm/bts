"""Base class for all BTS experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ExperimentDef:
    """Declarative experiment definition.

    Subclass and override hooks to define an experiment. The runner calls
    hooks in order: modify_features → modify_blend_configs →
    modify_training_params → (run walk-forward) → modify_strategy.

    Phase 0 experiments override run_diagnostic instead.
    """

    name: str
    phase: int  # 0=diagnostic, 1=screening, 2=forward-select
    category: str  # "feature", "model", "blend", "strategy", "diagnostic"
    description: str
    dependencies: list[str] = field(default_factory=list)

    def modify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override to add/replace/remove features. Return modified DataFrame."""
        return df

    def modify_blend_configs(
        self, configs: list[tuple[str, list[str]]]
    ) -> list[tuple[str, list[str]]]:
        """Override to add/replace blend model configs."""
        return configs

    def modify_training_params(self, params: dict) -> dict:
        """Override to change LightGBM training params."""
        return params

    def modify_strategy(
        self, profiles_df: pd.DataFrame, quality_bins: object
    ) -> tuple[pd.DataFrame, object]:
        """Override to change strategy/calibration/MDP inputs."""
        return profiles_df, quality_bins

    def run_diagnostic(
        self, df: pd.DataFrame, profiles: dict[int, pd.DataFrame]
    ) -> dict:
        """Override for Phase 0 diagnostics. Return report dict."""
        raise NotImplementedError(
            f"{self.name}: Phase 0 experiments must implement run_diagnostic()"
        )

    def feature_cols(self) -> list[str] | None:
        """Override to change FEATURE_COLS for this experiment. None = default."""
        return None

    def touches_features(self) -> bool:
        """Whether this experiment modifies features (requires recompute)."""
        return (
            type(self).modify_features is not ExperimentDef.modify_features
            or type(self).feature_cols is not ExperimentDef.feature_cols
        )

    def requires_per_model_capture(self) -> bool:
        """Whether this experiment needs per-model predictions captured.

        Override and return True for FWLS, Hedge, and other meta-learning
        experiments that operate on individual blend member outputs.
        """
        return False
