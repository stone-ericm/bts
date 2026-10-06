"""C2 step 2a §5.1: fixed, picklable stand-ins for the trained models (the design stubs the training leaves on both
sides). Each predicts P(hit | PA) from the rate-like features it is given, with fixed per-(seed, column) weights."""
from __future__ import annotations

import hashlib

import numpy as np


class FakeModel:
    def __init__(self, cols, seed: int = 0):
        self.cols = list(cols)
        self.seed = seed
        self.weights = [self._weight(c) for c in self.cols]
        # Serialized size like a small real model: pickling spans several writes (the partial-write fault needs it).
        self.padding = np.arange(20_000, dtype=np.float64)

    def _weight(self, col: str) -> float:
        if not any(k in col for k in ("hr_", "rate", "framing", "shrunk", "platoon")):
            return 0.0
        h = hashlib.sha256(f"{self.seed}:{col}".encode()).digest()
        return 0.5 + h[0] / 255.0

    def predict_proba(self, X):
        x = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        w = np.asarray(self.weights, dtype=float)
        denom = w.sum() or 1.0
        p = np.clip(0.04 + 0.9 * (x @ w) / denom, 0.02, 0.6)
        return np.column_stack([1.0 - p, p])


def train_model(df, feature_cols=None):
    from bts.features.compute import FEATURE_COLS
    return FakeModel(feature_cols or FEATURE_COLS, seed=0)


def train_blend(df, base_feature_cols=None, blend_configs=None, lgb_params=None):
    from bts.model.predict import _build_blend_configs
    configs = blend_configs or _build_blend_configs(base_feature_cols)
    return {cfg[0]: (FakeModel(cfg[1], seed=i + 1), cfg[1]) for i, cfg in enumerate(configs)}
