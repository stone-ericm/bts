"""Tests for LGB_PARAMS env-var gating.

BTS_LGBM_DETERMINISTIC=1 enables LightGBM's bit-exact-reproducibility flags
(`deterministic=True` + `force_row_wise=True`), needed to pool seeds across
providers. Default OFF preserves shipped P(57)=8.17% pooled. See
project_bts_oci_provider_add.md for the OCI drift evidence that motivated this.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


def test_lgb_params_deterministic_off_by_default(monkeypatch):
    monkeypatch.delenv("BTS_LGBM_DETERMINISTIC", raising=False)
    from bts.model import predict
    importlib.reload(predict)
    assert "deterministic" not in predict.LGB_PARAMS
    assert "force_row_wise" not in predict.LGB_PARAMS


def test_lgb_params_deterministic_on_via_env(monkeypatch):
    monkeypatch.setenv("BTS_LGBM_DETERMINISTIC", "1")
    from bts.model import predict
    importlib.reload(predict)
    assert predict.LGB_PARAMS["deterministic"] is True
    assert predict.LGB_PARAMS["force_row_wise"] is True


def test_lgb_params_deterministic_zero_treated_as_off(monkeypatch):
    monkeypatch.setenv("BTS_LGBM_DETERMINISTIC", "0")
    from bts.model import predict
    importlib.reload(predict)
    assert "deterministic" not in predict.LGB_PARAMS
    assert "force_row_wise" not in predict.LGB_PARAMS


def test_lgb_params_other_keys_preserved(monkeypatch):
    """Confirm the env-var path doesn't disturb existing param values."""
    monkeypatch.setenv("BTS_LGBM_DETERMINISTIC", "1")
    from bts.model import predict
    importlib.reload(predict)
    # Original values still present
    assert predict.LGB_PARAMS["n_estimators"] == 200
    assert predict.LGB_PARAMS["max_depth"] == 6
    assert predict.LGB_PARAMS["num_leaves"] == 31


def test_lgb_params_reload_resets_after_env_unset(monkeypatch):
    """Reload after unsetting brings flags back to absent."""
    monkeypatch.setenv("BTS_LGBM_DETERMINISTIC", "1")
    from bts.model import predict
    importlib.reload(predict)
    assert predict.LGB_PARAMS["deterministic"] is True

    monkeypatch.delenv("BTS_LGBM_DETERMINISTIC", raising=False)
    importlib.reload(predict)
    assert "deterministic" not in predict.LGB_PARAMS


def test_lgb_random_state_defaults_to_42(monkeypatch):
    monkeypatch.delenv("BTS_LGBM_RANDOM_STATE", raising=False)
    from bts.model import predict
    importlib.reload(predict)
    assert predict._lgbm_random_state() == 42


def test_lgb_random_state_overrides_from_env(monkeypatch):
    monkeypatch.setenv("BTS_LGBM_RANDOM_STATE", "314")
    from bts.model import predict
    importlib.reload(predict)
    assert predict._lgbm_random_state() == 314


class _FakeLGBMClassifier:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def fit(self, *_args, **_kwargs):
        self.feature_importances_ = np.array([0.0])
        return self

    def predict_proba(self, x):
        p = np.full(len(x), 0.5)
        return np.column_stack([1.0 - p, p])


def _tiny_training_frame():
    return pd.DataFrame({
        "season": [2019, 2020, 2021],
        "is_hit": [0, 1, 1],
        "feature": [0.1, 0.2, 0.3],
    })


def test_train_model_uses_lgb_random_state_env(monkeypatch):
    monkeypatch.setenv("BTS_LGBM_RANDOM_STATE", "2718")
    from bts.model import predict
    importlib.reload(predict)
    _FakeLGBMClassifier.instances = []
    monkeypatch.setattr(predict.lgb, "LGBMClassifier", _FakeLGBMClassifier)

    model = predict.train_model(_tiny_training_frame(), feature_cols=["feature"])

    assert model.kwargs["random_state"] == 2718


def test_train_blend_uses_lgb_random_state_env_for_each_model(monkeypatch):
    monkeypatch.setenv("BTS_LGBM_RANDOM_STATE", "1618")
    from bts.model import predict
    importlib.reload(predict)
    _FakeLGBMClassifier.instances = []
    monkeypatch.setattr(predict.lgb, "LGBMClassifier", _FakeLGBMClassifier)

    blend = predict.train_blend(
        _tiny_training_frame(),
        blend_configs=[
            ("one", ["feature"]),
            ("two", ["feature"]),
        ],
    )

    assert sorted(blend) == ["one", "two"]
    assert [m.kwargs["random_state"] for m, _cols in blend.values()] == [1618, 1618]


def test_evaluate_backtest_uses_lgb_random_state_env(monkeypatch):
    monkeypatch.setenv("BTS_LGBM_RANDOM_STATE", "808")
    from bts.model import predict
    importlib.reload(predict)
    from bts.evaluate import backtest
    importlib.reload(backtest)
    _FakeLGBMClassifier.instances = []
    monkeypatch.setattr(backtest.lgb, "LGBMClassifier", _FakeLGBMClassifier)
    monkeypatch.setattr(backtest, "FEATURE_COLS", ["feature"])

    df = pd.DataFrame({
        "date": ["2019-04-01", "2019-04-02", "2020-04-01", "2020-04-02"],
        "season": [2019, 2019, 2020, 2020],
        "is_hit": [0, 1, 0, 1],
        "feature": [0.1, 0.2, 0.3, 0.4],
        "batter_id": [1, 2, 1, 2],
        "game_pk": [10, 11, 20, 21],
    })

    backtest.walk_forward_evaluate(df, test_season=2020, retrain_every=1)

    assert _FakeLGBMClassifier.instances
    assert all(model.kwargs["random_state"] == 808 for model in _FakeLGBMClassifier.instances)
