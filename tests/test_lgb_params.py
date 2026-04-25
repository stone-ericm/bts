"""Tests for LGB_PARAMS env-var gating.

BTS_LGBM_DETERMINISTIC=1 enables LightGBM's bit-exact-reproducibility flags
(`deterministic=True` + `force_row_wise=True`), needed to pool seeds across
providers. Default OFF preserves shipped P(57)=8.17% pooled. See
project_bts_oci_provider_add.md for the OCI drift evidence that motivated this.
"""
from __future__ import annotations

import importlib


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
