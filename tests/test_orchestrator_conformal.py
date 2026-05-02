"""Tests for predict_local conformal integration."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fake_calibrators(tmp_path):
    """Create dated calibrator artifacts under data/conformal/ in tmp_path."""
    from bts.model.conformal import (
        BucketWilsonCalibrator,
        WeightedMondrianConformalCalibrator,
    )

    cal_dir = tmp_path / "data" / "conformal"
    cal_dir.mkdir(parents=True, exist_ok=True)

    wilson = BucketWilsonCalibrator(
        alphas=[0.05, 0.10, 0.20],
        bucket_lower={0.7: [0.62, 0.66, 0.70]},
        bucket_n={0.7: 100},
        bucket_hit_rate={0.7: 0.78},
    )
    conformal = WeightedMondrianConformalCalibrator(
        alphas=[0.05, 0.10, 0.20],
        bucket_quantiles={0.7: [0.18, 0.12, 0.06]},
        marginal_quantiles=[0.20, 0.15, 0.08],
        n_effective_per_bucket={0.7: 100.0},
    )
    joblib.dump(wilson, cal_dir / "wilson_calibrator_2026-05-01.pkl")
    joblib.dump(conformal, cal_dir / "calibrator_2026-05-01.pkl")
    return tmp_path


def test_predict_local_attaches_six_conformal_fields(fake_calibrators, monkeypatch):
    """When BTS_USE_CONFORMAL=1 and calibrators exist, predict_local
    appends 6 conformal columns to its output DataFrame."""
    from bts.orchestrator import _attach_conformal_lower_bounds

    monkeypatch.setenv("BTS_USE_CONFORMAL", "1")
    predictions = pd.DataFrame({
        "batter_id": [1, 2],
        "batter_name": ["X", "Y"],
        "p_game_hit": [0.71, 0.72],
    })
    out = _attach_conformal_lower_bounds(
        predictions, conformal_dir=fake_calibrators / "data" / "conformal",
    )
    expected_cols = [
        "p_game_hit_lower_conformal_95", "p_game_hit_lower_conformal_90",
        "p_game_hit_lower_conformal_80", "p_game_hit_lower_wilson_95",
        "p_game_hit_lower_wilson_90", "p_game_hit_lower_wilson_80",
    ]
    for col in expected_cols:
        assert col in out.columns, f"missing column {col}"
    # Sanity: bucket 0.70 → wilson_90 = 0.66
    assert out.loc[0, "p_game_hit_lower_wilson_90"] == 0.66


def test_predict_local_skips_conformal_when_env_unset(fake_calibrators, monkeypatch):
    from bts.orchestrator import _attach_conformal_lower_bounds

    monkeypatch.delenv("BTS_USE_CONFORMAL", raising=False)
    predictions = pd.DataFrame({"batter_id": [1], "p_game_hit": [0.71]})
    out = _attach_conformal_lower_bounds(
        predictions, conformal_dir=fake_calibrators / "data" / "conformal",
    )
    # Columns NOT added (env var not set)
    assert "p_game_hit_lower_conformal_95" not in out.columns


def test_predict_local_handles_missing_calibrator_gracefully(tmp_path, monkeypatch):
    from bts.orchestrator import _attach_conformal_lower_bounds

    monkeypatch.setenv("BTS_USE_CONFORMAL", "1")
    predictions = pd.DataFrame({"batter_id": [1], "p_game_hit": [0.71]})
    # No calibrator files exist; should add columns as None
    out = _attach_conformal_lower_bounds(
        predictions, conformal_dir=tmp_path / "no_such_dir",
    )
    assert "p_game_hit_lower_conformal_95" in out.columns
    assert pd.isna(out.loc[0, "p_game_hit_lower_conformal_95"])
