"""Tests for the validation gate script."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make scripts/ importable
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def _synth_calibration(n=200):
    """Synthesize backtest-style calibration data with date stratification."""
    rng = np.random.default_rng(42)
    base = date(2025, 5, 1)
    dates = [base + timedelta(days=int(rng.integers(0, 365))) for _ in range(n)]
    p = rng.uniform(0.6, 0.85, n)
    actual = (rng.uniform(0, 1, n) < p * 0.95).astype(int)
    return pd.DataFrame({
        "date": [d.isoformat() for d in dates],
        "rank": rng.integers(1, 11, n),
        "batter_id": rng.integers(1000, 9999, n),
        "p_game_hit": p,
        "actual_hit": actual,
        "n_pas": rng.integers(3, 6, n),
    })


def test_kfold_with_embargo_excludes_near_holdout_dates():
    from validate_conformal import kfold_indices_with_embargo

    df = _synth_calibration(n=200)
    folds = list(kfold_indices_with_embargo(df, k=5, embargo_days=7))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        # Train and test sets should not overlap
        assert set(train_idx).isdisjoint(set(test_idx))
        # Embargo: no train row within 7 days of any test row
        train_dates = pd.to_datetime(df.iloc[list(train_idx)]["date"]).dt.date
        test_dates = pd.to_datetime(df.iloc[list(test_idx)]["date"]).dt.date
        for td in test_dates.unique():
            min_gap = (train_dates - td).map(lambda x: abs(x.days)).min()
            assert min_gap > 7


def test_decision_matrix_includes_three_gates(tmp_path):
    from validate_conformal import build_decision_matrix
    cv_results = {
        "weighted_mondrian_conformal": {
            "marginal_coverage": [0.948, 0.901, 0.799],
            "marginal_coverage_ci": [(0.93, 0.96), (0.88, 0.92), (0.78, 0.82)],
            "bucket_coverage_violations_pct": [0.05, 0.07, 0.09],
            "median_width": [0.30, 0.12, 0.06],
            "median_width_ci_upper": [0.32, 0.13, 0.07],
        },
        "bucket_wilson": {
            "marginal_coverage": [0.95, 0.90, 0.80],
            "marginal_coverage_ci": [(0.94, 0.96), (0.89, 0.91), (0.79, 0.81)],
            "bucket_coverage_violations_pct": [0.04, 0.06, 0.08],
            "median_width": [0.10, 0.08, 0.05],
            "median_width_ci_upper": [0.11, 0.09, 0.06],
        },
    }
    matrix = build_decision_matrix(cv_results, alphas=[0.05, 0.10, 0.20])
    # Marginal + bucketed + tightness gates exist
    for method in matrix:
        for alpha_str in ("0.05", "0.10", "0.20"):
            assert "marginal" in matrix[method][alpha_str]
            assert "bucketed" in matrix[method][alpha_str]
            assert "tightness" in matrix[method][alpha_str]
            assert "ship" in matrix[method][alpha_str]
    # At alpha=0.05 conformal, tightness threshold is relaxed (no requirement)
    # so tightness gate=PASS even at width 0.30
    assert matrix["weighted_mondrian_conformal"]["0.05"]["tightness"] == "PASS"
