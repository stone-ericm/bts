import numpy as np
import pandas as pd

from bts.experiment.diagnostics import (
    StabilitySelectionDiagnostic,
    WassersteinDriftDiagnostic,
    StreakLengthDependenceDiagnostic,
    AFTShapeDiagnostic,
    ADWINChangepointDiagnostic,
)


def test_stability_selection_report_structure(mini_pa_df):
    # Need multiple seasons for stability selection
    df = mini_pa_df.copy()
    df2 = mini_pa_df.copy()
    df2["season"] = 2024
    df2["date"] = pd.date_range("2024-06-01", periods=5, freq="D").repeat(10)[:len(df2)]
    combined = pd.concat([df2, df], ignore_index=True)

    diag = StabilitySelectionDiagnostic()
    report = diag.run_diagnostic(combined, {})
    assert "feature_stability" in report
    assert isinstance(report["feature_stability"], dict)
    for feat, score in report["feature_stability"].items():
        assert 0 <= score <= 1, f"{feat}: {score}"


def test_wasserstein_drift_report_structure(mini_pa_df):
    df = mini_pa_df.copy()
    df2 = df.copy()
    df2["season"] = 2024
    combined = pd.concat([df, df2], ignore_index=True)

    diag = WassersteinDriftDiagnostic()
    report = diag.run_diagnostic(combined, {})
    assert "feature_drift" in report
    assert isinstance(report["feature_drift"], dict)


def test_streak_length_dependence(mock_profiles_df):
    diag = StreakLengthDependenceDiagnostic()
    report = diag.run_diagnostic(pd.DataFrame(), {2025: mock_profiles_df})
    assert "p1_by_streak_bucket" in report


def test_aft_shape_with_profiles(mock_profiles_df):
    diag = AFTShapeDiagnostic()
    report = diag.run_diagnostic(pd.DataFrame(), {2025: mock_profiles_df})
    assert "shape" in report
    if report["shape"] is not None:
        assert report["shape"] > 0
        assert "interpretation" in report


def test_adwin_with_profiles(mock_profiles_df):
    diag = ADWINChangepointDiagnostic()
    report = diag.run_diagnostic(pd.DataFrame(), {2025: mock_profiles_df})
    assert "n_changepoints" in report or "error" in report
