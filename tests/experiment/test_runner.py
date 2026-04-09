import json
from pathlib import Path

import pandas as pd

from bts.experiment.base import ExperimentDef
from bts.experiment.runner import (
    evaluate_pass_fail,
    run_diagnostics,
    sort_winners_by_p57,
)


# --- Phase 0 tests ---

class _MockDiagnostic(ExperimentDef):
    def run_diagnostic(self, df, profiles):
        return {"test_metric": 42, "stable_features": ["batter_hr_30g"]}


def test_run_diagnostics_saves_results(mini_pa_df, tmp_path):
    diag = _MockDiagnostic(
        name="mock_diag", phase=0, category="diagnostic",
        description="mock diagnostic",
    )
    results = run_diagnostics(
        experiments=[diag],
        pa_df=mini_pa_df,
        profiles={},
        results_dir=tmp_path / "results" / "phase0",
    )
    assert "mock_diag" in results
    assert results["mock_diag"]["test_metric"] == 42
    saved = json.loads((tmp_path / "results" / "phase0" / "mock_diag.json").read_text())
    assert saved["test_metric"] == 42


def test_run_diagnostics_empty_list(mini_pa_df, tmp_path):
    results = run_diagnostics(
        experiments=[],
        pa_df=mini_pa_df,
        profiles={},
        results_dir=tmp_path / "results" / "phase0",
    )
    assert results == {}


# --- Phase 1 tests ---

def test_evaluate_pass_fail_both_seasons_improve():
    diff = {
        "p_at_1_by_season": {
            "2024": {"baseline": 0.849, "variant": 0.855, "delta": 0.006},
            "2025": {"baseline": 0.859, "variant": 0.865, "delta": 0.006},
        },
        "streak_metrics": {"mean_max_streak": {"delta": 0.5}},
        "p_57_exact": {"delta": 0.001},
    }
    passed, reason = evaluate_pass_fail(diff)
    assert passed is True
    assert "both seasons" in reason.lower()


def test_evaluate_pass_fail_one_season_drops():
    diff = {
        "p_at_1_by_season": {
            "2024": {"baseline": 0.849, "variant": 0.840, "delta": -0.009},
            "2025": {"baseline": 0.859, "variant": 0.865, "delta": 0.006},
        },
        "streak_metrics": {"mean_max_streak": {"delta": 0.5}},
        "p_57_exact": {"delta": 0.001},
    }
    passed, reason = evaluate_pass_fail(diff)
    assert passed is False


def test_evaluate_pass_fail_neutral_p1_but_streak_and_exact_improve():
    diff = {
        "p_at_1_by_season": {
            "2024": {"baseline": 0.849, "variant": 0.848, "delta": -0.001},
            "2025": {"baseline": 0.859, "variant": 0.857, "delta": -0.002},
        },
        "streak_metrics": {"mean_max_streak": {"delta": 0.8}},
        "p_57_exact": {"delta": 0.0015},
    }
    passed, reason = evaluate_pass_fail(diff)
    assert passed is True
    assert "neutral" in reason.lower()


def test_evaluate_pass_fail_neutral_p1_but_streak_drops():
    diff = {
        "p_at_1_by_season": {
            "2024": {"baseline": 0.849, "variant": 0.848, "delta": -0.001},
            "2025": {"baseline": 0.859, "variant": 0.857, "delta": -0.002},
        },
        "streak_metrics": {"mean_max_streak": {"delta": -0.5}},
        "p_57_exact": {"delta": 0.0015},
    }
    passed, reason = evaluate_pass_fail(diff)
    assert passed is False


def test_evaluate_pass_fail_neutral_p1_but_exact_p57_drops():
    diff = {
        "p_at_1_by_season": {
            "2024": {"baseline": 0.849, "variant": 0.848, "delta": -0.001},
            "2025": {"baseline": 0.859, "variant": 0.857, "delta": -0.002},
        },
        "streak_metrics": {"mean_max_streak": {"delta": 0.5}},
        "p_57_exact": {"delta": -0.001},
    }
    passed, reason = evaluate_pass_fail(diff)
    assert passed is False


# --- Phase 2 tests ---

def test_sort_winners_by_p57():
    """Sorts passing experiments by mean_max_streak delta (despite the name)."""
    results = [
        {"name": "a", "passed": True, "diff": {"streak_metrics": {"mean_max_streak": {"delta": 0.5}}}},
        {"name": "b", "passed": True, "diff": {"streak_metrics": {"mean_max_streak": {"delta": 1.2}}}},
        {"name": "c", "passed": False, "diff": {"streak_metrics": {"mean_max_streak": {"delta": 2.0}}}},
        {"name": "d", "passed": True, "diff": {"streak_metrics": {"mean_max_streak": {"delta": 0.1}}}},
    ]
    winners = sort_winners_by_p57(results)
    assert len(winners) == 3  # c is excluded (not passed)
    assert winners[0]["name"] == "b"  # highest delta first
    assert winners[1]["name"] == "a"
    assert winners[2]["name"] == "d"
