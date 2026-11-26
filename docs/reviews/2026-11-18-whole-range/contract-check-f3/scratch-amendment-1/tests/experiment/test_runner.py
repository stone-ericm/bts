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


# --- Task 7a: dispatcher tests ---

def _dispatch_setup(monkeypatch, tmp_path):
    """Replace the two factored entry points with recorders so tests can
    verify which path got dispatched without running real walk-forwards."""
    import bts.experiment.runner_factored as rf
    calls = {"strategy": 0, "model_swap": 0}

    def fake_strategy(experiment, baseline_profiles, baseline_scorecard, results_dir):
        calls["strategy"] += 1
        return {"name": experiment.name, "scorecard": {}, "diff": {}, "passed": True, "reason": "fake-strategy"}

    def fake_model_swap(experiment, pa_df, baseline_scorecard, test_seasons, results_dir, retrain_every=7, cache_dir=None):
        calls["model_swap"] += 1
        return {"name": experiment.name, "scorecard": {}, "diff": {}, "passed": True, "reason": "fake-model-swap"}

    monkeypatch.setattr(rf, "run_strategy_experiment_fast", fake_strategy)
    monkeypatch.setattr(rf, "run_model_swap_experiment_fast", fake_model_swap)
    return calls


def test_dispatch_strategy_eligible_with_profiles_calls_strategy(monkeypatch, tmp_path):
    """use_factored=True + strategy-eligible + baseline_profiles set → strategy fast path called."""
    from bts.experiment.runner import run_single_screening
    from bts.experiment.registry import load_all_experiments, get_experiment

    load_all_experiments()
    exp = get_experiment("decision_calibration")  # truly strategy-only
    calls = _dispatch_setup(monkeypatch, tmp_path)

    profiles = pd.DataFrame({"date": [], "rank": [], "p_game_hit": [], "actual_hit": []})
    result = run_single_screening(
        exp, pa_df=pd.DataFrame(), baseline_scorecard={}, test_seasons=[2024, 2025],
        results_dir=tmp_path, baseline_profiles=profiles, use_factored=True,
    )
    assert calls["strategy"] == 1
    assert calls["model_swap"] == 0
    assert result["reason"] == "fake-strategy"


def test_dispatch_strategy_eligible_without_profiles_falls_through(monkeypatch, tmp_path, capsys):
    """use_factored=True + strategy-eligible + baseline_profiles=None → warn + fall through."""
    from bts.experiment.runner import run_single_screening
    from bts.experiment.registry import load_all_experiments, get_experiment

    load_all_experiments()
    exp = get_experiment("decision_calibration")
    calls = _dispatch_setup(monkeypatch, tmp_path)

    # Make the slow path also a no-op via monkeypatch so the test is fast
    import bts.simulate.backtest_blend as bb
    original = bb.blend_walk_forward
    bb.blend_walk_forward = lambda *a, **k: pd.DataFrame({"date": [], "rank": [], "p_game_hit": [], "actual_hit": []})
    try:
        run_single_screening(
            exp, pa_df=pd.DataFrame(), baseline_scorecard={}, test_seasons=[2024, 2025],
            results_dir=tmp_path, baseline_profiles=None, use_factored=True,
        )
    except Exception:
        # The slow path may raise on the empty df; we only care about the warn + dispatch counts
        pass
    finally:
        bb.blend_walk_forward = original

    assert calls["strategy"] == 0  # NOT called — fell through
    captured = capsys.readouterr()
    assert "WARN" in captured.err
    assert "decision_calibration" in captured.err
    assert "baseline_profiles=None" in captured.err


def test_dispatch_model_swap_eligible_calls_model_swap(monkeypatch, tmp_path):
    """use_factored=True + model-swap-eligible → model-swap fast path called."""
    from bts.experiment.runner import run_single_screening
    from bts.experiment.registry import load_all_experiments, get_experiment

    load_all_experiments()
    exp = get_experiment("catboost")  # appends a 13th config — model-swap eligible
    calls = _dispatch_setup(monkeypatch, tmp_path)

    result = run_single_screening(
        exp, pa_df=pd.DataFrame(), baseline_scorecard={}, test_seasons=[2024, 2025],
        results_dir=tmp_path, use_factored=True,
    )
    assert calls["model_swap"] == 1
    assert calls["strategy"] == 0
    assert result["reason"] == "fake-model-swap"


def test_dispatch_ineligible_falls_through(monkeypatch, tmp_path):
    """use_factored=True + ineligible (feature-mod) → neither fast path called."""
    from bts.experiment.runner import run_single_screening
    from bts.experiment.registry import load_all_experiments, get_experiment

    load_all_experiments()
    exp = get_experiment("wind_vector")  # modifies features
    calls = _dispatch_setup(monkeypatch, tmp_path)

    import bts.simulate.backtest_blend as bb
    original = bb.blend_walk_forward
    bb.blend_walk_forward = lambda *a, **k: pd.DataFrame({"date": [], "rank": [], "p_game_hit": [], "actual_hit": []})
    try:
        run_single_screening(
            exp, pa_df=pd.DataFrame(), baseline_scorecard={}, test_seasons=[2024, 2025],
            results_dir=tmp_path, baseline_profiles=pd.DataFrame(), use_factored=True,
        )
    except Exception:
        pass
    finally:
        bb.blend_walk_forward = original

    assert calls["strategy"] == 0
    assert calls["model_swap"] == 0


def test_dispatch_use_factored_false_skips_dispatch(monkeypatch, tmp_path):
    """use_factored=False → dispatch block skipped entirely; neither fast path called."""
    from bts.experiment.runner import run_single_screening
    from bts.experiment.registry import load_all_experiments, get_experiment

    load_all_experiments()
    exp = get_experiment("decision_calibration")  # strategy-eligible — would be called if dispatch fired
    calls = _dispatch_setup(monkeypatch, tmp_path)

    profiles = pd.DataFrame({"date": [], "rank": [], "p_game_hit": [], "actual_hit": []})
    import bts.simulate.backtest_blend as bb
    original = bb.blend_walk_forward
    bb.blend_walk_forward = lambda *a, **k: pd.DataFrame({"date": [], "rank": [], "p_game_hit": [], "actual_hit": []})
    try:
        run_single_screening(
            exp, pa_df=pd.DataFrame(), baseline_scorecard={}, test_seasons=[2024, 2025],
            results_dir=tmp_path, baseline_profiles=profiles, use_factored=False,
        )
    except Exception:
        pass
    finally:
        bb.blend_walk_forward = original

    assert calls["strategy"] == 0  # NOT called — use_factored=False
    assert calls["model_swap"] == 0
