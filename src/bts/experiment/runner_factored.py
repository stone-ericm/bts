"""Factored-runner variants that skip redundant work when safe.

These paths are bit-exact reproductions of src/bts/experiment/runner.py's
run_single_screening for specific experiment categories. Enabled via
cli flags; NOT the default path until bit-exact validation passes on
seed=42 for every experiment category they support.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from bts.experiment.base import ExperimentDef


def _is_eligible_for_strategy_fast_path(experiment: ExperimentDef) -> tuple[bool, str]:
    """Check if experiment is safe for profile-reuse path.

    Eligible iff: no feature changes, no blend config changes, no LGB param
    changes, no per-model capture needed. In that case the walk-forward
    output is bit-identical to baseline, so we can skip it.
    """
    from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
    if experiment.touches_features():
        return False, "modifies features"
    blends_after = experiment.modify_blend_configs(list(BLEND_CONFIGS))
    if blends_after != list(BLEND_CONFIGS):
        return False, "modifies blend_configs"
    params_after = experiment.modify_training_params(dict(LGB_PARAMS))
    if params_after != dict(LGB_PARAMS):
        return False, "modifies training params"
    if experiment.requires_per_model_capture():
        return False, "requires per-model capture"
    return True, "ok"


def run_strategy_experiment_fast(
    experiment: ExperimentDef,
    baseline_profiles: pd.DataFrame,
    baseline_scorecard: dict,
    results_dir: Path,
) -> dict:
    """Run a strategy-only experiment by reusing baseline profiles.

    Skips the walk-forward entirely and applies only modify_strategy.
    Raises ValueError if experiment is not strategy-only.
    """
    from bts.experiment.runner import evaluate_pass_fail, _save_json
    from bts.simulate.quality_bins import compute_bins
    from bts.validate.scorecard import compute_full_scorecard, diff_scorecards, save_scorecard

    eligible, reason = _is_eligible_for_strategy_fast_path(experiment)
    if not eligible:
        raise ValueError(f"{experiment.name} not eligible for fast strategy path: {reason}")

    print(f"\n[Phase 1 FAST-STRAT] {experiment.name}: {experiment.description}", file=sys.stderr)

    # Reuse baseline profiles 1:1
    profiles = baseline_profiles.copy()
    try:
        baseline_bins = compute_bins(profiles)
    except Exception:
        baseline_bins = None

    profiles, _ = experiment.modify_strategy(profiles, baseline_bins)
    scorecard = compute_full_scorecard(profiles)
    diff = diff_scorecards(baseline_scorecard, scorecard)
    passed, reason_msg = evaluate_pass_fail(diff)

    exp_dir = results_dir / experiment.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_scorecard(scorecard, exp_dir / "scorecard.json")
    _save_json(diff, exp_dir / "diff.json")
    (exp_dir / "summary.txt").write_text(f"{'PASS' if passed else 'FAIL'} | {reason_msg}")

    status = "PASS" if passed else "FAIL"
    print(f"  -> {status}: {reason_msg}", file=sys.stderr)

    return {
        "name": experiment.name,
        "scorecard": scorecard,
        "diff": diff,
        "passed": passed,
        "reason": reason_msg,
    }
