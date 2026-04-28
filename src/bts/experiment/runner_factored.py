"""Factored-runner variants that skip redundant work when safe.

These paths are bit-exact reproductions of src/bts/experiment/runner.py's
run_single_screening for specific experiment categories. Enabled via
cli flags; NOT the default path until bit-exact validation passes on
seed=42 for every experiment category they support.
"""
# Validation status:
#   - Eligibility detector: tested via fast refuse-tests in
#     tests/experiment/test_factored_runner_bitexact.py
#   - Bit-exact numeric equivalence to run_single_screening: deferred to
#     Task 6 of the throughput plan (Hetzner harness; ~60 min/parameter
#     wall on local Mac, infeasible to run in CI).
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


# ---------------------------------------------------------------------------
# Model-swap fast path
# ---------------------------------------------------------------------------
#
# Plan-vs-reality note: the throughput plan describes catboost / vrex / xendcg
# as "replace ONE of 12 blend configs". Empirically (see
# src/bts/experiment/models.py) they APPEND a 13th member without changing
# the existing 12 — so the cached subset is the full 12 baseline configs and
# only the new 13th config is trained fresh. The optimization is the same
# (12 cached, 1 fresh, ~12/13 of training compute saved on retrain days).


def _is_eligible_for_model_swap_fast_path(experiment: ExperimentDef) -> tuple[bool, str]:
    """Check if experiment is safe for the model-swap cache path.

    Eligible iff: no feature changes, no LGB param changes, no per-model
    capture, and modify_blend_configs returns the original 12 baseline
    configs as a contiguous prefix (i.e., only adds new configs, doesn't
    remove or reorder baseline ones).
    """
    from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
    if experiment.touches_features():
        return False, "modifies features"
    if experiment.requires_per_model_capture():
        return False, "requires per-model capture"
    params_after = experiment.modify_training_params(dict(LGB_PARAMS))
    if params_after != dict(LGB_PARAMS):
        return False, "modifies training params"
    blends_after = experiment.modify_blend_configs(list(BLEND_CONFIGS))
    if list(blends_after[: len(BLEND_CONFIGS)]) != list(BLEND_CONFIGS):
        return False, "modifies or removes baseline blend configs"
    if len(blends_after) <= len(BLEND_CONFIGS):
        return False, "doesn't add a new model (use strategy fast path instead)"
    return True, "ok"


def run_model_swap_experiment_fast(
    experiment: ExperimentDef,
    pa_df: "pd.DataFrame",
    baseline_scorecard: dict,
    test_seasons: list[int],
    results_dir: Path,
    retrain_every: int = 7,
    cache_dir: Path | None = None,
) -> dict:
    """Run a model-add experiment, reusing 12 baseline blend models from cache.

    First call per (seed × day) trains and caches the 12 baseline configs;
    subsequent model-add experiments load them and only train the new 13th
    config. Bit-exact equivalent to ``run_single_screening`` (modulo the
    timestamp field) because seeds + features + params are unchanged.

    Raises ValueError if experiment is not eligible (e.g., modifies features
    or training params).
    """
    import os
    from bts.experiment.runner import evaluate_pass_fail, _save_json
    from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
    from bts.simulate.backtest_blend import blend_walk_forward
    from bts.simulate.quality_bins import compute_bins
    from bts.validate.scorecard import (
        compute_full_scorecard, diff_scorecards, save_scorecard,
    )

    eligible, reason = _is_eligible_for_model_swap_fast_path(experiment)
    if not eligible:
        raise ValueError(f"{experiment.name} not eligible for fast model-swap path: {reason}")

    if cache_dir is None:
        cache_dir = Path("data/experiments/blend_cache")

    seed = int(os.environ.get("BTS_LGBM_RANDOM_STATE", "42"))

    # The new blend_configs include the original 12 baseline configs plus
    # whatever the experiment appends (catboost / vrex / xendcg / lambdarank
    # all append exactly one). The 12 baseline are the "reuse" set.
    new_blend_configs = experiment.modify_blend_configs(list(BLEND_CONFIGS))
    cache_reuse_configs = [cfg[0] for cfg in BLEND_CONFIGS]

    print(
        f"\n[Phase 1 FAST-MODEL-SWAP] {experiment.name}: "
        f"reusing {len(cache_reuse_configs)}/{len(new_blend_configs)} baseline configs",
        file=sys.stderr,
    )

    lgb_params = experiment.modify_training_params(dict(LGB_PARAMS))

    all_profiles = []
    for season in test_seasons:
        profiles = blend_walk_forward(
            pa_df, season,
            retrain_every=retrain_every,
            blend_configs=new_blend_configs,
            lgb_params=lgb_params,
            cache_dir=cache_dir,
            cache_seed=seed,
            cache_reuse_configs=cache_reuse_configs,
        )
        profiles["season"] = season
        all_profiles.append(profiles)

    combined_profiles = pd.concat(all_profiles, ignore_index=True)

    try:
        baseline_bins = compute_bins(combined_profiles)
    except Exception:
        baseline_bins = None
    combined_profiles, _ = experiment.modify_strategy(combined_profiles, baseline_bins)

    scorecard = compute_full_scorecard(combined_profiles)
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
