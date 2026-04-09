"""Experiment runner — executes phases 0, 1, and 2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bts.experiment.base import ExperimentDef


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default))


# ---------------------------------------------------------------------------
# Phase 0 — Diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(
    experiments: list[ExperimentDef],
    pa_df: pd.DataFrame,
    profiles: dict[int, pd.DataFrame],
    results_dir: Path,
) -> dict[str, dict]:
    """Run Phase 0 diagnostics and save reports.

    Args:
        experiments: List of Phase 0 ExperimentDef instances.
        pa_df: Feature-enriched PA DataFrame.
        profiles: {season: profiles_df} from existing backtests.
        results_dir: Directory to save JSON reports.

    Returns:
        {name: report_dict} for each diagnostic.
    """
    results: dict[str, dict] = {}
    for exp in experiments:
        print(f"[Phase 0] Running {exp.name}: {exp.description}", file=sys.stderr)
        report = exp.run_diagnostic(pa_df, profiles)
        _save_json(report, results_dir / f"{exp.name}.json")
        results[exp.name] = report
        print(f"  → Saved {results_dir / exp.name}.json", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Phase 1 — Screening
# ---------------------------------------------------------------------------

# Maximum allowed P@1 drop per season for "neutral" condition (0.3pp)
NEUTRAL_THRESHOLD = -0.003


def evaluate_pass_fail(diff: dict) -> tuple[bool, str]:
    """Evaluate whether an experiment passes screening.

    Pass if EITHER:
    1. P@1 improves on both 2024 AND 2025
    2. P@1 neutral on both (drop <= 0.3pp) AND MDP P(57) improves

    Returns:
        (passed, reason) tuple.
    """
    p1_by_season = diff.get("p_at_1_by_season", {})
    p57_diff = diff.get("p_57_mdp", {})

    season_deltas = {}
    for season_key, d in p1_by_season.items():
        season_deltas[str(season_key)] = d.get("delta", 0)

    if len(season_deltas) < 2:
        return False, "Missing season data"

    all_improve = all(d > 0 for d in season_deltas.values())
    all_neutral = all(d >= NEUTRAL_THRESHOLD for d in season_deltas.values())
    p57_improves = p57_diff.get("delta", 0) > 0

    if all_improve:
        return True, "P@1 improves on both seasons"
    if all_neutral and p57_improves:
        return True, "P@1 neutral, P(57) improves"
    return False, f"P@1 deltas: {season_deltas}, P(57) delta: {p57_diff.get('delta', 'N/A')}"


def run_single_screening(
    experiment: ExperimentDef,
    pa_df: pd.DataFrame,
    baseline_scorecard: dict,
    test_seasons: list[int],
    results_dir: Path,
    retrain_every: int = 7,
) -> dict:
    """Run a single Phase 1 experiment: walk-forward → scorecard → diff → pass/fail.

    Calls all 4 experiment hooks:
      1. modify_features (if touches_features)
      2. modify_blend_configs (always)
      3. modify_training_params (always)
      4. modify_strategy (after walk-forward, before scorecard)

    Returns dict with keys: scorecard, diff, passed, reason, name.
    """
    from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
    from bts.simulate.backtest_blend import blend_walk_forward
    from bts.simulate.quality_bins import compute_bins
    from bts.validate.scorecard import compute_full_scorecard, diff_scorecards, save_scorecard

    print(f"\n[Phase 1] {experiment.name}: {experiment.description}", file=sys.stderr)

    from bts.features.compute import FEATURE_COLS

    # 1. Apply feature modifications if needed
    df = pa_df
    if experiment.touches_features():
        print(f"  Recomputing features for {experiment.name}...", file=sys.stderr)
        df = experiment.modify_features(df.copy())

    # 2. Apply blend config modifications.
    # If experiment overrides feature_cols(), rewrite each existing blend
    # config's base features to use the experiment's feature set, preserving
    # any per-config additional features (e.g., Statcast variants).
    new_base_features = experiment.feature_cols()
    if new_base_features is not None:
        rewritten = []
        for config in BLEND_CONFIGS:
            name, cols = config[0], config[1]
            extras = [c for c in cols if c not in FEATURE_COLS]
            new_cols = list(new_base_features) + extras
            if len(config) == 3:
                rewritten.append((name, new_cols, config[2]))
            else:
                rewritten.append((name, new_cols))
        blend_configs = experiment.modify_blend_configs(rewritten)
    else:
        blend_configs = experiment.modify_blend_configs(list(BLEND_CONFIGS))

    # 3. Apply training param modifications
    lgb_params = experiment.modify_training_params(dict(LGB_PARAMS))

    capture_per_model = experiment.requires_per_model_capture()

    # Run walk-forward for each test season
    all_profiles = []
    for season in test_seasons:
        profiles = blend_walk_forward(
            df, season,
            retrain_every=retrain_every,
            blend_configs=blend_configs,
            lgb_params=lgb_params,
            capture_per_model=capture_per_model,
        )
        profiles["season"] = season
        all_profiles.append(profiles)

    combined_profiles = pd.concat(all_profiles, ignore_index=True)

    # 4. Apply strategy modifications (calibration, copula, etc.)
    try:
        baseline_bins = compute_bins(combined_profiles)
    except Exception:
        baseline_bins = None
    combined_profiles, _ = experiment.modify_strategy(combined_profiles, baseline_bins)

    # Compute scorecard and diff
    scorecard = compute_full_scorecard(combined_profiles)
    diff = diff_scorecards(baseline_scorecard, scorecard)
    passed, reason = evaluate_pass_fail(diff)

    # Save results
    exp_dir = results_dir / experiment.name
    save_scorecard(scorecard, exp_dir / "scorecard.json")
    _save_json(diff, exp_dir / "diff.json")
    summary = f"{'PASS' if passed else 'FAIL'} | {reason}"
    (exp_dir / "summary.txt").parent.mkdir(parents=True, exist_ok=True)
    (exp_dir / "summary.txt").write_text(summary)

    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  → {status}: {reason}", file=sys.stderr)

    return {
        "name": experiment.name,
        "scorecard": scorecard,
        "diff": diff,
        "passed": passed,
        "reason": reason,
    }


def run_screening(
    experiments: list[ExperimentDef],
    pa_df: pd.DataFrame,
    baseline_scorecard: dict,
    test_seasons: list[int],
    results_dir: Path,
    retrain_every: int = 7,
) -> list[dict]:
    """Run Phase 1 screening for all experiments."""
    results = []
    for exp in experiments:
        result = run_single_screening(
            exp, pa_df, baseline_scorecard, test_seasons,
            results_dir, retrain_every,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Phase 2 — Forward Stepwise Selection + Backward Elimination
# ---------------------------------------------------------------------------

def sort_winners_by_p57(results: list[dict]) -> list[dict]:
    """Filter to passing experiments and sort by P(57) improvement descending."""
    winners = [r for r in results if r.get("passed")]
    return sorted(
        winners,
        key=lambda r: r.get("diff", {}).get("p_57_mdp", {}).get("delta", 0),
        reverse=True,
    )


def run_selection(
    winners: list[dict],
    experiments_by_name: dict[str, ExperimentDef],
    pa_df: pd.DataFrame,
    test_seasons: list[int],
    results_dir: Path,
    retrain_every: int = 7,
) -> dict:
    """Run Phase 2: forward stepwise selection + backward elimination.

    Args:
        winners: Sorted list of Phase 1 results (passing only).
        experiments_by_name: {name: ExperimentDef} lookup.
        pa_df: Full PA DataFrame (with features already computed).
        test_seasons: Seasons to evaluate on.
        results_dir: Directory for phase2 results.

    Returns:
        Dict with forward_log, backward_log, final_scorecard, final_diff, included.
    """
    from bts.simulate.backtest_blend import blend_walk_forward
    from bts.validate.scorecard import compute_full_scorecard, diff_scorecards, save_scorecard

    print(f"\n[Phase 2] Forward selection with {len(winners)} candidates", file=sys.stderr)

    # Compute baseline
    baseline_profiles = []
    for season in test_seasons:
        profiles = blend_walk_forward(pa_df, season, retrain_every=retrain_every)
        profiles["season"] = season
        baseline_profiles.append(profiles)
    baseline_combined = pd.concat(baseline_profiles, ignore_index=True)
    baseline_scorecard = compute_full_scorecard(baseline_combined)
    current_p57 = baseline_scorecard.get("p_57_mdp", 0) or 0

    included: list[str] = []
    forward_log: list[dict] = []
    current_df = pa_df.copy()

    # Forward selection
    for winner in winners:
        name = winner["name"]
        exp = experiments_by_name[name]
        print(f"  Trying +{name}...", file=sys.stderr)

        candidate_df = exp.modify_features(current_df.copy())
        candidate_profiles = []
        for season in test_seasons:
            profiles = blend_walk_forward(candidate_df, season, retrain_every=retrain_every)
            profiles["season"] = season
            candidate_profiles.append(profiles)
        candidate_combined = pd.concat(candidate_profiles, ignore_index=True)
        candidate_scorecard = compute_full_scorecard(candidate_combined)
        candidate_p57 = candidate_scorecard.get("p_57_mdp", 0) or 0

        step = {
            "name": name,
            "p57_before": current_p57,
            "p57_after": candidate_p57,
            "delta": candidate_p57 - current_p57,
            "kept": candidate_p57 > current_p57,
        }
        forward_log.append(step)

        if candidate_p57 > current_p57:
            print(f"  ✓ Kept {name}: P(57) {current_p57:.4f} → {candidate_p57:.4f}", file=sys.stderr)
            included.append(name)
            current_df = candidate_df
            current_p57 = candidate_p57
        else:
            print(f"  ✗ Dropped {name}: P(57) did not improve", file=sys.stderr)

    print(f"\n  Forward selection: {len(included)} experiments included", file=sys.stderr)

    # Backward elimination
    backward_log: list[dict] = []
    for name in list(included):
        print(f"  Trying -{name}...", file=sys.stderr)
        test_df = pa_df.copy()
        for kept_name in included:
            if kept_name != name:
                test_df = experiments_by_name[kept_name].modify_features(test_df)

        test_profiles = []
        for season in test_seasons:
            profiles = blend_walk_forward(test_df, season, retrain_every=retrain_every)
            profiles["season"] = season
            test_profiles.append(profiles)
        test_combined = pd.concat(test_profiles, ignore_index=True)
        test_scorecard = compute_full_scorecard(test_combined)
        test_p57 = test_scorecard.get("p_57_mdp", 0) or 0

        step = {
            "name": name,
            "p57_with": current_p57,
            "p57_without": test_p57,
            "delta": current_p57 - test_p57,
            "kept": test_p57 < current_p57,
        }
        backward_log.append(step)

        if test_p57 >= current_p57:
            print(f"  ✗ Removed {name}: not needed", file=sys.stderr)
            included.remove(name)
        else:
            print(f"  ✓ Kept {name}: removing hurts P(57)", file=sys.stderr)

    # Final scorecard
    final_df = pa_df.copy()
    for name in included:
        final_df = experiments_by_name[name].modify_features(final_df)

    final_profiles = []
    for season in test_seasons:
        profiles = blend_walk_forward(final_df, season, retrain_every=retrain_every)
        profiles["season"] = season
        final_profiles.append(profiles)
    final_combined = pd.concat(final_profiles, ignore_index=True)
    final_scorecard = compute_full_scorecard(final_combined)
    final_diff = diff_scorecards(baseline_scorecard, final_scorecard)

    # Save
    results_dir.mkdir(parents=True, exist_ok=True)
    _save_json(forward_log, results_dir / "forward_selection_log.json")
    _save_json(backward_log, results_dir / "backward_elimination_log.json")
    save_scorecard(final_scorecard, results_dir / "final_scorecard.json")
    _save_json(final_diff, results_dir / "final_diff.json")

    print(f"\n  Final model: {included}", file=sys.stderr)
    print(f"  Final P(57): {final_scorecard.get('p_57_mdp', 'N/A')}", file=sys.stderr)

    return {
        "included": included,
        "forward_log": forward_log,
        "backward_log": backward_log,
        "final_scorecard": final_scorecard,
        "final_diff": final_diff,
    }
