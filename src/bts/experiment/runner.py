"""Experiment runner — executes phases 0, 1, and 2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bts.experiment.base import ExperimentDef


# ---------------------------------------------------------------------------
# Blend config composition (used by Phase 1 and Phase 2)
# ---------------------------------------------------------------------------

def compose_blend_args(
    experiments: list[ExperimentDef],
) -> tuple[list, dict, bool]:
    """Compose blend_configs / lgb_params / capture_per_model from stacked experiments.

    For each experiment that overrides ``feature_cols()``, its added columns
    are unioned into a per-blend-config base feature list. Per-config Statcast
    "extras" are preserved so single-Statcast variants still work. Each
    experiment's ``modify_blend_configs`` and ``modify_training_params`` are
    applied in order. ``capture_per_model`` is True if ANY experiment
    requires it.

    Returns:
        (blend_configs, lgb_params, capture_per_model)

    This was extracted from ``run_single_screening`` 2026-04-28 so Phase 2
    forward selection (``run_selection``) can apply the same logic. Before
    the extraction, ``run_selection`` called ``blend_walk_forward`` with
    the default BLEND_CONFIGS, so feature additions never reached training.
    """
    from bts.features.compute import FEATURE_COLS
    from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS

    # Union all added features (preserve insertion order for determinism).
    base_features = list(FEATURE_COLS)
    for exp in experiments:
        new_fc = exp.feature_cols()
        if new_fc is not None:
            for c in new_fc:
                if c not in base_features:
                    base_features.append(c)

    # Rewrite each blend config: replace its FEATURE_COLS slice with
    # base_features (which is FEATURE_COLS + experiment additions),
    # preserving any per-config extras (Statcast variant features).
    rewritten = []
    for config in BLEND_CONFIGS:
        name, cols = config[0], config[1]
        extras = [c for c in cols if c not in FEATURE_COLS]
        new_cols = list(base_features) + extras
        if len(config) == 3:
            rewritten.append((name, new_cols, config[2]))
        else:
            rewritten.append((name, new_cols))

    # Apply each experiment's modify_blend_configs in order.
    blend_configs = rewritten
    for exp in experiments:
        blend_configs = exp.modify_blend_configs(blend_configs)

    # Compose lgb_params by stacking each experiment's modifications.
    lgb_params = dict(LGB_PARAMS)
    for exp in experiments:
        lgb_params = exp.modify_training_params(lgb_params)

    capture_per_model = any(exp.requires_per_model_capture() for exp in experiments)

    return blend_configs, lgb_params, capture_per_model


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

    Uses a composite metric instead of MDP P(57), which has been shown
    to have ~80% coefficient of variation under tiny prediction noise
    (bin-boundary artifacts). The replacement metrics are much more
    stable: mean_max_streak (3% CoV), P99 streak (4% CoV), exact P(57)
    (27% CoV).

    Pass if EITHER:
    1. P@1 improves on BOTH 2024 AND 2025 (strict ranking improvement)
    2. P@1 neutral on both (drop <= 0.3pp) AND streak metrics improve:
       - mean_max_streak >= 0 (neutral or up)
       - exact P(57) > 0 (strictly improved)

    Returns:
        (passed, reason) tuple.
    """
    p1_by_season = diff.get("p_at_1_by_season", {})
    streak_diff = diff.get("streak_metrics", {})

    season_deltas = {}
    for season_key, d in p1_by_season.items():
        season_deltas[str(season_key)] = d.get("delta", 0)

    if len(season_deltas) < 2:
        return False, "Missing season data"

    all_improve = all(d > 0 for d in season_deltas.values())
    all_neutral = all(d >= NEUTRAL_THRESHOLD for d in season_deltas.values())

    if all_improve:
        return True, "P@1 improves on both seasons"

    # Composite secondary check: streak metrics under fixed strategy
    mean_streak_delta = streak_diff.get("mean_max_streak", {}).get("delta", 0) or 0
    exact_p57_delta = (diff.get("p_57_exact", {}) or {}).get("delta", 0) or 0

    if all_neutral and mean_streak_delta >= 0 and exact_p57_delta > 0:
        return True, (
            f"P@1 neutral, mean_streak {mean_streak_delta:+.2f}, "
            f"exact P(57) {exact_p57_delta:+.4f}"
        )

    return False, (
        f"P@1 deltas: {season_deltas}, "
        f"mean_streak: {mean_streak_delta:+.2f}, "
        f"exact P(57): {exact_p57_delta:+.4f}"
    )


def run_single_screening(
    experiment: ExperimentDef,
    pa_df: pd.DataFrame,
    baseline_scorecard: dict,
    test_seasons: list[int],
    results_dir: Path,
    retrain_every: int = 7,
    baseline_profiles: pd.DataFrame | None = None,
    use_factored: bool = False,
    blend_cache_dir: Path | None = None,
) -> dict:
    """Run a single Phase 1 experiment: walk-forward → scorecard → diff → pass/fail.

    Calls all 4 experiment hooks:
      1. modify_features (if touches_features)
      2. modify_blend_configs (always)
      3. modify_training_params (always)
      4. modify_strategy (after walk-forward, before scorecard)

    When ``use_factored=True``, dispatches to ``runner_factored`` fast paths
    where the experiment is eligible (strategy-only or model-add). Falls
    through to the full implementation otherwise.

    Args:
        baseline_profiles: Required for the strategy fast path. When
            ``use_factored=True`` and the experiment is strategy-only, the
            cached baseline walk-forward output is reused 1:1.
        use_factored: Opt-in to factored fast paths. Default False until
            Task 6's bit-exact validation harness confirms equivalence.
        blend_cache_dir: Cache directory for the model-swap fast path's
            12 baseline blend models per (seed × day). When None, the
            factored runner uses its own default.

    Returns dict with keys: scorecard, diff, passed, reason, name.
    """
    if use_factored:
        import sys as _sys
        from bts.experiment.runner_factored import (
            _is_eligible_for_strategy_fast_path,
            _is_eligible_for_model_swap_fast_path,
            run_strategy_experiment_fast,
            run_model_swap_experiment_fast,
        )
        strat_ok, _ = _is_eligible_for_strategy_fast_path(experiment)
        if strat_ok:
            if baseline_profiles is not None:
                return run_strategy_experiment_fast(
                    experiment, baseline_profiles, baseline_scorecard, results_dir,
                )
            # Strategy-eligible but caller forgot baseline_profiles. This is
            # a silent regression to the slow path — warn loudly so it shows
            # up in audit logs (especially after Task 7b flips default to True).
            print(
                f"  WARN: {experiment.name} is strategy-eligible but "
                f"baseline_profiles=None; falling through to slow path",
                file=_sys.stderr,
            )
        model_swap_ok, _ = _is_eligible_for_model_swap_fast_path(experiment)
        if model_swap_ok:
            return run_model_swap_experiment_fast(
                experiment, pa_df, baseline_scorecard, test_seasons, results_dir,
                retrain_every=retrain_every, cache_dir=blend_cache_dir,
            )
        # Fall through to current implementation when:
        # - experiment is not eligible for any fast path
        # - strategy-eligible but baseline_profiles=None (warned above)

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
    baseline_profiles: pd.DataFrame | None = None,
    use_factored: bool = False,
    blend_cache_dir: Path | None = None,
) -> list[dict]:
    """Run Phase 1 screening for all experiments.

    Args:
        baseline_profiles: Cached baseline walk-forward output, required
            for the strategy fast path when ``use_factored=True``.
        use_factored: Opt-in to factored fast paths in
            ``run_single_screening``. Default False.
        blend_cache_dir: Per-(seed × day) cache directory for the
            model-swap fast path.
    """
    results = []
    for exp in experiments:
        result = run_single_screening(
            exp, pa_df, baseline_scorecard, test_seasons,
            results_dir, retrain_every,
            baseline_profiles=baseline_profiles,
            use_factored=use_factored,
            blend_cache_dir=blend_cache_dir,
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Phase 2 — Forward Stepwise Selection + Backward Elimination
# ---------------------------------------------------------------------------

def sort_winners_by_p57(results: list[dict]) -> list[dict]:
    """Filter to passing experiments and sort by mean max streak delta.

    Despite the historical name, sorts by mean_max_streak (the most stable
    metric) rather than MDP P(57) (which is unreliable, ~80% CoV under noise).
    """
    winners = [r for r in results if r.get("passed")]
    return sorted(
        winners,
        key=lambda r: r.get("diff", {}).get("streak_metrics", {}).get(
            "mean_max_streak", {}).get("delta", 0),
        reverse=True,
    )


def run_selection(
    winners: list[dict],
    experiments_by_name: dict[str, ExperimentDef],
    pa_df: pd.DataFrame,
    test_seasons: list[int],
    results_dir: Path,
    retrain_every: int = 7,
    seeds: list[int] | None = None,
) -> dict:
    """Run Phase 2: forward stepwise selection + backward elimination.

    Args:
        winners: Sorted list of Phase 1 results (passing only).
        experiments_by_name: {name: ExperimentDef} lookup.
        pa_df: Full PA DataFrame (with features already computed).
        test_seasons: Seasons to evaluate on.
        results_dir: Directory for phase2 results.
        retrain_every: Walk-forward retrain interval.
        seeds: Optional list of seeds to pool across. If provided, each
            forward/backward step is evaluated at every seed and decisions
            use the mean ΔP(57) across paired seed comparisons. This was
            added 2026-04-28 because single-seed Phase 2 at production
            seed=42 hits a P(57) ceiling effect — seed=42's baseline P(57)
            is at the 95th percentile of the n=100 distribution, so real
            winners with positive pooled ΔP(57) get rejected. If seeds is
            None, behavior matches pre-2026-04-28 single-seed mode.

    Returns:
        Dict with forward_log, backward_log, final_scorecard, final_diff, included.
    """
    import os
    from statistics import mean as _mean
    from bts.simulate.backtest_blend import blend_walk_forward
    from bts.validate.scorecard import compute_full_scorecard, diff_scorecards, save_scorecard

    multi_seed = seeds is not None and len(seeds) > 0
    eval_seeds = list(seeds) if multi_seed else [None]

    print(
        f"\n[Phase 2] Forward selection with {len(winners)} candidates"
        + (f" (multi-seed n={len(seeds)})" if multi_seed else ""),
        file=sys.stderr,
    )

    def _walk(df, blend_configs, lgb_params, capture):
        """Run walk-forward across all test_seasons with the given blend args."""
        out = []
        for season in test_seasons:
            profiles = blend_walk_forward(
                df, season,
                retrain_every=retrain_every,
                blend_configs=blend_configs,
                lgb_params=lgb_params,
                capture_per_model=capture,
            )
            profiles["season"] = season
            out.append(profiles)
        return pd.concat(out, ignore_index=True)

    def _evaluate(df, stacked_experiments):
        """Run walk-forward + scorecard at each seed in eval_seeds.

        Returns (mean_p57, p57_per_seed_dict, scorecard_at_first_seed).
        scorecard_at_first_seed is the scorecard from the first seed, used
        for downstream diff/save (matching pre-existing single-seed semantics).
        """
        bc, lp, cap = compose_blend_args(stacked_experiments)
        p57_per_seed = {}
        first_scorecard = None
        first_combined = None
        for seed in eval_seeds:
            prev_env = None
            if seed is not None:
                prev_env = os.environ.get("BTS_LGBM_RANDOM_STATE")
                os.environ["BTS_LGBM_RANDOM_STATE"] = str(seed)
            try:
                combined = _walk(df, bc, lp, cap)
            finally:
                if seed is not None:
                    if prev_env is None:
                        os.environ.pop("BTS_LGBM_RANDOM_STATE", None)
                    else:
                        os.environ["BTS_LGBM_RANDOM_STATE"] = prev_env
            sc = compute_full_scorecard(combined)
            p57 = sc.get("p_57_mdp", 0) or 0
            seed_key = seed if seed is not None else "default"
            p57_per_seed[seed_key] = p57
            if first_scorecard is None:
                first_scorecard = sc
                first_combined = combined
        return _mean(p57_per_seed.values()), p57_per_seed, first_scorecard, first_combined

    # Compute baseline at every seed (or single default seed when seeds=None).
    current_p57, current_p57_per_seed, baseline_scorecard, baseline_combined = _evaluate(pa_df, [])

    included: list[str] = []
    forward_log: list[dict] = []
    current_df = pa_df.copy()

    # Forward selection — pooled mean ΔP(57) across seeds drives keep/drop.
    for winner in winners:
        name = winner["name"]
        exp = experiments_by_name[name]
        print(f"  Trying +{name}...", file=sys.stderr)

        candidate_df = exp.modify_features(current_df.copy())
        stacked = [experiments_by_name[n] for n in included] + [exp]
        candidate_p57, candidate_p57_per_seed, _, _ = _evaluate(candidate_df, stacked)

        # Paired delta per seed → mean. Falls back to single-value subtraction
        # in single-seed mode (the dicts have one key each).
        per_seed_deltas = {
            k: candidate_p57_per_seed[k] - current_p57_per_seed[k]
            for k in candidate_p57_per_seed
        }
        mean_delta = _mean(per_seed_deltas.values())

        step = {
            "name": name,
            "p57_before": current_p57,
            "p57_after": candidate_p57,
            "delta": mean_delta,
            "kept": mean_delta > 0,
        }
        if multi_seed:
            step["p57_before_per_seed"] = current_p57_per_seed
            step["p57_after_per_seed"] = candidate_p57_per_seed
            step["delta_per_seed"] = per_seed_deltas
        forward_log.append(step)

        if mean_delta > 0:
            print(
                f"  ✓ Kept {name}: pooled mean ΔP(57) {mean_delta:+.5f}"
                f" ({current_p57:.4f} → {candidate_p57:.4f})",
                file=sys.stderr,
            )
            included.append(name)
            current_df = candidate_df
            current_p57 = candidate_p57
            current_p57_per_seed = candidate_p57_per_seed
        else:
            print(
                f"  ✗ Dropped {name}: pooled mean ΔP(57) {mean_delta:+.5f} (not positive)",
                file=sys.stderr,
            )

    print(f"\n  Forward selection: {len(included)} experiments included", file=sys.stderr)

    # Backward elimination — remove each in turn, see if pooled P(57) holds.
    backward_log: list[dict] = []
    for name in list(included):
        print(f"  Trying -{name}...", file=sys.stderr)
        test_df = pa_df.copy()
        for kept_name in included:
            if kept_name != name:
                test_df = experiments_by_name[kept_name].modify_features(test_df)

        stacked = [experiments_by_name[n] for n in included if n != name]
        test_p57, test_p57_per_seed, _, _ = _evaluate(test_df, stacked)

        per_seed_deltas = {
            k: current_p57_per_seed[k] - test_p57_per_seed[k]
            for k in test_p57_per_seed
        }
        mean_loss = _mean(per_seed_deltas.values())

        step = {
            "name": name,
            "p57_with": current_p57,
            "p57_without": test_p57,
            "delta": mean_loss,
            "kept": mean_loss > 0,
        }
        if multi_seed:
            step["p57_with_per_seed"] = current_p57_per_seed
            step["p57_without_per_seed"] = test_p57_per_seed
            step["delta_per_seed"] = per_seed_deltas
        backward_log.append(step)

        if mean_loss <= 0:
            print(f"  ✗ Removed {name}: pooled removing helps or neutral", file=sys.stderr)
            included.remove(name)
            current_p57 = test_p57
            current_p57_per_seed = test_p57_per_seed
        else:
            print(f"  ✓ Kept {name}: pooled removing hurts P(57) by {mean_loss:+.5f}", file=sys.stderr)

    # Final scorecard — blend args for the FINAL included set, evaluated at
    # the first seed (single-seed mode) or first eval seed (multi-seed mode).
    # The full per-seed P(57) array is logged in forward/backward logs already.
    final_df = pa_df.copy()
    for name in included:
        final_df = experiments_by_name[name].modify_features(final_df)

    final_stacked = [experiments_by_name[n] for n in included]
    final_p57, final_p57_per_seed, final_scorecard, _ = _evaluate(final_df, final_stacked)
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
