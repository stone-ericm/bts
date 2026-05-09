"""CLI commands for the experiment framework."""

import json
import sys
from pathlib import Path

import click


RESULTS_BASE = Path("experiments/results")
DEFAULT_TEST_SEASONS = "2024,2025"


def _parse_season_list(value: str, flag_name: str) -> list[int]:
    seasons: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            seasons.append(int(item))
        except ValueError as exc:
            raise click.UsageError(
                f"{flag_name} must be a comma-separated list of integer seasons."
            ) from exc
    if not seasons:
        raise click.UsageError(f"{flag_name} must include at least one season.")
    if len(set(seasons)) != len(seasons):
        raise click.UsageError(f"{flag_name} must not contain duplicate seasons.")
    return seasons


def _resolve_season_split(
    *,
    test_seasons: str | None,
    selection_seasons: str | None,
    outer_eval_seasons: str | None,
) -> tuple[list[int], list[int] | None, dict | None]:
    """Resolve legacy or opt-in split season arguments."""
    has_selection = selection_seasons is not None
    has_outer = outer_eval_seasons is not None
    if has_selection != has_outer:
        raise click.UsageError(
            "--selection-seasons and --outer-eval-seasons must be supplied together."
        )
    if has_selection and test_seasons is not None:
        raise click.UsageError(
            "--test-seasons cannot be combined with --selection-seasons/--outer-eval-seasons."
        )

    if has_selection and selection_seasons is not None and outer_eval_seasons is not None:
        selection = _parse_season_list(selection_seasons, "--selection-seasons")
        outer = _parse_season_list(outer_eval_seasons, "--outer-eval-seasons")
        overlap = sorted(set(selection) & set(outer))
        if overlap:
            raise click.UsageError(
                "--selection-seasons and --outer-eval-seasons must be disjoint; "
                f"overlap: {overlap}"
            )
        metadata = {
            "split_mode": "season_level_selection_outer_eval",
            "selection_seasons": selection,
            "outer_eval_seasons": outer,
            "lockbox_used": False,
            "lockbox_manifest": None,
            "production_deploy_claim": False,
        }
        return selection, outer, metadata

    seasons = _parse_season_list(test_seasons or DEFAULT_TEST_SEASONS, "--test-seasons")
    return seasons, None, None


@click.group()
def experiment():
    """Frontier experiment framework — diagnostics, screening, selection."""
    pass


@experiment.command()
@click.option("--data-dir", default="data/processed", type=click.Path(),
              help="Processed parquet directory")
@click.option("--profiles-dir", default="data/simulation", type=click.Path(),
              help="Existing backtest profiles directory")
def diagnostics(data_dir: str, profiles_dir: str):
    """Run Phase 0 diagnostics."""
    import pandas as pd
    from bts.features.compute import compute_all_features
    from bts.experiment.registry import list_experiments, load_all_experiments
    from bts.experiment.runner import run_diagnostics

    load_all_experiments()
    diags = list_experiments(phase=0)
    if not diags:
        click.echo("No Phase 0 diagnostics registered.")
        return

    click.echo(f"Running {len(diags)} diagnostics...")

    proc = Path(data_dir)
    dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
    if not dfs:
        raise click.ClickException("No parquet files found. Run 'bts data build' first.")
    df = pd.concat(dfs, ignore_index=True)
    df = compute_all_features(df)

    profiles = {}
    prof_path = Path(profiles_dir)
    for p in prof_path.glob("backtest_*.parquet"):
        season = int(p.stem.split("_")[1])
        profiles[season] = pd.read_parquet(p)

    results = run_diagnostics(diags, df, profiles, RESULTS_BASE / "phase0")
    click.echo(f"\nDiagnostics complete. {len(results)} reports saved.")
    for name, report in results.items():
        click.echo(f"  {name}: {list(report.keys())[:5]}...")


@experiment.command()
@click.option("--data-dir", default="data/processed", type=click.Path())
@click.option("--subset", default=None, help="Comma-separated experiment names to run")
@click.option("--retrain-every", default=7, type=int)
@click.option(
    "--test-seasons",
    default=None,
    help="Comma-separated test seasons for the legacy path (default: 2024,2025)",
)
@click.option(
    "--selection-seasons",
    default=None,
    help="Comma-separated seasons used for opt-in feature/model selection.",
)
@click.option(
    "--outer-eval-seasons",
    default=None,
    help="Comma-separated later seasons reserved for the final outer evaluation.",
)
@click.option(
    "--use-factored/--no-use-factored",
    default=True,
    help="Use factored-runner fast paths where eligible "
    "(default: True; flipped 2026-04-28 after Stage 2 v2 PASS — "
    "AX102 validated all 32 experiments byte-equivalent at atol=1e-10)",
)
@click.option(
    "--blend-cache-dir",
    default=None,
    type=click.Path(),
    help="Override default cache dir for the model-swap fast path "
    "(default: data/experiments/blend_cache). Only consulted with --use-factored.",
)
@click.option(
    "--seeds",
    default=None,
    help="Comma-separated seeds to run in-process (e.g. '42,43,44'). When "
    "provided, run_screening is invoked once per seed with results in "
    "phase1/seed_<N>/ subdirs. Mutually exclusive with --seed-set. "
    "Default (no flag) preserves single-seed-via-env-var behavior used "
    "by audit_driver.",
)
@click.option(
    "--seed-set",
    default=None,
    help="Named seed manifest from data/seed_sets/<name>.json (e.g. "
    "'canonical-n10'). Mutually exclusive with --seeds.",
)
def screen(
    data_dir: str,
    subset: str | None,
    retrain_every: int,
    test_seasons: str | None,
    selection_seasons: str | None,
    outer_eval_seasons: str | None,
    use_factored: bool,
    blend_cache_dir: str | None,
    seeds: str | None,
    seed_set: str | None,
):
    """Run Phase 1 independent screening."""
    import os as _os
    import pandas as pd
    from bts.features.compute import compute_all_features
    from bts.experiment.registry import list_experiments, load_all_experiments, get_experiment
    from bts.experiment.runner import run_screening
    from bts.experiment.reporting import format_phase1_table
    from bts.validate.scorecard import compute_full_scorecard, save_scorecard
    from bts.simulate.backtest_blend import blend_walk_forward

    if seeds and seed_set:
        raise click.UsageError(
            "--seeds and --seed-set are mutually exclusive; pass at most one."
        )

    seed_list: list[int] | None = None
    if seeds:
        seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
        click.echo(f"Multi-seed Phase 1: running across {len(seed_list)} explicit seeds")
    elif seed_set:
        manifest_path = Path("data/seed_sets") / f"{seed_set}.json"
        if not manifest_path.exists():
            available = sorted(p.stem for p in Path("data/seed_sets").glob("*.json"))
            raise click.UsageError(
                f"Seed set '{seed_set}' not found at {manifest_path}. "
                f"Available: {available}"
            )
        manifest = json.loads(manifest_path.read_text())
        seed_list = [int(s) for s in manifest["seeds"]]
        click.echo(
            f"Multi-seed Phase 1: running across {len(seed_list)} seeds "
            f"from seed-set '{seed_set}'"
        )

    load_all_experiments()
    seasons, outer_seasons, split_metadata = _resolve_season_split(
        test_seasons=test_seasons,
        selection_seasons=selection_seasons,
        outer_eval_seasons=outer_eval_seasons,
    )

    if subset:
        experiments = [get_experiment(n.strip()) for n in subset.split(",")]
    else:
        experiments = list_experiments(phase=1)

    if not experiments:
        click.echo("No Phase 1 experiments to run.")
        return

    if split_metadata is None:
        click.echo(f"Screening {len(experiments)} experiments on seasons {seasons}")
    else:
        click.echo(
            f"Screening {len(experiments)} experiments on selection seasons {seasons}; "
            f"outer-eval seasons reserved for final stack: {outer_seasons}"
        )

    proc = Path(data_dir)
    dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)
    df = compute_all_features(df)

    def _screen_at(results_dir: Path) -> list[dict]:
        """Run baseline + screening for one (seed-implicit-via-env-var) call."""
        baseline_path = results_dir / "baseline_scorecard.json"
        baseline_combined: pd.DataFrame | None = None

        if baseline_path.exists() and not use_factored and split_metadata is None:
            baseline_scorecard = json.loads(baseline_path.read_text())
            click.echo(f"  Loaded cached baseline scorecard from {baseline_path}.")
        else:
            if use_factored and baseline_path.exists():
                click.echo(
                    "  Computing baseline profiles (cached scorecard exists, but "
                    "--use-factored requires in-memory profiles)..."
                )
            else:
                click.echo(f"  Computing baseline scorecard at {baseline_path}...")
            baseline_profiles_list = []
            for season in seasons:
                profiles = blend_walk_forward(df, season, retrain_every=retrain_every)
                profiles["season"] = season
                baseline_profiles_list.append(profiles)
            baseline_combined = pd.concat(baseline_profiles_list, ignore_index=True)
            baseline_scorecard = compute_full_scorecard(baseline_combined)
            if split_metadata is not None:
                from bts.experiment.runner import attach_split_metadata

                baseline_scorecard = attach_split_metadata(
                    baseline_scorecard, split_metadata, "selection_only"
                )
            save_scorecard(baseline_scorecard, baseline_path)

        return run_screening(
            experiments, df, baseline_scorecard, seasons,
            results_dir, retrain_every,
            baseline_profiles=baseline_combined if use_factored else None,
            use_factored=use_factored,
            blend_cache_dir=Path(blend_cache_dir) if blend_cache_dir else None,
            split_metadata=split_metadata,
            artifact_role="selection_only",
        )

    if seed_list is None:
        results = _screen_at(RESULTS_BASE / "phase1")
        click.echo(format_phase1_table(results))
    else:
        prev_env = _os.environ.get("BTS_LGBM_RANDOM_STATE")
        try:
            for seed in seed_list:
                _os.environ["BTS_LGBM_RANDOM_STATE"] = str(seed)
                seed_dir = RESULTS_BASE / "phase1" / f"seed_{seed}"
                click.echo(f"\n=== Seed {seed} ===")
                results = _screen_at(seed_dir)
                click.echo(format_phase1_table(results))
        finally:
            if prev_env is None:
                _os.environ.pop("BTS_LGBM_RANDOM_STATE", None)
            else:
                _os.environ["BTS_LGBM_RANDOM_STATE"] = prev_env
        click.echo(
            f"\nMulti-seed run complete: results in {RESULTS_BASE}/phase1/seed_*/. "
            f"Aggregate offline (e.g. via screen_pooled_n10 analysis script)."
        )


@experiment.command("export-candidate-artifacts")
@click.option("--data-dir", default="data/processed", type=click.Path(),
              help="Processed parquet directory")
@click.option("--candidate", required=True,
              help="Experiment name to export against the production baseline")
@click.option("--seasons", default=DEFAULT_TEST_SEASONS,
              help="Comma-separated historical seasons to materialize")
@click.option("--output-dir", required=True, type=click.Path(),
              help="Directory for manifest.json and paired profile parquets")
@click.option("--retrain-every", default=7, type=int)
@click.option("--top-n", default=10, type=int,
              help="Number of ranked candidates to keep per slate")
def export_candidate_artifacts(
    data_dir: str,
    candidate: str,
    seasons: str,
    output_dir: str,
    retrain_every: int,
    top_n: int,
):
    """Export paired production/candidate ranked-slate artifacts."""
    import pandas as pd
    from bts.features.compute import compute_all_features
    from bts.experiment.registry import load_all_experiments, get_experiment
    from bts.experiment.artifacts import materialize_candidate_profile_pair

    load_all_experiments()
    candidate_exp = get_experiment(candidate)
    season_list = _parse_season_list(seasons, "--seasons")

    proc = Path(data_dir)
    dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
    if not dfs:
        raise click.ClickException("No parquet files found. Run 'bts data build' first.")
    df = compute_all_features(pd.concat(dfs, ignore_index=True))

    click.echo(
        f"Exporting {candidate} vs production for seasons {season_list} "
        f"to {output_dir}"
    )
    manifest = materialize_candidate_profile_pair(
        pa_df=df,
        candidate=candidate_exp,
        seasons=season_list,
        output_dir=output_dir,
        retrain_every=retrain_every,
        top_n=top_n,
        data_dir=data_dir,
    )
    click.echo(f"Saved manifest: {Path(output_dir) / 'manifest.json'}")
    for variant, paths_by_season in manifest["profile_paths"].items():
        for season, rel_path in paths_by_season.items():
            click.echo(f"  {variant} {season}: {Path(output_dir) / rel_path}")


@experiment.command("compare-candidate-artifacts")
@click.option("--artifact-dir", required=True, type=click.Path(exists=True),
              help="Directory containing manifest.json from export-candidate-artifacts")
@click.option("--mc-trials", default=10_000, type=int,
              help="Monte Carlo trials for scorecard computation")
@click.option("--season-length", default=180, type=int,
              help="Days per simulated season")
@click.option("--save", "save_path", default=None, type=click.Path(),
              help="Comparison JSON path (default: ARTIFACT_DIR/comparison.json)")
def compare_candidate_artifacts(
    artifact_dir: str,
    mc_trials: int,
    season_length: int,
    save_path: str | None,
):
    """Compare a frozen production/candidate ranked-slate artifact pair."""
    from bts.experiment.artifacts import compare_candidate_profile_pair

    comparison = compare_candidate_profile_pair(
        artifact_dir=artifact_dir,
        mc_trials=mc_trials,
        season_length=season_length,
        save_path=save_path,
    )
    primary_delta = comparison.get("primary_delta")
    if primary_delta is None:
        delta_text = "N/A"
    else:
        delta_text = f"{primary_delta:+.6f}"
    click.echo(f"Saved comparison: {comparison['comparison_path']}")
    click.echo(f"Primary delta ({comparison['primary_metric']}): {delta_text}")


@experiment.command("verify-candidate-artifacts")
@click.option("--artifact-dir", required=True, type=click.Path(exists=True),
              help="Directory containing manifest.json and profile parquets")
@click.option("--expected-run-kind", default=None,
              help="Require a specific manifest run_kind")
@click.option("--expected-candidate", default=None,
              help="Require a specific candidate name")
@click.option("--expected-date", default=None,
              help="Require a specific live artifact date (YYYY-MM-DD)")
@click.option("--expected-git-commit", default=None,
              help="Require the frozen git commit recorded in manifest/profiles")
@click.option("--expected-top-n", default=None, type=int,
              help="Require exactly this many rows per variant/date")
@click.option("--require-live-preoutcome", is_flag=True,
              help="Require live_forward_preoutcome posture and null outcomes")
@click.option("--save", "save_path", default=None, type=click.Path(),
              help="Optional verification report JSON path")
def verify_candidate_artifacts(
    artifact_dir: str,
    expected_run_kind: str | None,
    expected_candidate: str | None,
    expected_date: str | None,
    expected_git_commit: str | None,
    expected_top_n: int | None,
    require_live_preoutcome: bool,
    save_path: str | None,
):
    """Verify paired production/candidate ranked-slate artifacts."""
    from bts.experiment.artifacts import verify_candidate_artifact_pair

    report = verify_candidate_artifact_pair(
        artifact_dir=artifact_dir,
        expected_run_kind=expected_run_kind,
        expected_candidate=expected_candidate,
        expected_date=expected_date,
        expected_git_commit=expected_git_commit,
        expected_top_n=expected_top_n,
        require_live_preoutcome=require_live_preoutcome,
        save_path=save_path,
    )
    status = "PASS" if report["ok"] else "FAIL"
    click.echo(
        f"Candidate artifact verification: {status} "
        f"({report['failure_count']} failures)"
    )
    click.echo(f"Manifest: {Path(artifact_dir) / 'manifest.json'}")
    if save_path is not None:
        click.echo(f"Saved verification: {save_path}")
    if not report["ok"]:
        failed = [check for check in report["checks"] if check["status"] != "pass"]
        for check in failed[:10]:
            detail = f" — {check['detail']}" if check.get("detail") else ""
            click.echo(f"  FAIL {check['name']}{detail}")
        raise click.ClickException("candidate artifact verification failed")


@experiment.command("resolve-live-candidate-artifacts")
@click.option("--artifact-dir", required=True, type=click.Path(exists=True),
              help="Directory containing a live_forward_preoutcome manifest")
@click.option("--output-dir", required=True, type=click.Path(),
              help="Directory for the resolved artifact copy")
@click.option("--data-dir", default="data/processed", type=click.Path(),
              help="Processed parquet directory containing pa_YEAR.parquet")
@click.option("--allow-partial", is_flag=True,
              help="Write a resolved copy even when some outcomes are missing")
@click.option("--overwrite", is_flag=True,
              help="Replace an existing resolved manifest in --output-dir")
@click.option("--save", "save_path", default=None, type=click.Path(),
              help="Optional resolution report JSON path")
def resolve_live_candidate_artifacts(
    artifact_dir: str,
    output_dir: str,
    data_dir: str,
    allow_partial: bool,
    overwrite: bool,
    save_path: str | None,
):
    """Join post-game outcomes onto a live-forward artifact copy."""
    from bts.experiment.artifacts import resolve_live_candidate_artifact_pair

    try:
        report = resolve_live_candidate_artifact_pair(
            artifact_dir=artifact_dir,
            output_dir=output_dir,
            data_dir=data_dir,
            allow_partial=allow_partial,
            overwrite=overwrite,
            save_path=save_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    status = "COMPLETE" if report["complete"] else "PARTIAL"
    click.echo(
        f"Resolved live candidate artifacts: {status} "
        f"({report['missing_count']} missing outcomes)"
    )
    click.echo(f"Source manifest: {report['source_manifest']}")
    click.echo(f"Resolved manifest: {report['resolved_manifest']}")
    if save_path is not None:
        click.echo(f"Saved resolution: {save_path}")
    if report["missing_count"] and not allow_partial:
        raise click.ClickException("live candidate artifact resolution incomplete")


@experiment.command("export-live-candidate-artifacts")
@click.option("--date", required=True, help="Prediction date (YYYY-MM-DD)")
@click.option("--candidate", required=True,
              help="Experiment name to export against the production baseline")
@click.option("--output-dir", required=True, type=click.Path(),
              help="Directory for manifest.json and paired live profile parquets")
@click.option("--data-dir", default="data/processed", type=click.Path(),
              help="Processed parquet directory")
@click.option("--top-n", default=10, type=int,
              help="Number of ranked candidates to keep in the pre-outcome slate")
@click.option("--refresh-data/--no-refresh-data", default=False,
              help="Refresh current-season data before production prediction "
                   "(default: no refresh; run after routine data refresh)")
def export_live_candidate_artifacts(
    date: str,
    candidate: str,
    output_dir: str,
    data_dir: str,
    top_n: int,
    refresh_data: bool,
):
    """Export pre-outcome production/candidate ranked slates for one date."""
    from bts.experiment.registry import load_all_experiments, get_experiment
    from bts.experiment.artifacts import materialize_live_candidate_profile_pair

    load_all_experiments()
    candidate_exp = get_experiment(candidate)
    click.echo(
        f"Exporting live pre-outcome {candidate} vs production for {date} "
        f"to {output_dir}"
    )
    manifest = materialize_live_candidate_profile_pair(
        date=date,
        candidate=candidate_exp,
        output_dir=output_dir,
        data_dir=data_dir,
        top_n=top_n,
        refresh_data=refresh_data,
    )
    click.echo(f"Saved manifest: {Path(output_dir) / 'manifest.json'}")
    for variant, paths_by_key in manifest["profile_paths"].items():
        for key, rel_path in paths_by_key.items():
            click.echo(f"  {variant} {key}: {Path(output_dir) / rel_path}")


@experiment.command()
@click.option("--data-dir", default="data/processed", type=click.Path())
@click.option("--retrain-every", default=7, type=int)
@click.option(
    "--test-seasons",
    default=None,
    help="Comma-separated test seasons for the legacy path (default: 2024,2025)",
)
@click.option(
    "--selection-seasons",
    default=None,
    help="Comma-separated seasons used for opt-in feature/model selection.",
)
@click.option(
    "--outer-eval-seasons",
    default=None,
    help="Comma-separated later seasons reserved for the final outer evaluation.",
)
@click.option(
    "--seeds",
    default=None,
    help="Comma-separated seeds to pool across (e.g. '42,43,44'). "
    "When provided, decisions use mean ΔP(57) across paired seed comparisons "
    "instead of single-seed P(57). Mutually exclusive with --seed-set. "
    "Recommended after 2026-04-28 because single-seed=42 is at the 95th "
    "percentile of the n=100 baseline distribution, creating a P(57) ceiling "
    "that rejects real winners.",
)
@click.option(
    "--seed-set",
    default=None,
    help="Named seed manifest (e.g. 'canonical-n10') loaded from "
    "data/seed_sets/<name>.json. Convenience over --seeds for the "
    "stable canonical sets. Mutually exclusive with --seeds.",
)
@click.option(
    "--keep-t-threshold",
    default=1.5,
    type=float,
    help="Minimum |t-stat| required to keep an experiment in multi-seed mode. "
    "Default 1.5. Ignored in single-seed mode (no t-stat available).",
)
@click.option(
    "--min-effect-size",
    default=None,
    type=float,
    help="Optional escape hatch: keep an experiment regardless of t-stat if "
    "|mean ΔP(57)| >= min-effect-size. Useful when n is small enough that "
    "t-stat is low-power but the effect itself is large.",
)
def select(
    data_dir: str,
    retrain_every: int,
    test_seasons: str | None,
    selection_seasons: str | None,
    outer_eval_seasons: str | None,
    seeds: str | None,
    seed_set: str | None,
    keep_t_threshold: float,
    min_effect_size: float | None,
):
    """Run Phase 2 forward stepwise selection."""
    if seeds and seed_set:
        raise click.UsageError(
            "--seeds and --seed-set are mutually exclusive; pass at most one."
        )
    import pandas as pd
    from bts.features.compute import compute_all_features
    from bts.experiment.registry import load_all_experiments, get_experiment
    from bts.experiment.runner import run_selection, sort_winners_by_p57
    from bts.experiment.reporting import format_phase2_log

    load_all_experiments()
    seasons, outer_seasons, split_metadata = _resolve_season_split(
        test_seasons=test_seasons,
        selection_seasons=selection_seasons,
        outer_eval_seasons=outer_eval_seasons,
    )

    phase1_dir = RESULTS_BASE / "phase1"
    results = []
    for exp_dir in sorted(phase1_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        summary_path = exp_dir / "summary.txt"
        diff_path = exp_dir / "diff.json"
        if not summary_path.exists() or not diff_path.exists():
            continue
        summary_text = summary_path.read_text()
        diff = json.loads(diff_path.read_text())
        results.append({
            "name": exp_dir.name,
            "passed": summary_text.startswith("PASS"),
            "diff": diff,
        })

    winners = sort_winners_by_p57(results)
    if not winners:
        click.echo("No winners from Phase 1. Nothing to select.")
        return

    if split_metadata is None:
        click.echo(f"Forward selection with {len(winners)} winners")
    else:
        click.echo(
            f"Forward selection with {len(winners)} winners on selection seasons {seasons}; "
            f"outer-eval seasons reserved for final stack: {outer_seasons}"
        )

    proc = Path(data_dir)
    dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)
    df = compute_all_features(df)

    experiments_by_name = {}
    for w in winners:
        experiments_by_name[w["name"]] = get_experiment(w["name"])

    seed_list = None
    if seeds:
        seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
        click.echo(f"Multi-seed Phase 2: pooling across {len(seed_list)} seeds")
    elif seed_set:
        manifest_path = Path("data/seed_sets") / f"{seed_set}.json"
        if not manifest_path.exists():
            available = sorted(p.stem for p in Path("data/seed_sets").glob("*.json"))
            raise click.UsageError(
                f"Seed set '{seed_set}' not found at {manifest_path}. "
                f"Available: {available}"
            )
        manifest = json.loads(manifest_path.read_text())
        seed_list = [int(s) for s in manifest["seeds"]]
        click.echo(
            f"Multi-seed Phase 2: pooling across {len(seed_list)} seeds "
            f"from seed-set '{seed_set}'"
        )

    selection_result = run_selection(
        winners, experiments_by_name, df, seasons,
        RESULTS_BASE / "phase2", retrain_every,
        seeds=seed_list,
        keep_t_threshold=keep_t_threshold,
        min_effect_size=min_effect_size,
        outer_eval_seasons=outer_seasons,
        split_metadata=split_metadata,
    )

    click.echo(format_phase2_log(selection_result))


@experiment.command()
def summary():
    """Print results summary across all phases."""
    from bts.experiment.reporting import format_phase1_table, format_phase2_log

    phase1_dir = RESULTS_BASE / "phase1"
    if not phase1_dir.exists():
        click.echo("No Phase 1 results found. Run 'bts experiment screen' first.")
        return

    results = []
    for exp_dir in sorted(phase1_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        diff_path = exp_dir / "diff.json"
        summary_path = exp_dir / "summary.txt"
        if not diff_path.exists():
            continue
        diff = json.loads(diff_path.read_text())
        passed = summary_path.read_text().startswith("PASS") if summary_path.exists() else False
        results.append({"name": exp_dir.name, "passed": passed, "diff": diff})

    if results:
        click.echo(format_phase1_table(results))

    phase2_path = RESULTS_BASE / "phase2" / "forward_selection_log.json"
    if phase2_path.exists():
        sel = json.loads(phase2_path.read_text())
        back_path = RESULTS_BASE / "phase2" / "backward_elimination_log.json"
        backward = json.loads(back_path.read_text()) if back_path.exists() else []
        final_path = RESULTS_BASE / "phase2" / "final_scorecard.json"
        final_sc = json.loads(final_path.read_text()) if final_path.exists() else {}
        click.echo(format_phase2_log({
            "forward_log": sel,
            "backward_log": backward,
            "final_scorecard": final_sc,
            "included": [s["name"] for s in sel if s.get("kept")],
        }))
