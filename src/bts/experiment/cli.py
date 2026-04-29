"""CLI commands for the experiment framework."""

import json
import sys
from pathlib import Path

import click


RESULTS_BASE = Path("experiments/results")


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
@click.option("--test-seasons", default="2024,2025", help="Comma-separated test seasons")
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
def screen(
    data_dir: str,
    subset: str | None,
    retrain_every: int,
    test_seasons: str,
    use_factored: bool,
    blend_cache_dir: str | None,
):
    """Run Phase 1 independent screening."""
    import pandas as pd
    from bts.features.compute import compute_all_features
    from bts.experiment.registry import list_experiments, load_all_experiments, get_experiment
    from bts.experiment.runner import run_screening
    from bts.experiment.reporting import format_phase1_table
    from bts.validate.scorecard import compute_full_scorecard, save_scorecard
    from bts.simulate.backtest_blend import blend_walk_forward

    load_all_experiments()
    seasons = [int(s.strip()) for s in test_seasons.split(",")]

    if subset:
        experiments = [get_experiment(n.strip()) for n in subset.split(",")]
    else:
        experiments = list_experiments(phase=1)

    if not experiments:
        click.echo("No Phase 1 experiments to run.")
        return

    click.echo(f"Screening {len(experiments)} experiments on seasons {seasons}")

    proc = Path(data_dir)
    dfs = [pd.read_parquet(p) for p in sorted(proc.glob("pa_*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)
    df = compute_all_features(df)

    baseline_path = RESULTS_BASE / "phase1" / "baseline_scorecard.json"
    baseline_combined: pd.DataFrame | None = None

    if baseline_path.exists() and not use_factored:
        baseline_scorecard = json.loads(baseline_path.read_text())
        click.echo("Loaded cached baseline scorecard.")
    else:
        # Compute baseline profiles when:
        #   - cached scorecard missing (always required), OR
        #   - use_factored=True (strategy fast path needs profiles in-memory)
        if use_factored and baseline_path.exists():
            click.echo(
                "Computing baseline profiles (cached scorecard exists, but "
                "--use-factored requires in-memory profiles)..."
            )
        else:
            click.echo("Computing baseline scorecard...")
        baseline_profiles_list = []
        for season in seasons:
            profiles = blend_walk_forward(df, season, retrain_every=retrain_every)
            profiles["season"] = season
            baseline_profiles_list.append(profiles)
        baseline_combined = pd.concat(baseline_profiles_list, ignore_index=True)
        baseline_scorecard = compute_full_scorecard(baseline_combined)
        save_scorecard(baseline_scorecard, baseline_path)

    results = run_screening(
        experiments, df, baseline_scorecard, seasons,
        RESULTS_BASE / "phase1", retrain_every,
        baseline_profiles=baseline_combined if use_factored else None,
        use_factored=use_factored,
        blend_cache_dir=Path(blend_cache_dir) if blend_cache_dir else None,
    )

    click.echo(format_phase1_table(results))


@experiment.command()
@click.option("--data-dir", default="data/processed", type=click.Path())
@click.option("--retrain-every", default=7, type=int)
@click.option("--test-seasons", default="2024,2025")
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
    test_seasons: str,
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
    seasons = [int(s.strip()) for s in test_seasons.split(",")]

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

    click.echo(f"Forward selection with {len(winners)} winners")

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
