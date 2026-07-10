import json
import click
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def _today_et() -> str:
    """Today's date (YYYY-MM-DD) in US Eastern — the contest's timezone.

    Default 'today' for scheduler day selection. Deliberately NOT UTC: UTC rolls
    over 4-5h ahead of ET, so a scheduler restart between ~8pm and midnight ET
    would otherwise initialize tomorrow's run_day and abandon tonight's result
    polling + late-slate pick delivery (audit finding O2).
    """
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def _tomorrow_et() -> str:
    """Tomorrow's date (YYYY-MM-DD) in US Eastern — the contest's timezone.

    Default for `bts preview`. Deliberately NOT UTC: between ~8pm and midnight
    ET, UTC has already rolled to tomorrow, so utcnow()+1day targets the day
    AFTER tomorrow and an evening/recovery preview writes the wrong slate
    (GPT-5.6 audit F15 — same class as _today_et / audit finding O2).
    """
    return (datetime.now(ZoneInfo("America/New_York")) + timedelta(days=1)).strftime("%Y-%m-%d")


@click.group()
def cli():
    """Beat the Streak v2 — PA-level MLB hit prediction."""
    pass


from bts.simulate.cli import simulate
cli.add_command(simulate)

from bts.experiment.cli import experiment
cli.add_command(experiment)

from bts.leaderboard.cli import leaderboard
cli.add_command(leaderboard)

from bts.data.backup_cli import backup
cli.add_command(backup)


@cli.group()
def validate():
    """Validation and benchmarking commands."""
    pass


@validate.command()
@click.option("--profiles-dir", default="data/simulation", type=click.Path(exists=True),
              help="Directory with backtest_*.parquet files")
@click.option("--mc-trials", default=10_000, type=int,
              help="Monte Carlo trials for streak simulation")
@click.option("--season-length", default=180, type=int,
              help="Days per simulated season")
@click.option("--save", "save_path", default=None, type=click.Path(),
              help="Save scorecard JSON to this path (default: auto-timestamped)")
@click.option("--diff", "diff_path", default=None, type=click.Path(exists=True),
              help="Baseline scorecard JSON to diff against")
@click.option("--manifest", "manifest_path", default=None, type=click.Path(exists=True),
              help="Split manifest JSON (SOTA #5). When provided, runs per-fold "
                   "scorecard on each fold's holdout slice and saves "
                   "fold_scorecards JSON; lockbox is held out and aggregate "
                   "metrics are deferred.")
def scorecard(
    profiles_dir: str,
    mc_trials: int,
    season_length: int,
    save_path: str | None,
    diff_path: str | None,
    manifest_path: str | None,
):
    """Compute and display the BTS model validation scorecard.

    Loads all backtest_*.parquet files, computes P@K, miss analysis,
    calibration, and streak metrics. Saves a JSON artifact.
    """
    if manifest_path is not None and diff_path is not None:
        raise click.UsageError(
            "--manifest and --diff are mutually exclusive: aggregate-fold "
            "diffing is deferred (SOTA #5 Phase 0/1 produces per-fold "
            "scorecards only; run --diff in non-manifest mode)"
        )

    import json as _json
    from datetime import datetime, timezone
    from rich.console import Console
    from rich.table import Table

    from bts.validate.scorecard import (
        compute_full_scorecard,
        save_scorecard,
        diff_scorecards,
    )

    console = Console()
    profiles_path = Path(profiles_dir)

    # --- Load profiles ---
    parquet_files = sorted(profiles_path.glob("backtest_*.parquet"))
    if not parquet_files:
        click.echo(f"No backtest_*.parquet files found in {profiles_dir}", err=True)
        raise SystemExit(1)

    import pandas as pd

    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        # Infer season from filename (backtest_YYYY.parquet)
        stem = pf.stem  # e.g. "backtest_2025"
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            df["season"] = int(parts[-1])
        dfs.append(df)

    profiles_df = pd.concat(dfs, ignore_index=True)
    console.print(f"[bold]Loaded {len(parquet_files)} profile files "
                  f"({len(profiles_df):,} rows, {profiles_df['date'].nunique()} days)[/bold]")

    # --- Manifest mode (SOTA #5): per-fold scorecards on holdout slices ---
    if manifest_path is not None:
        from bts.validate.scorecard import compute_scorecard_over_manifest
        console.print(f"[bold cyan]Manifest mode:[/bold cyan] {manifest_path}")
        console.print("Computing per-fold scorecards (lockbox held out, aggregate deferred)...")
        manifest_result = compute_scorecard_over_manifest(
            profiles_df, manifest_path,
            mc_trials=mc_trials, season_length=season_length,
        )
        # Display compact per-fold summary
        fold_table = Table(title="Per-Fold Scorecard (manifest mode)")
        fold_table.add_column("Fold", justify="right")
        fold_table.add_column("Train days", justify="right")
        fold_table.add_column("Holdout days", justify="right")
        fold_table.add_column("P@1", justify="right")
        for fs in manifest_result["fold_scorecards"]:
            p1 = fs["scorecard"].get("precision", {}).get(1)
            fold_table.add_row(
                str(fs["fold_idx"]),
                str(fs["train_n_dates"]),
                str(fs["holdout_n_dates"]),
                f"{p1:.1%}" if p1 is not None else "N/A",
            )
        console.print(fold_table)
        console.print(
            f"  [dim]lockbox: {manifest_result['lockbox']['start_date']} .. "
            f"{manifest_result['lockbox']['end_date']} "
            f"({manifest_result['lockbox']['description']})[/dim]"
        )
        # Save and exit before standard scorecard render
        if save_path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            save_path = f"data/validation/scorecard_manifest_{ts}.json"
        from bts.validate.scorecard import save_scorecard
        saved = save_scorecard(manifest_result, save_path)
        console.print(f"\n[green]Manifest scorecard saved to {saved}[/green]")
        return

    # --- Compute scorecard ---
    console.print(f"Computing scorecard (mc_trials={mc_trials:,}, "
                  f"season_length={season_length})...")
    sc = compute_full_scorecard(profiles_df, mc_trials=mc_trials, season_length=season_length)

    # --- Display: P@K table ---
    console.print()
    prec_table = Table(title="Precision @ K")
    prec_table.add_column("K", justify="right")
    prec_table.add_column("P@K", justify="right")
    for k, val in sorted(sc["precision"].items()):
        prec_table.add_row(str(k), f"{val:.1%}")
    console.print(prec_table)

    # --- Display: P@1 by season ---
    if sc.get("p_at_1_by_season"):
        season_table = Table(title="P@1 by Season")
        season_table.add_column("Season", justify="right")
        season_table.add_column("P@1", justify="right")
        for season_key, val in sorted(sc["p_at_1_by_season"].items()):
            season_table.add_row(str(season_key), f"{val:.1%}")
        console.print(season_table)

    # --- Display: P(57) ---
    console.print()
    p57_mc = sc["streak_metrics"].get("p_57_monte_carlo")
    p57_exact = sc.get("p_57_exact")
    p57_mdp = sc.get("p_57_mdp")
    console.print("[bold]P(57) estimates:[/bold]")
    console.print(f"  Monte Carlo ({mc_trials:,} trials): "
                  f"{p57_mc:.4%}" if p57_mc is not None else "  Monte Carlo: N/A")
    console.print(f"  Exact (absorbing chain):  "
                  f"{p57_exact:.4%}" if p57_exact is not None else "  Exact: N/A")
    console.print(f"  MDP optimal:              "
                  f"{p57_mdp:.4%}" if p57_mdp is not None else "  MDP: N/A")

    # --- Display: Miss analysis ---
    console.print()
    ma = sc["miss_analysis"]
    console.print("[bold]Miss Analysis (rank-1):[/bold]")
    console.print(f"  Miss days: {ma['n_miss_days']}")
    if ma.get("rank_2_hit_rate_on_miss") is not None:
        console.print(f"  Rank-2 hit rate on miss days: {ma['rank_2_hit_rate_on_miss']:.1%}")
    if ma.get("mean_p_hit_on_miss") is not None:
        console.print(f"  Mean predicted P(hit) on miss days: {ma['mean_p_hit_on_miss']:.3f}")
    if ma.get("mean_p_hit_on_hit") is not None:
        console.print(f"  Mean predicted P(hit) on hit days:  {ma['mean_p_hit_on_hit']:.3f}")

    # --- Display: Streak distribution ---
    console.print()
    sm = sc["streak_metrics"]
    streak_table = Table(title="Streak Distribution (Monte Carlo)")
    streak_table.add_column("Metric")
    streak_table.add_column("Value", justify="right")
    streak_table.add_row("Mean max streak", f"{sm['mean_max_streak']:.1f}")
    streak_table.add_row("Median max streak", str(sm["median_max_streak"]))
    streak_table.add_row("P90 max streak", str(sm["p90_max_streak"]))
    streak_table.add_row("P99 max streak", str(sm["p99_max_streak"]))
    streak_table.add_row("Longest replay streak", str(sm["longest_replay_streak"]))
    console.print(streak_table)

    # --- Display: Probabilistic forecast evaluation (SOTA #12) ---
    ps = sc.get("proper_scoring")
    if ps:
        console.print()
        ps_table = Table(title="Probabilistic Scoring (proper scores + top-bin calibration)")
        ps_table.add_column("Metric")
        ps_table.add_column("all_top10", justify="right")
        ps_table.add_column("rank1", justify="right")
        for label, key, fmt in [
            ("n", "n", "{:d}"),
            ("Log loss", "log_loss", "{:.4f}"),
            ("Brier score", "brier", "{:.4f}"),
        ]:
            ps_table.add_row(label, fmt.format(ps["all_top10"][key]), fmt.format(ps["rank1"][key]))
        for label, section, key, fmt in [
            ("Reliability", "decomposition", "reliability", "{:.4f}"),
            ("Resolution", "decomposition", "resolution", "{:.4f}"),
            ("Uncertainty", "decomposition", "uncertainty", "{:.4f}"),
            ("Top-bin mean p", "top_bin", "mean_p", "{:.4f}"),
            ("Top-bin mean y", "top_bin", "mean_y", "{:.4f}"),
            ("Top-bin gap (p − y)", "top_bin", "gap", "{:+.4f}"),
            ("Top-bin n", "top_bin", "n", "{:d}"),
        ]:
            ps_table.add_row(
                label,
                fmt.format(ps["all_top10"][section][key]),
                fmt.format(ps["rank1"][section][key]),
            )
        console.print(ps_table)
        meta = ps["metadata"]
        console.print(
            f"  [dim]bins={meta['n_bins']} ({meta['binning']}), "
            f"intervals={meta['interval_method']}[/dim]"
        )

    # --- Save ---
    if save_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        save_path = f"data/validation/scorecard_{ts}.json"
    saved = save_scorecard(sc, save_path)
    console.print(f"\n[green]Scorecard saved to {saved}[/green]")

    # --- Diff ---
    if diff_path:
        baseline = _json.loads(Path(diff_path).read_text())
        diffs = diff_scorecards(baseline, sc)
        console.print()
        diff_table = Table(title=f"Delta vs baseline: {diff_path}")
        diff_table.add_column("Field")
        diff_table.add_column("Baseline", justify="right")
        diff_table.add_column("Variant", justify="right")
        diff_table.add_column("Delta", justify="right")

        def _add_diff_rows(section_label: str, diff_dict: dict):
            for field, d in diff_dict.items():
                label = f"{section_label}.{field}" if section_label else str(field)
                delta_str = f"{d['delta']:+.4f}"
                color = "green" if d["delta"] > 0 else "red" if d["delta"] < 0 else ""
                colored_delta = f"[{color}]{delta_str}[/{color}]" if color else delta_str
                diff_table.add_row(
                    label,
                    f"{d['baseline']:.4f}",
                    f"{d['variant']:.4f}",
                    colored_delta,
                )

        for key, val in diffs.items():
            if isinstance(val, dict) and "delta" in val:
                # Top-level scalar diff (p_57_exact, p_57_mdp)
                _add_diff_rows("", {key: val})
            elif isinstance(val, dict):
                _add_diff_rows(key, val)

        console.print(diff_table)


@validate.command("split-manifest")
@click.option("--profiles-dir", default="data/simulation", type=click.Path(exists=True),
              help="Directory with backtest_*.parquet files")
@click.option("--lockbox-season", default=None, type=int,
              help="Season to source the lockbox from (default: latest tracked "
                   "complete season, skipping any partial in-progress season)")
@click.option("--lockbox-game-days", default=30, type=int,
              help="Number of game-days at end of lockbox season to reserve")
@click.option("--n-folds", default=5, type=int)
@click.option("--purge-game-days", default=7, type=int)
@click.option("--embargo-game-days", default=7, type=int,
              help="Recorded in manifest; only meaningful in deferred symmetric mode")
@click.option("--min-train-game-days", default=365, type=int,
              help="Floor on first fold's train size (game-days)")
@click.option("--min-complete-season-dates", default=150, type=int,
              help="Threshold for considering a season 'complete' when "
                   "auto-resolving the default lockbox season")
@click.option("--output", required=True, type=click.Path(),
              help="Output JSON path for the split manifest")
def split_manifest_cmd(
    profiles_dir: str,
    lockbox_season: int | None,
    lockbox_game_days: int,
    n_folds: int,
    purge_game_days: int,
    embargo_game_days: int,
    min_train_game_days: int,
    min_complete_season_dates: int,
    output: str,
):
    """Build a purged blocked CV split manifest with a reserved lockbox.

    Reads backtest_*.parquet files, resolves the lockbox season (skipping
    any partial in-progress season unless --lockbox-season is explicit),
    carves the last `--lockbox-game-days` of that season as the lockbox,
    and writes a deterministic manifest JSON.
    """
    from rich.console import Console
    import pandas as pd
    from bts.validate.splits import (
        make_purged_blocked_cv,
        save_manifest,
        resolve_default_lockbox_season,
        default_lockbox_for_season,
        collect_universe_dates,
    )

    console = Console()
    profiles_path = Path(profiles_dir)
    parquet_files = sorted(profiles_path.glob("backtest_*.parquet"))
    if not parquet_files:
        click.echo(f"No backtest_*.parquet files found in {profiles_dir}", err=True)
        raise SystemExit(1)

    # Group dates by season inferred from filename
    dates_by_season: dict[int, list] = {}
    for pf in parquet_files:
        stem = pf.stem
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            season = int(parts[-1])
        else:
            continue
        df = pd.read_parquet(pf, columns=["date"])
        season_dates = sorted({d.date() if hasattr(d, "date") else d
                                for d in pd.to_datetime(df["date"])})
        dates_by_season[season] = season_dates

    if not dates_by_season:
        click.echo("No seasons inferred from filenames", err=True)
        raise SystemExit(1)

    # Resolve lockbox season
    if lockbox_season is None:
        try:
            lockbox_season = resolve_default_lockbox_season(
                dates_by_season,
                min_complete_season_dates=min_complete_season_dates,
            )
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    lockbox = default_lockbox_for_season(
        dates_by_season, lockbox_season, n_game_days=lockbox_game_days
    )

    # Build universe restricted to dates <= lockbox.end_date by default.
    # This prevents partial-current-season files (e.g., an in-progress 2026
    # backtest beside complete 2025) from leaking post-lockbox dates into
    # any fold's train or holdout.
    universe = collect_universe_dates(dates_by_season, lockbox)

    folds = make_purged_blocked_cv(
        universe,
        n_folds=n_folds,
        purge_game_days=purge_game_days,
        embargo_game_days=embargo_game_days,
        min_train_game_days=min_train_game_days,
        lockbox=lockbox,
    )

    saved = save_manifest(
        folds, lockbox, output,
        purge_game_days=purge_game_days,
        embargo_game_days=embargo_game_days,
        min_train_game_days=min_train_game_days,
        mode="rolling_origin",
        universe_dates=universe,
    )

    console.print(f"[bold]Split manifest saved to {saved}[/bold]")
    console.print(
        f"  Lockbox season: [cyan]{lockbox_season}[/cyan]  "
        f"(range {lockbox.start_date} .. {lockbox.end_date}, "
        f"{lockbox_game_days} game-days)"
    )
    console.print(
        f"  Folds: {n_folds}  "
        f"purge={purge_game_days}gd  embargo={embargo_game_days}gd  "
        f"min_train={min_train_game_days}gd"
    )
    console.print(
        f"  Universe: {len(universe)} dates ({universe[0]} .. {universe[-1]})"
    )


@validate.command("conformal-gate")
@click.option("--profiles-dir", default="data/simulation", type=click.Path(exists=True),
              help="Directory with backtest_*.parquet files")
@click.option("--manifest", "manifest_path", required=True, type=click.Path(exists=True),
              help="Split manifest JSON (from `bts validate split-manifest`)")
@click.option("--methods", default="bucket_wilson,weighted_mondrian_conformal",
              help="Comma-separated calibrator methods")
@click.option("--alphas", default="0.05,0.10,0.20",
              help="Comma-separated significance levels (1 - target coverage)")
@click.option("--bucket-width", default=0.025, type=float)
@click.option("--min-bucket-n", default=30, type=int)
@click.option("--validity-tolerance", default=0.01, type=float)
@click.option("--tightness-threshold", default=0.30, type=float)
@click.option("--wilson-alpha", default=0.05, type=float,
              help="One-sided Wilson alpha for the OBSERVED-rate CI in the gate")
@click.option("--output", required=True, type=click.Path(),
              help="Output JSON path for the v2 gate result")
def conformal_gate_cmd(
    profiles_dir: str,
    manifest_path: str,
    methods: str,
    alphas: str,
    bucket_width: float,
    min_bucket_n: int,
    validity_tolerance: float,
    tightness_threshold: float,
    wilson_alpha: float,
    output: str,
):
    """Run the redesigned conformal-lower-bound gate (SOTA #11 P0/P1).

    Replaces the broken-for-binary-y per-row coverage metric with per-bucket
    lower-bound calibration validity (Wilson_lower >= mean_bound - tolerance).
    Composes #5 manifest (lockbox held out). #12 reliability machinery is
    used as DIAGNOSTICS only, not as a shipping gate.
    """
    import pandas as pd
    from rich.console import Console
    from rich.table import Table
    from bts.validate.conformal_gate import run_gate_matrix

    console = Console()
    profiles_path = Path(profiles_dir)
    parquet_files = sorted(profiles_path.glob("backtest_*.parquet"))
    if not parquet_files:
        click.echo(f"No backtest_*.parquet files in {profiles_dir}", err=True)
        raise SystemExit(1)
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        stem = pf.stem
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            df["season"] = int(parts[-1])
        dfs.append(df)
    profiles_df = pd.concat(dfs, ignore_index=True)
    console.print(f"[bold]Loaded {len(profiles_df):,} profile rows from "
                  f"{len(parquet_files)} files[/bold]")

    methods_list = tuple(m.strip() for m in methods.split(","))
    alphas_list = tuple(float(a.strip()) for a in alphas.split(","))

    result = run_gate_matrix(
        profiles_df, manifest_path,
        methods=methods_list, alphas=alphas_list,
        bucket_width=bucket_width, min_bucket_n=min_bucket_n,
        validity_tolerance=validity_tolerance,
        tightness_threshold=tightness_threshold,
        wilson_alpha=wilson_alpha,
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    matrix_table = Table(title=f"Conformal Gate v2 — {result['verdict']}")
    matrix_table.add_column("Method")
    matrix_table.add_column("Alpha", justify="right")
    matrix_table.add_column("Verdict", justify="center")
    matrix_table.add_column("Fail reasons (head)", overflow="fold")
    for cell_key, cell in result["method_alpha_matrix"].items():
        verdict_color = {
            "PASS": "green",
            "FAIL": "red",
            "INSUFFICIENT_DATA": "yellow",
        }.get(cell["verdict"], "white")
        reasons = "; ".join(cell["fail_reasons"][:2]) if cell["fail_reasons"] else ""
        matrix_table.add_row(
            cell["method"],
            f"{cell['alpha']:.2f}",
            f"[{verdict_color}]{cell['verdict']}[/{verdict_color}]",
            reasons,
        )
    console.print(matrix_table)
    console.print(
        f"  [dim]lockbox: {result['lockbox']['start_date']} .. "
        f"{result['lockbox']['end_date']} | "
        f"ship_set: {len(result['ship_set'])} cell(s)[/dim]"
    )
    console.print(f"\n[green]Gate output saved to {out_path}[/green]")


@validate.command("policy-value-eval")
@click.option("--profiles-dir", default="data/simulation", type=click.Path(exists=True),
              help="Directory with backtest_*.parquet files")
@click.option("--manifest", "manifest_path", required=True, type=click.Path(exists=True),
              help="Split manifest JSON (from `bts validate split-manifest`)")
@click.option("--target-policy", type=click.Choice(["mdp_optimal", "always_skip", "always_rank1"]),
              default="mdp_optimal",
              help="Target policy to evaluate (P0/P1 supports 3 baselines; named-strategy adapter deferred to P1.5+)")
@click.option("--season-length", default=180, type=int)
@click.option("--late-phase-days", default=30, type=int)
@click.option("--min-bin-n", default=200, type=int,
              help="Diagnostic threshold for SPARSE_HOLDOUT_SUPPORT flag")
@click.option("--n-bins", default=5, type=int)
@click.option("--output", required=True, type=click.Path(),
              help="Output JSON path for the policy_value_eval_v1 result")
def policy_value_eval_cmd(
    profiles_dir: str,
    manifest_path: str,
    target_policy: str,
    season_length: int,
    late_phase_days: int,
    min_bin_n: int,
    n_bins: int,
    output: str,
):
    """Run the per-fold policy-value evaluation contract (SOTA #13 P0/P1).

    Solves the target policy on each fold's train slice (or uses a baseline
    table for always_skip / always_rank1), evaluates the fixed policy against
    fold holdout bins via `evaluate_mdp_policy`, computes terminal-MC replay
    on holdout profiles as a cross-check. Lockbox held out per #5;
    aggregate_deferred=true.
    """
    import pandas as pd
    from rich.console import Console
    from rich.table import Table
    from bts.validate.ope_eval import evaluate_target_policy_on_manifest

    console = Console()
    profiles_path = Path(profiles_dir)
    parquet_files = sorted(profiles_path.glob("backtest_*.parquet"))
    if not parquet_files:
        click.echo(f"No backtest_*.parquet files in {profiles_dir}", err=True)
        raise SystemExit(1)
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        stem = pf.stem
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            df["season"] = int(parts[-1])
        dfs.append(df)
    profiles_df = pd.concat(dfs, ignore_index=True)
    console.print(f"[bold]Loaded {len(profiles_df):,} profile rows from "
                  f"{len(parquet_files)} files[/bold]")

    result = evaluate_target_policy_on_manifest(
        profiles_df, manifest_path,
        target_policy_name=target_policy,
        season_length=season_length,
        late_phase_days=late_phase_days,
        min_bin_n=min_bin_n,
        n_bins=n_bins,
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Strict JSON guard (Codex #111 #3): allow_nan=False fails closed if a
    # future metric becomes non-finite, surfacing the bug rather than emitting
    # the non-strict 'Infinity'/'NaN' tokens.
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str, allow_nan=False)
    )

    fold_table = Table(title=f"Policy-Value Eval — target_policy={target_policy}")
    fold_table.add_column("Fold", justify="right")
    fold_table.add_column("Train days", justify="right")
    fold_table.add_column("Holdout days", justify="right")
    fold_table.add_column("V_pi", justify="right")
    fold_table.add_column("V_replay", justify="right")
    fold_table.add_column("|disagree|", justify="right")
    fold_table.add_column("flag", justify="center")
    for fr in result["fold_results"]:
        fold_table.add_row(
            str(fr["fold_idx"]),
            str(fr["n_train_dates"]),
            str(fr["n_holdout_dates"]),
            f"{fr['V_pi']:.4f}",
            f"{fr['V_replay']:.4f}",
            f"{fr['disagreement_abs']:.4f}",
            fr["sparse_support"]["verdict_flag"],
        )
    console.print(fold_table)
    console.print(
        f"  [dim]lockbox: {result['lockbox']['start_date']} .. "
        f"{result['lockbox']['end_date']} | "
        f"aggregate_deferred=true[/dim]"
    )
    console.print(f"\n[green]Policy-value-eval output saved to {out_path}[/green]")


@validate.command("rare-event-ce-is")
@click.option("--profiles-dir", default="data/simulation", type=click.Path(exists=True),
              help="Directory with backtest_*.parquet files")
@click.option("--manifest", "manifest_path", required=True, type=click.Path(exists=True),
              help="Split manifest JSON (from `bts validate split-manifest`)")
@click.option("--n-rounds-train", default=8, type=int,
              help="CE rounds on fold-train profiles to learn theta")
@click.option("--n-per-round-train", default=5000, type=int)
@click.option("--n-final-train", default=2000, type=int,
              help="Final IS sample size on TRAIN (point estimate is discarded; "
                   "smaller default than holdout since we only keep theta_final)")
@click.option("--n-final-holdout", default=20000, type=int,
              help="Final IS sample size on HOLDOUT for the fixed-window estimate")
@click.option("--seed", default=42, type=int)
@click.option("--streak-threshold", default=57, type=int,
              help="Threshold for the rare event (default 57 to match exact_p57's "
                   "hard-coded absorbing state)")
@click.option("--min-ess", default=1000.0, type=float,
              help="Diagnostic threshold (NOT a gate)")
@click.option("--max-weight-share", default=0.1, type=float,
              help="Diagnostic threshold (NOT a gate)")
@click.option("--output", required=True, type=click.Path(),
              help="Output JSON path for the rare_event_ce_is_v1 result")
def rare_event_ce_is_cmd(
    profiles_dir: str,
    manifest_path: str,
    n_rounds_train: int,
    n_per_round_train: int,
    n_final_train: int,
    n_final_holdout: int,
    seed: int,
    streak_threshold: int,
    min_ess: float,
    max_weight_share: float,
    output: str,
):
    """Run the per-fold CE-IS rare-event evaluation contract (SOTA #14 P0/P1).

    Black-box wrapper of `bts.simulate.rare_event_mc.estimate_p57_with_ceis`
    over a #5 manifest. Per fold: CE rounds learn theta on fold-train
    profiles (the train point estimate is discarded — there is no public
    tune-only API in v1); final IS estimate runs on fold-holdout profiles
    with `n_rounds=0, theta=train_theta`. Lockbox held out per #5;
    aggregate_deferred=true.

    Estimand: P(max consecutive rank-1 hits >= streak_threshold) over the
    ordered fold-holdout date sequence under independent Bernoulli rank-1
    hits. Horizon = n_holdout_dates per fold. NOT comparable to #13's V_pi.
    """
    import pandas as pd
    from rich.console import Console
    from rich.table import Table
    from bts.validate.rare_event_mc_eval import evaluate_ceis_on_manifest

    console = Console()
    profiles_path = Path(profiles_dir)
    parquet_files = sorted(profiles_path.glob("backtest_*.parquet"))
    if not parquet_files:
        click.echo(f"No backtest_*.parquet files in {profiles_dir}", err=True)
        raise SystemExit(1)
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        stem = pf.stem
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            df["season"] = int(parts[-1])
        dfs.append(df)
    profiles_df = pd.concat(dfs, ignore_index=True)
    console.print(f"[bold]Loaded {len(profiles_df):,} profile rows from "
                  f"{len(parquet_files)} files[/bold]")

    result = evaluate_ceis_on_manifest(
        profiles_df, manifest_path,
        n_rounds_train=n_rounds_train,
        n_per_round_train=n_per_round_train,
        n_final_train=n_final_train,
        n_final_holdout=n_final_holdout,
        seed=seed,
        streak_threshold=streak_threshold,
        min_ess=min_ess,
        max_weight_share_threshold=max_weight_share,
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Strict JSON — fails closed if a metric becomes non-finite
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str, allow_nan=False)
    )

    fold_table = Table(
        title=f"CE-IS Rare-Event MC — streak_threshold={streak_threshold}, "
              f"horizon=n_holdout_dates"
    )
    fold_table.add_column("Fold", justify="right")
    fold_table.add_column("Train days", justify="right")
    fold_table.add_column("Holdout days", justify="right")
    fold_table.add_column("Estimate", justify="right")
    fold_table.add_column("CI [lo, hi]", justify="right")
    fold_table.add_column("ESS", justify="right")
    fold_table.add_column("MaxWS", justify="right")
    fold_table.add_column("flag", justify="center")
    for fr in result["fold_results"]:
        d = fr["diagnostics"]
        fold_table.add_row(
            str(fr["fold_idx"]),
            str(fr["n_train_dates"]),
            str(fr["n_holdout_dates"]),
            f"{fr['fixed_window_estimate']:.4e}",
            f"[{fr['ci_lower']:.2e}, {fr['ci_upper']:.2e}]",
            f"{d['ess']:.0f}",
            f"{d['max_weight_share']:.3f}",
            d["verdict_flag"],
        )
    console.print(fold_table)
    console.print(
        f"  [dim]lockbox: {result['lockbox']['start_date']} .. "
        f"{result['lockbox']['end_date']} | "
        f"aggregate_deferred=true | "
        f"NOT comparable to #13 V_pi[/dim]"
    )
    console.print(f"\n[green]CE-IS eval output saved to {out_path}[/green]")


@validate.command("falsification-harness")
@click.option("--profiles-glob", default="data/simulation/profiles_seed*_season*.parquet",
              help="Glob for v2.5+ profile parquets (must contain a 'season' column)")
@click.option("--pa-glob", default="data/simulation/pa_predictions_*.parquet",
              help="Glob for PA-level prediction parquets")
@click.option("--output", default="data/validation/falsification_harness.json",
              type=click.Path(), help="Output verdict JSON path")
@click.option("--n-bootstrap", default=2000, type=int,
              help="Bootstrap replicates for OPE CIs and dependence CIs")
@click.option("--n-final", default=20000, type=int,
              help="Final IS sample size for CE-IS rare-event MC")
@click.option("--headline-p57", default=0.0817, type=float,
              help="In-sample headline P(57) to defend (default: 8.17%)")
@click.option("--n-block-bootstrap", default=0, type=int,
              help="Profile-level block-bootstrap replicates for pooled CI (default 0 = use 5-fold percentile).")
@click.option("--expected-block-length", default=7, type=int,
              help="Mean block length (days) for stationary bootstrap when --n-block-bootstrap > 0.")
def falsification_harness_cmd(
    profiles_glob, pa_glob, output, n_bootstrap, n_final, headline_p57,
    n_block_bootstrap, expected_block_length,
):
    """Run the BTS 8.17% falsification harness.

    Wires DR-OPE (fixed-policy + pipeline), CE-IS rare-event MC, and
    PA + cross-game dependence diagnostics into a single verdict JSON.
    See data/validation/falsification_harness.json for output.
    """
    import pandas as pd
    from scripts.run_falsification_harness import run_harness

    profile_paths = sorted(Path().glob(profiles_glob))
    pa_paths = sorted(Path().glob(pa_glob))
    if not profile_paths:
        raise click.ClickException(f"No profiles found matching: {profiles_glob}")
    if not pa_paths:
        raise click.ClickException(f"No PA files found matching: {pa_glob}")

    profiles = pd.concat(pd.read_parquet(p) for p in profile_paths)
    pa_df = pd.concat(pd.read_parquet(p) for p in pa_paths)
    out = run_harness(
        profiles, pa_df,
        output_path=Path(output),
        headline_p57_in_sample=headline_p57,
        n_bootstrap=n_bootstrap,
        n_final=n_final,
        n_block_bootstrap=n_block_bootstrap,
        expected_block_length=expected_block_length,
    )
    click.echo(json.dumps(out, indent=2))


@cli.group()
def data():
    """Data pipeline commands."""
    pass


@data.command()
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--data-dir", default="data/raw", type=click.Path(), help="Output directory")
@click.option("--delay", default=0.5, type=float, help="Seconds between API requests")
def pull(start: str, end: str, data_dir: str, delay: float):
    """Pull game feeds from MLB Stats API."""
    from bts.data.pull import pull_feeds

    output = Path(data_dir)
    click.echo(f"Pulling games from {start} to {end} into {output}/")
    paths = pull_feeds(start, end, output, delay=delay)
    click.echo(f"Done. {len(paths)} game feeds downloaded.")


@data.command()
@click.option("--seasons", required=True, help="Comma-separated seasons (e.g., 2023,2024,2025)")
@click.option("--raw-dir", default="data/raw", type=click.Path(), help="Raw data directory")
@click.option("--out-dir", default="data/processed", type=click.Path(), help="Output directory")
def build(seasons: str, raw_dir: str, out_dir: str):
    """Build PA-level Parquet from raw game feeds."""
    from bts.data.build import build_season

    raw = Path(raw_dir)
    out = Path(out_dir)

    for season_str in seasons.split(","):
        season = int(season_str.strip())
        output_path = out / f"pa_{season}.parquet"
        click.echo(f"Building {output_path} from {raw}/{season}/...")
        df = build_season(raw, output_path, season)
        click.echo(f"  {len(df)} plate appearances written.")


@data.command(name="enrich-weather")
@click.option("--data-dir", default="data/raw", type=click.Path(), help="Raw data directory")
@click.option("--seasons", required=True, help="Comma-separated seasons (e.g., 2023,2024,2025)")
@click.option("--delay", default=0.3, type=float, help="Seconds between API requests")
def enrich_weather_cmd(data_dir: str, seasons: str, delay: float):
    """Enrich game feeds with atmospheric data from Open-Meteo."""
    from bts.data.pull import enrich_weather

    raw = Path(data_dir)
    for season_str in seasons.split(","):
        season = int(season_str.strip())
        season_dir = raw / str(season)
        if not season_dir.exists():
            click.echo(f"Skipping {season}: no raw data at {season_dir}")
            continue
        click.echo(f"Enriching {season} weather data...")
        count = enrich_weather(season_dir, delay=delay)
        click.echo(f"  {count} games enriched.")


@data.command(name="collect-lineup-times")
@click.option("--date", default=None, help="Date (YYYY-MM-DD, default today ET)")
@click.option("--out-dir", default="data/lineup_posting_times", type=click.Path(),
              help="Output directory for JSONL state files")
def data_collect_lineup_times(date, out_dir):
    """Poll MLB API once for lineup confirmation times on the given date.

    Designed to be called every 5 minutes via systemd timer or cron.
    Each call is a single poll pass across all games that still need
    confirmation. JSONL file is updated in place with accumulating data.
    """
    from datetime import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo
    from bts.data.lineup_collect import collect_for_date

    if date is None:
        date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    state = collect_for_date(date=date, out_dir=Path(out_dir))
    n_both = sum(
        1 for g in state.games.values()
        if g.first_away_confirmed_utc and g.first_home_confirmed_utc
    )
    click.echo(f"{date}: {n_both}/{len(state.games)} games fully confirmed")


@data.command(name="analyze-lineup-times")
@click.option("--in-dir", default="data/lineup_posting_times", type=click.Path())
@click.option("--from-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--to-date", required=True, help="End date (YYYY-MM-DD)")
def data_analyze_lineup_times(in_dir, from_date, to_date):
    """Report lineup-posting-time distribution for a date range.

    Prints percentiles and a short histogram-style summary. Use to inform
    scheduler timing configuration (lineup_check_offset_min, fallback_deadline_min).
    """
    from pathlib import Path
    from bts.data.lineup_analyze import load_samples_from_jsonl, compute_distribution

    samples = load_samples_from_jsonl(Path(in_dir), from_date, to_date)
    dist = compute_distribution(samples)

    click.echo(f"Lineup posting time distribution ({from_date} to {to_date})")
    click.echo(f"  n = {dist.n} samples")
    if dist.n == 0:
        click.echo("  (no samples — check data/lineup_posting_times/ has data for this range)")
        return
    click.echo(f"  mean   = {dist.mean:.0f} min before first pitch")
    click.echo(f"  p10    = {dist.p10:.0f}")
    click.echo(f"  p25    = {dist.p25:.0f}")
    click.echo(f"  p50    = {dist.p50:.0f}")
    click.echo(f"  p75    = {dist.p75:.0f}")
    click.echo(f"  p90    = {dist.p90:.0f}")
    click.echo(f"  p95    = {dist.p95:.0f}")
    click.echo(f"  p99    = {dist.p99:.0f}")
    click.echo("")
    click.echo("Interpretation:")
    click.echo(f"  To capture p95 of lineups at lock time, use lineup_check_offset_min >= {int(dist.p95) + 5}")
    click.echo(f"  For fallback_deadline_min, accept up to p90 ({int(dist.p90)}) loss of confirmed data")


@data.command(name="backfill-lineup-times")
@click.option("--picks-dir", default="data/picks", type=click.Path(exists=True))
def data_backfill_lineup_times(picks_dir):
    """Extract coarse lineup-time samples from existing Pi5 scheduler state.

    Coarse (5-15 min resolution) but real data to bootstrap the distribution
    analysis before the collection script has accumulated a week of data.
    Combine output with results from 'bts data analyze-lineup-times'.
    """
    from pathlib import Path
    from bts.data.lineup_analyze import backfill_from_scheduler_state, compute_distribution

    samples = backfill_from_scheduler_state(Path(picks_dir))
    dist = compute_distribution(samples)
    click.echo(f"Bootstrap from Pi5 scheduler state: n={dist.n}")
    if dist.n:
        click.echo(f"  p50={dist.p50:.0f}, p90={dist.p90:.0f}, p95={dist.p95:.0f}")


@data.command(name="sync-to-r2")
@click.option("--processed-dir", default="data/processed", type=click.Path())
@click.option("--models-dir", default="data/models", type=click.Path())
@click.option("--prune/--no-prune", "do_prune", default=True,
              help="After a successful sync, delete unreferenced content-addressed "
                   "objects older than 7 days (age guard protects in-flight syncs)")
def data_sync_to_r2(processed_dir, models_dir, do_prune):
    """Upload local parquets + lookup cache to R2, atomically updating manifest."""
    from pathlib import Path
    from bts.data.sync import R2Client, prune_unreferenced, sync_to_r2

    processed = Path(processed_dir)
    models = Path(models_dir)
    if not processed.exists():
        raise click.ClickException(
            f"Directory {processed} does not exist. Run from the BTS repo root, "
            f"or pass --processed-dir to override."
        )
    if not models.exists():
        raise click.ClickException(
            f"Directory {models} does not exist. Run from the BTS repo root, "
            f"or pass --models-dir to override."
        )

    client = R2Client.from_env()
    manifest = sync_to_r2(
        client=client,
        processed_dir=processed,
        models_dir=models,
    )
    click.echo(f"Sync complete: {len(manifest['files'])} files, schema={manifest['schema_version']}")
    if do_prune:
        report = prune_unreferenced(client)
        click.echo(
            f"Prune: {len(report['deleted'])} unreferenced objects deleted, "
            f"{len(report['kept_recent'])} kept (age guard)"
        )


@data.command(name="sync-from-r2")
@click.option("--processed-dir", default="data/processed", type=click.Path())
@click.option("--models-dir", default="data/models", type=click.Path())
def data_sync_from_r2(processed_dir, models_dir):
    """Download parquets + lookup cache from R2, verifying checksums."""
    from pathlib import Path
    from bts.data.sync import R2Client, sync_from_r2

    client = R2Client.from_env()
    manifest = sync_from_r2(
        client=client,
        processed_dir=Path(processed_dir),
        models_dir=Path(models_dir),
    )
    click.echo(
        f"Sync complete: {len(manifest['files'])} files, "
        f"git_sha={manifest.get('git_sha', 'unknown')[:12]}"
    )


@data.command(name="verify-manifest")
def data_verify_manifest():
    """Check R2 manifest state without modifying anything (tripwire mode)."""
    from bts.data.sync import R2Client, verify_manifest

    client = R2Client.from_env()
    report = verify_manifest(client)
    if not report["exists"]:
        click.echo("Manifest not found in R2.", err=True)
        raise SystemExit(2)
    if not report.get("version_supported", True):
        click.echo("Manifest version unsupported.", err=True)
        raise SystemExit(2)
    click.echo(f"branch:         {report['branch']}")
    click.echo(f"git_sha:        {report['git_sha']}")
    click.echo(f"schema_version: {report['schema_version']} "
               f"{'OK' if report['schema_version_match'] else 'MISMATCH'}")
    age_str = f"{report['age_hours']:.1f}h ago" if report.get('age_hours') is not None else "unknown age"
    click.echo(f"updated_at:     {report['updated_at']} ({age_str})")
    click.echo(f"n_files:        {report['n_files']}")
    click.echo(f"stale:          {report['stale']}")
    click.echo(f"objects_ok:     {report['objects_ok']} (existence+size only; "
               f"restores verify sha256)"
               + (f" (missing: {report['objects_missing']}, "
                  f"size mismatch: {report['objects_size_mismatch']})"
                  if not report['objects_ok'] else ""))
    if report['stale'] or not report['schema_version_match'] or not report['objects_ok']:
        raise SystemExit(1)


@data.command(name="archive-historical-raw")
@click.option("--raw-dir", default="data/raw", type=click.Path(exists=True))
@click.option("--exclude-season", multiple=True, type=int, default=None,
              help="Seasons to exclude (defaults to current year)")
@click.option("--tarball-key", default=None,
              help="R2 key for the archive (defaults to raw-archive-2017-{last year}.tar.gz)")
def data_archive_historical_raw(raw_dir, exclude_season, tarball_key):
    """One-shot: tar historical raw JSON and upload to R2 as cold archive."""
    from datetime import datetime
    from pathlib import Path
    from bts.data.sync import R2Client, archive_historical_raw

    current_year = datetime.now().year
    if not exclude_season:
        exclude_season = (current_year,)
    if tarball_key is None:
        tarball_key = f"raw-archive-2017-{current_year - 1}.tar.gz"

    client = R2Client.from_env()
    archive_historical_raw(
        client=client,
        raw_dir=Path(raw_dir),
        tarball_key=tarball_key,
        exclude_seasons=set(exclude_season),
    )
    click.echo(f"Archive uploaded: {tarball_key}")


@cli.command()
@click.option("--date", required=True, help="Date to predict (YYYY-MM-DD)")
@click.option("--data-dir", default="data/processed", type=click.Path(), help="Processed data directory")
@click.option("--picks-dir", default="data/picks", type=click.Path(), help="Picks output directory")
@click.option("--models-dir", default="data/models", type=click.Path(), help="Cached models directory")
@click.option("--top", default=10, type=int, help="Number of ranked picks to show")
@click.option("--dry-run", is_flag=True, help="Print rankings only — don't save pick or post to Bluesky")
def run(date: str, data_dir: str, picks_dir: str, models_dir: str, top: int, dry_run: bool):
    """Run daily BTS automation: predict, save pick, post to Bluesky.

    Picks the highest-ranked batter from the 12-model blend.
    MDP policy determines skip/single/double.
    Use --dry-run to preview rankings without saving or posting.
    """
    import pandas as pd
    from datetime import datetime, timezone
    from bts.model.predict import run_pipeline, save_blend, load_blend
    from bts.contest_state import load_decision_streak_state
    from bts.picks import get_game_statuses_detailed, save_pick
    from bts.posting import format_post, post_to_bluesky, should_post_now
    from bts.strategy import select_pick

    picks_path = Path(picks_dir)
    models_path = Path(models_dir)

    # Step 1: Run prediction pipeline (with model caching)
    click.echo(f"[{datetime.now(timezone.utc).strftime('%H:%M UTC')}] Running predictions for {date}...")
    cache_path = models_path / f"blend_{date}.pkl"
    cached_blend = None
    if cache_path.exists():
        click.echo(f"  Loading cached model from {cache_path}")
        cached_blend = load_blend(cache_path)

    try:
        predictions = run_pipeline(
            date, data_dir,
            cached_blend=cached_blend,
            save_blend_path=cache_path if not cached_blend else None,
        )
    except RuntimeError as e:
        click.echo(f"ERROR: {e}", err=True)
        return
    except Exception as e:
        click.echo(f"ERROR: Pipeline failed — {e}", err=True)
        return

    if predictions.empty:
        click.echo("No games found for this date.")
        return

    # Print ranked picks
    click.echo(f"\n{'='*80}")
    click.echo(f"BTS PICKS — {date}")
    click.echo(f"{'='*80}")
    click.echo(f"{'#':<4} {'Batter':<22} {'Team':<5} {'Pos':>3} {'vs Pitcher':<22} {'P(PA)':>6} {'P(Game)':>7}  {'Flags'}")
    click.echo(f"{'-'*80}")
    shown = 0
    for _, row in predictions.iterrows():
        if shown >= top:
            break
        if pd.isna(row.get("p_game_hit")):
            continue
        p_pa = row.get("p_hit_pa", row.get("p_game_hit", 0))
        click.echo(
            f"{shown+1:<4} {row['batter_name']:<22} {row['team']:<5} "
            f"{int(row.get('lineup', 0)):>3} {row['pitcher_name']:<22} "
            f"{p_pa:>5.1%} {row['p_game_hit']:>6.1%}  {row.get('flags', '')}"
        )
        shown += 1

    if dry_run:
        click.echo("\n  (--dry-run: not saving or posting)")
        return

    # Step 2: Apply strategy (streak-aware thresholds)
    decision_state = load_decision_streak_state(picks_path)
    try:
        game_statuses_detailed = get_game_statuses_detailed(date)
    except Exception:
        game_statuses_detailed = None
    result = select_pick(
        predictions,
        date,
        picks_path,
        streak=decision_state.streak,
        saver_available=decision_state.saver_available,
        allow_double=decision_state.allow_double,
        game_statuses_detailed=game_statuses_detailed,
        require_detailed_statuses=True,
    ).pick_result

    if result is None:
        # Skip day — post to Bluesky with top pick info
        top = predictions.iloc[0] if not predictions.empty else None
        if top is not None and pd.notna(top.get("p_game_hit")):
            from bts.posting import format_skip_post, post_to_bluesky, should_post_now
            click.echo(f"Skipping — {top['batter_name']} ({top.get('team', '?')}) "
                       f"at {top['p_game_hit']:.1%} below threshold. Streak holds at {decision_state.streak}.")
            if not dry_run and should_post_now(top.get("game_time", ""), False):
                text = format_skip_post(top["batter_name"], top.get("team", "?"),
                                        top["p_game_hit"], decision_state.streak)
                try:
                    uri = post_to_bluesky(text)
                    click.echo(f"  Posted skip to Bluesky: {uri}")
                except Exception as e:
                    click.echo(f"  Bluesky skip post failed: {e}", err=True)
        else:
            click.echo(f"No valid picks available. Streak holds at {decision_state.streak}.")
        return

    if result.locked:
        reason = "already posted" if result.daily.bluesky_posted else "game started"
        click.echo(f"Pick locked: {result.daily.pick.batter_name} ({reason})")
        # Catch-up posting if needed
        if not result.daily.bluesky_posted:
            decision_state = load_decision_streak_state(picks_path)
            text = format_post(
                result.daily.pick.batter_name, result.daily.pick.team,
                result.daily.pick.pitcher_name, result.daily.pick.p_game_hit, decision_state.streak,
                result.daily.double_down.batter_name if result.daily.double_down else None,
                result.daily.double_down.p_game_hit if result.daily.double_down else None,
                result.daily.double_down.team if result.daily.double_down else None,
                result.daily.double_down.pitcher_name if result.daily.double_down else None,
            )
            try:
                uri = post_to_bluesky(text)
                result.daily.bluesky_posted = True
                result.daily.bluesky_uri = uri
                save_pick(result.daily, picks_path)
                click.echo(f"  Posted to Bluesky (catch-up): {uri}")
            except Exception as e:
                click.echo(f"  Bluesky catch-up post failed: {e}", err=True)
        return

    # New or updated pick — attach provenance v1 fields per Codex #168.
    daily = result.daily
    from bts.picks import attach_provenance
    from bts.simulate.mdp import DEFAULT_POLICY_PATH
    attach_provenance(
        daily,
        blend_path=models_path / f"blend_{date}.pkl",
        policy_path=DEFAULT_POLICY_PATH,
    )
    click.echo(f"Pick: {daily.pick.batter_name} ({daily.pick.p_game_hit:.1%}) "
               f"vs {daily.pick.pitcher_name}")
    if daily.double_down:
        p_both = daily.pick.p_game_hit * daily.double_down.p_game_hit
        click.echo(f"  DOUBLE DOWN: + {daily.double_down.batter_name} "
                    f"({daily.double_down.p_game_hit:.1%}), P(both): {p_both:.1%}")

    save_pick(daily, picks_path)
    click.echo(f"  Saved to {picks_path / f'{date}.json'}")

    # Post to Bluesky if appropriate
    decision_state = load_decision_streak_state(picks_path)
    if should_post_now(daily.pick.game_time, daily.bluesky_posted):
        text = format_post(
            daily.pick.batter_name, daily.pick.team, daily.pick.pitcher_name,
            daily.pick.p_game_hit, decision_state.streak,
            daily.double_down.batter_name if daily.double_down else None,
            daily.double_down.p_game_hit if daily.double_down else None,
            daily.double_down.team if daily.double_down else None,
            daily.double_down.pitcher_name if daily.double_down else None,
        )
        try:
            uri = post_to_bluesky(text)
            daily.bluesky_posted = True
            daily.bluesky_uri = uri
            save_pick(daily, picks_path)
            click.echo(f"  Posted to Bluesky: {uri}")
        except Exception as e:
            click.echo(f"  Bluesky post failed: {e}", err=True)
    else:
        click.echo("  Not posting yet (game not within 3h, not evening run)")


@cli.command()
@click.option("--date", help="Date to preview (YYYY-MM-DD). Defaults to tomorrow.")
@click.option("--data-dir", default="data/processed", type=click.Path())
@click.option("--picks-dir", default="data/picks", type=click.Path())
@click.option("--models-dir", default="data/models", type=click.Path())
def preview(date: str | None, data_dir: str, picks_dir: str, models_dir: str):
    """Save a preliminary pick for tomorrow using projected lineups.

    Runs the full prediction pipeline and saves the pick to disk,
    but never posts to Bluesky. The scheduler will re-evaluate and
    overwrite when confirmed lineups are available.

    Designed to run from the overnight cron (after 3am data refresh)
    so the dashboard shows a pending pick instead of blank.
    """
    from bts.contest_state import load_decision_streak_state
    from bts.model.predict import run_pipeline, load_blend
    from bts.picks import get_game_statuses_detailed, save_pick, load_pick
    from bts.strategy import select_pick

    if date is None:
        date = _tomorrow_et()

    picks_path = Path(picks_dir)
    models_path = Path(models_dir)

    # Don't overwrite a pick that already has a result or was posted
    existing = load_pick(date, picks_path)
    if existing and (existing.result or existing.bluesky_posted):
        click.echo(f"Pick for {date} already resolved or posted — skipping preview.")
        return

    click.echo(f"[preview] Running predictions for {date}...")
    cache_path = models_path / f"blend_{date}.pkl"
    cached_blend = None
    if cache_path.exists():
        cached_blend = load_blend(cache_path)

    try:
        predictions = run_pipeline(
            date, data_dir,
            cached_blend=cached_blend,
            save_blend_path=cache_path if not cached_blend else None,
        )
    except Exception as e:
        click.echo(f"[preview] Failed: {e}", err=True)
        return

    if predictions.empty:
        click.echo(f"[preview] No games found for {date}.")
        return

    decision_state = load_decision_streak_state(picks_path)
    try:
        game_statuses_detailed = get_game_statuses_detailed(date)
    except Exception:
        game_statuses_detailed = None
    result = select_pick(
        predictions,
        date,
        picks_path,
        streak=decision_state.streak,
        saver_available=decision_state.saver_available,
        allow_double=decision_state.allow_double,
        game_statuses_detailed=game_statuses_detailed,
        require_detailed_statuses=True,
    ).pick_result

    if result is None:
        top = predictions.iloc[0]
        click.echo(f"[preview] Skip day — {top['batter_name']} at {top['p_game_hit']:.1%} below threshold.")
        return

    daily = result.daily
    # Preview path also gets provenance v1 fields per Codex #168.
    from bts.picks import attach_provenance
    from bts.simulate.mdp import DEFAULT_POLICY_PATH
    models_path_preview = Path(models_dir)
    attach_provenance(
        daily,
        blend_path=models_path_preview / f"blend_{date}.pkl",
        policy_path=DEFAULT_POLICY_PATH,
    )
    save_pick(daily, picks_path)
    click.echo(f"[preview] {daily.pick.batter_name} ({daily.pick.team}) "
               f"{daily.pick.p_game_hit:.1%} vs {daily.pick.pitcher_name}")
    if daily.double_down:
        click.echo(f"[preview] + {daily.double_down.batter_name} ({daily.double_down.team}) "
                   f"{daily.double_down.p_game_hit:.1%}")
    click.echo(f"[preview] Saved to {picks_path / f'{date}.json'} (PROJECTED — scheduler will re-evaluate)")


@cli.command(name="set-contest-streak")
@click.option("--streak", required=True, type=int, help="Actual active BTS account streak.")
@click.option("--best-streak", type=int, default=None, help="Actual season-best streak.")
@click.option(
    "--saver-available/--saver-unavailable",
    default=None,
    help="Actual BTS streak-saver availability. Omit when unknown.",
)
@click.option("--source-date", default=None, help="Observation date YYYY-MM-DD; defaults to today ET.")
@click.option("--source", default="manual_cli", help="Short source label for the observation.")
@click.option("--username", default=None, help="BTS username for this account state.")
@click.option("--ttl-hours", type=float, default=24.0,
              help="Override lifetime in hours (default 24); after this, auto-fetch wins again.")
@click.option("--override-expires-at", default=None,
              help="Explicit override expiry (ISO-8601); overrides --ttl-hours.")
@click.option("--reason", default=None, help="Why this emergency override is set (e.g. 'API auth down').")
@click.option("--picks-dir", default="data/picks", type=click.Path())
def set_contest_streak(
    streak: int,
    best_streak: int | None,
    saver_available: bool | None,
    source_date: str | None,
    source: str,
    username: str | None,
    ttl_hours: float,
    override_expires_at: str | None,
    reason: str | None,
    picks_dir: str,
):
    """Write an EXPIRING manual contest-streak override.

    Emergency use only — automated `fetch-contest-streak` is the default source.
    The override beats auto only until it expires (default 24h), so a forgotten
    manual file can never silently freeze live picks again.
    """
    from datetime import date, datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    if streak < 0:
        raise click.BadParameter("streak must be non-negative", param_hint="--streak")
    if best_streak is not None and best_streak < streak:
        raise click.BadParameter(
            "best streak must be at least the active streak",
            param_hint="--best-streak",
        )

    if source_date is None:
        observed_date = datetime.now(ZoneInfo("America/New_York")).date()
    else:
        try:
            observed_date = date.fromisoformat(source_date)
        except ValueError as exc:
            raise click.BadParameter(
                "source date must be YYYY-MM-DD",
                param_hint="--source-date",
            ) from exc

    if override_expires_at is not None:
        try:
            expires_dt = datetime.fromisoformat(override_expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise click.BadParameter(
                "override-expires-at must be ISO-8601",
                param_hint="--override-expires-at",
            ) from exc
        if expires_dt.tzinfo is None:
            expires_dt = expires_dt.replace(tzinfo=timezone.utc)
    else:
        expires_dt = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)

    state = {
        "schema_version": "bts_contest_streak_manual_v2",
        "active_streak": streak,
        "source": source,
        "source_date": observed_date.isoformat(),
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "override_expires_at": expires_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if best_streak is not None:
        state["best_streak"] = best_streak
    if saver_available is not None:
        click.echo("Note: --saver-available/--saver-unavailable is deprecated and no longer "
                   "affects the live saver (use `bts saver-state`); ignoring it.")
    if username:
        state["username"] = username
    if reason:
        state["reason"] = reason

    path = Path(picks_dir) / "account_state" / "contest_streak.manual.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    # Streak Saver flag: safe cold-init / sound auto-earn from the (manual) best_streak.
    from bts.saver_state import maybe_auto_earn_saver, season_for
    from zoneinfo import ZoneInfo
    _season = season_for(observed_date, now_year=datetime.now(ZoneInfo("America/New_York")).year)
    maybe_auto_earn_saver(Path(picks_dir), best_streak=best_streak, season=_season)
    click.echo(
        f"Wrote {path}: active_streak={streak} source_date={observed_date.isoformat()} "
        f"override_expires_at={state['override_expires_at']}"
    )


@cli.command(name="saver-state")
@click.option("--show", is_flag=True, help="Print the current Streak Saver flag state.")
@click.option("--init", "init_state", type=click.Choice(["not_earned", "active", "used"]),
              default=None, help="Initialize the season's flag (only when uninitialized; --force to override).")
@click.option("--use", "mark_used", is_flag=True, help="Mark the saver used (active -> used).")
@click.option("--undo", is_flag=True, help="Undo a mark-used (used -> active).")
@click.option("--force", is_flag=True, help="With --init, overwrite an already-initialized state.")
@click.option("--season", type=int, default=None, help="Contest season (default: current ET year).")
@click.option("--picks-dir", default="data/picks", type=click.Path())
def saver_state_cmd(show, init_state, mark_used, undo, force, season, picks_dir):
    """Manage the one-time Streak Saver flag (the sole live saver authority)."""
    from zoneinfo import ZoneInfo
    from bts.saver_state import load_saver_state, transition_saver_state

    picks = Path(picks_dir)
    season = season or datetime.now(ZoneInfo("America/New_York")).year

    if init_state is not None:
        current = load_saver_state(picks, season=season).state
        if current == "uninitialized":
            transition_saver_state(picks, expected_prior="uninitialized", new_state=init_state,
                                   season=season, source="cli")
            click.echo(f"Initialized saver flag: {init_state} (season {season})")
        elif force:
            transition_saver_state(picks, expected_prior=current, new_state=init_state,
                                   season=season, source="cli", force=True)
            click.echo(f"Forced saver flag: {current} -> {init_state} (season {season})")
        else:
            raise click.ClickException(
                f"saver flag already initialized as {current!r}; use --force to override")
    elif mark_used:
        ok = transition_saver_state(picks, expected_prior="active", new_state="used",
                                    season=season, source="cli")
        click.echo("Marked saver used." if ok
                   else f"No-op: saver is {load_saver_state(picks, season=season).state}, not active.")
    elif undo:
        ok = transition_saver_state(picks, expected_prior="used", new_state="active",
                                    season=season, source="cli")
        click.echo("Undid mark-used (saver active again)." if ok
                   else f"No-op: saver is {load_saver_state(picks, season=season).state}, not used.")

    click.echo(f"saver_state: {load_saver_state(picks, season=season).state} (season {season})")


def _atomic_write_json(path, obj):
    """Write JSON atomically: temp file in the same dir, fsync, os.replace."""
    import os
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _fetch_rounds(client=None):
    """Fetch rounds.json (no auth) -> {roundId: date}. Patchable in tests."""
    import httpx
    from bts.leaderboard.endpoints import ROUNDS_URL, browser_headers
    from bts.leaderboard.scraper import parse_rounds_lookup
    client = client or httpx
    resp = client.get(ROUNDS_URL, headers=browser_headers(), timeout=30.0)
    resp.raise_for_status()
    return parse_rounds_lookup(resp.json())


def _fetch_bts_to_mlb(client=None):
    """Fetch players.json (no auth) -> {bts_player_id: mlb_feed_id}. Patchable."""
    import httpx
    from bts.leaderboard.endpoints import browser_headers
    client = client or httpx
    url = "https://mlb-play.mlbstatic.com/apps/beat-the-streak/game/json/players.json"
    resp = client.get(url, headers=browser_headers(), timeout=30.0)
    resp.raise_for_status()
    out: dict[int, int] = {}
    for p in resp.json().get("players", []):
        if p.get("id") is not None and p.get("feedId") is not None:
            out[int(p["id"])] = int(p["feedId"])
    return out


def _contest_fetch_alert(status_path, dm_recipient, msg, cooldown_hours=6):
    """DM on failure, throttled via status_path (>=cooldown_hours between DMs). Returns whether sent."""
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    last_alert = None
    if status_path.exists():
        try:
            last_alert = json.loads(status_path.read_text()).get("last_alert_at")
        except (json.JSONDecodeError, OSError):
            last_alert = None
    should_alert = True
    if last_alert:
        try:
            last_dt = datetime.fromisoformat(last_alert.replace("Z", "+00:00"))
            should_alert = (now - last_dt) >= timedelta(hours=cooldown_hours)
        except ValueError:
            should_alert = True
    sent = False
    if should_alert and dm_recipient:
        try:
            from bts.dm import send_dm
            send_dm(dm_recipient, f"BTS fetch-contest-streak failed: {msg}")
            sent = True
        except Exception as exc:
            click.echo(f"(DM failed: {exc})", err=True)
    record = {"last_error": msg, "last_error_at": now.isoformat().replace("+00:00", "Z")}
    # Only consume the cooldown when a DM was actually SENT (not on missing recipient / send failure),
    # so a later run with the recipient fixed can still alert on the transition.
    record["last_alert_at"] = now.isoformat().replace("+00:00", "Z") if sent else last_alert
    _atomic_write_json(status_path, record)
    return sent


@cli.command(name="check-pick-entered")
@click.option("--picks-dir", default="data/picks", help="Picks directory")
@click.option("--expected-username", default=None, help="Refuse if session identity differs")
@click.option("--dm-recipient", default=None, help="Bluesky handle for the not-entered DM")
@click.option("--window-min", default=75, type=int,
              help="Check only within this many minutes before first pitch")
@click.option("--now-et", default=None, help="Test override: naive ET datetime ISO")
def check_pick_entered(picks_dir, expected_username, dm_recipient, window_min, now_et):
    """DM if today's delivered pick was never entered in the MLB app.

    v2 (2026-07-03, RE-ENABLED in cron): entry is now detected from the UNION
    of two sources — the profile endpoint (settled rows only; v1's sole source
    and why v1 false-alarmed every pre-pitch day and was disabled 2026-06-12)
    and GET api/predictions (discovered via the app JS bundle), which exposes
    the pending same-day row before settlement. Any fetch failure skips
    quietly WITHOUT consuming the once-per-day marker, so a transient error
    can never produce the v1 false-alarm class — the next cron run retries.

    Runs from cron every 15 min; exits silently unless NOW is inside the
    pre-first-pitch window for today's locked pick. v3 (audit F1): "alerted"
    is non-terminal — every run re-verifies until the entry is confirmed or
    the cutoff passes, with throttled escalations at T-30/T-15 to cutoff and
    a one-time all-clear DM once the entry appears (marker with escalation
    ledger in data/health_state/pick_entry_check.json).
    """
    import sys
    import httpx
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from bts.picks import load_pick
    from bts.scheduler import _earliest_pick_game_et

    ET = ZoneInfo("America/New_York")
    now = (datetime.fromisoformat(now_et).replace(tzinfo=ET)
           if now_et else datetime.now(ET))
    today = now.date().isoformat()

    picks = Path(picks_dir)
    daily = load_pick(today, picks)
    if daily is None:
        click.echo(f"check-pick-entered: no pick for {today}; nothing to check")
        return

    # Only a COMMITTED/locked pick is something Eric was told to enter. The
    # scheduler rewrites {date}.json all day with previews/projections, and an
    # un-delivered pick can still be deferred (and its file deleted) minutes
    # later — so the file's mere existence must not trigger a "not entered" nag.
    # Gate on the same commit signal check-results uses: decision.json
    # (decision.scoreable) with a pick_was_delivered fallback. This covers the
    # delivered / private_locked / locked_unconfirmed commit states and stays
    # silent for previews/deferred picks. (2026-07-06 premature-DM fix.)
    from bts.daily_decision import is_scoreable_commit
    if not is_scoreable_commit(today, picks, daily):
        click.echo(f"check-pick-entered: pick for {today} not committed/locked; nothing to check")
        return

    first_pitch = _earliest_pick_game_et(daily)
    minutes_to_pitch = (first_pitch - now).total_seconds() / 60
    # BTS rejects submissions within 5 min of first pitch. Only check inside
    # [cutoff, window]: below the cutoff the pick can no longer be entered, so a
    # "Fix it now!" nag is useless (and its countdown would go negative).
    submit_cutoff_min = 5
    # Strict lower bound: at exactly first_pitch-5 the entry is already locked
    # (Codex review #8) — never DM "0 min to submit".
    if not (submit_cutoff_min < minutes_to_pitch <= window_min):
        click.echo(f"check-pick-entered: outside window ({minutes_to_pitch:.0f} min to pitch)")
        return

    status_path = picks.parent / "health_state" / "pick_entry_check.json"
    # "confirmed" is the ONLY terminal state (audit F1): an "alerted" day keeps
    # RE-VERIFYING on every run until the entry appears or the window closes —
    # a delivered warning is not verified remediation (the 7/08 missed DD leg
    # sailed past a single 18:00 DM). "dm_failed" stays fully retryable
    # (Codex review, 2026-07-03).
    prior = {}
    if status_path.exists():
        try:
            prior = json.loads(status_path.read_text())
        except (json.JSONDecodeError, OSError):
            prior = {}
        if prior.get("date") != today:
            prior = {}
        if prior.get("status") == "confirmed":
            click.echo(f"check-pick-entered: already confirmed for {today}")
            return
    was_alerted = prior.get("status") == "alerted"
    # Markers written before the escalation ladder lack the field: the initial
    # alert has by definition already fired on an "alerted" marker.
    prior_escalations = list(prior.get("escalations")
                             or (["initial"] if was_alerted else []))

    from bts.leaderboard.auth import (
        load_session_cookies, extract_uid, fetch_login_session, AuthError,
    )
    import bts.contest_fetch as _cf
    try:
        cookies = load_session_cookies()
        uid = extract_uid(cookies)
        session = fetch_login_session(uid=uid, cookies=cookies)
        if expected_username and session.username != expected_username:
            click.echo(f"check-pick-entered: identity mismatch ({session.username!r}); skipping", err=True)
            return
        success = _cf.fetch_profile(session.user_id, cookies, session.xsid)
        pending = _cf.fetch_pending_predictions(cookies, session.xsid)
        rounds = _fetch_rounds()
        bts_to_mlb = _fetch_bts_to_mlb()
    except (AuthError, httpx.HTTPError, KeyError, ValueError, TypeError,
            _cf.ContestFetchError) as exc:
        # Any fetch failure skips quietly WITHOUT writing the daily marker, so a
        # transient error can never suppress the real check (the v1 false-alarm
        # class) and the next */15 cron run retries.
        click.echo(f"check-pick-entered: fetch failed, skipping quietly: {exc}", err=True)
        return

    # Verify the DELIVERED pick(s) are what got entered — Eric always intends the
    # entered pick to equal the recommendation, so a wrong player or a missing
    # double-down slot is a real anomaly, not just "no pick at all".
    required_mlb_ids = {
        b for b in (daily.pick.batter_id,
                    daily.double_down.batter_id if daily.double_down else None)
        if b is not None
    }
    ok, reason = _cf.pick_entry_status(
        success, pending, rounds, now.date(), required_mlb_ids, bts_to_mlb)
    if ok and reason != "match":
        # present_unverified: entries exist but the crosswalk can't prove
        # identity. NOT terminal (Codex review #1): a wrong player hiding
        # behind a crosswalk gap must keep being re-verified until lock; a
        # later crosswalk refresh can still resolve it either way.
        _atomic_write_json(status_path, {"date": today, "status": "present_unverified",
                                         "reason": reason, "checked_at": now.isoformat(),
                                         "escalations": prior_escalations})
        click.echo(f"check-pick-entered: {today} entry present but identity "
                   f"unverified; will re-verify")
        return
    if ok:
        _atomic_write_json(status_path, {"date": today, "status": "confirmed",
                                         "reason": reason, "checked_at": now.isoformat()})
        if (was_alerted or "initial" in prior_escalations) and dm_recipient:
            # One-time all-clear on the alerted -> confirmed transition: the
            # operator got a scary DM; close the loop when the fix lands.
            try:
                import bts.dm
                bts.dm.send_dm(dm_recipient,
                               f"✅ BTS pick entry confirmed for {today} ({reason}).")
                click.echo(f"check-pick-entered: confirmation DM sent to {dm_recipient}")
            except Exception as exc:
                click.echo(f"check-pick-entered: confirmation DM failed: {exc}", err=True)
        click.echo(f"check-pick-entered: {today} pick entered ({reason})")
        return

    names = daily.pick.batter_name
    if daily.double_down:
        names += f" + DD {daily.double_down.batter_name}"
    # Report time to the submission cutoff (first pitch - 5), not to first pitch.
    minutes_to_cutoff = minutes_to_pitch - submit_cutoff_min

    # Escalation ladder (audit F1): the initial alert fires on first detection;
    # T-30 and T-15 (minutes to the submission cutoff) re-alert if the entry is
    # STILL missing. Each tier fires at most once; an alert sent at/below a
    # threshold consumes that tier too — one DM near the cutoff suffices.
    if "initial" not in prior_escalations:
        tier = "initial"
    elif "t30" not in prior_escalations and minutes_to_cutoff <= 30:
        tier = "t30"
    elif "t15" not in prior_escalations and minutes_to_cutoff <= 15:
        tier = "t15"
    else:
        _atomic_write_json(status_path, {"date": today, "status": "alerted",
                                         "reason": reason, "checked_at": now.isoformat(),
                                         "escalations": prior_escalations})
        click.echo(f"check-pick-entered: still not entered ({reason}); "
                   f"re-verifying each run until cutoff")
        sys.exit(1)

    if tier == "initial":
        lead = ("BTS pick NOT entered" if reason == "no_pick"
                else "BTS entry does NOT match the recommended pick")
        msg = (f"\u26a0\ufe0f {lead} in MLB app: {names} — first pitch "
               f"{first_pitch.strftime('%-I:%M %p ET')} "
               f"({minutes_to_cutoff:.0f} min to submit). Fix it now!")
    else:
        msg = (f"\u23f0 STILL NOT entered ({reason}): {names} — "
               f"{minutes_to_cutoff:.0f} min left to submit!")
    dm_sent = False
    if dm_recipient:
        try:
            import bts.dm
            bts.dm.send_dm(dm_recipient, msg)
            dm_sent = True
            click.echo(f"check-pick-entered: DM sent to {dm_recipient}")
        except Exception as exc:
            click.echo(f"check-pick-entered: DM failed: {exc}", err=True)
    # Consume tiers ONLY when the alert actually went out (or there's no
    # recipient to reach) — a failed DM must stay retryable at every tier
    # (Codex review, 2026-07-03).
    if dm_sent or not dm_recipient:
        consumed = set(prior_escalations) | {"initial"}
        for t_name, threshold in (("t30", 30), ("t15", 15)):
            if minutes_to_cutoff <= threshold:
                consumed.add(t_name)
        _atomic_write_json(status_path, {"date": today, "status": "alerted",
                                         "reason": reason, "checked_at": now.isoformat(),
                                         "escalations": sorted(consumed)})
    elif tier == "initial" and not was_alerted:
        _atomic_write_json(status_path, {"date": today, "status": "dm_failed",
                                         "reason": reason, "checked_at": now.isoformat()})
    else:
        # Failed escalation DM: keep the marker unchanged so the tier retries.
        _atomic_write_json(status_path, {"date": today, "status": "alerted",
                                         "reason": reason, "checked_at": now.isoformat(),
                                         "escalations": prior_escalations})
    sys.exit(1)


@cli.command(name="fetch-contest-streak")
@click.option("--picks-dir", default="data/picks", type=click.Path())
@click.option("--expected-username", default=None,
              help="Require this BTS username (identity guard against wrong cookies).")
@click.option("--dm-recipient", default=None, help="Bluesky handle for throttled failure alerts.")
@click.option("--dry-run", is_flag=True, help="Print would-write without writing.")
def fetch_contest_streak(picks_dir, expected_username, dm_recipient, dry_run):
    """Fetch the real MLB BTS account streak and write contest_streak.json (atomic).

    Fails safe: on auth/cookie/HTTP/shape/identity failure it NEVER overwrites the
    prior good observation; it alerts (throttled DM) and exits nonzero. A current
    activeStreak IS written even when the per-round predictions array lags the counter
    (the snapshot/coverage split); contest_state labels lagged/stale downstream.
    """
    import sys
    import httpx
    from datetime import datetime, timezone
    from bts.leaderboard.auth import (
        load_session_cookies, extract_uid, fetch_login_session, AuthError,
    )
    from bts.contest_fetch import (
        fetch_profile, derive_source_date, build_observation, ContestFetchError,
    )

    picks = Path(picks_dir)
    out_path = picks / "account_state" / "contest_streak.json"
    status_path = picks.parent / "health_state" / "contest_streak_fetch_status.json"

    def _fail(msg, code=2):
        click.echo(f"fetch-contest-streak: {msg}", err=True)
        _contest_fetch_alert(status_path, dm_recipient, msg)
        sys.exit(code)

    # 1. auth + identity guard
    try:
        cookies = load_session_cookies()
        uid = extract_uid(cookies)
        session = fetch_login_session(uid=uid, cookies=cookies)
    except AuthError as exc:
        _fail(f"auth/cookie error — refresh via capture_bts_cookies.py on Mac. ({exc})")
    except httpx.HTTPError as exc:
        _fail(f"auth network error: {exc}")

    if expected_username and session.username != expected_username:
        _fail(f"identity mismatch: got {session.username!r}, expected {expected_username!r}; refusing to write")

    # prior-observation identity guard: never overwrite another account's good observation
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
        except (json.JSONDecodeError, OSError):
            prior = {}
        prior_user, prior_uid = prior.get("username"), prior.get("user_id")
        if (prior_user is not None and prior_user != session.username) or \
           (prior_uid is not None and prior_uid != session.user_id):
            _fail(f"prior contest_streak.json is {prior_user!r}/{prior_uid}, not session "
                  f"{session.username!r}/{session.user_id}; refusing to overwrite")

    # 2. profile + 3. currentness proof + build — ANY shape/fetch error must ALERT, not crash silently
    try:
        success = fetch_profile(session.user_id, cookies, session.xsid)
        predictions = success.get("predictions", [])
        rounds = _fetch_rounds()
        source_date = derive_source_date(predictions, rounds)  # may be None when the ledger lags
        observation = build_observation(
            success, source_date, session.user_id, session.username,
            datetime.now(timezone.utc),
        )
    except (httpx.HTTPError, AttributeError, TypeError, ValueError, KeyError, ContestFetchError) as exc:
        _fail(f"profile/rounds shape or fetch error: {exc}")

    # write (atomic) or dry-run — a current activeStreak is written even when the
    # predictions array lags; staleness is labeled by contest_state downstream, not
    # gated here (the snapshot/coverage split).
    summary = (f"active_streak={observation['active_streak']} "
               f"best_streak={observation['best_streak']} source_date={observation['source_date']}")
    if dry_run:
        click.echo(f"[dry-run] would write {out_path}: {summary}")
        return
    _atomic_write_json(out_path, observation)
    _atomic_write_json(status_path, {
        "last_success_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "last_error": None,
    })
    # persist the full per-round MLB ledger (append-only) for analysis + Phase-2 saver inference
    ledger_row = {
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "active_streak": success["activeStreak"],
        "best_streak": success["seasonBestStreak"],
        "source_date": source_date.isoformat() if source_date is not None else None,
        "predictions": predictions,
    }
    with (picks / "account_state" / "contest_ledger.jsonl").open("a") as _fh:
        _fh.write(json.dumps(ledger_row) + "\n")
    # Streak Saver flag: safe cold-init below 10 + sound auto-earn at 10 from the reliable
    # seasonBestStreak. Never auto-inits `active` from a cold file at >=10 (fail-closed).
    from bts.saver_state import maybe_auto_earn_saver, season_for
    from zoneinfo import ZoneInfo
    _season = season_for(source_date, now_year=datetime.now(ZoneInfo("America/New_York")).year)
    maybe_auto_earn_saver(picks, best_streak=observation["best_streak"], season=_season)
    click.echo(f"wrote {out_path}: {summary}")


@cli.command(name="predict-json")
@click.option("--date", required=True, help="Date to predict (YYYY-MM-DD)")
@click.option("--data-dir", default="data/processed", type=click.Path(), help="Processed data directory")
@click.option("--models-dir", default="data/models", type=click.Path(), help="Cached models directory")
def predict_json(date: str, data_dir: str, models_dir: str):
    """Run predictions and output JSON to stdout.

    Worker command for remote orchestration. Outputs a JSON array of
    ranked predictions. All log messages go to stderr.
    """
    import json as _json
    import sys
    from datetime import datetime, timezone
    from bts.model.predict import run_pipeline, save_blend, load_blend

    models_path = Path(models_dir)

    click.echo(
        f"[{datetime.now(timezone.utc).strftime('%H:%M UTC')}] "
        f"Running predictions for {date}...",
        err=True,
    )

    cache_path = models_path / f"blend_{date}.pkl"
    cached_blend = None
    if cache_path.exists():
        click.echo(f"  Loading cached model from {cache_path}", err=True)
        cached_blend = load_blend(cache_path)

    try:
        predictions = run_pipeline(
            date, data_dir,
            cached_blend=cached_blend,
            save_blend_path=cache_path if not cached_blend else None,
        )
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    if predictions.empty:
        click.echo("[]")
        return

    # Select columns needed by the orchestrator
    columns = [
        "batter_name", "batter_id", "team", "lineup",
        "pitcher_name", "pitcher_id", "game_pk", "game_time",
        "p_hit_pa", "p_game_hit", "flags",
    ]
    output_cols = [c for c in columns if c in predictions.columns]
    output = predictions[output_cols].to_dict(orient="records")

    # Clean up NaN/None for JSON serialization
    for row in output:
        for k, v in row.items():
            if isinstance(v, float) and (v != v):  # NaN check
                row[k] = None
            elif hasattr(v, 'item'):  # numpy scalar
                row[k] = v.item()

    click.echo(_json.dumps(output, indent=2))


@cli.command()
@click.option("--date", required=True, help="Date to predict (YYYY-MM-DD)")
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True), help="Orchestrator config TOML file")
def orchestrate(date: str, config_path: str):
    """Orchestrate predictions across compute tiers (Pi5 command).

    Cascades through SSH tiers (Mac -> Alienware -> Cloud), applies
    pick strategy, saves pick, posts to Bluesky. DMs on total failure.
    """
    from bts.orchestrator import orchestrate as _orchestrate

    success = _orchestrate(Path(config_path), date)
    if not success:
        raise SystemExit(1)


@cli.command()
@click.option("--date", default=None, help="Date to schedule (YYYY-MM-DD, default: today)")
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True), help="Orchestrator config TOML file")
@click.option("--dry-run", is_flag=True, help="Show schedule without executing")
def schedule(date: str | None, config_path: str, dry_run: bool):
    """Run the dynamic lineup scheduler for a day.

    Fetches the MLB schedule, computes lineup check times (game_time - 45min),
    sleeps between checks, runs predictions when new lineups confirm, and
    posts to Bluesky when lock conditions are met.
    """
    from bts.orchestrator import load_config
    from bts.scheduler import run_day

    if date is None:
        date = _today_et()

    config = load_config(Path(config_path))
    run_day(date=date, config=config, dry_run=dry_run)


@cli.command(name="check-results")
@click.option("--date", required=True, help="Date to check results for (YYYY-MM-DD)")
@click.option("--picks-dir", default="data/picks", type=click.Path(), help="Picks directory")
@click.option(
    "--shadow-status-output",
    type=click.Path(),
    help=(
        "Write context-stack shadow monitoring status JSON. "
        "Defaults to <picks parent>/validation/context_stack_shadow_status.json."
    ),
)
def check_results(date: str, picks_dir: str, shadow_status_output: str | None):
    """Check if yesterday's pick got a hit and update the streak.

    Designed to run via cron at 1am ET (after all games finish).
    """
    from pathlib import Path
    from bts.picks import (
        load_pick, update_streak, save_pick, load_streak,
        load_shadow_pick, save_shadow_pick,
        active_streak_results, effective_daily_result,
        iter_daily_pick_slots, resolve_daily_slot_results,
        scoring_lock,
    )

    picks_path = Path(picks_dir)
    daily = load_pick(date, picks_path)

    def write_shadow_status_artifact() -> None:
        shadow_files = list(picks_path.glob("*.shadow.json"))
        if not shadow_files:
            return
        try:
            from bts.shadow_eval import build_shadow_cycle_status, write_manifest_json
            output_path = (
                Path(shadow_status_output)
                if shadow_status_output
                else picks_path.parent / "validation" / "context_stack_shadow_status.json"
            )
            status = build_shadow_cycle_status(picks_path)
            write_manifest_json(status, output_path)
            click.echo(f"  Shadow status: {output_path}")
        except Exception as e:
            click.echo(f"WARNING: Failed to write shadow status — {e}", err=True)

    if daily is None:
        click.echo(f"No pick found for {date}.")
        return

    def reconcile_shadow_result() -> tuple[bool, str | None]:
        """Resolve the shadow pick independently from production streak state."""
        shadow = load_shadow_pick(date, picks_path)
        if shadow is None or shadow.result in ("hit", "miss", "void"):
            return False, shadow.result if shadow else None

        try:
            slot_results = resolve_daily_slot_results(shadow, date)
        except Exception as e:
            click.echo(f"ERROR: Failed to check shadow result — {e}", err=True)
            return False, None

        if slot_results is None:
            click.echo(
                "WARNING: Shadow pick has an active game not final or batter not found. "
                "Shadow result unchanged."
            )
            return False, None

        shadow.slot_results = slot_results
        shadow.result = effective_daily_result(slot_results)
        save_shadow_pick(shadow, picks_path)
        shadow_names = [
            slot.batter_name
            for slot_key, slot in iter_daily_pick_slots(shadow)
            if slot_results.get(slot_key) != "void"
        ]
        void_names = [
            slot.batter_name
            for slot_key, slot in iter_daily_pick_slots(shadow)
            if slot_results.get(slot_key) == "void"
        ]
        if void_names:
            click.echo(f"  Shadow void: {', '.join(void_names)}")
        click.echo(
            f"  Shadow: {' + '.join(shadow_names) if shadow_names else 'all picks void'} — "
            f"{shadow.result.upper()}"
        )
        return True, shadow.result

    # Skip if scheduler already resolved this pick (avoid double-counting streak)
    if daily.result in ("hit", "miss", "void"):
        reconcile_shadow_result()
        write_shadow_status_artifact()
        click.echo(f"Already resolved: {daily.pick.batter_name} — {daily.result}. Skipping.")
        return

    # GH #144: only score a committed pick. A stale preview / pre-lock / undelivered
    # <date>.json must not advance the streak. Shadow still reconciles on this exit.
    from bts.daily_decision import is_scoreable_commit

    scoreable = is_scoreable_commit(date, picks_path, daily)

    if not scoreable:
        reconcile_shadow_result()
        write_shadow_status_artifact()
        click.echo(f"{date}: decision was not a committed pick (skip / undelivered) — not scoring.")
        return

    for _, slot in iter_daily_pick_slots(daily):
        click.echo(f"Checking {slot.batter_name} (game {slot.game_pk})...")

    try:
        slot_results = resolve_daily_slot_results(daily, date)
    except Exception as e:
        click.echo(f"ERROR: Failed to check game result — {e}", err=True)
        return

    if slot_results is None:
        click.echo("WARNING: Active pick game not final or batter not found. "
                   "Streak unchanged. Check manually.")
        return

    results = active_streak_results(slot_results)

    # Serialize the streak read-modify-write against the daemon's result
    # polling (review F13): re-check INSIDE the lock — the daemon may have
    # scored this date while we were resolving (the pre-check above ran
    # before the network fetch). fresh is adopted before mutation so a
    # concurrent metadata update isn't clobbered, and fresh=None fails closed
    # (review #6). Shadow reconciliation (network) stays OUTSIDE the lock
    # (review #5).
    skip_reason = None
    with scoring_lock(picks_path):
        fresh = load_pick(date, picks_path)
        if fresh is None:
            skip_reason = ("Pick file disappeared during scoring; failing "
                           "closed (no streak update).")
        elif fresh.result in ("hit", "miss", "void"):
            skip_reason = (f"Already resolved by another scorer: {fresh.result}. "
                           f"Skipping streak update.")
        else:
            daily = fresh
            new_streak = update_streak(results, picks_path) if results else load_streak(picks_path)
            daily.slot_results = slot_results
            daily.result = effective_daily_result(slot_results)
            save_pick(daily, picks_path)
    if skip_reason is not None:
        click.echo(skip_reason)
        reconcile_shadow_result()
        write_shadow_status_artifact()
        return

    reconcile_shadow_result()
    write_shadow_status_artifact()

    # Report
    void_names = [
        slot.batter_name
        for slot_key, slot in iter_daily_pick_slots(daily)
        if slot_results.get(slot_key) == "void"
    ]
    if void_names:
        click.echo(f"VOID: {', '.join(void_names)}.")

    if daily.result == "void":
        click.echo(f"All picks void. Streak unchanged: {new_streak}")
    elif daily.result == "hit":
        hit_names = [
            slot.batter_name
            for slot_key, slot in iter_daily_pick_slots(daily)
            if slot_results.get(slot_key) == "hit"
        ]
        click.echo(f"HIT! {' + '.join(hit_names)}. Streak: {new_streak}")
    else:
        miss_names = [
            slot.batter_name
            for slot_key, slot in iter_daily_pick_slots(daily)
            if slot_results.get(slot_key) == "miss"
        ]
        click.echo(f"MISS: {', '.join(miss_names)}. Streak reset to 0.")

    # Bluesky result reply is handled by the scheduler's result polling.
    # This cron safety net only updates the local pick file.


@cli.command(name="reconcile")
@click.option("--picks-dir", default="data/picks", type=click.Path(), help="Picks directory")
@click.option("--lookback", default=8, type=int, help="Days to look back (default: 8)")
def reconcile(picks_dir: str, lookback: int):
    """Re-check recent picks for scoring changes (hit overturned to error).

    Looks back 8 days by default. If a result changed, updates the pick file,
    recalculates the streak, and reports corrections.
    """
    from bts.picks import reconcile_results, load_streak

    picks_path = Path(picks_dir)
    click.echo(f"Reconciling last {lookback} days of picks...")
    corrections = reconcile_results(picks_path, lookback_days=lookback)

    if not corrections:
        streak = load_streak(picks_path)
        click.echo(f"No scoring changes detected. Streak: {streak}")
    else:
        streak = load_streak(picks_path)
        click.echo(f"CORRECTIONS FOUND ({len(corrections)}):")
        for c in corrections:
            click.echo(f"  {c['date']}: {c['batter']} — {c['old_result']} -> {c['new_result']}")
        click.echo(f"Streak recalculated: {streak}")


@cli.command("park-drag-refresh")
@click.option("--root", default="data/external/park_drag", show_default=True,
              help="Root of the external park_drag artifact directory.")
@click.option("--lookback-days", default=3, show_default=True,
              help="Re-fetch window (days) before the store's newest date.")
def park_drag_refresh_cmd(root: str, lookback_days: int):
    """Refresh the park_drag_delta external table (daily cron on the box).

    Fetches recent four-seam pitches + game weather, recomputes game-level
    drag, rebuilds the serving-correct export atomically. Failures exit
    non-zero and land in producer_status.json (the park_drag_freshness health
    source surfaces them)."""
    import json as _json
    from pathlib import Path as _Path

    from bts.features.park_drag_producer import refresh

    summary = refresh(_Path(root), lookback_days=lookback_days)
    click.echo(_json.dumps(summary, default=str))
    if not summary.get("ok"):
        raise SystemExit(1)


@cli.command(name="shadow-report")
@click.option("--picks-dir", default="data/picks", type=click.Path(), help="Picks directory")
def shadow_report(picks_dir: str):
    """Compare shadow model picks against production picks.

    Reads {date}.json and {date}.shadow.json pairs from the picks directory.
    Reports agreement rate, disagreement details, and production's day-level
    hit rate (the DD-aware streak-advancing rate, not true top-1 P@1 — a
    double-down day only counts as "hit" when BOTH picks hit).
    """
    import json as _json
    from pathlib import Path

    picks_path = Path(picks_dir)
    shadow_files = sorted(picks_path.glob("*.shadow.json"))

    if not shadow_files:
        click.echo("No shadow pick pairs found.")
        return

    pairs = []
    for sf in shadow_files:
        date = sf.name.replace(".shadow.json", "")
        prod_file = picks_path / f"{date}.json"
        if not prod_file.exists():
            continue
        prod = _json.loads(prod_file.read_text())
        shadow = _json.loads(sf.read_text())
        from bts.shadow_eval import SHADOW_MODEL_NAME
        if (shadow.get("shadow_model_version")
                or "context_stack_shadow_v1") != SHADOW_MODEL_NAME:
            # prior feature-stack version: excluded, same rule as shadow_eval
            continue
        pairs.append((date, prod, shadow))

    if not pairs:
        click.echo("No shadow pick pairs found (shadow files exist but no matching production files).")
        return

    agrees = 0
    disagrees = []
    prod_hits = 0
    shadow_hits = 0
    prod_resolved = 0
    shadow_resolved = 0

    for date, prod, shadow in pairs:
        prod_name = prod["pick"]["batter_name"]
        shadow_name = shadow["pick"]["batter_name"]
        prod_result = prod.get("result")
        shadow_result = shadow.get("result")

        if prod_name == shadow_name:
            agrees += 1
        else:
            disagrees.append((date, prod_name, prod.get("pick", {}).get("p_game_hit"),
                              shadow_name, shadow.get("pick", {}).get("p_game_hit"),
                              prod_result))

        if prod_result in ("hit", "miss"):
            prod_resolved += 1
            if prod_result == "hit":
                prod_hits += 1
        if shadow_result in ("hit", "miss"):
            shadow_resolved += 1
            if shadow_result == "hit":
                shadow_hits += 1

    total = len(pairs)
    pct = agrees / total * 100

    click.echo(f"Shadow Model Report ({total} days, {30 - total} remaining to threshold)")
    click.echo(f"{'='*60}")
    click.echo(f"Agreement rate: {agrees}/{total} ({pct:.1f}%)")
    if prod_resolved > 0:
        click.echo(
            f"Production day hit rate (DD-aware): "
            f"{prod_hits}/{prod_resolved} ({prod_hits/prod_resolved*100:.1f}%)"
        )
    if shadow_resolved > 0:
        click.echo(
            f"Shadow recorded day hit rate: "
            f"{shadow_hits}/{shadow_resolved} ({shadow_hits/shadow_resolved*100:.1f}%)"
        )
    if shadow_resolved < total:
        click.echo(
            f"Shadow results unresolved: {total - shadow_resolved}/{total} "
            f"(shadow hit rate incomplete until reconciliation/backfill)"
        )
    click.echo()

    if disagrees:
        click.echo(f"Disagreements ({len(disagrees)} days):")
        click.echo(f"{'Date':<12} {'Production':<20} {'Shadow':<20} {'Result'}")
        click.echo(f"{'-'*12} {'-'*20} {'-'*20} {'-'*8}")
        for date, pn, pp, sn, sp, res in disagrees:
            pp_str = f"{pp:.1%}" if pp else "?"
            sp_str = f"{sp:.1%}" if sp else "?"
            res_str = res or "pending"
            click.echo(f"{date:<12} {pn:<15} {pp_str:<4}  {sn:<15} {sp_str:<4}  {res_str}")


@cli.command(name="shadow-status")
@click.option("--picks-dir", default="data/picks", type=click.Path(), help="Picks directory")
@click.option("--output", type=click.Path(), help="Write status JSON to this path")
@click.option("--min-days", default=30, type=int, help="Resolved paired days needed before review")
def shadow_status(picks_dir: str, output: str | None, min_days: int):
    """Emit the live context-stack shadow monitoring status artifact.

    This is the lightweight daily monitor: it reads recorded production and
    shadow pick files only. Use ``shadow-backfill-results`` for the heavier
    DD-aware recompute/audit path.
    """
    from bts.shadow_eval import build_shadow_cycle_status, write_manifest_json

    status = build_shadow_cycle_status(Path(picks_dir), min_days=min_days)
    counts = status["counts"]
    click.echo(
        f"Shadow cycle status: {status['cycle_state']} "
        f"({counts['resolved_paired_days']}/{min_days} resolved paired days)"
    )
    click.echo(
        f"Files: {counts['shadow_files']} shadow, "
        f"{counts['paired_production_files']} paired production, "
        f"{counts['unresolved_shadow_results']} unresolved shadow"
    )
    quality = status["quality_recorded"]
    prod_rate = quality["production_day_hit_rate"]["rate"]
    shadow_rate = quality["shadow_day_hit_rate"]["rate"]
    gap = quality["shadow_minus_production_hit_rate"]["value"]
    if prod_rate is not None and shadow_rate is not None and gap is not None:
        click.echo(
            f"Recorded quality: production {prod_rate:.1%}, "
            f"shadow {shadow_rate:.1%}, gap {gap:+.1%} "
            f"(n={quality['n_evaluable_days']})"
        )
    if status["coverage"]["unresolved_shadow_dates"]:
        click.echo(
            "Unresolved shadow dates: "
            + ", ".join(status["coverage"]["unresolved_shadow_dates"])
        )
    if output:
        write_manifest_json(status, Path(output))
        click.echo(f"Wrote status: {output}")


@cli.command(name="skip-policy-shadow-update")
@click.option("--picks-dir", default="data/picks", type=click.Path(), help="Picks directory")
@click.option("--status-output", default="data/validation/skip_policy_shadow_status.json",
              type=click.Path(), help="Status artifact path")
@click.option("--date", default=None, help="Single decision date YYYY-MM-DD (else recent markers)")
@click.option("--no-reconcile", is_flag=True, help="Skip realized-outcome reconciliation")
def skip_policy_shadow_update(picks_dir, status_output, date, no_reconcile):
    """Record shadow entries from MDP skips in decision.json, reconcile outcomes, refresh status.

    Counterfactual SHADOW POLICY (docs/audit/2026-06-20-skip-policy-shadow.md): the scheduler
    writes `<date>/decision.json` (action=skip, source=mdp) at each genuine MDP skip; this reads
    those records, logs what taking the single would have done, and reconciles the realized hit —
    accumulating the live band hit rate vs the calibrated ~0.744 breakeven to settle whether the
    streak>=8 skip rule costs streaks. Run nightly from cron.
    """
    from datetime import datetime, timezone
    from bts.shadow_eval import _current_git_commit
    from bts import skip_policy_shadow as sps

    picks_path = Path(picks_dir)
    now = datetime.now(timezone.utc)

    pruned = sps.prune_superseded(picks_path)   # drop superseded skips (decision.json flipped to pick)
    if pruned:
        click.echo(f"Pruned {len(pruned)} superseded record(s): {', '.join(pruned)}")

    if date:
        rec = sps.record_skip_from_decision(date, picks_path, now=now)
        if rec is None:
            click.echo(f"{date}: no MDP skip in decision.json (production picked, or no decision); nothing to record.")
        else:
            click.echo(f"Recorded {date}: deployed=skip shadow=single (divergent)")
    else:
        recorded = sps.record_pending_skips(picks_path, lookback_days=10, now=now)
        click.echo(f"Recorded {len(recorded)} new skip decision(s): {', '.join(recorded) or 'none'}")

    if not no_reconcile:
        n = sps.reconcile_pending(picks_path, hit_checker=sps.make_hit_checker(), now=now)
        click.echo(f"Reconciled {n} pending outcome(s).")

    status = sps.write_status(picks_path, status_output, git_commit=_current_git_commit())
    v = status["shadow_band_hit_rate"]
    rate = f"{v['rate']:.1%}" if v["rate"] is not None else "n/a"
    click.echo(f"Status: {status['counts']['divergent_days']} divergent days, {v['resolved']} resolved, "
               f"verdict={v['verdict']} (rate={rate} vs breakeven {v['breakeven_p']}). Wrote {status_output}")
    aged = status["counts"].get("aged_superseded_records") or []
    if aged:
        # cron runs only `update` (round-3 F3) — the contradiction warning
        # must land in cron.log, not only in the manual status command.
        click.echo(f"⚠ AGED CONTRADICTIONS excluded from sample "
                   f"(decision.json no longer mdp-skip; investigate): {aged}")


@cli.command(name="skip-policy-shadow-status")
@click.option("--status-file", default="data/validation/skip_policy_shadow_status.json", type=click.Path())
def skip_policy_shadow_status(status_file):
    """Print the skip-policy shadow verdict (is the streak>=8 skip rule costing streaks?)."""
    path = Path(status_file)
    if not path.exists():
        click.echo(f"No status artifact at {status_file}; run `bts skip-policy-shadow-update` first.")
        return
    s = json.loads(path.read_text())
    c, v = s["counts"], s["shadow_band_hit_rate"]
    click.echo(f"skip-policy shadow ({s['schema_version']}) generated {s['generated_at']}")
    click.echo(f"  divergent days: {c['divergent_days']}  ({c['resolved_divergent']} resolved, {c['pending']} pending)")
    rate = f"{v['rate']:.1%}" if v["rate"] is not None else "n/a"
    ci = f" CI[{v['wilson_ci'][0]:.1%},{v['wilson_ci'][1]:.1%}]" if v["wilson_ci"] else ""
    click.echo(f"  skipped-band realized hit rate: {rate}{ci}  vs breakeven {v['breakeven_p']:.3f}  (monitoring only)")
    basis = v.get("verdict_basis") or {}
    if basis.get("checkpoint"):
        bci = basis.get("ci")
        bci_str = f" CI[{bci[0]:.1%},{bci[1]:.1%}]" if bci else ""
        click.echo(f"  VERDICT: {v['verdict']}  — pre-registered look at n={basis['checkpoint']} "
                   f"({basis.get('hits_used')}/{basis.get('n_used')}{bci_str}, z={basis.get('z')})")
    elif "verdict_basis" not in v and v["verdict"] != "insufficient_n":
        click.echo(f"  VERDICT: {v['verdict']}  — LEGACY v1 artifact (retired nightly-Wilson "
                   f"rule); rerun `bts skip-policy-shadow-update` for a pre-registered verdict")
    else:
        click.echo(f"  VERDICT: {v['verdict']}  — no pre-registered look fired yet")
    aged = c.get("aged_superseded_records") or []
    if aged:
        click.echo(f"  ⚠ AGED CONTRADICTIONS (excluded from sample; decision.json no longer mdp-skip): {aged}")


@cli.command(name="shadow-backfill-results")
@click.option("--picks-dir", default="data/picks", type=click.Path(), help="Picks directory")
@click.option("--raw-dir", type=click.Path(), help="Cached raw game-feed directory; defaults to picks parent/raw")
@click.option("--output", type=click.Path(), help="Write manifest JSON to this path")
@click.option("--apply", "apply_changes", is_flag=True, help="Apply reviewed manifest changes")
@click.option("--backup-dir", type=click.Path(), help="Required with --apply; stores pre-change shadow files")
@click.option("--bootstrap", default=10_000, type=int, help="Bootstrap replicates for hit-rate gap CI")
def shadow_backfill_results(
    picks_dir: str,
    raw_dir: str | None,
    output: str | None,
    apply_changes: bool,
    backup_dir: str | None,
    bootstrap: int,
):
    """Dry-run or apply DD-aware shadow-result recomputation.

    The default mode is read-only: recompute every *.shadow.json result using
    cached game JSON before MLB API fallback, emit an audit manifest, and
    calculate paired shadow-vs-production quality metrics.

    Rollback after an apply: copy *.shadow.json files from --backup-dir back to
    --picks-dir.
    """
    from bts.shadow_eval import (
        apply_shadow_backfill_manifest,
        build_shadow_backfill_manifest,
        write_manifest_json,
    )

    if apply_changes and not backup_dir:
        raise click.UsageError("--backup-dir is required with --apply")

    manifest = build_shadow_backfill_manifest(
        Path(picks_dir),
        raw_dir=Path(raw_dir) if raw_dir else None,
        n_bootstrap=bootstrap,
    )
    counts = manifest["counts"]
    mode = "APPLY" if apply_changes else "DRY RUN"
    click.echo(
        f"Shadow result backfill {mode}: "
        f"{counts['shadow_files']} files, {counts['resolved']} resolved, "
        f"{counts['unresolved']} unresolved, {counts['errors']} errors, "
        f"{counts['would_change']} would change"
    )
    classes = counts["change_class"]
    click.echo(
        "Change classes: "
        f"new={classes['new']}, changed={classes['changed']}, "
        f"unchanged={classes['unchanged']}, skipped={classes['skipped']}, "
        f"error={classes['error']}"
    )
    if classes["changed"]:
        click.echo(
            "Review required: changed rows overwrite an existing shadow result "
            "with a DD-aware recomputation."
        )
        changed_dates = [
            row["date"] for row in manifest["rows"]
            if row["change_class"] == "changed"
        ]
        click.echo(f"Changed dates: {', '.join(changed_dates)}")

    quality = manifest["quality_if_applied"]
    prod_rate = quality["production_day_hit_rate"]["rate"]
    shadow_rate = quality["shadow_day_hit_rate"]["rate"]
    gap = quality["shadow_minus_production_hit_rate"]["value"]
    if prod_rate is not None and shadow_rate is not None and gap is not None:
        click.echo(
            f"Quality if applied: production {prod_rate:.1%}, "
            f"shadow {shadow_rate:.1%}, gap {gap:+.1%} "
            f"(n={quality['n_evaluable_days']})"
        )
    else:
        click.echo("Quality if applied: no fully evaluable paired days")

    if apply_changes:
        apply_result = apply_shadow_backfill_manifest(
            manifest,
            backup_dir=Path(backup_dir),
        )
        manifest["mode"] = "apply"
        manifest["apply_result"] = apply_result
        click.echo(
            f"Applied {len(apply_result['applied'])} changes; "
            f"skipped {len(apply_result['skipped'])}"
        )
    else:
        click.echo("No files changed. Re-run with --apply and --backup-dir after review.")

    if output:
        write_manifest_json(manifest, Path(output))
        click.echo(f"Wrote manifest: {output}")


@cli.group()
def state():
    """State management: export / regenerate / verify BTS state."""


@state.command(name="export")
@click.option("--picks-dir", default="data/picks", type=click.Path(exists=True))
@click.option("--to", "output_path", default="data/state/initial-state.json", type=click.Path())
def state_export(picks_dir, output_path):
    """Export current state to a committable snapshot file.

    Refuses to run if any pick in picks-dir is unresolved. Used at
    the moment of cloud migration cutover to freeze pre-migration history.
    """
    from pathlib import Path
    from bts.state.export import export_initial_state, UnresolvedPickError

    try:
        snapshot = export_initial_state(
            picks_dir=Path(picks_dir),
            output_path=Path(output_path),
        )
    except UnresolvedPickError as e:
        click.echo(str(e), err=True)
        raise SystemExit(2)

    click.echo(
        f"Exported {len(snapshot['historical_picks'])} picks to {output_path}\n"
        f"  cutoff_date: {snapshot['cutoff_date']}\n"
        f"  streak_at_cutoff: {snapshot['streak_at_cutoff']}\n"
        f"  saver_available: {snapshot['saver_available']}"
    )


@state.command(name="regenerate")
@click.option("--snapshot", default="data/state/initial-state.json",
              type=click.Path(exists=True))
@click.option("--handle", default="beatthestreakbot.bsky.social")
@click.option("--out-picks-dir", default="data/picks", type=click.Path())
def state_regenerate(snapshot, handle, out_picks_dir):
    """Rebuild BTS state from committed snapshot + Bluesky post history.

    Used for disaster recovery when production picks are lost (e.g.
    server rebuild) or during migration between providers. Post-cutoff
    data comes from Bluesky; pre-cutoff data comes from the committed
    initial snapshot.
    """
    from pathlib import Path
    from bts.state.regenerate import regenerate

    summary = regenerate(
        snapshot_path=Path(snapshot),
        bluesky_handle=handle,
        out_picks_dir=Path(out_picks_dir),
    )
    click.echo("Regeneration complete:")
    for k, v in summary.items():
        click.echo(f"  {k}: {v}")


@state.command(name="verify")
@click.option("--live-dir", default="data/picks", type=click.Path(exists=True))
@click.option("--snapshot", default="data/state/initial-state.json",
              type=click.Path(exists=True))
@click.option("--handle", default="beatthestreakbot.bsky.social")
def state_verify(live_dir, snapshot, handle):
    """Regenerate state to a temp dir and diff against live state.

    Run periodically as a drift check. Exits 0 if clean, 1 if drift found.
    """
    import tempfile
    from pathlib import Path
    from bts.state.regenerate import regenerate
    from bts.state.verify import diff_pick_files

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "picks"
        summary = regenerate(
            snapshot_path=Path(snapshot),
            bluesky_handle=handle,
            out_picks_dir=tmp_path,
        )
        report = diff_pick_files(Path(live_dir), tmp_path)

    if report.is_clean:
        click.echo(f"Drift check CLEAN. {summary['snapshot_picks']} snapshot + {summary['bluesky_picks']} Bluesky picks.")
        return

    click.echo("Drift detected:", err=True)
    for issue in report.issues:
        click.echo(f"  - {issue}", err=True)
    raise SystemExit(1)
