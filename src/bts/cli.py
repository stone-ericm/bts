import json
import click
from pathlib import Path


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
def scorecard(
    profiles_dir: str,
    mc_trials: int,
    season_length: int,
    save_path: str | None,
    diff_path: str | None,
):
    """Compute and display the BTS model validation scorecard.

    Loads all backtest_*.parquet files, computes P@K, miss analysis,
    calibration, and streak metrics. Saves a JSON artifact.
    """
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
def data_sync_to_r2(processed_dir, models_dir):
    """Upload local parquets + lookup cache to R2, atomically updating manifest."""
    from pathlib import Path
    from bts.data.sync import R2Client, sync_to_r2

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
    if report['stale'] or not report['schema_version_match']:
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
    from bts.picks import save_pick, load_streak
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
    streak = load_streak(picks_path)
    result = select_pick(predictions, date, picks_path, streak=streak)

    if result is None:
        # Skip day — post to Bluesky with top pick info
        top = predictions.iloc[0] if not predictions.empty else None
        if top is not None and pd.notna(top.get("p_game_hit")):
            from bts.posting import format_skip_post, post_to_bluesky, should_post_now
            click.echo(f"Skipping — {top['batter_name']} ({top.get('team', '?')}) "
                       f"at {top['p_game_hit']:.1%} below threshold. Streak holds at {streak}.")
            if not dry_run and should_post_now(top.get("game_time", ""), False):
                text = format_skip_post(top["batter_name"], top.get("team", "?"),
                                        top["p_game_hit"], streak)
                try:
                    uri = post_to_bluesky(text)
                    click.echo(f"  Posted skip to Bluesky: {uri}")
                except Exception as e:
                    click.echo(f"  Bluesky skip post failed: {e}", err=True)
        else:
            click.echo(f"No valid picks available. Streak holds at {streak}.")
        return

    if result.locked:
        reason = "already posted" if result.daily.bluesky_posted else "game started"
        click.echo(f"Pick locked: {result.daily.pick.batter_name} ({reason})")
        # Catch-up posting if needed
        if not result.daily.bluesky_posted:
            streak = load_streak(picks_path)
            text = format_post(
                result.daily.pick.batter_name, result.daily.pick.team,
                result.daily.pick.pitcher_name, result.daily.pick.p_game_hit, streak,
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

    # New or updated pick
    daily = result.daily
    click.echo(f"Pick: {daily.pick.batter_name} ({daily.pick.p_game_hit:.1%}) "
               f"vs {daily.pick.pitcher_name}")
    if daily.double_down:
        p_both = daily.pick.p_game_hit * daily.double_down.p_game_hit
        click.echo(f"  DOUBLE DOWN: + {daily.double_down.batter_name} "
                    f"({daily.double_down.p_game_hit:.1%}), P(both): {p_both:.1%}")

    save_pick(daily, picks_path)
    click.echo(f"  Saved to {picks_path / f'{date}.json'}")

    # Post to Bluesky if appropriate
    streak = load_streak(picks_path)
    if should_post_now(daily.pick.game_time, daily.bluesky_posted):
        text = format_post(
            daily.pick.batter_name, daily.pick.team, daily.pick.pitcher_name,
            daily.pick.p_game_hit, streak,
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
    from datetime import datetime, timedelta, timezone
    from bts.model.predict import run_pipeline, load_blend
    from bts.picks import save_pick, load_pick, load_streak
    from bts.strategy import select_pick

    if date is None:
        tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
        date = tomorrow.strftime("%Y-%m-%d")

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

    streak = load_streak(picks_path)
    result = select_pick(predictions, date, picks_path, streak=streak)

    if result is None:
        top = predictions.iloc[0]
        click.echo(f"[preview] Skip day — {top['batter_name']} at {top['p_game_hit']:.1%} below threshold.")
        return

    daily = result.daily
    save_pick(daily, picks_path)
    click.echo(f"[preview] {daily.pick.batter_name} ({daily.pick.team}) "
               f"{daily.pick.p_game_hit:.1%} vs {daily.pick.pitcher_name}")
    if daily.double_down:
        click.echo(f"[preview] + {daily.double_down.batter_name} ({daily.double_down.team}) "
                   f"{daily.double_down.p_game_hit:.1%}")
    click.echo(f"[preview] Saved to {picks_path / f'{date}.json'} (PROJECTED — scheduler will re-evaluate)")


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
    from datetime import datetime, timezone
    from bts.orchestrator import load_config
    from bts.scheduler import run_day

    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    config = load_config(Path(config_path))
    run_day(date=date, config=config, dry_run=dry_run)


@cli.command(name="check-results")
@click.option("--date", required=True, help="Date to check results for (YYYY-MM-DD)")
@click.option("--picks-dir", default="data/picks", type=click.Path(), help="Picks directory")
def check_results(date: str, picks_dir: str):
    """Check if yesterday's pick got a hit and update the streak.

    Designed to run via cron at 1am ET (after all games finish).
    """
    from pathlib import Path
    from bts.picks import load_pick, check_hit, update_streak, save_pick

    picks_path = Path(picks_dir)
    daily = load_pick(date, picks_path)

    if daily is None:
        click.echo(f"No pick found for {date}.")
        return

    # Skip if scheduler already resolved this pick (avoid double-counting streak)
    if daily.result in ("hit", "miss"):
        click.echo(f"Already resolved: {daily.pick.batter_name} — {daily.result}. Skipping.")
        return

    # Check primary pick
    click.echo(f"Checking {daily.pick.batter_name} (game {daily.pick.game_pk})...")
    try:
        primary_result = check_hit(
            daily.pick.game_pk, daily.pick.batter_id,
            batter_name=daily.pick.batter_name,
            date=date, team=daily.pick.team,
        )
    except Exception as e:
        click.echo(f"ERROR: Failed to check game result — {e}", err=True)
        return

    if primary_result is None:
        # Could be game not final OR batter scratched
        click.echo(f"WARNING: {daily.pick.batter_name} not found in boxscore or game not final. "
                    f"Streak unchanged. Check manually.")
        return

    results = [primary_result]

    # Check double-down if applicable
    if daily.double_down:
        click.echo(f"Checking {daily.double_down.batter_name} (game {daily.double_down.game_pk})...")
        try:
            double_result = check_hit(
                daily.double_down.game_pk, daily.double_down.batter_id,
                batter_name=daily.double_down.batter_name,
                date=date, team=daily.double_down.team,
            )
        except Exception as e:
            click.echo(f"ERROR: Failed to check double-down result — {e}", err=True)
            return
        if double_result is None:
            click.echo(f"WARNING: {daily.double_down.batter_name} not found in boxscore or game not final. "
                        f"Streak unchanged. Check manually.")
            return
        results.append(double_result)

    # Update streak
    new_streak = update_streak(results, picks_path)

    # Save result back to pick file
    daily.result = "hit" if all(results) else "miss"
    save_pick(daily, picks_path)

    # Check shadow pick result if shadow file exists
    from bts.picks import load_shadow_pick, save_shadow_pick
    shadow = load_shadow_pick(date, picks_path)
    if shadow and shadow.result is None:
        shadow_hit = check_hit(
            shadow.pick.game_pk, shadow.pick.batter_id,
            batter_name=shadow.pick.batter_name, date=date, team=shadow.pick.team,
        )
        if shadow_hit is not None:
            shadow.result = "hit" if shadow_hit else "miss"
            save_shadow_pick(shadow, picks_path)
            click.echo(f"  Shadow: {shadow.pick.batter_name} — {'HIT' if shadow_hit else 'MISS'}")

    # Report
    if all(results):
        hit_names = [daily.pick.batter_name]
        if daily.double_down:
            hit_names.append(daily.double_down.batter_name)
        click.echo(f"HIT! {' + '.join(hit_names)}. Streak: {new_streak}")
    else:
        miss_names = []
        if not results[0]:
            miss_names.append(daily.pick.batter_name)
        if len(results) > 1 and not results[1]:
            miss_names.append(daily.double_down.batter_name)
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
        pairs.append((date, prod, shadow))

    if not pairs:
        click.echo("No shadow pick pairs found (shadow files exist but no matching production files).")
        return

    agrees = 0
    disagrees = []
    prod_hits = 0
    shadow_hits = 0
    resolved = 0

    for date, prod, shadow in pairs:
        prod_name = prod["pick"]["batter_name"]
        shadow_name = shadow["pick"]["batter_name"]
        prod_result = prod.get("result")

        if prod_name == shadow_name:
            agrees += 1
        else:
            disagrees.append((date, prod_name, prod.get("pick", {}).get("p_game_hit"),
                              shadow_name, shadow.get("pick", {}).get("p_game_hit"),
                              prod_result))

        if prod_result in ("hit", "miss"):
            resolved += 1
            if prod_result == "hit":
                prod_hits += 1
            if prod_name == shadow_name:
                if prod_result == "hit":
                    shadow_hits += 1

    total = len(pairs)
    pct = agrees / total * 100

    click.echo(f"Shadow Model Report ({total} days, {30 - total} remaining to threshold)")
    click.echo(f"{'='*60}")
    click.echo(f"Agreement rate: {agrees}/{total} ({pct:.1f}%)")
    if resolved > 0:
        click.echo(
            f"Production day hit rate (DD-aware): "
            f"{prod_hits}/{resolved} ({prod_hits/resolved*100:.1f}%)"
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
