import click
from pathlib import Path


@click.group()
def cli():
    """Beat the Streak v2 — PA-level MLB hit prediction."""
    pass


from bts.simulate.cli import simulate
cli.add_command(simulate)


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




@cli.command()
@click.option("--date", required=True, help="Date to predict (YYYY-MM-DD)")
@click.option("--data-dir", default="data/processed", type=click.Path(), help="Processed data directory")
@click.option("--picks-dir", default="data/picks", type=click.Path(), help="Picks output directory")
@click.option("--models-dir", default="data/models", type=click.Path(), help="Cached models directory")
@click.option("--top", default=10, type=int, help="Number of ranked picks to show")
@click.option("--dry-run", is_flag=True, help="Print rankings only — don't save pick or post to Bluesky")
def run(date: str, data_dir: str, picks_dir: str, models_dir: str, top: int, dry_run: bool):
    """Run daily BTS automation: predict, save pick, post to Bluesky.

    Designed to run via cron at 11am, 4pm, and 7:30pm ET.
    Uses densest-bucket strategy with 78% override threshold.
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

    # Reply to original Bluesky post with result
    if daily.bluesky_uri:
        from bts.posting import format_result_reply, reply_to_bluesky
        reply_text = format_result_reply(daily.result, new_streak)
        try:
            reply_uri = reply_to_bluesky(reply_text, daily.bluesky_uri)
            click.echo(f"  Result reply posted: {reply_uri}")
        except Exception as e:
            click.echo(f"  Result reply failed: {e}", err=True)
