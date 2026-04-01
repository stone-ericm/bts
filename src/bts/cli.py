import click
from pathlib import Path


@click.group()
def cli():
    """Beat the Streak v2 — PA-level MLB hit prediction."""
    pass


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
    from pathlib import Path
    from datetime import datetime, timezone
    from bts.model.predict import run_pipeline, save_blend, load_blend
    from bts.picks import (
        pick_from_row, save_pick, load_pick, load_streak,
        get_game_statuses, DailyPick,
    )
    from bts.posting import format_post, post_to_bluesky, should_post_now

    picks_path = Path(picks_dir)
    models_path = Path(models_dir)
    DOUBLE_THRESHOLD = 0.65

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
    import pandas as pd
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
        flags = row.get("flags", "")
        p_pa = row.get("p_hit_pa", row.get("p_game_hit", 0))
        click.echo(
            f"{shown+1:<4} {row['batter_name']:<22} {row['team']:<5} "
            f"{int(row.get('lineup', 0)):>3} {row['pitcher_name']:<22} "
            f"{p_pa:>5.1%} {row['p_game_hit']:>6.1%}  {flags}"
        )
        shown += 1

    # Step 2: Load current state
    if dry_run:
        click.echo("\n  (--dry-run: not saving or posting)")
        return

    current = load_pick(date, picks_path)
    streak = load_streak(picks_path)

    # Step 3: Filter to games not yet started
    try:
        statuses = get_game_statuses(date)
    except Exception as e:
        click.echo(f"ERROR: Failed to fetch game statuses — {e}", err=True)
        return
    not_started = predictions["game_pk"].map(lambda pk: statuses.get(pk) == "P")
    available = predictions[not_started]

    if available.empty:
        if current:
            click.echo(f"All games started. Pick locked: {current.pick.batter_name}")
        else:
            click.echo("All games started, no pick was made.")
        return

    # Step 4: Check if current pick is locked (game started OR already posted)
    if current and (statuses.get(current.pick.game_pk) != "P" or current.bluesky_posted):
        reason = "game started" if statuses.get(current.pick.game_pk) != "P" else "already posted"
        click.echo(f"Pick locked: {current.pick.batter_name} ({reason})")
        # Still post if we haven't yet — don't lose the Bluesky post
        if not current.bluesky_posted and not dry_run:
            text = format_post(
                current.pick.batter_name, current.pick.team, current.pick.pitcher_name,
                current.pick.p_game_hit, streak,
                current.double_down.batter_name if current.double_down else None,
                current.double_down.p_game_hit if current.double_down else None,
            )
            try:
                uri = post_to_bluesky(text)
                current.bluesky_posted = True
                current.bluesky_uri = uri
                save_pick(current, picks_path)
                click.echo(f"  Posted to Bluesky (catch-up): {uri}")
            except Exception as e:
                click.echo(f"  Bluesky catch-up post failed: {e}", err=True)
        return

    # Step 5: Densest bucket + asymmetric override strategy
    #
    # Rule: pick from the densest window (most games) by default.
    # Non-densest picks must exceed 78% to override. Densest picks need no threshold.
    #
    # Chronologically across the 3 daily runs (11am/4pm/7:30pm):
    # - Each run identifies the densest window from games NOT YET STARTED
    # - If the top pick is from the densest window → lock it (no threshold)
    # - If the top pick is from a non-densest window AND >78% → override, lock it
    # - If the top pick is from a non-densest window with projected lineups
    #   (e.g., west coast at 4pm) → tentatively pick, re-confirm at next run
    #
    # Validated at 86.9% avg P@1 across 6 seasons (2020-2025).
    OVERRIDE_THRESHOLD = 0.78
    valid = available[available["p_game_hit"].notna()]
    if valid.empty:
        click.echo("No batters with valid predictions available.")
        return

    if "game_time" in valid.columns:
        from datetime import timedelta as _td
        def _et_hour(gt):
            try:
                from datetime import datetime as _dt
                utc = _dt.fromisoformat(str(gt).replace("Z", "+00:00"))
                return (utc - _td(hours=4)).hour
            except:
                return 18  # default to prime
        valid = valid.copy()
        valid["_et_hour"] = valid["game_time"].apply(_et_hour)
        early = valid[valid["_et_hour"] < 16]
        prime = valid[(valid["_et_hour"] >= 16) & (valid["_et_hour"] < 20)]
        west = valid[valid["_et_hour"] >= 20]

        buckets = {"early": early, "prime": prime, "west": west}
        densest_name = max(buckets, key=lambda k: len(buckets[k]))
        densest = buckets[densest_name]

        # Top pick overall
        top_overall = valid.iloc[0]
        top_window = "early" if top_overall["_et_hour"] < 16 else ("prime" if top_overall["_et_hour"] < 20 else "west")

        if top_window == densest_name:
            # Top pick is from the densest window — take it, no threshold needed
            click.echo(f"  Densest window: {densest_name} ({len(densest)} batters) — top pick is here")
            valid = densest
        elif top_overall["p_game_hit"] > OVERRIDE_THRESHOLD:
            # Top pick is from a non-densest window but exceeds override threshold
            click.echo(f"  Override: {top_overall['batter_name']} ({top_overall['p_game_hit']:.1%}) "
                        f"from {top_window} beats densest ({densest_name})")
            # Keep full pool — the override pick is already ranked #1
        else:
            # Top pick is from a non-densest window and below threshold — use densest instead
            click.echo(f"  Densest window: {densest_name} ({len(densest)} batters) — "
                        f"top pick ({top_window}, {top_overall['p_game_hit']:.1%}) below {OVERRIDE_THRESHOLD:.0%} threshold")
            valid = densest

    best_row = valid.iloc[0]
    new_pick = pick_from_row(best_row)

    if current and current.pick.batter_id == new_pick.batter_id:
        click.echo(f"Confirmed: {new_pick.batter_name} ({new_pick.p_game_hit:.1%})")
    elif current:
        click.echo(f"Upgraded: {current.pick.batter_name} -> {new_pick.batter_name} "
                    f"({current.pick.p_game_hit:.1%} -> {new_pick.p_game_hit:.1%})")
    else:
        click.echo(f"Pick: {new_pick.batter_name} ({new_pick.p_game_hit:.1%}) "
                    f"vs {new_pick.pitcher_name}")

    # Step 6: Check for double-down
    double_pick = None
    if len(valid) >= 2:
        second_row = valid.iloc[1]
        p_both = best_row["p_game_hit"] * second_row["p_game_hit"]
        if p_both >= DOUBLE_THRESHOLD:
            double_pick = pick_from_row(second_row)
            click.echo(f"  DOUBLE DOWN: + {double_pick.batter_name} "
                        f"({double_pick.p_game_hit:.1%}), P(both): {p_both:.1%}")

    # Step 7: Build runner-up info
    runner_up = None
    if len(valid) >= 2:
        ru = valid.iloc[1]
        runner_up = {"batter_name": ru["batter_name"], "p_game_hit": float(ru["p_game_hit"])}

    # Step 8: Save pick (streak is NOT stored — read from streak.json at post time)
    daily = DailyPick(
        date=date,
        run_time=datetime.now(timezone.utc).isoformat(),
        pick=new_pick,
        double_down=double_pick,
        runner_up=runner_up,
        bluesky_posted=current.bluesky_posted if current else False,
        bluesky_uri=current.bluesky_uri if current else None,
    )
    save_pick(daily, picks_path)
    click.echo(f"  Saved to {picks_path / f'{date}.json'}")

    # Step 9: Post to Bluesky if appropriate (read streak fresh from streak.json)
    if should_post_now(new_pick.game_time, daily.bluesky_posted):
        text = format_post(
            new_pick.batter_name, new_pick.team, new_pick.pitcher_name,
            new_pick.p_game_hit, streak,
            double_pick.batter_name if double_pick else None,
            double_pick.p_game_hit if double_pick else None,
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
        primary_result = check_hit(daily.pick.game_pk, daily.pick.batter_id)
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
            double_result = check_hit(daily.double_down.game_pk, daily.double_down.batter_id)
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
