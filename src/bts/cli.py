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
@click.option("--top", default=15, type=int, help="Number of picks to show")
@click.option("--no-opener-check", is_flag=True, help="Skip opener detection (faster)")
def predict(date: str, data_dir: str, top: int, no_opener_check: bool):
    """Generate ranked BTS picks for a date."""
    import pandas as pd
    from bts.model.predict import run_pipeline

    click.echo(f"Running prediction pipeline for {date}...")
    try:
        picks = run_pipeline(date, data_dir, check_openers=not no_opener_check)
    except RuntimeError as e:
        click.echo(str(e))
        return

    if picks.empty:
        click.echo("No games found for this date.")
        return

    click.echo(f"\n{'='*80}")
    click.echo(f"BTS PICKS — {date}")
    click.echo(f"{'='*80}")
    click.echo(f"{'#':<4} {'Batter':<22} {'Team':<5} {'Pos':>3} {'vs Pitcher':<22} {'P(PA)':>6} {'P(Game)':>7}  {'Flags'}")
    click.echo(f"{'-'*80}")

    shown = 0
    for _, row in picks.iterrows():
        if shown >= top:
            break
        if pd.isna(row.get("p_game_hit")):
            continue
        flags = row.get("flags", "")
        click.echo(
            f"{shown+1:<4} {row['batter_name']:<22} {row['team']:<5} "
            f"{int(row['lineup']):>3} {row['pitcher_name']:<22} "
            f"{row['p_hit_pa']:>5.1%} {row['p_game_hit']:>6.1%}  {flags}"
        )
        shown += 1

    # Recommendation: 1 or 2 picks based on P(both hit)
    DOUBLE_THRESHOLD = 0.65
    best = picks.iloc[0]
    valid_picks = picks[picks["p_game_hit"].notna()]

    if len(valid_picks) >= 2:
        second = valid_picks.iloc[1]
        p_both = best["p_game_hit"] * second["p_game_hit"]

        if p_both >= DOUBLE_THRESHOLD:
            click.echo(f"\nDOUBLE DOWN: {best['batter_name']} ({best['p_game_hit']:.1%}) "
                        f"+ {second['batter_name']} ({second['p_game_hit']:.1%})")
            click.echo(f"  P(both hit): {p_both:.1%}")
        else:
            click.echo(f"\nSingle pick: {best['batter_name']} ({best['p_game_hit']:.1%})")
            click.echo(f"  P(both hit) with #{second['batter_name']}: {p_both:.1%} (below {DOUBLE_THRESHOLD:.0%} threshold)")
    else:
        click.echo(f"\nSingle pick: {best['batter_name']} ({best['p_game_hit']:.1%})")

    if best.get("flags"):
        click.echo(f"  WARNING: {best['flags']}")


@cli.command()
@click.option("--date", required=True, help="Date of the pick (YYYY-MM-DD)")
@click.option("--batter", required=True, help="Batter name")
@click.option("--team", required=True, help="Team abbreviation")
@click.option("--pitcher", required=True, help="Opposing pitcher name")
@click.option("--pct", required=True, type=float, help="P(game hit) percentage")
@click.option("--streak", required=True, type=int, help="Current streak count")
@click.option("--double", default=None, help="Second pick name for double down (optional)")
@click.option("--double-pct", default=None, type=float, help="Second pick P(game hit)")
@click.option("--dry-run", is_flag=True, help="Print post text without posting")
def post(date: str, batter: str, team: str, pitcher: str, pct: float,
         streak: int, double: str, double_pct: float, dry_run: bool):
    """Post today's pick to Bluesky."""
    from bts.posting import format_post, post_to_bluesky

    p_game = pct / 100
    double_p = double_pct / 100 if double_pct else None
    text = format_post(batter, team, pitcher, p_game, streak, double, double_p)

    if dry_run:
        click.echo(f"Would post:\n{text}")
        return

    try:
        uri = post_to_bluesky(text)
        click.echo(f"Posted: {uri}")
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
