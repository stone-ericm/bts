# src/bts/leaderboard/cli.py
"""CLI for the BTS leaderboard watcher.

  bts leaderboard scrape    — run the daily scrape
  bts leaderboard status    — last successful scrape, lag, errors
  bts leaderboard backfill  — re-walk historical visible picks for a user
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import click

from bts.leaderboard.auth import (
    AuthError, load_session_cookies, extract_uid, fetch_xsid,
)
from bts.leaderboard.scraper import (
    run as scraper_run,
    scrape_user_profile,
    scrape_static_lookups,
)
from bts.leaderboard.storage import append_user_picks

DEFAULT_OUTPUT_DIR = Path("data/leaderboard")


@click.group()
def leaderboard():
    """BTS leaderboard watcher commands."""
    pass


@leaderboard.command()
@click.option("--output-dir", type=click.Path(), default=str(DEFAULT_OUTPUT_DIR))
@click.option("--top-n", type=int, default=100)
@click.option("--dm-recipient", default=None,
              help="Bluesky handle for auth-failure notifications")
def scrape(output_dir: str, top_n: int, dm_recipient: str | None):
    """Run a full daily scrape: 4 leaderboards + per-user profiles for top-N."""
    try:
        cookies = load_session_cookies()
        uid = extract_uid(cookies)
        xsid = fetch_xsid(uid=uid, cookies=cookies)
    except AuthError as e:
        msg = f"BTS leaderboard scrape: auth/cookie error — refresh via capture_bts_cookies.py on Mac. ({e})"
        click.echo(msg, err=True)
        if dm_recipient:
            try:
                from bts.dm import send_dm
                send_dm(dm_recipient, msg)
            except Exception as dm_err:
                click.echo(f"(DM also failed: {dm_err})", err=True)
        sys.exit(2)
    scraper_run(cookies=cookies, xsid=xsid, output_dir=Path(output_dir), top_n=top_n)
    click.echo(f"scrape complete: {datetime.utcnow().isoformat()}Z")


@leaderboard.command()
@click.option("--output-dir", type=click.Path(), default=str(DEFAULT_OUTPUT_DIR))
def status(output_dir: str):
    """Report last successful scrape time + lag."""
    od = Path(output_dir)
    snaps = sorted((od / "leaderboard_snapshots").glob("*.parquet")) if od.exists() else []
    if not snaps:
        click.echo("no successful scrape yet")
        return
    latest = snaps[-1]
    click.echo(f"last scrape: {latest.name}")
    click.echo(f"size: {latest.stat().st_size} bytes")


@leaderboard.command()
@click.argument("username")
@click.argument("user_id", type=int)
@click.option("--output-dir", type=click.Path(), default=str(DEFAULT_OUTPUT_DIR))
def backfill(username: str, user_id: int, output_dir: str):
    """Re-fetch a single user's full visible picks log."""
    try:
        cookies = load_session_cookies()
        uid = extract_uid(cookies)
        xsid = fetch_xsid(uid=uid, cookies=cookies)
    except AuthError as e:
        click.echo(f"auth error: {e}", err=True)
        sys.exit(2)
    lookups = scrape_static_lookups(cookies)
    picks, _ = scrape_user_profile(user_id, cookies=cookies, xsid=xsid, lookups=lookups)
    user_path = Path(output_dir) / "user_picks" / f"{username}.parquet"
    append_user_picks(user_path, picks)
    click.echo(f"backfilled {len(picks)} picks for {username} (user_id={user_id})")
