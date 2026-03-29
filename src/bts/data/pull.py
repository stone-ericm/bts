"""MLB Stats API data pulling: schedule discovery and game feed download."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request

API_BASE = "https://statsapi.mlb.com"


def discover_games(start_date: str, end_date: str) -> list[dict]:
    """Discover completed MLB games in a date range.

    Args:
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD

    Returns:
        List of dicts with keys: gamePk, date
    """
    games = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        url = f"{API_BASE}/api/v1/schedule?sportId=1&date={date_str}"
        resp = urlopen(url, timeout=15)
        data = json.loads(resp.read())

        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                if game["status"]["detailedState"] == "Final":
                    games.append({
                        "gamePk": game["gamePk"],
                        "date": game.get("officialDate", date_str),
                    })

        current += timedelta(days=1)

    return games


def download_game_feed(game_pk: int, output_dir: Path) -> Path:
    """Download a single game feed JSON. Skips if already exists.

    Args:
        game_pk: MLB game PK identifier
        output_dir: Directory to write {game_pk}.json into

    Returns:
        Path to the JSON file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{game_pk}.json"

    if output_path.exists():
        return output_path

    url = f"{API_BASE}/api/v1.1/game/{game_pk}/feed/live"
    resp = urlopen(url, timeout=30)
    data = resp.read()

    output_path.write_bytes(data)
    return output_path


def pull_feeds(
    start_date: str,
    end_date: str,
    data_dir: Path,
    delay: float = 0.5,
) -> list[Path]:
    """Pull all game feeds for a date range.

    Args:
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        data_dir: Root data directory (files go to data_dir/{season}/{gamePk}.json)
        delay: Seconds between API requests

    Returns:
        List of paths to downloaded JSON files
    """
    games = discover_games(start_date, end_date)
    paths = []

    for i, game in enumerate(games):
        season = game["date"][:4]
        output_dir = data_dir / season
        path = download_game_feed(game["gamePk"], output_dir)
        paths.append(path)

        if delay > 0 and i < len(games) - 1:
            time.sleep(delay)

    return paths
