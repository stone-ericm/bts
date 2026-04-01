"""MLB Stats API data pulling: schedule discovery and game feed download."""

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request

API_BASE = "https://statsapi.mlb.com"
OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"


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
    for attempt in range(3):
        try:
            resp = urlopen(url, timeout=30)
            data = resp.read()
            output_path.write_bytes(data)
            return output_path
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


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
    failed = []

    for i, game in enumerate(games):
        season = game["date"][:4]
        output_dir = data_dir / season
        try:
            path = download_game_feed(game["gamePk"], output_dir)
            paths.append(path)
        except Exception as e:
            failed.append(game["gamePk"])
            print(f"  SKIP {game['gamePk']}: {e}", file=sys.stderr)

        if delay > 0 and i < len(games) - 1:
            time.sleep(delay)

    if failed:
        print(f"  {len(failed)} games failed: {failed}", file=sys.stderr)

    return paths


def enrich_weather(season_dir: Path, delay: float = 0.3) -> int:
    """Fetch atmospheric data from Open-Meteo for all games in a season directory.

    Writes {gamePk}_weather.json sidecar files with pressure, humidity, dewpoint.
    Skips games that already have weather files.

    Returns:
        Number of games enriched
    """
    count = 0
    game_files = sorted(season_dir.glob("*.json"))

    for game_path in game_files:
        if game_path.stem.endswith("_weather"):
            continue

        weather_path = season_dir / f"{game_path.stem}_weather.json"
        if weather_path.exists():
            continue

        feed = json.loads(game_path.read_text())
        game_data = feed.get("gameData", {})

        coords = (
            game_data.get("venue", {})
            .get("location", {})
            .get("defaultCoordinates", {})
        )
        lat = coords.get("latitude")
        lon = coords.get("longitude")
        date = game_data.get("datetime", {}).get("officialDate")

        if not all([lat, lon, date]):
            continue

        url = (
            f"{OPEN_METEO_BASE}"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={date}&end_date={date}"
            f"&hourly=surface_pressure,relative_humidity_2m,dewpoint_2m"
        )

        try:
            resp = urlopen(url, timeout=15)
            data = json.loads(resp.read())
            hourly = data.get("hourly", {})

            pressures = [p for p in hourly.get("surface_pressure", []) if p is not None]
            humidities = [h for h in hourly.get("relative_humidity_2m", []) if h is not None]
            dewpoints = [d for d in hourly.get("dewpoint_2m", []) if d is not None]

            weather_data = {
                "surface_pressure": sum(pressures) / len(pressures) if pressures else None,
                "relative_humidity": sum(humidities) / len(humidities) if humidities else None,
                "dewpoint": sum(dewpoints) / len(dewpoints) if dewpoints else None,
            }

            weather_path.write_text(json.dumps(weather_data))
            count += 1

            if delay > 0:
                time.sleep(delay)

        except Exception:
            continue

    return count
