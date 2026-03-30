"""Parse MLB game feed JSON into plate appearance records."""

import json
import re
from pathlib import Path

import pandas as pd

from bts.data.schema import HIT_EVENTS, PA_ENDING_EVENTS


def _parse_wind(wind_str: str) -> tuple[int | None, str | None]:
    """Parse '9 mph, Out To CF' into (9, 'Out To CF')."""
    if not wind_str:
        return None, None
    match = re.match(r"(\d+)\s*mph,?\s*(.*)", wind_str)
    if match:
        return int(match.group(1)), match.group(2).strip()
    return None, wind_str


def _get_lineup_positions(boxscore: dict) -> dict[int, int]:
    """Build a map of batter_id -> lineup_position from boxscore.

    battingOrder is stored as "100", "200", etc. Divide by 100.
    Pinch hitters and substitutes may have values like "101" -> still position 1.
    """
    positions = {}
    for side in ("away", "home"):
        players = boxscore["teams"][side].get("players", {})
        for key, player in players.items():
            batting_order = player.get("battingOrder")
            if batting_order:
                batter_id = player["person"]["id"]
                positions[batter_id] = int(batting_order) // 100
    return positions


def _get_hp_umpire_id(boxscore: dict) -> int | None:
    """Extract home plate umpire ID from officials list."""
    for official in boxscore.get("officials", []):
        if official.get("officialType") == "Home Plate":
            return official["official"]["id"]
    return None


def parse_game_feed(feed: dict) -> list[dict]:
    """Parse a game feed into a list of plate appearance dicts.

    Each dict has all columns defined in schema.PA_COLUMNS.
    Pitch sequence fields are stored as Python lists.
    """
    game_data = feed["gameData"]
    live_data = feed["liveData"]
    boxscore = live_data["boxscore"]

    game_pk = game_data["game"]["pk"]
    date = game_data["datetime"]["officialDate"]
    season = int(game_data["game"]["season"])
    venue_id = game_data["venue"]["id"]
    roof_type = game_data["venue"].get("fieldInfo", {}).get("roofType")

    weather = game_data.get("weather", {})
    weather_temp_str = weather.get("temp")
    weather_temp = int(weather_temp_str) if weather_temp_str else None
    wind_speed, wind_dir = _parse_wind(weather.get("wind", ""))

    hp_umpire_id = _get_hp_umpire_id(boxscore)
    lineup_positions = _get_lineup_positions(boxscore)
    away_team_id = game_data.get("teams", {}).get("away", {}).get("id")
    home_team_id = game_data.get("teams", {}).get("home", {}).get("id")

    rows = []
    for play in live_data["plays"]["allPlays"]:
        event_type = play["result"].get("eventType", "")
        if event_type not in PA_ENDING_EVENTS:
            continue

        batter_id = play["matchup"]["batter"]["id"]
        pitcher_id = play["matchup"]["pitcher"]["id"]
        bat_side = play["matchup"].get("batSide", {}).get("code")
        pitch_hand = play["matchup"].get("pitchHand", {}).get("code")
        is_home = play["about"]["halfInning"] == "bottom"
        count = play.get("count", {})

        pitch_types = []
        pitch_calls = []
        pitch_px = []
        pitch_pz = []
        pitch_speeds = []
        pitch_end_speeds = []
        pitch_spin_rates = []
        pitch_extensions = []
        pitch_break_vertical = []
        pitch_break_horizontal = []
        sz_top = None
        sz_bottom = None
        launch_speed = None
        launch_angle = None
        trajectory = None
        hardness = None
        total_distance = None
        challenge_player_id = None
        challenge_role = None
        challenge_overturned = None
        challenge_team_batting = None

        for event in play.get("playEvents", []):
            if not event.get("isPitch"):
                continue
            details = event.get("details", {})
            pitch_data = event.get("pitchData", {})
            coords = pitch_data.get("coordinates", {})

            pitch_types.append(details.get("type", {}).get("code", "UN"))
            pitch_calls.append(details.get("call", {}).get("code", ""))
            pitch_px.append(coords.get("pX"))
            pitch_pz.append(coords.get("pZ"))
            pitch_speeds.append(pitch_data.get("startSpeed"))
            pitch_end_speeds.append(pitch_data.get("endSpeed"))
            breaks = pitch_data.get("breaks", {})
            pitch_spin_rates.append(breaks.get("spinRate"))
            pitch_extensions.append(pitch_data.get("extension"))
            pitch_break_vertical.append(breaks.get("breakVertical"))
            pitch_break_horizontal.append(breaks.get("breakHorizontal"))

            sz_top = pitch_data.get("strikeZoneTop", sz_top)
            sz_bottom = pitch_data.get("strikeZoneBottom", sz_bottom)

            hit_data = event.get("hitData")
            if hit_data:
                launch_speed = hit_data.get("launchSpeed")
                launch_angle = hit_data.get("launchAngle")
                trajectory = hit_data.get("trajectory")
                hardness = hit_data.get("hardness")
                total_distance = hit_data.get("totalDistance")

            # ABS challenge data (2026+). Only MJ type = standard ball/strike challenge.
            # NH and MA types are non-player reviews (umpire-initiated, foul ball, etc).
            rd = event.get("reviewDetails")
            if rd and rd.get("reviewType") == "MJ":
                challenge_player_id = rd.get("player", {}).get("id")
                challenge_overturned = rd.get("isOverturned")
                # Determine challenger's role in this PA
                if challenge_player_id == batter_id:
                    challenge_role = "batter"
                elif challenge_player_id == pitcher_id:
                    challenge_role = "pitcher"
                else:
                    challenge_role = "catcher"  # ~95% of non-batter/pitcher challenges
                # Determine if the challenging team is batting
                challenge_tid = rd.get("challengeTeamId")
                if challenge_tid is not None:
                    batting_tid = away_team_id if not is_home else home_team_id
                    challenge_team_batting = (challenge_tid == batting_tid)

        rows.append({
            "game_pk": game_pk,
            "date": date,
            "season": season,
            "batter_id": batter_id,
            "pitcher_id": pitcher_id,
            "bat_side": bat_side,
            "pitch_hand": pitch_hand,
            "lineup_position": lineup_positions.get(batter_id),
            "is_home": is_home,
            "hp_umpire_id": hp_umpire_id,
            "venue_id": venue_id,
            "pitch_count": len(pitch_types),
            "pitch_types": pitch_types,
            "pitch_calls": pitch_calls,
            "pitch_px": pitch_px,
            "pitch_pz": pitch_pz,
            "sz_top": sz_top,
            "sz_bottom": sz_bottom,
            "final_count_balls": count.get("balls", 0),
            "final_count_strikes": count.get("strikes", 0),
            "launch_speed": launch_speed,
            "launch_angle": launch_angle,
            "trajectory": trajectory,
            "hardness": hardness,
            "total_distance": total_distance,
            "pitch_speeds": pitch_speeds,
            "pitch_end_speeds": pitch_end_speeds,
            "pitch_spin_rates": pitch_spin_rates,
            "pitch_extensions": pitch_extensions,
            "pitch_break_vertical": pitch_break_vertical,
            "pitch_break_horizontal": pitch_break_horizontal,
            "challenge_player_id": challenge_player_id,
            "challenge_role": challenge_role,
            "challenge_overturned": challenge_overturned,
            "challenge_team_batting": challenge_team_batting,
            "event_type": event_type,
            "is_hit": 1 if event_type in HIT_EVENTS else 0,
            "weather_temp": weather_temp,
            "weather_wind_speed": wind_speed,
            "weather_wind_dir": wind_dir,
            "roof_type": roof_type,
            "atm_pressure": None,
            "humidity": None,
        })

    return rows


REGULAR_SEASON_TYPE = "R"


def build_season(
    raw_dir: Path,
    output_path: Path,
    season: int,
    game_types: set[str] | None = None,
) -> pd.DataFrame:
    """Build PA-level Parquet from raw game feed JSONs for one season.

    Args:
        raw_dir: Root raw directory (looks in raw_dir/{season}/*.json)
        output_path: Path to write the Parquet file
        season: Year to process
        game_types: Set of game type codes to include (default: {"R"} for regular season only).
            Codes: R=regular, S=spring training, E=exhibition, F=wild card,
            D=division series, L=league championship, W=world series, A=all-star.

    Returns:
        The built DataFrame
    """
    if game_types is None:
        game_types = {REGULAR_SEASON_TYPE}

    season_dir = raw_dir / str(season)
    if not season_dir.exists():
        raise FileNotFoundError(f"No raw data for season {season} at {season_dir}")

    all_rows = []
    skipped = 0
    json_files = sorted(season_dir.glob("*.json"))

    for json_path in json_files:
        # Skip weather sidecar files
        if json_path.stem.endswith("_weather"):
            continue
        feed = json.loads(json_path.read_text())

        # Filter by game type
        game_info = feed.get("gameData", {}).get("game", {})
        gt = game_info.get("type", "")
        if gt not in game_types:
            skipped += 1
            continue

        # Skip 7-inning doubleheader games (2020-2021 COVID rule).
        # The API reports scheduledInnings=9 even for these, so we detect
        # them by: doubleheader flag + actual innings played <= 7.
        dh = game_info.get("doubleHeader", "N")
        if dh in ("Y", "S"):
            plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
            max_inning = max((p["about"]["inning"] for p in plays), default=9)
            if max_inning <= 7:
                skipped += 1
                continue

        rows = parse_game_feed(feed)

        # Merge weather sidecar if present
        weather_path = json_path.parent / f"{json_path.stem}_weather.json"
        if weather_path.exists():
            weather = json.loads(weather_path.read_text())
            for row in rows:
                row["atm_pressure"] = weather.get("surface_pressure")
                row["humidity"] = weather.get("relative_humidity")

        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df
