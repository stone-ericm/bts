"""Dynamic lineup scheduler for BTS.

Replaces fixed cron runs with game-time-aware lineup checks.
Checks lineups 45 min before each game, clusters nearby checks,
and commits picks only when confirmed lineup + gap threshold met.
"""

import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
from zoneinfo import ZoneInfo

from bts.util import retry_urlopen
from bts.picks import API_BASE

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def fetch_schedule(date: str) -> list[dict]:
    """Fetch today's MLB schedule. Returns list of game dicts."""
    resp = json.loads(retry_urlopen(
        f"{API_BASE}/api/v1/schedule?sportId=1&date={date}&hydrate=probablePitcher",
        timeout=15,
    ).read())
    games = []
    for d in resp.get("dates", []):
        games.extend(d.get("games", []))
    return games


def _game_time_et(game: dict) -> datetime:
    """Extract game time as ET datetime."""
    utc = datetime.fromisoformat(game["gameDate"].replace("Z", "+00:00"))
    return utc.astimezone(ET)


def compute_run_times(
    games: list[dict],
    offset_min: int = 45,
    cluster_min: int = 10,
) -> list[dict]:
    """Compute clustered lineup check times from game schedule.

    For each game, the check time is game_time - offset_min.
    Checks within cluster_min of each other are merged into one run.

    Returns list of {"time_et": datetime, "game_pks": [int, ...]}
    sorted by time.
    """
    if not games:
        return []

    checks = []
    for g in games:
        et = _game_time_et(g)
        check_time = et - timedelta(minutes=offset_min)
        checks.append({"time_et": check_time, "game_pk": g["gamePk"]})

    checks.sort(key=lambda c: c["time_et"])

    clusters = []
    current = {"time_et": checks[0]["time_et"], "game_pks": [checks[0]["game_pk"]]}

    for c in checks[1:]:
        if (c["time_et"] - current["time_et"]) <= timedelta(minutes=cluster_min):
            current["game_pks"].append(c["game_pk"])
        else:
            clusters.append(current)
            current = {"time_et": c["time_et"], "game_pks": [c["game_pk"]]}

    clusters.append(current)
    return clusters


def detect_doubleheader_game2s(games: list[dict]) -> set[int]:
    """Detect game 2 of doubleheaders (fluid start time).

    Returns set of game_pks that are doubleheader game 2s.
    Detected by finding two games with the same away+home team pair.
    """
    team_games = {}
    for g in games:
        away = g["teams"]["away"]["team"]["name"]
        home = g["teams"]["home"]["team"]["name"]
        key = (away, home)
        team_games.setdefault(key, []).append(g)

    game2s = set()
    for key, team_g in team_games.items():
        if len(team_g) >= 2:
            team_g.sort(key=lambda x: _game_time_et(x))
            for g in team_g[1:]:
                game2s.add(g["gamePk"])

    return game2s


def compute_wakeup_time(
    games: list[dict],
    default_hour_et: int = 10,
    early_buffer_min: int = 60,
) -> datetime:
    """Compute scheduler wake-up time based on earliest game.

    If any game starts before the default init hour, wakes up
    early_buffer_min before the earliest game.
    """
    today_et = datetime.now(ET).replace(hour=default_hour_et, minute=0, second=0, microsecond=0)

    if not games:
        return today_et

    earliest = min(_game_time_et(g) for g in games)
    early_wake = earliest - timedelta(minutes=early_buffer_min)

    if early_wake < today_et:
        return early_wake

    return today_et


def check_confirmed_lineups(game_pks: list[int]) -> dict[int, bool]:
    """Check which games have confirmed lineups posted.

    A lineup is confirmed if the boxscore has players with battingOrder set.
    Returns {game_pk: has_confirmed_lineup}.
    """
    results = {}
    for pk in game_pks:
        try:
            resp = json.loads(retry_urlopen(
                f"{API_BASE}/api/v1.1/game/{pk}/feed/live",
                timeout=15,
            ).read())
            has_lineup = False
            for side in ("away", "home"):
                players = resp["liveData"]["boxscore"]["teams"][side]["players"]
                for pid, pdata in players.items():
                    if pdata.get("battingOrder"):
                        has_lineup = True
                        break
                if has_lineup:
                    break
            results[pk] = has_lineup
        except Exception:
            results[pk] = False

    return results


@dataclass
class SchedulerState:
    """Daily scheduler state, persisted to JSON."""
    date: str
    schedule_fetched_at: str
    games: list[dict]  # [{game_pk, game_time_et, lineup_confirmed, is_doubleheader_game2}]
    confirmed_game_pks: list[int]
    runs_completed: list[dict]  # [{time, new_lineups, skipped}]
    pick_locked: bool
    pick_locked_at: str | None
    result_status: str | None  # "final", "suspended", "unresolved", None
    next_wakeup: str | None  # ISO for next day's wake-up


def save_state(state: SchedulerState, picks_dir: Path) -> Path:
    """Save scheduler state to JSON."""
    date_dir = picks_dir / state.date
    date_dir.mkdir(parents=True, exist_ok=True)
    path = date_dir / "scheduler_state.json"
    path.write_text(json.dumps(asdict(state), indent=2))
    return path


def load_state(date: str, picks_dir: Path) -> SchedulerState | None:
    """Load scheduler state from JSON. Returns None if not found."""
    path = picks_dir / date / "scheduler_state.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return SchedulerState(**data)


def count_new_confirmations(
    game_pks: list[int],
    previously_confirmed: set[int],
) -> int:
    """Check for new lineup confirmations since last check.

    Updates previously_confirmed in place. Returns count of newly confirmed games.
    """
    statuses = check_confirmed_lineups(game_pks)
    new_count = 0
    for pk, confirmed in statuses.items():
        if confirmed and pk not in previously_confirmed:
            previously_confirmed.add(pk)
            new_count += 1
    return new_count
