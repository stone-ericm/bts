"""Pick persistence, streak tracking, and MLB API helpers for BTS automation."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from bts.util import retry_urlopen

API_BASE = "https://statsapi.mlb.com"


@dataclass
class Pick:
    batter_name: str
    batter_id: int
    team: str
    lineup_position: int
    pitcher_name: str
    pitcher_id: int | None
    p_game_hit: float
    flags: list[str]
    projected_lineup: bool
    game_pk: int
    game_time: str  # ISO 8601 UTC


@dataclass
class DailyPick:
    date: str
    run_time: str
    pick: Pick
    double_down: Pick | None
    runner_up: dict | None  # {"batter_name": str, "p_game_hit": float}
    bluesky_posted: bool = False
    bluesky_uri: str | None = None
    result: str | None = None  # "hit", "miss", or None (pending)


def pick_from_row(row) -> Pick:
    """Create a Pick from a prediction DataFrame row."""
    flags_str = row.get("flags", "")
    flags = [f.strip() for f in flags_str.split(",") if f.strip()] if flags_str else []
    return Pick(
        batter_name=row["batter_name"],
        batter_id=int(row["batter_id"]),
        team=row["team"],
        lineup_position=int(row["lineup"]),
        pitcher_name=row["pitcher_name"],
        pitcher_id=int(row["pitcher_id"]) if row.get("pitcher_id") else None,
        p_game_hit=float(row["p_game_hit"]),
        flags=flags,
        projected_lineup="PROJECTED" in flags_str,
        game_pk=int(row["game_pk"]),
        game_time=row["game_time"],
    )


def save_pick(daily: DailyPick, picks_dir: Path) -> Path:
    """Save daily pick to JSON file."""
    picks_dir.mkdir(parents=True, exist_ok=True)
    path = picks_dir / f"{daily.date}.json"
    path.write_text(json.dumps(asdict(daily), indent=2))
    return path


def load_pick(date: str, picks_dir: Path) -> DailyPick | None:
    """Load daily pick from JSON file. Returns None if not found."""
    path = picks_dir / f"{date}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return DailyPick(
        date=data["date"],
        run_time=data["run_time"],
        pick=Pick(**data["pick"]),
        double_down=Pick(**data["double_down"]) if data["double_down"] else None,
        runner_up=data["runner_up"],
        bluesky_posted=data.get("bluesky_posted", False),
        bluesky_uri=data.get("bluesky_uri"),
        result=data.get("result"),
    )


def load_streak(picks_dir: Path) -> int:
    """Load current streak count. Returns 0 if no streak file."""
    path = picks_dir / "streak.json"
    if not path.exists():
        return 0
    return json.loads(path.read_text()).get("streak", 0)


def save_streak(streak: int, picks_dir: Path) -> None:
    """Save current streak count."""
    picks_dir.mkdir(parents=True, exist_ok=True)
    path = picks_dir / "streak.json"
    path.write_text(json.dumps({
        "streak": streak,
        "updated": datetime.now(timezone.utc).isoformat(),
    }))


def update_streak(results: list[bool], picks_dir: Path) -> int:
    """Update streak based on pick results.

    Single pick: [True] -> +1, [False] -> 0
    Double-down: [True, True] -> +2, anything else -> 0
    """
    current = load_streak(picks_dir)
    new = current + len(results) if all(results) else 0
    save_streak(new, picks_dir)
    return new


def get_game_statuses(date: str) -> dict[int, str]:
    """Get game statuses for all games on a date.

    Returns {game_pk: abstractGameCode} where codes are:
        P = Preview (not started), L = Live, F = Final
    """
    resp = json.loads(retry_urlopen(
        f"{API_BASE}/api/v1/schedule?sportId=1&date={date}",
        timeout=15,
    ).read())
    statuses = {}
    for d in resp.get("dates", []):
        for g in d.get("games", []):
            statuses[g["gamePk"]] = g["status"]["abstractGameCode"]
    return statuses


def check_hit(game_pk: int, batter_id: int) -> bool | None:
    """Check if a batter got a hit in a game.

    Returns True (hit), False (no hit), or None (game not final OR batter
    not found in boxscore, e.g. scratched).
    """
    resp = json.loads(retry_urlopen(
        f"{API_BASE}/api/v1.1/game/{game_pk}/feed/live",
        timeout=15,
    ).read())
    status = resp["gameData"]["status"]["abstractGameCode"]
    if status != "F":
        return None

    # Check boxscore batting stats
    for side in ("away", "home"):
        players = resp["liveData"]["boxscore"]["teams"][side]["players"]
        key = f"ID{batter_id}"
        if key in players:
            hits = players[key].get("stats", {}).get("batting", {}).get("hits", 0)
            return hits > 0
    # Batter not in boxscore — likely scratched
    return None
