"""Pick persistence, streak tracking, and MLB API helpers for BTS automation."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


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
    streak: int
    bluesky_posted: bool = False
    bluesky_uri: str | None = None


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
        streak=data["streak"],
        bluesky_posted=data.get("bluesky_posted", False),
        bluesky_uri=data.get("bluesky_uri"),
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
