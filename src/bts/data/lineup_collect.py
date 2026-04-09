"""Poll MLB API for lineup confirmation times.

Runs as a periodic cron/timer. For each game scheduled on a given date,
polls the feed endpoint and records the first time both sides have
confirmed lineups (battingOrder populated for at least one player).
"""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from bts.picks import API_BASE
from bts.util import retry_urlopen


@dataclass
class LineupPollResult:
    """Result of polling one game's current lineup status."""
    game_pk: int
    away_confirmed: bool
    home_confirmed: bool


def poll_game_lineup(game_pk: int) -> LineupPollResult:
    """Fetch one game's feed and check lineup confirmation for each side.

    A side is 'confirmed' if at least one player has battingOrder set to
    a non-empty value. Returns both=False on any API error.
    """
    try:
        raw = retry_urlopen(
            f"{API_BASE}/api/v1.1/game/{game_pk}/feed/live",
            timeout=15,
        ).read()
        data = json.loads(raw)
    except Exception:
        return LineupPollResult(game_pk=game_pk, away_confirmed=False, home_confirmed=False)

    away_confirmed = False
    home_confirmed = False
    try:
        players_by_side = data["liveData"]["boxscore"]["teams"]
        for player in players_by_side.get("away", {}).get("players", {}).values():
            if player.get("battingOrder"):
                away_confirmed = True
                break
        for player in players_by_side.get("home", {}).get("players", {}).values():
            if player.get("battingOrder"):
                home_confirmed = True
                break
    except (KeyError, TypeError):
        pass

    return LineupPollResult(
        game_pk=game_pk,
        away_confirmed=away_confirmed,
        home_confirmed=home_confirmed,
    )


@dataclass
class GameCollectionEntry:
    """Per-game collection state within one day."""
    game_pk: int
    game_time_et: str
    first_away_confirmed_utc: Optional[str] = None
    first_home_confirmed_utc: Optional[str] = None
    poll_count: int = 0


class CollectionState:
    """Stateful tracker for one day of lineup-time collection."""

    def __init__(self, date: str):
        self.date = date
        self.games: dict[int, GameCollectionEntry] = {}

    def record_poll(
        self,
        game_pk: int,
        game_time_et: str,
        poll_time_utc: datetime,
        away_confirmed: bool,
        home_confirmed: bool,
    ) -> None:
        """Update state with one poll result. First confirmation is sticky."""
        entry = self.games.get(game_pk)
        if entry is None:
            entry = GameCollectionEntry(game_pk=game_pk, game_time_et=game_time_et)
            self.games[game_pk] = entry

        if away_confirmed and entry.first_away_confirmed_utc is None:
            entry.first_away_confirmed_utc = poll_time_utc.isoformat()
        if home_confirmed and entry.first_home_confirmed_utc is None:
            entry.first_home_confirmed_utc = poll_time_utc.isoformat()
        entry.poll_count += 1

    def write_jsonl(self, out_dir: Path) -> Path:
        """Write all known entries to {date}.jsonl (one JSON object per line)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.date}.jsonl"
        lines = []
        for entry in sorted(self.games.values(), key=lambda e: e.game_pk):
            lines.append(json.dumps({
                "game_pk": entry.game_pk,
                "game_time_et": entry.game_time_et,
                "first_away_confirmed_utc": entry.first_away_confirmed_utc,
                "first_home_confirmed_utc": entry.first_home_confirmed_utc,
                "poll_count": entry.poll_count,
            }))
        out_path.write_text("\n".join(lines) + "\n" if lines else "")
        return out_path


def run_collection_tick(
    state: CollectionState,
    now_utc: datetime,
) -> None:
    """Poll games that still need confirmation. Updates state in place.

    Skips games where both sides are already confirmed (no work to do).
    """
    for game_pk, entry in list(state.games.items()):
        if entry.first_away_confirmed_utc and entry.first_home_confirmed_utc:
            continue
        result = poll_game_lineup(game_pk)
        state.record_poll(
            game_pk=game_pk,
            game_time_et=entry.game_time_et,
            poll_time_utc=now_utc,
            away_confirmed=result.away_confirmed,
            home_confirmed=result.home_confirmed,
        )
