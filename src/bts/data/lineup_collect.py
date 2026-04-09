"""Poll MLB API for lineup confirmation times.

Runs as a periodic cron/timer. For each game scheduled on a given date,
polls the feed endpoint and records the first time both sides have
confirmed lineups (battingOrder populated for at least one player).
"""
import json
from dataclasses import dataclass

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
