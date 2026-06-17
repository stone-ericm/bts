from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import httpx

from bts.leaderboard.endpoints import USER_AGENT, USER_PROFILE_URL_TEMPLATE

# MLB profile settles rounds as hit / not_hit / void; "miss" kept for legacy/local safety.
RESOLVED = {"hit", "not_hit", "miss", "void"}


class ContestFetchError(Exception):
    """Raised when the fetched contest profile cannot be trusted."""


def fetch_profile(
    user_id: int,
    cookies: dict[str, str],
    xsid: str,
    *,
    client: Any = httpx,
) -> dict:
    """Fetch the MLB BTS user profile and return its success payload."""
    url = USER_PROFILE_URL_TEMPLATE.format(user_id=user_id, xsid=xsid)
    response = client.get(
        url,
        cookies=cookies,
        timeout=30.0,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    response.raise_for_status()
    return response.json()["success"]


def derive_source_date(
    predictions: list[dict],
    rounds: dict[int, date],
) -> date | None:
    """Return the latest round date proven by a settled profile prediction."""
    settled_dates: list[date] = []
    for prediction in predictions:
        if prediction.get("result") not in RESOLVED:
            continue
        round_id = prediction.get("roundId")
        if round_id is None:
            continue
        round_date = rounds.get(int(round_id))
        if round_date is not None:
            settled_dates.append(round_date)
    return max(settled_dates) if settled_dates else None


def has_prediction_for(success: dict, rounds: dict[int, date], target: date) -> bool:
    """True if the profile has ANY prediction (pending included) for `target`.

    The did-the-pick-actually-get-entered check (2026-06-12 incident: a pick
    our system delivered was never entered in the MLB app; nothing alerted
    until the streak froze a day later). Unlike derive_source_date this does
    NOT filter to resolved results — a pending row proves entry.
    """
    for prediction in success.get("predictions", []):
        round_id = prediction.get("roundId")
        if round_id is None:
            continue
        if rounds.get(int(round_id)) == target:
            return True
    return False


def _require_streak_int(success: dict, field: str) -> int:
    value = success.get(field)
    if type(value) is not int:
        raise ContestFetchError(f"{field} must be an integer")
    if value < 0:
        raise ContestFetchError(f"{field} must be non-negative")
    return value


def validate_fetch(success: dict) -> None:
    """Reject malformed or internally inconsistent streak fields."""
    active_streak = _require_streak_int(success, "activeStreak")
    best_streak = _require_streak_int(success, "seasonBestStreak")
    if best_streak < active_streak:
        raise ContestFetchError("seasonBestStreak must be >= activeStreak")


def _format_recorded_at(recorded_at: datetime) -> str:
    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=timezone.utc)
    return recorded_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_observation(
    success: dict,
    source_date: date | None,
    user_id: int | None,
    username: str | None,
    recorded_at: datetime,
) -> dict:
    """Build the auto contest-streak observation persisted by the CLI."""
    if source_date is None:
        raise ContestFetchError("source_date is required")
    validate_fetch(success)
    return {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": success["activeStreak"],
        "best_streak": success["seasonBestStreak"],
        "source": "mlb_bts_profile",
        "source_date": source_date.isoformat(),
        "recorded_at": _format_recorded_at(recorded_at),
        "user_id": user_id,
        "username": username,
        "saver_available": None,
    }
