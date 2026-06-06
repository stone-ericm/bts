from __future__ import annotations

from datetime import date
from typing import Any

import httpx

from bts.leaderboard.endpoints import USER_AGENT, USER_PROFILE_URL_TEMPLATE

RESOLVED = {"hit", "miss", "void"}


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
