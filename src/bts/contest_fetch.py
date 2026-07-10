from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import httpx

from bts.leaderboard.endpoints import (
    PREDICTIONS_URL_TEMPLATE,
    USER_PROFILE_URL_TEMPLATE,
    browser_headers,
)

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
        headers=browser_headers(),
    )
    response.raise_for_status()
    return response.json()["success"]


def fetch_pending_predictions(
    cookies: dict[str, str],
    xsid: str,
    *,
    client: Any = httpx,
) -> list[dict]:
    """Fetch OWN current-round predictions, pending rows included.

    GET api/predictions is the only discovered endpoint that shows a same-day
    entry before settlement (the profile endpoint is settled-only — the reason
    check-pick-entered v1 false-alarmed and was disabled 2026-06-12). Rows are
    flat {roundId, unitId, playerId, number, result, ...}; result is null
    while pending.
    """
    url = PREDICTIONS_URL_TEMPLATE.format(xsid=xsid)
    response = client.get(
        url,
        cookies=cookies,
        timeout=30.0,
        headers=browser_headers(),
    )
    response.raise_for_status()
    body = response.json()
    # Distinguish "authenticated, no pending pick" (success.predictions == [])
    # from schema drift / an error envelope. A drifted 200 must NOT collapse to
    # [] — that would read as "no pick entered" and fire a false alarm (the v1
    # class). Raise so the caller's fetch-failed path skips quietly instead.
    if not isinstance(body, dict) or "success" not in body:
        raise ContestFetchError(f"predictions response missing 'success': {str(body)[:200]}")
    success = body["success"]
    if not isinstance(success, dict) or "predictions" not in success:
        raise ContestFetchError("predictions response missing success.predictions")
    predictions = success["predictions"]
    if not isinstance(predictions, list):
        raise ContestFetchError("success.predictions is not a list")
    return predictions


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


def entered_bts_player_ids(
    profile_success: dict,
    pending_rows: list[dict],
    rounds: dict[int, date],
    target: date,
) -> list[int]:
    """BTS playerIds entered for `target`, from BOTH endpoint shapes.

    The profile endpoint nests picks under `predictions[].roundPredictions[]`
    (settled-only); GET /predictions returns FLAT rows {roundId, playerId, ...}
    (includes pending). Union both so a same-day entry is found pre-settlement.
    """
    def _round_matches(round_id) -> bool:
        if round_id is None:
            return False
        return rounds.get(int(round_id)) == target

    ids: list[int] = []
    for pred in profile_success.get("predictions", []):
        if _round_matches(pred.get("roundId")):
            for rp in pred.get("roundPredictions", []):
                if rp.get("playerId") is not None:
                    ids.append(int(rp["playerId"]))
    for row in pending_rows:
        if _round_matches(row.get("roundId")) and row.get("playerId") is not None:
            ids.append(int(row["playerId"]))
    return ids


def pick_entry_status(
    profile_success: dict,
    pending_rows: list[dict],
    rounds: dict[int, date],
    target: date,
    required_mlb_ids: set[int],
    bts_to_mlb: dict[int, int],
) -> tuple[bool, str]:
    """Is the DELIVERED pick entered? Returns (ok, reason).

    Eric always intends the entered pick to equal the recommendation, so a
    mismatch (wrong player, or a missing double-down slot) is a real anomaly to
    surface — NOT a false alarm. But our BTS->MLB crosswalk can be incomplete;
    when we can't resolve an entered row we fall back to presence-only so OUR
    gap never fires a false "not entered". Reasons:
      no_pick            — nothing entered for target -> alert
      match              — every required MLB id is present -> ok
      present_unverified — something entered but crosswalk can't confirm -> ok
      mismatch           — resolved, and a required id is missing -> alert
    """
    entered_bts = entered_bts_player_ids(profile_success, pending_rows, rounds, target)
    if not entered_bts:
        return False, "no_pick"
    if required_mlb_ids and len(entered_bts) < len(required_mlb_ids):
        # Fewer rows entered than slots delivered: a slot is missing no matter
        # how the crosswalk resolves identities — unverifiable identity must
        # not mask a missing double-down (Codex review 2026-07-09 #1).
        return False, "mismatch"
    if not required_mlb_ids:
        return True, "present_unverified"  # nothing to match against
    entered_mlb = {bts_to_mlb[b] for b in entered_bts if b in bts_to_mlb}
    if required_mlb_ids <= entered_mlb:
        return True, "match"
    unresolved = [b for b in entered_bts if b not in bts_to_mlb]
    if unresolved or not bts_to_mlb:
        return True, "present_unverified"  # can't prove a mismatch -> don't false-alarm
    return False, "mismatch"


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
    """Build the auto contest-streak observation persisted by the CLI.

    Snapshot/coverage split: the activeStreak snapshot persists even when ledger
    coverage is unknown. The per-round predictions array lags MLB's own counter, so
    ``source_date`` may be None; contest_state treats a null source_date as stale
    (conservative) rather than discarding a current streak.
    """
    validate_fetch(success)
    return {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": success["activeStreak"],
        "best_streak": success["seasonBestStreak"],
        "source": "mlb_bts_profile",
        "source_date": source_date.isoformat() if source_date is not None else None,
        "recorded_at": _format_recorded_at(recorded_at),
        "user_id": user_id,
        "username": username,
        "saver_available": None,
    }
