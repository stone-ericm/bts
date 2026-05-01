"""Discovered MLB.com BTS API endpoints (Phase 1 complete 2026-05-01).

Auth flow (must execute in order):
  1. Load session cookies from platform keychain (see auth.py).
  2. Extract `uid` from the `oktaid` cookie.
  3. POST {uid, platform: "web"} to AUTH_LOGIN_URL — response body contains xSid.
  4. Use cookies + xSid query param for all data calls below.

The xSid format is `<24hex>_<unix_seconds>`. It expires; treat it as a
short-lived bearer that's refreshed at the start of each scrape run.

Each leaderboard tab maps to a distinct `ranksType` query value:
  Active Streak  -> ACTIVE_STREAK
  All Season     -> SEASON_BEST_STREAK
  All Time       -> OVERALL_BEST_STREAK
The "Yesterday" tab uses a different URL shape (LEADERBOARD_ROUND_URL_TEMPLATE)
parameterized by round_id (yesterday's round, lookup via rounds.json).

Per-user picks + season stats come from a SINGLE endpoint
(USER_PROFILE_URL_TEMPLATE). This is simpler than the original spec's
two-endpoint assumption.

pick_date is NOT in the picks payload directly. Predictions carry roundId;
rounds.json maps roundId -> date. The scraper joins these.
"""
from __future__ import annotations

# Auth
AUTH_LOGIN_URL: str = (
    "https://mlb-play.mlbstatic.com/apps/beat-the-streak/game/api/auth/login"
)

# Leaderboard for season-wide rankings (Active Streak, All Season, All Time)
LEADERBOARD_URL_TEMPLATE: str = (
    "https://mlb-play.mlbstatic.com/apps/beat-the-streak/game/api/rank/leaderboard"
    "?season={season}&page={page}&limit={limit}&usersType=ALL"
    "&ranksType={ranks_type}&xSid={xsid}"
)

# Leaderboard for a specific round (used by the "Yesterday" tab)
LEADERBOARD_ROUND_URL_TEMPLATE: str = (
    "https://mlb-play.mlbstatic.com/apps/beat-the-streak/game/api/rank/leaderboard"
    "/round/{round_id}?page={page}&limit={limit}&usersType=ALL"
    "&ranksType={ranks_type}&xSid={xsid}"
)

# Per-user profile: combines picks history + season stats in one response.
# This replaces the spec's separate USER_PICKS + USER_STATS endpoints.
USER_PROFILE_URL_TEMPLATE: str = (
    "https://mlb-play.mlbstatic.com/apps/beat-the-streak/game/api/rank/user"
    "/{user_id}/profile?xSid={xsid}"
)

# Static JSON file: maps roundId -> date. Not auth-required, refreshed by MLB.
ROUNDS_URL: str = (
    "https://mlb-play.mlbstatic.com/apps/beat-the-streak/game/json/rounds.json"
)

# Internal tab names mapped to MLB's ranksType query values
RANKS_TYPE_BY_TAB: dict[str, str] = {
    "active_streak": "ACTIVE_STREAK",
    "all_season": "SEASON_BEST_STREAK",
    "all_time": "OVERALL_BEST_STREAK",
    # "yesterday" uses LEADERBOARD_ROUND_URL_TEMPLATE with the previous day's
    # round_id (look up via rounds.json) + ranksType=ACTIVE_STREAK
    "yesterday": "ACTIVE_STREAK",
}

# Cookie holding the Okta-issued uid (passed in auth/login POST body)
OKTAID_COOKIE_NAME: str = "oktaid"

# Platform value passed in auth/login POST body
AUTH_LOGIN_PLATFORM: str = "web"

# Default User-Agent for outbound requests
USER_AGENT: str = (
    "bts-leaderboard-watcher/1.0 (research; contact: stone.ericm@gmail.com)"
)
