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

# Own account's CURRENT-round predictions, INCLUDING pending (unsettled) rows.
# Discovered 2026-07-03 via the app JS bundle (Api.Prediction.get). This is the
# endpoint the profile endpoint is NOT: profiles carry settled rows only, while
# /predictions exposes the same-day entry pre-settlement — the check-pick-entered
# v2 data source. Self only; other users' pending playerIds are server-redacted.
PREDICTIONS_URL_TEMPLATE: str = (
    "https://mlb-play.mlbstatic.com/apps/beat-the-streak/game/api/predictions"
    "?xSid={xsid}"
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

# Browser-fidelity request identity (2026-07-03). This is Eric's OWN authorized
# account reading a public MLB game leaderboard for a personal project; the goal
# is only to keep the traffic from standing out as an obvious bot so the account
# isn't rate-limited or throttled at the higher request volume deep pagination
# needs. We present a normal Chrome-on-macOS session (matching Eric's real
# environment) with the headers the BTS single-page app itself sends, rather
# than a self-identifying scraper UA. No IP spoofing / proxy rotation / control
# circumvention — just request hygiene. `USER_AGENT` is kept as a back-compat
# alias but now equals the browser UA so ALL calls share one consistent identity.
BROWSER_UA: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
USER_AGENT: str = BROWSER_UA

BTS_APP_REFERER: str = "https://www.mlb.com/apps/beat-the-streak/game"


def browser_headers(accept: str = "application/json, text/plain, */*") -> dict[str, str]:
    """Headers a real BTS-app XHR carries — one consistent browser identity."""
    return {
        "User-Agent": BROWSER_UA,
        "Accept": accept,
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": BTS_APP_REFERER,
        "Origin": "https://www.mlb.com",
        "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
    }
