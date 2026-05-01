# src/bts/leaderboard/endpoints.py
"""Discovered MLB.com BTS API endpoints.

Filled in during Phase 1 by reverse-engineering the BTS app via Chrome
DevTools or the superpowers-chrome MCP. Each constant should be a
Python-format-string template parameterized on the runtime args
(date, username, tab, page).

When you discover a new endpoint, add it here, document the response
shape in models.py, and write a fixture in tests/leaderboard/fixtures/.
"""

# TODO Phase 1 — fill in actual URLs after DevTools observation
LEADERBOARD_URL_TEMPLATE: str = ""
USER_PICKS_URL_TEMPLATE: str = ""
USER_STATS_URL_TEMPLATE: str = ""
