"""Scraper tests using captured HTTP fixtures."""
from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from bts.leaderboard.scraper import (
    parse_leaderboard_response,
    parse_user_profile_response,
    parse_rounds_lookup,
    StaticLookups,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


class TestParseLeaderboardResponse:
    def test_active_streak_parses(self):
        body = _load("leaderboard_active_streak.json")
        rows = parse_leaderboard_response(
            body, tab="active_streak", captured_at=datetime(2026, 5, 1, 14, 0),
        )
        assert len(rows) > 0
        # Top row is rank 1 highest streak
        top = rows[0]
        assert top.rank == 1
        assert top.streak >= rows[-1].streak  # ranking descending
        assert all(r.tab == "active_streak" for r in rows)
        assert top.username == "tombrady12"

    def test_yesterday_tab_uses_round_response(self):
        body = _load("leaderboard_yesterday_round.json")
        rows = parse_leaderboard_response(
            body, tab="yesterday", captured_at=datetime(2026, 5, 1, 14, 0),
        )
        assert len(rows) > 0
        assert all(r.tab == "yesterday" for r in rows)


class TestParseRoundsLookup:
    def test_returns_round_to_date_dict(self):
        body = _load("rounds.json")
        lookup = parse_rounds_lookup(body)
        # rounds.json has round 823 -> 2026-03-25
        assert lookup[823] == date(2026, 3, 25)


class TestParseUserProfileResponse:
    def test_parses_tombrady12_profile(self):
        body = _load("user_profile_595403_tombrady12.json")
        rounds = parse_rounds_lookup(_load("rounds.json"))
        lookups = StaticLookups(rounds=rounds, players={}, units={}, squads={})
        picks, stats = parse_user_profile_response(
            body, captured_at=datetime(2026, 5, 1, 14, 0),
            user_id_unused=595403, lookups=lookups,
        )
        # 36 visible picks for tombrady12's 35-game streak (1 per round, may differ by 1)
        assert len(picks) >= 30
        # Stats
        assert stats.best_streak == 35
        assert stats.active_streak == 35
        assert stats.pick_accuracy_pct == 100.0
        # First pick (most recent): roundId 859, result "hit", streak 35, primary
        most_recent = max(picks, key=lambda p: p.round_id)
        assert most_recent.round_id == 859
        assert most_recent.result == "hit"
        assert most_recent.streak_after == 35
        assert most_recent.pick_number == 1
        assert most_recent.bts_player_id == 1419
        assert most_recent.pick_date == date(2026, 4, 30)  # via rounds_lookup

    def test_resolves_names_when_lookups_populated(self):
        body = _load("user_profile_595403_tombrady12.json")
        rounds = parse_rounds_lookup(_load("rounds.json"))
        # Minimal hand-crafted players + units + squads
        players = {1419: {"id": 1419, "feedId": 666176, "name": "Juan Soto", "squadId": 14}}
        squads = {14: {"id": 14, "abbreviation": "NYM"}, 5: {"id": 5, "abbreviation": "WSH"}}
        units = {465: {"id": 465, "homeSquadId": 14, "awaySquadId": 5}}
        lookups = StaticLookups(rounds=rounds, players=players, units=units, squads=squads)
        picks, _ = parse_user_profile_response(
            body, captured_at=datetime(2026, 5, 1, 14, 0),
            user_id_unused=595403, lookups=lookups,
        )
        most_recent = max(picks, key=lambda p: p.round_id)
        assert most_recent.batter_name == "Juan Soto"
        assert most_recent.batter_id == 666176  # feedId
        assert most_recent.batter_team == "NYM"
        assert most_recent.opponent_team == "WSH"
        assert most_recent.home_or_away == "home"

    def test_unresolved_lookups_leave_optional_fields_none(self):
        body = _load("user_profile_595403_tombrady12.json")
        rounds = parse_rounds_lookup(_load("rounds.json"))
        lookups = StaticLookups(rounds=rounds, players={}, units={}, squads={})
        picks, _ = parse_user_profile_response(
            body, captured_at=datetime(2026, 5, 1, 14, 0),
            user_id_unused=595403, lookups=lookups,
        )
        most_recent = max(picks, key=lambda p: p.round_id)
        assert most_recent.batter_name is None
        assert most_recent.batter_team is None


class TestScrapeLeaderboardWithMock:
    def test_passes_cookies_and_xsid_to_get(self):
        from bts.leaderboard.scraper import scrape_leaderboard
        body = _load("leaderboard_active_streak.json")
        with patch("bts.leaderboard.scraper.httpx.get") as mock_get:
            resp = MagicMock(status_code=200)
            resp.json.return_value = body
            resp.raise_for_status = lambda: None
            mock_get.return_value = resp
            scrape_leaderboard(tab="active_streak", cookies={"a": "b"}, xsid="x_123")
        url = mock_get.call_args.args[0]
        assert "ranksType=ACTIVE_STREAK" in url
        assert "xSid=x_123" in url
        kwargs = mock_get.call_args.kwargs
        assert kwargs["cookies"] == {"a": "b"}
        assert "bts-leaderboard-watcher" in kwargs["headers"]["User-Agent"]
