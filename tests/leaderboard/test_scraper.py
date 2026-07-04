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
        # Browser-fidelity identity (2026-07-03): looks like Chrome, not a bot UA.
        assert "Chrome/" in kwargs["headers"]["User-Agent"]
        assert kwargs["headers"]["Referer"].endswith("beat-the-streak/game")


class TestParseUserId:
    def test_user_id_populated_from_rank_rows(self):
        body = _load("leaderboard_active_streak.json")
        rows = parse_leaderboard_response(
            body, tab="active_streak", captured_at=datetime(2026, 5, 1, 14, 0))
        assert rows[0].user_id == body["success"]["ranks"][0]["userId"]
        assert all(r.user_id is not None for r in rows)


def _rank_entry(uid: int, rank: int, streak: int) -> dict:
    return {"userId": uid, "rank": rank, "username": f"u{uid}",
            "activeStreak": streak, "streak": streak}


def _body(entries: list[dict]) -> dict:
    return {"success": {"ranks": entries}}


class TestDeepScrapeRun:
    """run() with deep active-streak pagination + expanded profile tier."""

    def _setup(self, monkeypatch, tmp_path, pages: dict[int, list[dict]],
               fail_pages: set[int] = frozenset()):
        import bts.leaderboard.scraper as scraper_mod
        from bts.leaderboard.models import SeasonStats

        requested_pages: list[int] = []
        profiled: list[int] = []

        def fake_get_json(url: str, cookies=None, **kw):
            if "ranksType=ACTIVE_STREAK" in url:
                page = int(url.split("page=")[1].split("&")[0])
                requested_pages.append(page)
                if page in fail_pages:
                    raise RuntimeError(f"boom page {page}")
                return _body(pages.get(page, []))
            if "ranksType=SEASON_BEST_STREAK" in url:
                return _body([_rank_entry(101, 1, 20), _rank_entry(102, 2, 19)])
            if "ranksType=OVERALL_BEST_STREAK" in url:
                return _body([_rank_entry(201, 1, 40), _rank_entry(202, 2, 39)])
            raise AssertionError(f"unexpected _get_json url: {url}")

        def fake_profile(user_id, cookies, xsid, lookups):
            profiled.append(user_id)
            return [], SeasonStats(
                captured_at=datetime(2026, 7, 3, 14, 0), username="unknown",
                best_streak=0, active_streak=0, pick_accuracy_pct=0.0)

        monkeypatch.setattr(scraper_mod, "_get_json", fake_get_json)
        monkeypatch.setattr(scraper_mod, "_deep_page_pause", lambda: None)
        monkeypatch.setattr(scraper_mod, "scrape_static_lookups",
                            lambda cookies: StaticLookups())
        monkeypatch.setattr(scraper_mod, "scrape_user_profile", fake_profile)
        return scraper_mod, requested_pages, profiled

    def test_deep_pagination_dedupes_and_stops_on_min_streak(self, monkeypatch, tmp_path):
        import pyarrow.parquet as pq
        pages = {
            1: [_rank_entry(1, 1, 30), _rank_entry(2, 2, 29), _rank_entry(3, 3, 28)],
            2: [_rank_entry(3, 3, 28), _rank_entry(4, 4, 27), _rank_entry(5, 5, 26)],
            3: [_rank_entry(6, 6, 25), _rank_entry(7, 7, 4), _rank_entry(8, 8, 4)],
            4: [_rank_entry(99, 9, 3)],  # must never be requested
        }
        scraper_mod, requested_pages, profiled = self._setup(monkeypatch, tmp_path, pages)
        scraper_mod.run(
            cookies={}, xsid="x", output_dir=tmp_path, top_n=2,
            today=date(2026, 7, 3), deep_limit=3, deep_max_pages=10,
            deep_min_streak=5, profile_top_n=20,  # high: don't let the cap bite here
        )
        assert requested_pages == [1, 2, 3]  # stop AFTER min-streak page, no page 4
        snap = pq.read_table(
            tmp_path / "leaderboard_snapshots" / "2026-07-03.parquet").to_pandas()
        active = snap[snap["tab"] == "active_streak"]
        assert sorted(active["user_id"]) == [1, 2, 3, 4, 5, 6, 7, 8]  # deduped uid 3
        assert len(snap) == 8 + 2 + 2  # deep + all_season + all_time (yesterday skipped)
        # profile tier under a generous cap: all 8 deep active + both other tabs
        assert sorted(profiled) == [1, 2, 3, 4, 5, 6, 7, 8, 101, 102, 201, 202]

    def test_deep_stops_on_short_page(self, monkeypatch, tmp_path):
        pages = {1: [_rank_entry(1, 1, 30), _rank_entry(2, 2, 29)]}  # < deep_limit
        scraper_mod, requested_pages, _ = self._setup(monkeypatch, tmp_path, pages)
        scraper_mod.run(cookies={}, xsid="x", output_dir=tmp_path, top_n=2,
                        today=date(2026, 7, 3), deep_limit=3, deep_max_pages=10,
                        deep_min_streak=5, profile_top_n=4)
        assert requested_pages == [1]

    def test_deep_page_error_keeps_partial(self, monkeypatch, tmp_path):
        import pyarrow.parquet as pq
        pages = {1: [_rank_entry(1, 1, 30), _rank_entry(2, 2, 29), _rank_entry(3, 3, 28)]}
        scraper_mod, requested_pages, profiled = self._setup(
            monkeypatch, tmp_path, pages, fail_pages={2})
        scraper_mod.run(cookies={}, xsid="x", output_dir=tmp_path, top_n=2,
                        today=date(2026, 7, 3), deep_limit=3, deep_max_pages=10,
                        deep_min_streak=5, profile_top_n=4)
        assert requested_pages == [1, 2]
        snap = pq.read_table(
            tmp_path / "leaderboard_snapshots" / "2026-07-03.parquet").to_pandas()
        assert len(snap[snap["tab"] == "active_streak"]) == 3  # page-1 rows kept
        assert 1 in profiled and 101 in profiled  # scrape continued past the failure

    def test_profiles_capped_at_profile_top_n(self, monkeypatch, tmp_path):
        # 3 deep active users + 2 each from all_season/all_time = 7 tracked, but
        # profile_top_n=4 must cap profile fetches at 4 (footprint bound), and the
        # deep ACTIVE users (inserted first) must be among those kept.
        pages = {1: [_rank_entry(1, 1, 30), _rank_entry(2, 2, 29), _rank_entry(3, 3, 4)]}
        scraper_mod, requested_pages, profiled = self._setup(monkeypatch, tmp_path, pages)
        scraper_mod.run(cookies={}, xsid="x", output_dir=tmp_path, top_n=2,
                        today=date(2026, 7, 3), deep_limit=3, deep_max_pages=10,
                        deep_min_streak=5, profile_top_n=4)
        assert len(profiled) == 4
        assert {1, 2, 3} <= set(profiled)  # all 3 deep-active users profiled

    def test_deep_disabled_is_legacy_single_page(self, monkeypatch, tmp_path):
        import pyarrow.parquet as pq
        pages = {1: [_rank_entry(1, 1, 30), _rank_entry(2, 2, 29)]}
        scraper_mod, requested_pages, profiled = self._setup(monkeypatch, tmp_path, pages)
        scraper_mod.run(cookies={}, xsid="x", output_dir=tmp_path, top_n=2,
                        today=date(2026, 7, 3), deep_limit=3, deep_max_pages=0,
                        deep_min_streak=5, profile_top_n=20)  # high: don't let the cap bite
        assert requested_pages == [1]
        snap = pq.read_table(
            tmp_path / "leaderboard_snapshots" / "2026-07-03.parquet").to_pandas()
        assert len(snap[snap["tab"] == "active_streak"]) == 2
        assert sorted(profiled) == [1, 2, 101, 102, 201, 202]
