"""Tests for bts.web rendering helpers — focused on _render_pa_cell.

Separated from test_web_audit_progress.py (which is scoped to the audit
endpoint) to avoid mixing concerns.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from bts.web import _render_pa_cell


class TestRenderPaCellPlaceholder:
    """Placeholder branch (pa is None) — driven by lineup_status + batters_away."""

    def test_on_deck(self):
        html = _render_pa_cell(None, lineup_status="on_deck", batters_away=1)
        assert "ON DECK" in html

    def test_in_hole(self):
        html = _render_pa_cell(None, lineup_status="in_hole", batters_away=2)
        assert "IN THE HOLE" in html

    def test_upcoming_distance_5(self):
        html = _render_pa_cell(None, lineup_status="upcoming", batters_away=5)
        assert "5 batters away" in html

    def test_out_of_game(self):
        html = _render_pa_cell(None, lineup_status="out_of_game")
        assert "OUT" in html

    def test_not_in_lineup(self):
        html = _render_pa_cell(None, lineup_status="not_in_lineup")
        assert "Not in lineup" in html

    def test_at_bat_renders_blank(self):
        html = _render_pa_cell(None, lineup_status="at_bat", batters_away=0)
        assert "ON DECK" not in html
        assert "OUT" not in html
        assert "batters" not in html

    def test_pre_game_renders_blank(self):
        html = _render_pa_cell(None, lineup_status="pre_game")
        assert "ON DECK" not in html
        assert "OUT" not in html

    def test_final_renders_blank(self):
        html = _render_pa_cell(None, lineup_status="final")
        assert "ON DECK" not in html
        assert "OUT" not in html

    def test_default_args_render_blank(self):
        html = _render_pa_cell(None)
        assert "ON DECK" not in html
        assert "OUT" not in html


class TestRenderPaCellFilledPrecedence:
    """Filled-cell branch must IGNORE lineup_status / batters_away args."""

    def test_filled_hit_ignores_lineup_status(self):
        pa = {
            "result": "Single",
            "is_hit": True,
            "rbi": 0,
            "pitches": [],
            "in_progress": False,
        }
        html = _render_pa_cell(pa, lineup_status="on_deck", batters_away=1)
        assert "ON DECK" not in html
        assert "Single" in html

    def test_filled_out_ignores_lineup_status(self):
        pa = {
            "result": "Strikeout",
            "is_hit": False,
            "out_number": 1,
            "rbi": 0,
            "pitches": [],
            "in_progress": False,
        }
        html = _render_pa_cell(pa, lineup_status="out_of_game")
        # The placeholder OUT badge is not added in filled cells
        assert ">OUT<" not in html

    def test_in_progress_pa_ignores_lineup_status(self):
        pa = {
            "in_progress": True,
            "pitches": [{"is_strike": True}, {"is_strike": False}],
        }
        html = _render_pa_cell(pa, lineup_status="on_deck", batters_away=1)
        assert "ON DECK" not in html
        assert "AB" in html


def test_dashboard_polling_refreshes_live_game_section(monkeypatch):
    import bts.web

    today = datetime.now().strftime("%Y-%m-%d")
    monkeypatch.setattr(bts.web, "load_streak", lambda: 0)
    monkeypatch.setattr(bts.web, "fetch_bluesky_posts", lambda: [])
    monkeypatch.setattr(bts.web, "load_scheduler_state", lambda date: {})
    monkeypatch.setattr(bts.web, "load_all_picks", lambda: [{
        "date": today,
        "pick": {
            "batter_name": "Test Batter",
            "batter_id": 1,
            "team": "BOS",
            "pitcher_name": "Pitcher",
            "p_game_hit": 0.7,
            "game_pk": 123,
            "game_time": f"{today}T23:00:00+00:00",
        },
        "double_down": None,
        "result": None,
        "bluesky_posted": False,
    }])
    monkeypatch.setattr(bts.web, "_build_live_game_data", lambda pick_data: ([
        {
            "game_status": "L",
            "inning": "Top 1st",
            "away_team": "BOS",
            "home_team": "NYY",
            "score": {"away": 0, "home": 0},
            "batters": [],
        }
    ], {
        "game_status": "L",
        "inning": "Top 1st",
        "away_team": "BOS",
        "home_team": "NYY",
        "score": {"away": 0, "home": 0},
        "batters": [],
    }))

    html = bts.web.render_page()

    assert 'id="live-game-section"' in html
    assert 'data-game-status="L"' in html
    assert 'document.getElementById("live-game-section")' in html
    assert 'fetch("/api/live-html?date=" + date)' in html
    assert 'document.getElementById("scorecard")' not in html


def test_render_page_shows_void_slot(monkeypatch):
    import bts.web

    today = datetime.now().strftime("%Y-%m-%d")
    monkeypatch.setattr(bts.web, "load_streak", lambda: 4)
    monkeypatch.setattr(bts.web, "fetch_bluesky_posts", lambda: [])
    monkeypatch.setattr(bts.web, "load_scheduler_state", lambda date: {"pick_locked": True})
    monkeypatch.setattr(bts.web, "_build_live_game_data", lambda pick_data: ([], None))
    monkeypatch.setattr(bts.web, "load_all_picks", lambda: [{
        "date": today,
        "pick": {
            "batter_name": "Voided Batter",
            "batter_id": 1,
            "team": "BOS",
            "pitcher_name": "Pitcher",
            "p_game_hit": 0.7,
            "game_pk": 123,
            "game_time": f"{today}T23:00:00+00:00",
        },
        "double_down": {
            "batter_name": "Active Batter",
            "batter_id": 2,
            "team": "ATH",
            "pitcher_name": "Pitcher 2",
            "p_game_hit": 0.69,
            "game_pk": 456,
            "game_time": f"{today}T23:10:00+00:00",
        },
        "result": "hit",
        "slot_results": {"pick": "void", "double_down": "hit"},
        "bluesky_posted": True,
    }])

    html = bts.web.render_page()

    assert "Voided Batter" in html
    assert '<span title="Void">VOID</span>' in html
    assert "HIT" in html
