"""Tests for bts.web rendering helpers — focused on _render_pa_cell.

Separated from test_web_audit_progress.py (which is scoped to the audit
endpoint) to avoid mixing concerns.
"""
from __future__ import annotations

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
        assert "5 batters" in html

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
