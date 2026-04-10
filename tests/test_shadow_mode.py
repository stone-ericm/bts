"""Tests for scheduler shadow mode."""
import json
from pathlib import Path

import pytest


def test_shadow_mode_writes_to_shadow_dir(tmp_path):
    """In shadow mode, picks are written to data/shadow/{date}/ not data/picks/."""
    from bts.picks import save_pick_shadow

    shadow_dir = tmp_path / "shadow"
    pick_data = {
        "date": "2026-04-10",
        "pick": {"batter_name": "Test", "batter_id": 100, "team": "NYY"},
        "result": None,
    }

    save_pick_shadow(pick_data, shadow_dir=shadow_dir, source="fly")

    out = shadow_dir / "2026-04-10" / "fly.json"
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["pick"]["batter_name"] == "Test"


def test_shadow_pick_round_trips(tmp_path):
    """Shadow pick files can be loaded back as JSON."""
    from bts.picks import save_pick_shadow

    shadow_dir = tmp_path / "shadow"
    pick_data = {
        "date": "2026-04-10",
        "pick": {"batter_name": "Hoerner", "batter_id": 200, "team": "CHC",
                 "pitcher_name": "Pitcher", "pitcher_id": 300, "game_pk": 999,
                 "game_time": "2026-04-10T19:05:00-04:00", "p_game_hit": 0.85},
        "double_down": None,
        "result": None,
    }

    path = save_pick_shadow(pick_data, shadow_dir=shadow_dir, source="fly")
    loaded = json.loads(path.read_text())
    assert loaded["pick"]["p_game_hit"] == 0.85
    assert loaded["double_down"] is None
