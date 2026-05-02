"""Tests for Pick dataclass conformal field extensions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bts.picks import Pick, DailyPick, save_pick, load_pick


def _pick_with_conformal_fields(**overrides) -> Pick:
    base = dict(
        batter_name="Juan Soto", batter_id=665742, team="NYM",
        lineup_position=2, pitcher_name="X Pitcher", pitcher_id=999,
        p_game_hit=0.78, flags=[], projected_lineup=False,
        game_pk=12345, game_time="2026-05-01T19:10:00+00:00",
        pitcher_team="WSH",
        p_game_hit_lower_conformal_95=0.62,
        p_game_hit_lower_conformal_90=0.68,
        p_game_hit_lower_conformal_80=0.72,
        p_game_hit_lower_wilson_95=0.66,
        p_game_hit_lower_wilson_90=0.70,
        p_game_hit_lower_wilson_80=0.74,
    )
    base.update(overrides)
    return Pick(**base)


class TestPickConformalFields:
    def test_pick_accepts_six_lower_bound_fields(self):
        p = _pick_with_conformal_fields()
        assert p.p_game_hit_lower_conformal_95 == 0.62
        assert p.p_game_hit_lower_wilson_95 == 0.66

    def test_pick_defaults_to_none_when_omitted(self):
        # Backward-compat: existing callers don't pass conformal fields
        p = Pick(
            batter_name="X", batter_id=1, team="A", lineup_position=1,
            pitcher_name="Y", pitcher_id=2, p_game_hit=0.7, flags=[],
            projected_lineup=False, game_pk=1, game_time="2026-05-01T00:00:00+00:00",
        )
        assert p.p_game_hit_lower_conformal_95 is None
        assert p.p_game_hit_lower_wilson_80 is None

    def test_save_and_load_roundtrip_preserves_fields(self, tmp_path):
        p = _pick_with_conformal_fields()
        daily = DailyPick(
            date="2026-05-01", run_time="2026-05-01T00:00:00+00:00",
            pick=p, double_down=None, runner_up=None,
        )
        save_pick(daily, tmp_path)
        loaded = load_pick("2026-05-01", tmp_path)
        assert loaded.pick.p_game_hit_lower_conformal_90 == 0.68
        assert loaded.pick.p_game_hit_lower_wilson_80 == 0.74

    def test_loads_old_pick_file_without_conformal_fields(self, tmp_path):
        # Backward compat: a JSON file written by older code (without these fields)
        # should still load
        old_pick_json = {
            "date": "2026-04-15",
            "run_time": "2026-04-15T00:00:00+00:00",
            "pick": {
                "batter_name": "X", "batter_id": 1, "team": "A",
                "lineup_position": 1, "pitcher_name": "Y", "pitcher_id": 2,
                "p_game_hit": 0.7, "flags": [], "projected_lineup": False,
                "game_pk": 1, "game_time": "2026-04-15T00:00:00+00:00",
                "pitcher_team": None,
            },
            "double_down": None, "runner_up": None,
            "bluesky_posted": False, "bluesky_uri": None, "result": None,
        }
        (tmp_path / "2026-04-15.json").write_text(json.dumps(old_pick_json))
        loaded = load_pick("2026-04-15", tmp_path)
        assert loaded.pick.p_game_hit == 0.7
        assert loaded.pick.p_game_hit_lower_conformal_95 is None
