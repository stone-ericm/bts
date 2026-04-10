"""Tests for shadow pick save/load."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from bts.picks import DailyPick, Pick, save_shadow_pick, load_shadow_pick


def _make_daily(date="2026-04-10"):
    pick = Pick(
        batter_name="Luis Arraez", batter_id=650333, team="SF",
        lineup_position=2, pitcher_name="Shane Baz", pitcher_id=669358,
        p_game_hit=0.767, flags=[], projected_lineup=False,
        game_pk=824858, game_time="2026-04-10T23:15:00Z", pitcher_team="BAL",
    )
    return DailyPick(
        date=date, run_time="2026-04-10T22:32:41Z", pick=pick,
        double_down=None, runner_up=None,
    )


class TestSaveShadowPick:
    def test_saves_to_shadow_json(self, tmp_path):
        daily = _make_daily()
        path = save_shadow_pick(daily, tmp_path)
        assert path == tmp_path / "2026-04-10.shadow.json"
        assert path.exists()

    def test_content_matches_daily(self, tmp_path):
        daily = _make_daily()
        save_shadow_pick(daily, tmp_path)
        data = json.loads((tmp_path / "2026-04-10.shadow.json").read_text())
        assert data["pick"]["batter_name"] == "Luis Arraez"
        assert data["pick"]["p_game_hit"] == pytest.approx(0.767, abs=0.001)


class TestLoadShadowPick:
    def test_load_existing(self, tmp_path):
        daily = _make_daily()
        save_shadow_pick(daily, tmp_path)
        loaded = load_shadow_pick("2026-04-10", tmp_path)
        assert loaded is not None
        assert loaded.pick.batter_name == "Luis Arraez"

    def test_load_missing_returns_none(self, tmp_path):
        assert load_shadow_pick("2026-04-10", tmp_path) is None
