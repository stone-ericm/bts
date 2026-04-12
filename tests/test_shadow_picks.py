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

    def test_load_preserves_bluesky_posted(self, tmp_path):
        """load_shadow_pick must honor the file's bluesky_posted field.

        Previously this function force-zeroed bluesky_posted and bluesky_uri,
        which masked a silent corruption bug: when the scheduler's shadow
        pipeline accidentally wrote production's DailyPick (with
        bluesky_posted=true) to the shadow file, the next load would overwrite
        those fields to false on save-back, hiding the corruption from the
        30-day shadow eval. Honest loads make corruption visible.
        """
        # Simulate a corrupted shadow file with bluesky_posted=true on disk
        corrupted = {
            "date": "2026-04-12",
            "run_time": "2026-04-12T17:19:41.741015+00:00",
            "pick": {
                "batter_name": "Brendan Donovan", "batter_id": 680977,
                "team": "SEA", "lineup_position": 1,
                "pitcher_name": "Cody Bolton", "pitcher_id": 675989,
                "p_game_hit": 0.7169, "flags": ["OPENER"],
                "projected_lineup": False, "game_pk": 823154,
                "game_time": "2026-04-12T20:10:00Z", "pitcher_team": "HOU",
            },
            "double_down": None, "runner_up": None,
            "bluesky_posted": True,
            "bluesky_uri": "at://did:plc:test/post/corrupt",
            "result": None,
        }
        (tmp_path / "2026-04-12.shadow.json").write_text(json.dumps(corrupted))

        loaded = load_shadow_pick("2026-04-12", tmp_path)
        assert loaded is not None
        assert loaded.bluesky_posted is True
        assert loaded.bluesky_uri == "at://did:plc:test/post/corrupt"
