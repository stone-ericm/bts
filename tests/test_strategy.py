"""Tests for BTS pick strategy (densest bucket + override)."""

import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd


# Force heuristic mode in all strategy tests — MDP policy file may exist on dev machines
@pytest.fixture(autouse=True)
def _disable_mdp():
    with patch("bts.strategy._load_mdp", return_value=None):
        yield

from bts.picks import Pick, DailyPick, save_pick


def _predictions(rows):
    """Build a predictions DataFrame from simplified row dicts."""
    defaults = {
        "batter_id": 100001,
        "team": "NYM",
        "lineup": 1,
        "pitcher_name": "Test Pitcher",
        "pitcher_id": 200001,
        "game_pk": 778899,
        "game_time": "2026-04-01T23:10:00Z",  # 7:10pm ET — prime window
        "p_hit_pa": 0.30,
        "flags": "",
    }
    full_rows = []
    for i, r in enumerate(rows):
        row = {**defaults, **r}
        row.setdefault("batter_name", f"Batter {i+1}")
        row.setdefault("p_game_hit", 0.75 - i * 0.02)
        full_rows.append(row)
    return pd.DataFrame(full_rows)


class TestSelectPick:
    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    def test_basic_pick(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Jacob Wilson", "p_game_hit": 0.83},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result is not None
        assert not result.locked
        assert result.daily.pick.batter_name == "Jacob Wilson"
        assert result.daily.pick.p_game_hit == 0.83

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    def test_double_down_when_threshold_met(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Wilson", "p_game_hit": 0.82, "game_pk": 778899},
            {"batter_name": "Mangum", "p_game_hit": 0.81, "game_pk": 778900},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result.daily.double_down is not None
        assert result.daily.double_down.batter_name == "Mangum"

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    def test_no_double_down_below_threshold(self, mock_statuses, tmp_path):
        """P(both) = 0.82 * 0.66 = 0.5412 < 0.55 (streak 0 threshold)."""
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Wilson", "p_game_hit": 0.82, "game_pk": 778899},
            {"batter_name": "Mangum", "p_game_hit": 0.66, "game_pk": 778900},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result.daily.double_down is None

    @patch("bts.strategy.get_game_statuses", return_value={778899: "F"})
    def test_locked_when_game_started(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        existing = DailyPick(
            date="2026-04-01",
            run_time="2026-04-01T15:00:00+00:00",
            pick=Pick(
                batter_name="Wilson", batter_id=100001, team="ATH",
                lineup_position=1, pitcher_name="Suarez", pitcher_id=200001,
                p_game_hit=0.76, flags=[], projected_lineup=False,
                game_pk=778899, game_time="2026-04-01T23:10:00Z",
            ),
            double_down=None, runner_up=None,
        )
        save_pick(existing, tmp_path)

        preds = _predictions([{"batter_name": "Wilson", "game_pk": 778899}])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result.locked
        assert result.daily.pick.batter_name == "Wilson"

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P"})
    def test_locked_when_already_posted(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        existing = DailyPick(
            date="2026-04-01",
            run_time="2026-04-01T15:00:00+00:00",
            pick=Pick(
                batter_name="Wilson", batter_id=100001, team="ATH",
                lineup_position=1, pitcher_name="Suarez", pitcher_id=200001,
                p_game_hit=0.76, flags=[], projected_lineup=False,
                game_pk=778899, game_time="2026-04-01T23:10:00Z",
            ),
            double_down=None, runner_up=None,
            bluesky_posted=True, bluesky_uri="at://did:plc:test/post/123",
        )
        save_pick(existing, tmp_path)

        preds = _predictions([{"batter_name": "Wilson", "game_pk": 778899}])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result.locked

    @patch("bts.strategy.get_game_statuses", return_value={778899: "F"})
    def test_all_games_started_no_prior_pick(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        preds = _predictions([{"game_pk": 778899}])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result is None

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    def test_runner_up_populated(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Wilson", "p_game_hit": 0.84, "game_pk": 778899},
            {"batter_name": "Mangum", "p_game_hit": 0.81, "game_pk": 778900},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result.daily.runner_up is not None
        assert result.daily.runner_up["batter_name"] == "Mangum"

    @patch("bts.strategy.get_game_statuses", return_value={
        778899: "P", 778900: "P", 778901: "P", 778902: "P",
    })
    def test_override_from_non_densest_window(self, mock_statuses, tmp_path):
        """A non-densest pick above 78% should override the densest window."""
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Early Star", "p_game_hit": 0.85,
             "game_pk": 778899, "game_time": "2026-04-01T17:10:00Z"},
            {"batter_name": "Prime 1", "p_game_hit": 0.74,
             "game_pk": 778900, "game_time": "2026-04-01T23:10:00Z"},
            {"batter_name": "Prime 2", "p_game_hit": 0.72,
             "game_pk": 778901, "game_time": "2026-04-01T23:40:00Z"},
            {"batter_name": "Prime 3", "p_game_hit": 0.70,
             "game_pk": 778902, "game_time": "2026-04-02T00:10:00Z"},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result.daily.pick.batter_name == "Early Star"

    @patch("bts.strategy.get_game_statuses", return_value={
        778899: "P", 778900: "P", 778901: "P", 778902: "P",
    })
    def test_no_override_below_threshold(self, mock_statuses, tmp_path):
        """A non-densest pick below 78% override should not override.

        The densest window top pick (0.82) is above skip threshold, so
        the day is playable. The non-densest pick (0.77) is below the
        override threshold (0.78), so densest window is used.
        """
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Early OK", "p_game_hit": 0.77,
             "game_pk": 778899, "game_time": "2026-04-01T17:10:00Z"},
            {"batter_name": "Prime 1", "p_game_hit": 0.82,
             "game_pk": 778900, "game_time": "2026-04-01T23:10:00Z"},
            {"batter_name": "Prime 2", "p_game_hit": 0.81,
             "game_pk": 778901, "game_time": "2026-04-01T23:40:00Z"},
            {"batter_name": "Prime 3", "p_game_hit": 0.80,
             "game_pk": 778902, "game_time": "2026-04-02T00:10:00Z"},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result.daily.pick.batter_name == "Prime 1"

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P"})
    def test_skip_below_threshold(self, mock_statuses, tmp_path):
        """Top pick below 0.80 skip threshold → skip day (return None)."""
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Wilson", "p_game_hit": 0.78},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result is None

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    def test_no_double_during_sprint(self, mock_statuses, tmp_path):
        """At streak 50+, no doubling even with strong picks."""
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Wilson", "p_game_hit": 0.88, "game_pk": 778899},
            {"batter_name": "Mangum", "p_game_hit": 0.85, "game_pk": 778900},
        ])
        # P(both) = 0.748 > any normal threshold, but sprint = no doubling
        result = select_pick(preds, "2026-04-01", tmp_path, streak=50)

        assert result is not None
        assert result.daily.double_down is None

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    def test_double_during_lockdown(self, mock_statuses, tmp_path):
        """At streak 35, doubling at 0.65 threshold still active."""
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Wilson", "p_game_hit": 0.85, "game_pk": 778899},
            {"batter_name": "Mangum", "p_game_hit": 0.82, "game_pk": 778900},
        ])
        # P(both) = 0.697 > 0.65 lockdown threshold
        result = select_pick(preds, "2026-04-01", tmp_path, streak=35)

        assert result.daily.double_down is not None

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P"})
    def test_empty_predictions(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        preds = pd.DataFrame()
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result is None
