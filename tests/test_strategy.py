"""Tests for BTS pick strategy."""

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

    @patch("bts.picks.get_game_statuses_detailed", return_value={
        778899: {"abstract": "F", "detailed": "Final"},
    })
    @patch("bts.strategy.get_game_statuses", return_value={778899: "F"})
    def test_locked_when_game_started(self, mock_statuses, _detailed_statuses, tmp_path):
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

    @patch("bts.picks.get_game_statuses_detailed", return_value={
        778899: {"abstract": "F", "detailed": "Postponed"},
        778900: {"abstract": "P", "detailed": "Pre-Game"},
    })
    @patch("bts.strategy.get_game_statuses", return_value={778899: "F", 778900: "P"})
    def test_existing_postponed_unposted_reselects_fresh_candidate(
        self, mock_statuses, _detailed_statuses, tmp_path
    ):
        from bts.strategy import select_pick

        existing = DailyPick(
            date="2026-04-01",
            run_time="2026-04-01T15:00:00+00:00",
            pick=Pick(
                batter_name="Stale Pick", batter_id=100001, team="ATH",
                lineup_position=1, pitcher_name="Suarez", pitcher_id=200001,
                p_game_hit=0.76, flags=[], projected_lineup=False,
                game_pk=778899, game_time="2026-04-01T23:10:00Z",
            ),
            double_down=None, runner_up=None,
        )
        save_pick(existing, tmp_path)

        preds = _predictions([
            {"batter_name": "Fresh Pick", "p_game_hit": 0.84, "game_pk": 778900},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result is not None
        assert result.locked is False
        assert result.daily.pick.batter_name == "Fresh Pick"

    @patch("bts.picks.get_game_statuses_detailed", return_value={
        778899: {"abstract": "P", "detailed": "Pre-Game"},
    })
    @patch("bts.strategy.get_game_statuses", return_value={778899: "P"})
    def test_existing_preview_unposted_does_not_lock(self, mock_statuses, _detailed_statuses, tmp_path):
        from bts.strategy import select_pick

        existing = DailyPick(
            date="2026-04-01",
            run_time="2026-04-01T15:00:00+00:00",
            pick=Pick(
                batter_name="Morning Pick", batter_id=100001, team="ATH",
                lineup_position=1, pitcher_name="Suarez", pitcher_id=200001,
                p_game_hit=0.76, flags=[], projected_lineup=True,
                game_pk=778899, game_time="2026-04-01T23:10:00Z",
            ),
            double_down=None, runner_up=None,
        )
        save_pick(existing, tmp_path)

        preds = _predictions([
            {"batter_name": "Refreshed Pick", "p_game_hit": 0.84, "game_pk": 778899},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result is not None
        assert result.locked is False
        assert result.daily.pick.batter_name == "Refreshed Pick"

    @pytest.mark.parametrize(
        ("abstract", "detailed"),
        [
            ("L", "Warmup"),
            ("L", "In Progress"),
            ("L", "Suspended"),
            ("F", "Final"),
            ("F", "Game Over"),
        ],
    )
    def test_existing_started_or_final_statuses_lock(self, abstract, detailed, tmp_path):
        from bts.strategy import select_pick

        existing = DailyPick(
            date="2026-04-01",
            run_time="2026-04-01T15:00:00+00:00",
            pick=Pick(
                batter_name="Locked Pick", batter_id=100001, team="ATH",
                lineup_position=1, pitcher_name="Suarez", pitcher_id=200001,
                p_game_hit=0.76, flags=[], projected_lineup=False,
                game_pk=778899, game_time="2026-04-01T23:10:00Z",
            ),
            double_down=None, runner_up=None,
        )
        save_pick(existing, tmp_path)

        preds = _predictions([{"batter_name": "New Pick", "game_pk": 778899}])
        with patch("bts.picks.get_game_statuses_detailed", return_value={
            778899: {"abstract": abstract, "detailed": detailed},
        }), patch("bts.strategy.get_game_statuses", return_value={778899: abstract}):
            result = select_pick(preds, "2026-04-01", tmp_path)

        assert result is not None
        assert result.locked is True
        assert result.daily.pick.batter_name == "Locked Pick"

    @patch("bts.picks.get_game_statuses", side_effect=OSError("abstract unavailable"))
    @patch("bts.picks.get_game_statuses_detailed", side_effect=OSError("detailed unavailable"))
    def test_existing_status_lookup_failure_locks(self, _detailed_statuses, _abstract_statuses, tmp_path):
        from bts.strategy import select_pick

        existing = DailyPick(
            date="2026-04-01",
            run_time="2026-04-01T15:00:00+00:00",
            pick=Pick(
                batter_name="Conservative Pick", batter_id=100001, team="ATH",
                lineup_position=1, pitcher_name="Suarez", pitcher_id=200001,
                p_game_hit=0.76, flags=[], projected_lineup=False,
                game_pk=778899, game_time="2026-04-01T23:10:00Z",
            ),
            double_down=None, runner_up=None,
        )
        save_pick(existing, tmp_path)

        preds = _predictions([{"batter_name": "New Pick", "game_pk": 778899}])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result is not None
        assert result.locked is True
        assert result.daily.pick.batter_name == "Conservative Pick"

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

    @patch("bts.picks.get_game_statuses_detailed", return_value={
        778899: {"abstract": "P", "detailed": "Pre-Game"},
    })
    def test_locked_when_pick_already_dm_notified(self, mock_statuses, tmp_path):
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
            notification_sent=True,
            notification_channel="bluesky_dm",
            notification_id="msg-123",
        )
        save_pick(existing, tmp_path)

        preds = _predictions([{"batter_name": "Wilson", "game_pk": 778899}])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result.locked

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    def test_for_shadow_ignores_locked_production(self, mock_statuses, tmp_path):
        """When called with for_shadow=True, select_pick must NOT short-circuit
        on production's lock state — the shadow model must always compute its
        own pick from its own predictions, even when production is already
        locked and posted.

        Regression: previously, the shadow path's select_pick call would load
        the production pick file, see bluesky_posted=True, and return the
        production DailyPick. The shadow predictions were silently discarded
        and {date}.shadow.json became a copy of {date}.json.
        """
        from bts.strategy import select_pick

        # Production is LOCKED with bluesky_posted=True and game still in "P"
        existing = DailyPick(
            date="2026-04-01",
            run_time="2026-04-01T15:00:00+00:00",
            pick=Pick(
                batter_name="Production Pick", batter_id=100001, team="ATH",
                lineup_position=1, pitcher_name="Suarez", pitcher_id=200001,
                p_game_hit=0.76, flags=[], projected_lineup=False,
                game_pk=778899, game_time="2026-04-01T23:10:00Z",
            ),
            double_down=None, runner_up=None,
            bluesky_posted=True, bluesky_uri="at://did:plc:test/post/123",
        )
        save_pick(existing, tmp_path)

        # Shadow predictions: completely different top batter in a different game
        preds = _predictions([
            {"batter_name": "Shadow Top", "p_game_hit": 0.82, "game_pk": 778900},
            {"batter_name": "Shadow Second", "p_game_hit": 0.79, "game_pk": 778899},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path, for_shadow=True)

        assert result is not None
        assert result.locked is False
        assert result.daily.pick.batter_name == "Shadow Top"
        assert result.daily.pick.p_game_hit == 0.82
        assert result.daily.bluesky_posted is False
        assert result.daily.bluesky_uri is None

    @patch("bts.strategy.get_game_statuses", return_value={778899: "F"})
    def test_all_games_started_no_prior_pick(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        preds = _predictions([{"game_pk": 778899}])
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result is None

    def test_default_coarse_path_does_not_lookup_detailed_status(self, tmp_path):
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Coarse Path", "p_game_hit": 0.83, "game_pk": 778899},
        ])
        with patch("bts.picks.get_game_statuses_detailed", side_effect=AssertionError), \
             patch("bts.strategy.get_game_statuses", return_value={778899: "P"}):
            result = select_pick(preds, "2026-04-01", tmp_path)

        assert result is not None
        assert result.daily.pick.batter_name == "Coarse Path"

    def test_strict_detailed_lookup_failure_does_not_use_coarse_statuses(self, tmp_path):
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Coarse Unsafe", "p_game_hit": 0.83, "game_pk": 778899},
        ])
        with patch("bts.strategy.get_game_statuses_detailed", side_effect=OSError), \
             patch("bts.strategy.get_game_statuses", return_value={778899: "P"}) as mock_coarse:
            result = select_pick(
                preds,
                "2026-04-01",
                tmp_path,
                require_detailed_statuses=True,
            )

        assert result is None
        mock_coarse.assert_not_called()

    def test_strict_detailed_lookup_failure_locks_existing_pick(self, tmp_path):
        from bts.strategy import select_pick

        existing = DailyPick(
            date="2026-04-01",
            run_time="2026-04-01T15:00:00+00:00",
            pick=Pick(
                batter_name="Conservative Existing", batter_id=100001, team="ATH",
                lineup_position=1, pitcher_name="Suarez", pitcher_id=200001,
                p_game_hit=0.76, flags=[], projected_lineup=False,
                game_pk=778899, game_time="2026-04-01T23:10:00Z",
            ),
            double_down=None, runner_up=None,
        )
        save_pick(existing, tmp_path)

        preds = _predictions([{"batter_name": "New Pick", "game_pk": 778899}])
        with patch("bts.strategy.get_game_statuses_detailed", side_effect=OSError), \
             patch("bts.strategy.get_game_statuses", return_value={778899: "P"}) as mock_coarse:
            result = select_pick(
                preds,
                "2026-04-01",
                tmp_path,
                require_detailed_statuses=True,
            )

        assert result is not None
        assert result.locked is True
        assert result.daily.pick.batter_name == "Conservative Existing"
        mock_coarse.assert_not_called()

    @pytest.mark.parametrize("detailed", ["Postponed", "Cancelled", "Canceled"])
    def test_detailed_void_candidate_excluded_even_when_abstract_preview(
        self, detailed, tmp_path
    ):
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Voided Top", "p_game_hit": 0.91, "game_pk": 778899},
            {"batter_name": "Fresh Pick", "p_game_hit": 0.84, "game_pk": 778900},
        ])
        result = select_pick(
            preds,
            "2026-04-01",
            tmp_path,
            game_statuses_detailed={
                778899: {"abstract": "P", "detailed": detailed},
                778900: {"abstract": "P", "detailed": "Pre-Game"},
            },
        )

        assert result is not None
        assert result.locked is False
        assert result.daily.pick.batter_name == "Fresh Pick"

    def test_detailed_missing_candidate_excluded(self, tmp_path):
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Missing Top", "p_game_hit": 0.91, "game_pk": 778899},
            {"batter_name": "Fresh Pick", "p_game_hit": 0.84, "game_pk": 778900},
        ])
        result = select_pick(
            preds,
            "2026-04-01",
            tmp_path,
            game_statuses_detailed={
                778900: {"abstract": "P", "detailed": "Pre-Game"},
            },
        )

        assert result is not None
        assert result.locked is False
        assert result.daily.pick.batter_name == "Fresh Pick"

    def test_detailed_preview_candidates_match_coarse_selection(self, tmp_path):
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Top Preview", "p_game_hit": 0.84, "game_pk": 778899},
            {"batter_name": "Second Preview", "p_game_hit": 0.81, "game_pk": 778900},
        ])
        coarse_statuses = {778899: "P", 778900: "P"}
        detailed_statuses = {
            778899: {"abstract": "P", "detailed": "Pre-Game"},
            778900: {"abstract": "P", "detailed": "Pre-Game"},
        }

        with patch("bts.strategy.get_game_statuses", return_value=coarse_statuses):
            coarse_result = select_pick(preds, "2026-04-01", tmp_path / "coarse")
        detailed_result = select_pick(
            preds,
            "2026-04-01",
            tmp_path / "detailed",
            game_statuses_detailed=detailed_statuses,
        )

        assert coarse_result is not None
        assert detailed_result is not None
        assert detailed_result.daily.pick.batter_name == coarse_result.daily.pick.batter_name
        assert detailed_result.daily.double_down is not None
        assert coarse_result.daily.double_down is not None
        assert detailed_result.daily.double_down.batter_name == coarse_result.daily.double_down.batter_name

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
    def test_picks_highest_p_game_hit_regardless_of_time(self, mock_statuses, tmp_path):
        """select_pick always picks the highest P(game_hit) batter."""
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

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
    def test_allow_double_false_downgrades_to_single(self, mock_statuses, tmp_path):
        """Live stale-contest fallback can forbid automatic double-downs."""
        from bts.strategy import select_pick

        preds = _predictions([
            {"batter_name": "Wilson", "p_game_hit": 0.82, "game_pk": 778899},
            {"batter_name": "Mangum", "p_game_hit": 0.81, "game_pk": 778900},
        ])
        result = select_pick(preds, "2026-04-01", tmp_path, allow_double=False)

        assert result is not None
        assert result.daily.pick.batter_name == "Wilson"
        assert result.daily.double_down is None

    @patch("bts.strategy.get_game_statuses", return_value={778899: "P"})
    def test_empty_predictions(self, mock_statuses, tmp_path):
        from bts.strategy import select_pick

        preds = pd.DataFrame()
        result = select_pick(preds, "2026-04-01", tmp_path)

        assert result is None


class TestShouldLock:
    def test_locks_when_all_confirmed(self):
        from bts.strategy import should_lock

        top_pick = {"p_game_hit": 0.82, "projected_lineup": False, "game_pk": 100}
        all_picks = [
            {"p_game_hit": 0.82, "projected_lineup": False, "game_pk": 100},
            {"p_game_hit": 0.79, "projected_lineup": False, "game_pk": 200},
        ]
        assert should_lock(top_pick, all_picks, early_lock_gap=0.03) is True

    def test_locks_when_gap_exceeds_threshold(self):
        from bts.strategy import should_lock

        top_pick = {"p_game_hit": 0.85, "projected_lineup": False, "game_pk": 100}
        all_picks = [
            {"p_game_hit": 0.85, "projected_lineup": False, "game_pk": 100},
            {"p_game_hit": 0.80, "projected_lineup": True, "game_pk": 200},
        ]
        # Gap is 0.05, threshold is 0.03 — lock
        assert should_lock(top_pick, all_picks, early_lock_gap=0.03) is True

    def test_waits_when_gap_below_threshold(self):
        from bts.strategy import should_lock

        top_pick = {"p_game_hit": 0.83, "projected_lineup": False, "game_pk": 100}
        all_picks = [
            {"p_game_hit": 0.83, "projected_lineup": False, "game_pk": 100},
            {"p_game_hit": 0.82, "projected_lineup": True, "game_pk": 200},
        ]
        # Gap is 0.01, threshold is 0.03 — wait
        assert should_lock(top_pick, all_picks, early_lock_gap=0.03) is False

    def test_waits_when_top_pick_is_projected(self):
        from bts.strategy import should_lock

        top_pick = {"p_game_hit": 0.85, "projected_lineup": True, "game_pk": 100}
        all_picks = [
            {"p_game_hit": 0.85, "projected_lineup": True, "game_pk": 100},
        ]
        assert should_lock(top_pick, all_picks, early_lock_gap=0.03) is False


# --- Hardening: game_pk normalization so a NaN / type-mismatched value can't
#     masquerade as a different game and trigger a same-game (correlated) double ---

@patch("bts.strategy.get_game_statuses", return_value={778899: "P", "778899": "P"})
def test_no_double_for_same_game_mixed_type_game_pk(mock_statuses, tmp_path):
    """Same game in mixed int/str representation ("778899" vs 778899) must not be
    treated as two different games. Without normalization, NaN != NaN / str != int
    both evaluate True and yield a junk/same-game double-down."""
    from bts.strategy import select_pick

    preds = _predictions([
        {"batter_name": "Wilson", "p_game_hit": 0.82, "game_pk": 778899},
        {"batter_name": "Same Game Str", "p_game_hit": 0.81, "game_pk": "778899"},
    ])
    result = select_pick(preds, "2026-04-01", tmp_path)

    assert result.daily.double_down is None
    assert result.daily.runner_up is None


@patch("bts.strategy._mdp_action_from", return_value="double")
@patch("bts.strategy.get_game_statuses", return_value={778899: "P"})
def test_mdp_double_with_single_game_yields_safe_single(mock_statuses, mock_mdp, tmp_path):
    """MDP can return 'double' even when only one game is available; with no
    executable different-game second pick it must resolve to a safe single (no
    double_down), not an inconsistent double-with-no-partner."""
    from bts.strategy import select_pick

    preds = _predictions([
        {"batter_name": "Solo", "p_game_hit": 0.85, "game_pk": 778899},
    ])
    result = select_pick(preds, "2026-04-01", tmp_path)

    assert result is not None
    assert result.daily.pick.batter_name == "Solo"
    assert result.daily.double_down is None


@patch("bts.strategy.get_game_statuses", return_value={778899: "P", 778900: "P"})
def test_double_down_still_works_for_genuinely_different_games(mock_statuses, tmp_path):
    """Guard against over-correction: real different-game pairs must still double."""
    from bts.strategy import select_pick

    preds = _predictions([
        {"batter_name": "Wilson", "p_game_hit": 0.82, "game_pk": 778899},
        {"batter_name": "Mangum", "p_game_hit": 0.81, "game_pk": 778900},
    ])
    result = select_pick(preds, "2026-04-01", tmp_path)

    assert result.daily.double_down is not None
    assert result.daily.double_down.batter_name == "Mangum"
