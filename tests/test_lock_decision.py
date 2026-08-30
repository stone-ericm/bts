"""_lock_decision_from_predictions returns a LockDecision that says WHY a lock is
blocked and, for a gap block, WHICH game's confirmation could change the pick.
The fallback planner defers only for that window (2026-08-30)."""
from unittest.mock import patch

import pandas as pd

from bts.picks import DailyPick, Pick

PREVIEW = {"abstract": "P", "detailed": "Scheduled"}


def _daily(primary_projected=False, dd=None):
    p = Pick(batter_name="Kwan", batter_id=1, team="CLE", lineup_position=1, pitcher_name="L",
             pitcher_id=2, p_game_hit=0.757, flags=[], projected_lineup=primary_projected,
             game_pk=100, game_time="2026-08-30T17:40:00Z")
    return DailyPick(date="2026-08-30", run_time="x", pick=p, double_down=dd, runner_up=None)


def _preds(contender_p=0.741, contender_flags="PROJECTED lineup"):
    return pd.DataFrame([
        {"batter_name": "Kwan", "game_pk": 100, "p_game_hit": 0.757, "flags": ""},
        {"batter_name": "Arraez", "game_pk": 300, "p_game_hit": contender_p, "flags": contender_flags},
    ])


@patch("bts.picks.get_game_statuses_detailed", return_value={100: PREVIEW, 200: PREVIEW, 300: PREVIEW})
def test_gap_block_names_the_contender_game(_st):
    from bts.scheduler import _lock_decision_from_predictions
    d = _lock_decision_from_predictions(_preds(), _daily(), "2026-08-30", 0.03)
    assert d.should_lock is False and d.block_reason == "gap" and d.contender_game_pk == 300
    assert d.should_lock_ungated is False and abs(d.best_projected - 0.741) < 1e-9


@patch("bts.picks.get_game_statuses_detailed", return_value={100: PREVIEW, 300: PREVIEW})
def test_gap_passed_locks(_st):
    from bts.scheduler import _lock_decision_from_predictions
    d = _lock_decision_from_predictions(_preds(contender_p=0.70), _daily(), "2026-08-30", 0.03)
    assert d.should_lock is True and d.block_reason is None and d.contender_game_pk is None


@patch("bts.picks.get_game_statuses_detailed", return_value={100: PREVIEW, 300: PREVIEW})
def test_primary_projected_block(_st):
    from bts.scheduler import _lock_decision_from_predictions
    d = _lock_decision_from_predictions(_preds(), _daily(primary_projected=True), "2026-08-30", 0.03)
    assert d.should_lock is False and d.block_reason == "primary_projected"


@patch("bts.picks.get_game_statuses_detailed", return_value={100: PREVIEW, 200: PREVIEW, 300: PREVIEW})
def test_dd_projected_is_gate_only(_st):
    from bts.scheduler import _lock_decision_from_predictions
    dd = Pick(batter_name="DD", batter_id=5, team="X", lineup_position=1, pitcher_name="P",
              pitcher_id=6, p_game_hit=0.72, flags=["PROJECTED lineup"], projected_lineup=True,
              game_pk=200, game_time="2026-08-30T20:05:00Z")
    d = _lock_decision_from_predictions(_preds(contender_p=0.70), _daily(dd=dd), "2026-08-30", 0.03)
    assert d.should_lock is False and d.should_lock_ungated is True and d.block_reason == "dd_projected"


@patch("bts.picks.get_game_statuses_detailed",
       return_value={100: PREVIEW, 300: PREVIEW, 400: PREVIEW})
def test_all_in_gap_contender_games_are_reported(_st):
    """Codex r2 F7: every projected contender within early_lock_gap can change the
    decision when it confirms — the planner needs all their games, not just the top."""
    from bts.scheduler import _lock_decision_from_predictions
    preds = pd.DataFrame([
        {"batter_name": "Kwan", "game_pk": 100, "p_game_hit": 0.757, "flags": ""},
        {"batter_name": "A", "game_pk": 300, "p_game_hit": 0.750, "flags": "PROJECTED lineup"},
        {"batter_name": "B", "game_pk": 400, "p_game_hit": 0.740, "flags": "PROJECTED lineup"},
        {"batter_name": "C", "game_pk": 400, "p_game_hit": 0.600, "flags": "PROJECTED lineup"},
    ])
    d = _lock_decision_from_predictions(preds, _daily(), "2026-08-30", 0.03)
    assert d.block_reason == "gap" and d.contender_game_pk == 300
    assert set(d.contender_game_pks) == {300, 400}          # 0.600 is outside the gap


@patch("bts.picks.get_game_statuses_detailed", side_effect=RuntimeError("down"))
def test_status_failure(_st):
    from bts.scheduler import _lock_decision_from_predictions
    d = _lock_decision_from_predictions(_preds(), _daily(), "2026-08-30", 0.03)
    assert d.should_lock is False and d.block_reason == "status_failure"


@patch("bts.picks.get_game_statuses_detailed",
       return_value={100: {"abstract": "L", "detailed": "In Progress"}, 300: PREVIEW})
def test_selected_slot_started_is_slot_unavailable(_st):
    from bts.scheduler import _lock_decision_from_predictions
    d = _lock_decision_from_predictions(_preds(), _daily(), "2026-08-30", 0.03)
    assert d.should_lock is False and d.block_reason == "slot_unavailable"
