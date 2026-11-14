"""The delivery chokepoint never delivers a pick whose game MLB lists as postponed or cancelled (incident L03).

2026: when the fallback's refresh failed, the cached pick was delivered for a game already postponed. The guard
sits beside the cutoff guard in _deliver_and_lock_pick, so every caller and every delivery mode goes through it.
"""
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from bts.picks import load_pick, save_pick
from bts.scheduler import _deliver_and_lock_pick
from tests.test_late_delivery_guard import DATE, ET, _daily, _state, cfg  # noqa: F401  (cfg is a fixture)

BEFORE_CUTOFF = datetime(2026, 8, 30, 13, 0, 0, tzinfo=ET)
PICK_GAME, DD_GAME = 824393, 824959


def _deliver(cfg, tmp_path, daily, statuses, mode="dm"):
    cfg["scheduler"]["pick_delivery"] = mode
    save_pick(daily, tmp_path)
    state = _state()
    with patch("bts.scheduler._delivery_game_statuses", side_effect=statuses), \
         patch("bts.scheduler._now_et", return_value=BEFORE_CUTOFF), \
         patch("bts.scheduler._trigger_live_forward_capture_on_lock"), \
         patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=0)), \
         patch("bts.dm.send_dm", return_value="msg-1") as dm, \
         patch("bts.health.alert.dispatch_dm_for_health_alerts", return_value=True) as alert:
        ok = _deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, "fallback")
    return ok, state, dm, alert


@pytest.mark.parametrize("state_name", ["Postponed", "Cancelled", "Canceled"])
def test_a_postponed_or_cancelled_game_is_refused_archived_and_alerted(cfg, tmp_path, state_name):
    ok, state, dm, alert = _deliver(cfg, tmp_path, _daily(),
                                    lambda date: {PICK_GAME: {"abstract": "F", "detailed": state_name}})
    assert ok is False and state.pick_locked is False
    dm.assert_not_called()
    assert load_pick(DATE, tmp_path) is None                                  # the dead candidate is removed
    assert list(tmp_path.glob("**/*refused_delivery*"))                       # and archived
    assert state.delivery_refusals[-1]["void_games"] == {str(PICK_GAME): state_name}
    (alerts, _recipient), _kw = alert.call_args
    assert alerts[0].level == "CRITICAL" and alerts[0].source == "postponed_pick"
    assert state_name in alerts[0].message and str(PICK_GAME) in alerts[0].message
    assert "Kwan" not in alerts[0].message              # nothing in the alert can be read as a pick


def test_a_playable_game_is_delivered(cfg, tmp_path):
    ok, state, dm, alert = _deliver(cfg, tmp_path, _daily(),
                                    lambda date: {PICK_GAME: {"abstract": "P", "detailed": "Scheduled"}})
    assert ok is True and state.pick_locked is True
    dm.assert_called_once()
    alert.assert_not_called()


def test_a_double_down_leg_on_a_postponed_game_refuses_the_whole_pick(cfg, tmp_path):
    daily = _daily(dd_utc="2026-08-30T20:05:00Z")
    ok, state, dm, _alert = _deliver(cfg, tmp_path, daily, lambda date: {
        PICK_GAME: {"abstract": "P", "detailed": "Scheduled"}, DD_GAME: {"abstract": "F", "detailed": "Postponed"}})
    assert ok is False and not dm.called
    assert state.delivery_refusals[-1]["void_games"] == {str(DD_GAME): "Postponed"}


def test_an_unreadable_schedule_delivers(cfg, tmp_path):
    """The status read fails in the same outages that make a refresh fail; a postponed pick is a Pass, not a
    miss, so an unverifiable game is delivered rather than the day lost."""
    def down(date):
        raise OSError("MLB schedule unreachable")
    ok, state, dm, alert = _deliver(cfg, tmp_path, _daily(), down)
    assert ok is True and state.pick_locked is True
    dm.assert_called_once()
    alert.assert_not_called()


def test_a_game_missing_from_the_schedule_answer_delivers(cfg, tmp_path):
    ok, _state_, dm, _alert = _deliver(cfg, tmp_path, _daily(), lambda date: {})
    assert ok is True and dm.called


def test_private_mode_also_refuses_a_postponed_game(cfg, tmp_path):
    ok, state, _dm, alert = _deliver(cfg, tmp_path, _daily(),
                                     lambda date: {PICK_GAME: {"abstract": "F", "detailed": "Postponed"}},
                                     mode="private")
    assert ok is False and state.pick_locked is False and alert.called
