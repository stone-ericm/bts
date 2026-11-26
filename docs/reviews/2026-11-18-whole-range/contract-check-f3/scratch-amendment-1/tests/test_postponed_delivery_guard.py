"""The delivery chokepoint refuses only picks with no playable game (incident L03).

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
         patch("bts.posting.post_to_bluesky", return_value="uri-1"), \
         patch("bts.health.alert.dispatch_dm_for_health_alerts", return_value=True) as alert:
        ok = _deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, "fallback")
    return ok, state, dm, alert


@pytest.mark.parametrize("state_name", ["Postponed", "Cancelled", "Canceled"])
@pytest.mark.parametrize("mode", ["dm", "public", "private"])
def test_a_postponed_or_cancelled_game_is_refused_archived_and_alerted(cfg, tmp_path, state_name, mode):
    ok, state, dm, alert = _deliver(cfg, tmp_path, _daily(),
                                    lambda date: {PICK_GAME: {"abstract": "F", "detailed": state_name}}, mode=mode)
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


def test_a_postponed_double_down_leg_leaves_the_delivered_double_unchanged(cfg, tmp_path):
    """Review pp1 B2: a Pass leg must not discard the playable leg at the final fallback."""
    from bts.daily_decision import load_decision
    daily = _daily(dd_utc="2026-08-30T20:05:00Z")
    ok, state, dm, alert = _deliver(cfg, tmp_path, daily, lambda date: {
        PICK_GAME: {"abstract": "P", "detailed": "Scheduled"}, DD_GAME: {"abstract": "F", "detailed": "Postponed"}})
    assert ok is True and state.pick_locked is True
    dm.assert_called_once()
    assert "Kwan" in dm.call_args[0][1] and "McNeil" in dm.call_args[0][1]
    saved = load_pick(DATE, tmp_path)
    assert saved.pick.batter_name == "Kwan" and saved.double_down.batter_name == "McNeil"
    dec = load_decision(DATE, tmp_path)
    assert dec["action"] == "double" and dec["double_down"]["batter_name"] == "McNeil" and dec["scoreable"] is True
    assert not state.delivery_refusals
    assert not list(tmp_path.glob("**/*refused_delivery*"))
    alert.assert_not_called()


def test_a_postponed_primary_leaves_the_delivered_double_unchanged(cfg, tmp_path):
    daily = _daily(dd_utc="2026-08-30T20:05:00Z")
    ok, state, dm, _alert = _deliver(cfg, tmp_path, daily, lambda date: {
        PICK_GAME: {"abstract": "F", "detailed": "Postponed"}, DD_GAME: {"abstract": "P", "detailed": "Scheduled"}})
    assert ok is True
    saved = load_pick(DATE, tmp_path)
    assert saved.pick.batter_name == "Kwan" and saved.double_down.batter_name == "McNeil"
    assert "McNeil" in dm.call_args[0][1] and "Kwan" in dm.call_args[0][1]


@pytest.mark.parametrize("mode", ["dm", "public", "private"])
@pytest.mark.parametrize("same_game", [False, True])
def test_a_double_with_both_legs_postponed_is_refused(cfg, tmp_path, mode, same_game):
    daily = _daily(dd_utc="2026-08-30T20:05:00Z")
    if same_game:
        daily.double_down.game_pk = PICK_GAME
    ok, state, dm, alert = _deliver(cfg, tmp_path, daily, lambda date: {
        PICK_GAME: {"abstract": "F", "detailed": "Postponed"}, DD_GAME: {"abstract": "F", "detailed": "Cancelled"}}, mode=mode)
    assert ok is False and not dm.called and load_pick(DATE, tmp_path) is None
    assert state.delivery_refusals[-1]["reason"] == "game_not_playable"
    assert list(tmp_path.glob("**/*refused_delivery*"))
    alerts = alert.call_args.args[0]
    assert len(alerts) == 1 and alerts[0].level == "CRITICAL"
    assert "Kwan" not in alerts[0].message and "McNeil" not in alerts[0].message
    assert str(PICK_GAME) in alerts[0].message and "Postponed" in alerts[0].message
    if not same_game:
        assert str(DD_GAME) in alerts[0].message and "Cancelled" in alerts[0].message


def test_a_partial_void_double_still_faces_the_original_earliest_cutoff(cfg, tmp_path):
    """The whole double keeps its original cutoff, including the void leg (a known pre-existing limit)."""
    daily = _daily(dd_utc="2026-08-30T20:05:00Z")           # primary 13:40 ET (cutoff 13:35), double 16:05 ET
    cfg["scheduler"]["pick_delivery"] = "dm"
    save_pick(daily, tmp_path)
    state = _state()
    with patch("bts.scheduler._delivery_game_statuses", side_effect=lambda date: {
            PICK_GAME: {"abstract": "F", "detailed": "Postponed"}, DD_GAME: {"abstract": "P", "detailed": "Scheduled"}}), \
         patch("bts.scheduler._now_et", return_value=datetime(2026, 8, 30, 13, 36, 0, tzinfo=ET)), \
         patch("bts.scheduler._trigger_live_forward_capture_on_lock"), \
         patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=0)), \
         patch("bts.dm.send_dm", return_value="msg-1") as dm, \
         patch("bts.health.alert.dispatch_dm_for_health_alerts", return_value=True):
        ok = _deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, "fallback")
    assert ok is False and not dm.called
    assert [r["reason"] for r in state.delivery_refusals] == ["past_submission_cutoff"]


def test_the_health_check_words_each_refusal_by_its_reason(tmp_path):
    """Review pp1 N2: a postponement refusal used to render as 'None min past the None cutoff'."""
    import json
    from datetime import date as _date
    from bts.health import late_delivery
    day = tmp_path / DATE
    day.mkdir()
    (day / "scheduler_state.json").write_text(json.dumps({"delivery_refusals": [
        {"reason": "game_not_playable", "batter": "Kwan", "label": "fallback", "void_games": {"824393": "Postponed"}},
        {"batter": "Kwan", "label": "lineup", "late_min": 1.2, "cutoff_et": "2026-08-30T13:35:00-04:00"}]}))
    alerts = late_delivery.check(tmp_path, today=_date(2026, 8, 30))
    assert [a.level for a in alerts] == ["CRITICAL", "CRITICAL"]
    assert "game 824393 is Postponed" in alerts[0].message and "None" not in alerts[0].message
    assert all("reduced" not in a.message for a in alerts)
    assert "1.2 min past" in alerts[1].message                               # a record without a reason is a late one


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


def test_the_decision_record_keeps_the_original_double_selection(cfg, tmp_path):
    """The partial-void guard must keep the original selection and its double metadata."""
    from bts.daily_decision import load_decision
    from tests.test_scheduler_decision_record_integration import _sel
    from bts.scheduler import _row_from_daily
    daily = _daily(dd_utc="2026-08-30T20:05:00Z")
    cfg["scheduler"]["pick_delivery"] = "dm"
    save_pick(daily, tmp_path)
    state = _state()
    selection = _sel("double", "mdp", primary=_row_from_daily(daily.pick), double=_row_from_daily(daily.double_down))
    with patch("bts.scheduler._delivery_game_statuses", side_effect=lambda date: {
            PICK_GAME: {"abstract": "P", "detailed": "Scheduled"}, DD_GAME: {"abstract": "F", "detailed": "Postponed"}}), \
         patch("bts.scheduler._now_et", return_value=BEFORE_CUTOFF), \
         patch("bts.scheduler._trigger_live_forward_capture_on_lock"), \
         patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=0)), \
         patch("bts.dm.send_dm", return_value="msg-1"), \
         patch("bts.health.alert.dispatch_dm_for_health_alerts", return_value=True):
        assert _deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, "fallback", selection=selection) is True
    dec = load_decision(DATE, tmp_path)
    assert dec["action"] == "double" and dec["double_down"]["batter_name"] == "McNeil"
    assert dec["source"] == "mdp"
    assert dec["primary"]["batter_name"] == "Kwan" and dec["primary"]["game_pk"] == PICK_GAME
