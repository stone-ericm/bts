"""Fail-closed delivery guard: never deliver a pick at/after its submission cutoff.

2026-08-30: the daemon DM'd Kwan at 13:36:14 for a 13:40 first pitch (cutoff 13:35).
The guard lives at the ONE delivery chokepoint (_deliver_and_lock_pick) so no
caller — lineup lock, in-loop fallback, final fallback — can send a dead pick.
"""
import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from bts.picks import DailyPick, Pick, load_pick, save_pick
from bts.scheduler import SchedulerState, _deliver_and_lock_pick

ET = ZoneInfo("America/New_York")
DATE = "2026-08-30"
CONFIG = {"scheduler": {"pick_delivery": "dm"}, "bluesky": {"dm_recipient": "eric.test"},
          "orchestrator": {"picks_dir": "PLACEHOLDER"}}


def _state():
    return SchedulerState(date=DATE, schedule_fetched_at="x", games=[], confirmed_game_pks=[],
                          runs_completed=[], pick_locked=False, pick_locked_at=None,
                          result_status=None, next_wakeup=None)


def _daily(dd_utc=None):
    pick = Pick(batter_name="Kwan", batter_id=680757, team="CLE", lineup_position=1,
                pitcher_name="Lugo", pitcher_id=607625, p_game_hit=0.7566, flags=[],
                projected_lineup=False, game_pk=824393, game_time="2026-08-30T17:40:00Z")
    dd = None
    if dd_utc:
        dd = Pick(batter_name="McNeil", batter_id=643446, team="ATH", lineup_position=2,
                  pitcher_name="Bassitt", pitcher_id=605135, p_game_hit=0.7428, flags=[],
                  projected_lineup=False, game_pk=824959, game_time=dd_utc)
    return DailyPick(date=DATE, run_time="2026-08-30T17:20:00+00:00", pick=pick,
                     double_down=dd, runner_up=None)


@pytest.fixture
def cfg(tmp_path):
    c = json.loads(json.dumps(CONFIG))
    c["orchestrator"]["picks_dir"] = str(tmp_path)
    return c


@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=0))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.dm.send_dm", return_value="msg-1")
@patch("bts.scheduler._now_et")
def test_one_second_before_cutoff_delivers(mock_now, mock_dm, _cap, _dss, cfg, tmp_path):
    mock_now.return_value = datetime(2026, 8, 30, 13, 34, 59, tzinfo=ET)
    daily = _daily()
    save_pick(daily, tmp_path)
    state = _state()
    ok = _deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, "lineup")
    assert ok is True and state.pick_locked is True
    mock_dm.assert_called_once()
    assert load_pick(DATE, tmp_path).delivered_at == mock_now.return_value.isoformat()


@patch("bts.health.alert.dispatch_dm_for_health_alerts", return_value=True)
@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=0))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.dm.send_dm", return_value="msg-1")
@patch("bts.scheduler._now_et")
def test_at_cutoff_refuses_archives_and_alerts(mock_now, mock_dm, _cap, _dss, mock_alert,
                                               cfg, tmp_path, capsys):
    mock_now.return_value = datetime(2026, 8, 30, 13, 35, 0, tzinfo=ET)   # == cutoff
    daily = _daily()
    save_pick(daily, tmp_path)
    state = _state()
    ok = _deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, "lineup")
    assert ok is False
    assert state.pick_locked is False
    mock_dm.assert_not_called()
    assert not (tmp_path / f"{DATE}.json").exists()          # removed so later cycles re-pick
    archives = list((tmp_path / DATE).glob("refused_delivery_*.json"))
    assert len(archives) == 1
    body = json.loads(archives[0].read_text())
    assert body["refused_delivery"]["reason"] == "past_submission_cutoff"
    assert body["refused_delivery"]["label"] == "lineup"
    assert state.delivery_refusals and state.delivery_refusals[0]["batter"] == "Kwan"
    # persisted on scheduler_state.json for the EOD late_delivery audit
    persisted = json.loads((tmp_path / DATE / "scheduler_state.json").read_text())
    assert persisted["delivery_refusals"][0]["late_min"] == 0.0
    alerts = mock_alert.call_args.args[0]
    assert alerts[0].level == "CRITICAL" and alerts[0].source == "late_delivery"
    assert "DELIVERY REFUSED" in capsys.readouterr().err


@patch("bts.health.alert.dispatch_dm_for_health_alerts", return_value=True)
@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=0))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.dm.send_dm", return_value="msg-1")
@patch("bts.scheduler._now_et")
def test_earlier_double_down_sets_the_cutoff(mock_now, mock_dm, _cap, _dss, _alert, cfg, tmp_path):
    # primary 16:05 ET, DD 13:40 ET → cutoff 13:35 from the DD
    daily = _daily(dd_utc="2026-08-30T17:40:00Z")
    daily.pick.game_time = "2026-08-30T20:05:00Z"
    save_pick(daily, tmp_path)
    mock_now.return_value = datetime(2026, 8, 30, 13, 36, tzinfo=ET)
    assert _deliver_and_lock_pick(daily, cfg, tmp_path, _state(), DATE, "fallback") is False
    mock_dm.assert_not_called()


@patch("bts.health.alert.dispatch_dm_for_health_alerts", return_value=True)
@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=0))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.scheduler._now_et")
def test_private_mode_is_guarded_too(mock_now, _cap, _dss, _alert, cfg, tmp_path):
    """A private lock past the cutoff is just as unenterable as a DM."""
    cfg["scheduler"]["pick_delivery"] = "private"
    daily = _daily()
    save_pick(daily, tmp_path)
    mock_now.return_value = datetime(2026, 8, 30, 13, 40, tzinfo=ET)
    state = _state()
    assert _deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, "lineup") is False
    assert state.pick_locked is False


@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=0))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.dm.send_dm", return_value="msg-1")
@patch("bts.scheduler._now_et")
def test_already_delivered_pick_still_locks_after_cutoff(mock_now, mock_dm, _cap, _dss, cfg, tmp_path):
    """Evidence path: a pick that WAS delivered earlier re-locks on restart even after cutoff."""
    mock_now.return_value = datetime(2026, 8, 30, 14, 0, tzinfo=ET)
    daily = _daily()
    daily.notification_sent = True
    daily.notification_id = "old"
    save_pick(daily, tmp_path)
    state = _state()
    assert _deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, "lineup") is True
    assert state.pick_locked is True
    mock_dm.assert_not_called()


@patch("bts.health.alert.dispatch_dm_for_health_alerts", return_value=True)
@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=0))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.dm.send_dm", return_value="msg-1")
@patch("bts.scheduler._now_et")
def test_clock_crossing_cutoff_between_guard_and_send_refuses(mock_now, mock_dm, _cap, _dss,
                                                              mock_alert, cfg, tmp_path):
    """Codex r2 F1: the top-of-function guard passed, but the contest-state fetch
    ate the window — the pre-send re-check must refuse, and must not leave the
    delivery_attempted crash marker set."""
    times = iter([datetime(2026, 8, 30, 13, 34, 0, tzinfo=ET)])   # guard passes...
    late = datetime(2026, 8, 30, 13, 35, 30, tzinfo=ET)           # ...everything after is late
    mock_now.side_effect = lambda: next(times, late)
    daily = _daily()
    save_pick(daily, tmp_path)
    state = _state()
    ok = _deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, "lineup")
    assert ok is False and state.pick_locked is False
    mock_dm.assert_not_called()
    assert daily.delivery_attempted is False
    assert list((tmp_path / DATE).glob("refused_delivery_*.json"))


@patch("bts.health.alert.dispatch_dm_for_health_alerts", return_value=True)
@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=0))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.dm.send_dm", return_value="msg-1")
@patch("bts.scheduler._now_et")
def test_unconfirmed_prior_attempt_pages_critical(mock_now, mock_dm, _cap, _dss,
                                                  mock_alert, cfg, tmp_path):
    """Codex r2 F2 (partial): the crash-idempotency lock is silent today; it must
    page so the operator verifies the send and the MLB entry immediately."""
    mock_now.return_value = datetime(2026, 8, 30, 13, 0, tzinfo=ET)
    daily = _daily()
    daily.delivery_attempted = True
    save_pick(daily, tmp_path)
    state = _state()
    ok = _deliver_and_lock_pick(daily, cfg, tmp_path, state, DATE, "lineup")
    assert ok is False and state.pick_locked is True
    mock_dm.assert_not_called()
    alerts = mock_alert.call_args.args[0]
    assert alerts[0].level == "CRITICAL"
    assert "outcome unknown" in alerts[0].message.lower()
    assert "never confirmed" in alerts[0].message.lower()


def test_legacy_pick_file_without_delivered_at_loads(tmp_path):
    """Old <date>.json files predate the field; load_pick backfills None."""
    daily = _daily()
    save_pick(daily, tmp_path)
    body = json.loads((tmp_path / f"{DATE}.json").read_text())
    body.pop("delivered_at", None)
    (tmp_path / f"{DATE}.json").write_text(json.dumps(body))
    assert load_pick(DATE, tmp_path).delivered_at is None
