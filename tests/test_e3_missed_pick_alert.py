"""E3: missed-pick early alert — DM the operator in-window if delivery failed."""
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from bts.scheduler import _maybe_alert_missed_pick, _alert_missed_pick, SchedulerState
from bts.picks import DailyPick, Pick, save_pick

ET = timezone(timedelta(hours=-4))
CONFIG = {"orchestrator": {"picks_dir": None}, "bluesky": {"dm_recipient": "me.bsky.social"}}


def _daily(**kw):
    d = DailyPick(
        date="2026-04-06", run_time="2026-04-06T19:29:00+00:00",
        pick=Pick(batter_name="Hoerner", batter_id=1, team="CHC",
                  lineup_position=1, pitcher_name="Baz", pitcher_id=2,
                  p_game_hit=0.73, flags=[], projected_lineup=False,
                  game_pk=100, game_time="2026-04-06T20:10:00-04:00"),
        double_down=None, runner_up=None,
    )
    for k, v in kw.items():
        setattr(d, k, v)
    return d


def _cfg(tmp_path):
    return {"orchestrator": {"picks_dir": str(tmp_path)}, "bluesky": {"dm_recipient": "me.bsky.social"}}


def _no_skip_state():
    return SchedulerState(
        date="2026-04-06", schedule_fetched_at="t", games=[], confirmed_game_pks=[],
        runs_completed=[], pick_locked=False, pick_locked_at=None,
        result_status=None, next_wakeup=None,
    )


@patch("bts.scheduler._alert_missed_pick")
@patch("bts.scheduler._watchdog_ping_sleep")
@patch("bts.scheduler._now_et")
def test_alerts_when_undelivered_near_first_pitch(mock_now, _sleep, mock_alert, tmp_path):
    mock_now.return_value = datetime(2026, 4, 6, 20, 5, tzinfo=ET)  # 5 min to game (past 10-min window)
    save_pick(_daily(), tmp_path)
    _maybe_alert_missed_pick(_cfg(tmp_path), "2026-04-06", tmp_path, 10, None, _no_skip_state())
    mock_alert.assert_called_once()


@patch("bts.scheduler._alert_missed_pick")
@patch("bts.scheduler._now_et")
def test_no_alert_when_delivered(mock_now, mock_alert, tmp_path):
    mock_now.return_value = datetime(2026, 4, 6, 20, 5, tzinfo=ET)
    save_pick(_daily(bluesky_posted=True, bluesky_uri="at://x"), tmp_path)
    _maybe_alert_missed_pick(_cfg(tmp_path), "2026-04-06", tmp_path, 10, None, _no_skip_state())
    mock_alert.assert_not_called()


def test_no_alert_on_skip_day_final_skip_candidate(tmp_path, monkeypatch):
    """A deliberate MDP skip (final_skip_candidate set) is NOT a missed pick — no alert."""
    dispatched = []
    monkeypatch.setattr("bts.scheduler._alert_missed_pick",
                        lambda *a, **kw: dispatched.append(True))
    state = SchedulerState(
        date="2026-04-06", schedule_fetched_at="t", games=[], confirmed_game_pks=[],
        runs_completed=[], pick_locked=False, pick_locked_at=None,
        result_status=None, next_wakeup=None,
        final_skip_candidate={"best_batter": "Hoerner", "best_p": 0.73},
    )
    _maybe_alert_missed_pick(_cfg(tmp_path), "2026-04-06", tmp_path, 10, None, state)
    assert dispatched == [], "missed-pick alert must not fire on a skip day"


def test_no_alert_on_skip_day_skip_summary(tmp_path, monkeypatch):
    """An in-game skip (skip_summary set) is NOT a missed pick — no alert."""
    dispatched = []
    monkeypatch.setattr("bts.scheduler._alert_missed_pick",
                        lambda *a, **kw: dispatched.append(True))
    state = SchedulerState(
        date="2026-04-06", schedule_fetched_at="t", games=[], confirmed_game_pks=[],
        runs_completed=[], pick_locked=False, pick_locked_at=None,
        result_status=None, next_wakeup=None,
        skip_summary={"best_batter": "Hoerner", "best_p": 0.73, "streak": 5},
    )
    _maybe_alert_missed_pick(_cfg(tmp_path), "2026-04-06", tmp_path, 10, None, state)
    assert dispatched == [], "missed-pick alert must not fire when skip_summary is set"


def test_no_alert_when_scoreable_decision_on_disk(monkeypatch, tmp_path):
    """The on-disk decision.json is the authority (crash between decision write
    and state save leaves committed_pick_written stale). A scoreable commit on
    disk suppresses the alert regardless of in-memory state (Codex r2 #3)."""
    from bts.daily_decision import write_decision

    dispatched = []
    monkeypatch.setattr("bts.scheduler._alert_missed_pick",
                        lambda *a, **kw: dispatched.append(True))
    monkeypatch.setattr("bts.scheduler._now_et",
                        lambda: datetime(2026, 4, 6, 20, 5, tzinfo=ET))
    save_pick(_daily(), tmp_path)  # pick file itself carries no delivery flags
    write_decision("2026-04-06", tmp_path, action="single", source="mdp",
                   primary={"batter_id": 1, "batter_name": "Hoerner", "team": "CHC",
                            "game_pk": 100, "p_game_hit": 0.73},
                   delivery_status="private_locked", scoreable=True)
    _maybe_alert_missed_pick(_cfg(tmp_path), "2026-04-06", tmp_path, 10, None, _no_skip_state())
    assert dispatched == [], "a scoreable on-disk commit must suppress the alert"


def test_true_missed_pick_still_alerts(monkeypatch, tmp_path):
    """No skip state + undelivered pick → alert MUST still fire (regression guard)."""
    dispatched = []
    monkeypatch.setattr("bts.scheduler._alert_missed_pick",
                        lambda *a, **kw: dispatched.append(True))
    monkeypatch.setattr("bts.scheduler._watchdog_ping_sleep", lambda s: None)
    # Game time in the past so wait_s ≤ 0 and the function doesn't actually sleep
    monkeypatch.setattr("bts.scheduler._now_et",
                        lambda: datetime(2026, 4, 6, 20, 5, tzinfo=ET))
    save_pick(_daily(), tmp_path)  # undelivered
    _maybe_alert_missed_pick(_cfg(tmp_path), "2026-04-06", tmp_path, 10, None, _no_skip_state())
    assert dispatched, "a genuine missed pick with no skip state must still trigger the alert"


@patch("bts.health.alert.dispatch_dm_for_health_alerts")
def test_alert_sender_dispatches_missed_pick_critical(mock_dispatch, tmp_path):
    _alert_missed_pick(_cfg(tmp_path), _daily(), mins_to_game=5)
    mock_dispatch.assert_called_once()
    alerts = mock_dispatch.call_args[0][0]
    assert alerts[0].source == "missed_pick"
    assert alerts[0].level == "CRITICAL"
