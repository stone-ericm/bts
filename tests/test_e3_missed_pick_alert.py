"""E3: missed-pick early alert — DM the operator in-window if delivery failed."""
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from bts.scheduler import _maybe_alert_missed_pick, _alert_missed_pick
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


@patch("bts.scheduler._alert_missed_pick")
@patch("bts.scheduler._watchdog_ping_sleep")
@patch("bts.scheduler._now_et")
def test_alerts_when_undelivered_near_first_pitch(mock_now, _sleep, mock_alert, tmp_path):
    mock_now.return_value = datetime(2026, 4, 6, 20, 5, tzinfo=ET)  # 5 min to game (past 10-min window)
    save_pick(_daily(), tmp_path)
    _maybe_alert_missed_pick(_cfg(tmp_path), "2026-04-06", tmp_path, 10, None)
    mock_alert.assert_called_once()


@patch("bts.scheduler._alert_missed_pick")
@patch("bts.scheduler._now_et")
def test_no_alert_when_delivered(mock_now, mock_alert, tmp_path):
    mock_now.return_value = datetime(2026, 4, 6, 20, 5, tzinfo=ET)
    save_pick(_daily(bluesky_posted=True, bluesky_uri="at://x"), tmp_path)
    _maybe_alert_missed_pick(_cfg(tmp_path), "2026-04-06", tmp_path, 10, None)
    mock_alert.assert_not_called()


@patch("bts.health.alert.dispatch_dm_for_health_alerts")
def test_alert_sender_dispatches_missed_pick_critical(mock_dispatch, tmp_path):
    _alert_missed_pick(_cfg(tmp_path), _daily(), mins_to_game=5)
    mock_dispatch.assert_called_once()
    alerts = mock_dispatch.call_args[0][0]
    assert alerts[0].source == "missed_pick"
    assert alerts[0].level == "CRITICAL"
