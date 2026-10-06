"""Checklist C1 and C2 (docs/ops/2027-season-start.md), prerequisites of the C1 rank-2 watchdog (plan P4/P5).

C1: private mode must never touch a delivery transport. Both transports are patched with recorders, so a regression
cannot post for real and cannot pass silently.
C2: a legacy `scheduler.shadow_mode`-only config used to fall through to public posting; it is now refused.
"""
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from bts.picks import DailyPick, Pick
from bts.scheduler import SchedulerState, _deliver_and_lock_pick, _pick_delivery_mode

ET = ZoneInfo("America/New_York")


def _state():
    return SchedulerState(date="2026-04-06", schedule_fetched_at="x", games=[], confirmed_game_pks=[],
                          runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None,
                          next_wakeup=None)


def _daily():
    return DailyPick(date="2026-04-06", run_time="2026-04-06T19:29:00+00:00",
                     pick=Pick(batter_name="Hoerner", batter_id=1, team="CHC", lineup_position=1, pitcher_name="Baz",
                               pitcher_id=2, p_game_hit=0.73, flags=[], projected_lineup=False, game_pk=100,
                               game_time="2026-04-06T20:10:00Z"),
                     double_down=None, runner_up=None)


@pytest.mark.parametrize("config", [{"scheduler": {"pick_delivery": "private"}},
                                    {"scheduler": {"private_mode": True}},
                                    {"scheduler": {"posting_mode": "local"}}])
@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=4))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.dm.send_dm")
@patch("bts.posting.post_to_bluesky")
def test_private_mode_never_touches_a_transport(post, dm, _cap, _dss, config, tmp_path):
    with patch("bts.scheduler._now_et", return_value=datetime(2026, 4, 6, 15, 0, tzinfo=ET)):
        daily = _daily()
        _deliver_and_lock_pick(daily, config, tmp_path, _state(), "2026-04-06", "test")
    post.assert_not_called()
    dm.assert_not_called()
    assert daily.bluesky_posted is False and not daily.notification_sent


def test_a_legacy_shadow_mode_only_config_is_refused():
    with pytest.raises(ValueError, match="shadow_mode"):
        _pick_delivery_mode({"scheduler": {"shadow_mode": True}})
    with pytest.raises(ValueError, match="shadow_mode"):
        _pick_delivery_mode({"scheduler": {"shadow_mode": False}})


def test_an_explicit_delivery_mode_still_wins_and_shadow_model_is_unrelated():
    assert _pick_delivery_mode({"scheduler": {"shadow_mode": True, "pick_delivery": "private"}}) == "private"
    assert _pick_delivery_mode({"scheduler": {"shadow_model": True, "pick_delivery": "dm"}}) == "dm"
    assert _pick_delivery_mode({"scheduler": {"shadow_model": True}}) == "public"     # the shadow MODEL key


def test_run_day_refuses_a_legacy_config_before_any_work():
    from bts.scheduler import run_day
    with patch("bts.scheduler.fetch_schedule", side_effect=AssertionError("no work may start")) as fs:
        with pytest.raises(ValueError, match="shadow_mode"):
            run_day(date="2026-04-06", config={"scheduler": {"shadow_mode": True}}, dry_run=True)
    fs.assert_not_called()
