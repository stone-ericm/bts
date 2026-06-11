"""E2: delivery idempotency — a crash mid-send must not double-post on restart."""
from unittest.mock import patch, MagicMock

from bts.scheduler import _deliver_and_lock_pick, SchedulerState
from bts.picks import DailyPick, Pick

CONFIG = {"scheduler": {"pick_delivery": "public"}}


def _state(date="2026-04-06"):
    return SchedulerState(
        date=date, schedule_fetched_at="x", games=[], confirmed_game_pks=[],
        runs_completed=[], pick_locked=False, pick_locked_at=None,
        result_status=None, next_wakeup=None,
    )


def _daily(**kw):
    d = DailyPick(
        date="2026-04-06", run_time="2026-04-06T19:29:00+00:00",
        pick=Pick(batter_name="Hoerner", batter_id=1, team="CHC",
                  lineup_position=1, pitcher_name="Baz", pitcher_id=2,
                  p_game_hit=0.73, flags=[], projected_lineup=False,
                  game_pk=100, game_time="2026-04-06T20:10:00Z"),
        double_down=None, runner_up=None,
    )
    for k, v in kw.items():
        setattr(d, k, v)
    return d


@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=4))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.posting.post_to_bluesky", return_value="at://uri")
def test_unconfirmed_prior_attempt_does_not_repost(mock_post, _cap, _dss, tmp_path):
    """delivery_attempted=True but not delivered → daemon crashed mid-send → DON'T re-post."""
    daily = _daily(delivery_attempted=True)
    _deliver_and_lock_pick(daily, CONFIG, tmp_path, _state(), "2026-04-06", "test")
    mock_post.assert_not_called()


@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=4))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.posting.post_to_bluesky", return_value="at://uri")
def test_fresh_pick_posts_normally(mock_post, _cap, _dss, tmp_path):
    """A fresh pick (no prior attempt) posts and records success — guard must not over-trigger."""
    daily = _daily()
    ok = _deliver_and_lock_pick(daily, CONFIG, tmp_path, _state(), "2026-04-06", "test")
    assert ok is True
    mock_post.assert_called_once()
    assert daily.bluesky_posted is True


@patch("bts.contest_state.load_decision_streak_state", return_value=MagicMock(streak=4))
@patch("bts.scheduler._trigger_live_forward_capture_on_lock")
@patch("bts.posting.post_to_bluesky")
def test_caught_failure_allows_retry(mock_post, _cap, _dss, tmp_path):
    """A KNOWN failure (post raised, caught) clears the marker so the next cycle retries —
    only an uncaught crash should block. Protects the clear-on-failure logic."""
    mock_post.side_effect = [RuntimeError("network blip"), "at://uri"]
    daily = _daily()
    ok1 = _deliver_and_lock_pick(daily, CONFIG, tmp_path, _state(), "2026-04-06", "test")
    assert ok1 is False
    ok2 = _deliver_and_lock_pick(daily, CONFIG, tmp_path, _state(), "2026-04-06", "test")
    assert ok2 is True
    assert mock_post.call_count == 2
