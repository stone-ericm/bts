"""reconcile_results must respect the BTS correction cutoff (C-03).

Official Rules: Streak scores are tabulated by 08:00 ET the day after each game day and
"will not be revised or recalculated to reflect official MLB statistics changes or
corrections that occur after 8:00 a.m. ET the day following the impacted game." In 2026
MLB re-scored two Chandler Simpson singles as fielding errors days later (5/10, 8/20); the
nightly reconcile applied both at the day + 6 run and overwrote settled hits with misses.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from bts.picks import DailyPick, Pick, load_pick, reconcile_results, save_pick, save_streak

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def _plant_hit(picks_dir, day: str) -> None:
    pick = Pick(
        batter_name="Chandler Simpson", batter_id=802415, team="TB", lineup_position=1,
        pitcher_name="P", pitcher_id=456, p_game_hit=0.79, flags=[],
        projected_lineup=False, game_pk=822934, game_time=f"{day}T17:10:00Z",
    )
    save_pick(DailyPick(date=day, run_time=f"{day}T15:00:00Z", pick=pick, double_down=None,
                        runner_up=None, result="hit", slot_results={"pick": "hit"}), picks_dir)
    save_streak(1, picks_dir)


class SimClock:
    """Wall clock the test can move forward, e.g. while a feed request or a lock wait is in flight."""

    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now


def _reconcile_with_late_error(picks_dir, *, on_feed=None, **kwargs):
    """Run reconcile while MLB's current data says the play was an error (the batter has no hit)."""
    def feed(*args, **kw):
        if on_feed:
            on_feed()
        return "miss"
    with patch("bts.picks.get_game_statuses_detailed", return_value={}), \
         patch("bts.picks.check_hit", side_effect=feed) as spy:
        return reconcile_results(picks_dir, lookback_days=8, **kwargs), spy


def test_post_cutoff_rescoring_does_not_flip_a_settled_hit(tmp_path):
    """The C-03 incident (real clock, as the cron runs): the day + 6 run leaves the hit alone."""
    day = (date.today() - timedelta(days=6)).isoformat()
    _plant_hit(tmp_path, day)

    corrections, feed = _reconcile_with_late_error(tmp_path)

    assert corrections == []
    kept = load_pick(day, tmp_path)
    assert (kept.result, kept.slot_results) == ("hit", {"pick": "hit"})
    assert feed.call_count == 0  # a final date is not even re-fetched


def test_pre_cutoff_correction_is_still_applied(tmp_path):
    """A change made before 08:00 ET the next day still counts, so the 02:00 run applies it."""
    _plant_hit(tmp_path, "2026-08-20")

    corrections, _ = _reconcile_with_late_error(tmp_path, clock=SimClock(datetime(2026, 8, 21, 2, 0, tzinfo=ET)))

    assert [(c["date"], c["old_result"], c["new_result"]) for c in corrections] == [("2026-08-20", "hit", "miss")]
    assert load_pick("2026-08-20", tmp_path).result == "miss"


@pytest.mark.parametrize("day, when, applied", [
    # summer (EDT = UTC-4), given in ET and as the same instants in UTC
    ("2026-08-20", datetime(2026, 8, 21, 7, 59, tzinfo=ET), True),
    ("2026-08-20", datetime(2026, 8, 21, 8, 0, tzinfo=ET), False),
    ("2026-08-20", datetime(2026, 8, 21, 11, 59, tzinfo=UTC), True),
    ("2026-08-20", datetime(2026, 8, 21, 12, 0, tzinfo=UTC), False),
    # winter (EST = UTC-5): 07:30 ET is 12:30 UTC — a fixed UTC-4 cutoff would already be closed
    ("2026-01-10", datetime(2026, 1, 11, 12, 30, tzinfo=UTC), True),
    ("2026-01-10", datetime(2026, 1, 11, 13, 0, tzinfo=UTC), False),
    # spring-forward day 2026-03-08: the cutoff is 08:00 EDT = 12:00 UTC
    ("2026-03-07", datetime(2026, 3, 8, 11, 59, tzinfo=UTC), True),
    ("2026-03-07", datetime(2026, 3, 8, 12, 0, tzinfo=UTC), False),
    # fall-back day 2026-11-01: the cutoff is 08:00 EST = 13:00 UTC
    ("2026-10-31", datetime(2026, 11, 1, 12, 30, tzinfo=UTC), True),
    ("2026-10-31", datetime(2026, 11, 1, 13, 0, tzinfo=UTC), False),
])
def test_cutoff_is_0800_eastern_the_next_day(tmp_path, day, when, applied):
    _plant_hit(tmp_path, day)

    _reconcile_with_late_error(tmp_path, clock=SimClock(when))

    assert load_pick(day, tmp_path).result == ("miss" if applied else "hit")


def test_today_is_the_eastern_calendar_day(tmp_path):
    """23:30 ET on 8/20 is already 8/21 in UTC; 8/20 is still today in ET and is not reconciled."""
    _plant_hit(tmp_path, "2026-08-20")

    corrections, feed = _reconcile_with_late_error(tmp_path, clock=SimClock(datetime(2026, 8, 21, 3, 30, tzinfo=UTC)))

    assert corrections == [] and feed.call_count == 0
    assert load_pick("2026-08-20", tmp_path).result == "hit"


def test_feed_answer_arriving_after_the_cutoff_is_discarded(tmp_path):
    """Eligible at 07:59:59, but the feed answers at 08:00:01 — it may already carry a late change."""
    _plant_hit(tmp_path, "2026-08-20")
    clock = SimClock(datetime(2026, 8, 21, 7, 59, 59, tzinfo=ET))

    corrections, _ = _reconcile_with_late_error(
        tmp_path, clock=clock, on_feed=lambda: setattr(clock, "now", datetime(2026, 8, 21, 8, 0, 1, tzinfo=ET)))

    assert corrections == []
    assert load_pick("2026-08-20", tmp_path).result == "hit"


def test_write_landing_after_the_cutoff_is_refused(tmp_path):
    """Observed at 07:59:58, but waiting for the scoring lock pushes the write past 08:00."""
    import bts.picks as picks_mod
    _plant_hit(tmp_path, "2026-08-20")
    clock = SimClock(datetime(2026, 8, 21, 7, 59, 58, tzinfo=ET))
    real_lock = picks_mod.scoring_lock

    @contextmanager
    def slow_lock(picks_dir):
        clock.now = datetime(2026, 8, 21, 8, 0, 1, tzinfo=ET)
        with real_lock(picks_dir):
            yield

    with patch("bts.picks.scoring_lock", slow_lock):
        corrections, _ = _reconcile_with_late_error(tmp_path, clock=clock)

    assert corrections == []
    assert load_pick("2026-08-20", tmp_path).result == "hit"


def test_late_feed_answer_is_discarded_even_if_the_clock_steps_back(tmp_path):
    """The answer arrives at 08:00:01; a wall-clock step back (e.g. an NTP correction) puts the
    locked write at 07:59:59. Only the arrival-time check can stop this write."""
    import bts.picks as picks_mod
    _plant_hit(tmp_path, "2026-08-20")
    clock = SimClock(datetime(2026, 8, 21, 7, 59, 59, tzinfo=ET))
    real_lock = picks_mod.scoring_lock

    @contextmanager
    def lock_after_step_back(picks_dir):
        clock.now = datetime(2026, 8, 21, 7, 59, 59, tzinfo=ET)
        with real_lock(picks_dir):
            yield

    with patch("bts.picks.scoring_lock", lock_after_step_back):
        corrections, _ = _reconcile_with_late_error(
            tmp_path, clock=clock, on_feed=lambda: setattr(clock, "now", datetime(2026, 8, 21, 8, 0, 1, tzinfo=ET)))

    assert corrections == []
    assert load_pick("2026-08-20", tmp_path).result == "hit"


def test_replay_boundary_is_read_under_the_lock(tmp_path):
    """A run that starts at 23:59:59 and gets the lock after midnight replays through the new
    day, so the settled 8/20 hit stays in the local streak (codex r2 #2)."""
    import bts.picks as picks_mod
    from bts.picks import load_streak
    _plant_hit(tmp_path, "2026-08-20")  # local streak 1
    clock = SimClock(datetime(2026, 8, 20, 23, 59, 59, tzinfo=ET))
    real_lock = picks_mod.scoring_lock

    @contextmanager
    def lock_after_midnight(picks_dir):
        clock.now = datetime(2026, 8, 21, 0, 0, 1, tzinfo=ET)
        with real_lock(picks_dir):
            yield

    with patch("bts.picks.scoring_lock", lock_after_midnight):
        _reconcile_with_late_error(tmp_path, clock=clock)

    assert load_streak(tmp_path) == 1


def test_clock_without_timezone_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="timezone"):
        reconcile_results(tmp_path, clock=lambda: datetime(2026, 8, 21, 2, 0))
