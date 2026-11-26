from datetime import date

from bts.web import _streak_subtitle
from bts.contest_state import DecisionStreakState


def _state(status, streak=8, model=10, src_date=date(2026, 6, 16)):
    return DecisionStreakState(
        streak=streak, saver_available=False, allow_double=True, source="contest",
        status=status, model_streak=model, model_saver_available=True,
        contest_streak=streak, contest_saver_available=None, contest_source_date=src_date)


def test_lagged_subtitle_says_last_confirmed_through_date():
    sub = _streak_subtitle(_state("lagged"))
    assert "Last confirmed" in sub and "2026-06-16" in sub


def test_stale_subtitle_warns_may_be_lower():
    sub = _streak_subtitle(_state("stale"))
    assert "Last confirmed" in sub and "may be lower" in sub.lower()


def test_fresh_no_divergence_is_contest_state():
    assert _streak_subtitle(_state("fresh", streak=8, model=8)) == "Contest State"


def test_model_replay_shown_when_diverged():
    # the what-if model streak is surfaced as a labeled research note
    assert "Replay 10" in _streak_subtitle(_state("lagged"))


def test_error_message_first():
    assert _streak_subtitle(None, "bad contest state") == "Streak State Error"
