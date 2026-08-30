"""The contest submission cutoff (first pitch − 5 min) has ONE definition (bts.picks).

2026-08-30: the daemon DM'd Kwan at 13:36:14 for a 13:40 first pitch. The literal
`5` lived in cli.py and health/pick_entry.py; the scheduler never consulted it.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _daily(primary_utc="2026-08-30T17:40:00Z", dd_utc=None):
    from bts.picks import DailyPick, Pick
    pick = Pick(batter_name="Kwan", batter_id=680757, team="CLE", lineup_position=1,
                pitcher_name="Lugo", pitcher_id=607625, p_game_hit=0.7566, flags=[],
                projected_lineup=False, game_pk=824393, game_time=primary_utc)
    dd = None
    if dd_utc:
        dd = Pick(batter_name="McNeil", batter_id=643446, team="ATH", lineup_position=2,
                  pitcher_name="Bassitt", pitcher_id=605135, p_game_hit=0.7428, flags=[],
                  projected_lineup=False, game_pk=824959, game_time=dd_utc)
    return DailyPick(date="2026-08-30", run_time="2026-08-30T17:36:12+00:00",
                     pick=pick, double_down=dd, runner_up=None)


def test_constant_is_five_minutes():
    from bts.picks import SUBMISSION_CUTOFF_MIN
    assert SUBMISSION_CUTOFF_MIN == 5


def test_cutoff_is_earliest_slot_minus_five():
    from bts.picks import earliest_pick_game_et, submission_cutoff_et
    d = _daily(primary_utc="2026-08-30T20:05:00Z", dd_utc="2026-08-30T17:40:00Z")
    assert earliest_pick_game_et(d) == datetime(2026, 8, 30, 13, 40, tzinfo=ET)
    assert submission_cutoff_et(d) == datetime(2026, 8, 30, 13, 35, tzinfo=ET)


def test_single_pick_cutoff():
    from bts.picks import submission_cutoff_et
    assert submission_cutoff_et(_daily()) == datetime(2026, 8, 30, 13, 35, tzinfo=ET)


def test_naive_game_time_is_treated_as_utc():
    """Pick.game_time is documented UTC; a naive value must not be read in the
    host timezone (Codex r2 F8: under TZ=America/New_York a naive intended-UTC
    17:40 produced a cutoff four hours late)."""
    from bts.picks import submission_cutoff_et
    d = _daily(primary_utc="2026-08-30T17:40:00")   # tz-naive, intended UTC
    assert submission_cutoff_et(d) == datetime(2026, 8, 30, 13, 35, tzinfo=ET)


def test_scheduler_and_health_reuse_the_constant():
    from bts.health import pick_entry
    from bts.picks import SUBMISSION_CUTOFF_MIN
    from bts.scheduler import _earliest_pick_game_et
    assert pick_entry.SUBMIT_CUTOFF_MIN is SUBMISSION_CUTOFF_MIN
    d = _daily(primary_utc="2026-08-30T20:05:00Z", dd_utc="2026-08-30T17:40:00Z")
    assert _earliest_pick_game_et(d) == datetime(2026, 8, 30, 13, 40, tzinfo=ET)
