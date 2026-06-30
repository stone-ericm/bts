"""Suspended-game scoring in the production hit-check path.

A suspended-then-resumed game keeps its original officialDate and a cumulative final
boxscore, but per BTS rules the resumed portion is never evaluated. The production
scorer must grade from pre-suspension plays only, so a resumed-portion hit cannot
continue a streak. See docs/audit/2026-06-29-skip-threshold-and-discrimination.md.
"""
from bts.picks import _grade_pick_pre_suspension, _check_hit_in_game, _boxscore_hit

RESUME = "2026-06-17T18:00:00Z"
PRE = "2026-06-16T23:30:00Z"   # before resumeDateTime -> pre-suspension
POST = "2026-06-17T18:30:00Z"  # at/after resumeDateTime -> resumed portion


def _play(batter_id, event_type, start_time, full_name=None):
    return {
        "result": {"eventType": event_type},
        "matchup": {"batter": {"id": batter_id, "fullName": full_name or f"Batter {batter_id}"}},
        "about": {"startTime": start_time},
    }


def _feed(*, resume_dt=None, plays=None, boxscore_hits=None, batter_id=11):
    """Minimal game-feed resp. boxscore_hits sets the (cumulative) boxscore hit total."""
    away_players = {}
    if boxscore_hits is not None:
        away_players[f"ID{batter_id}"] = {
            "person": {"id": batter_id, "fullName": f"Batter {batter_id}"},
            "stats": {"batting": {"hits": boxscore_hits}},
        }
    return {
        "gameData": {
            "datetime": {"resumeDateTime": resume_dt} if resume_dt else {},
            "status": {"abstractGameCode": "F"},
        },
        "liveData": {
            "plays": {"allPlays": plays or []},
            "boxscore": {"teams": {"away": {"players": away_players}, "home": {"players": {}}}},
        },
    }


# ---- _grade_pick_pre_suspension ----

def test_grade_pre_suspension_hit_counts():
    resp = _feed(resume_dt=RESUME, plays=[_play(11, "single", PRE), _play(11, "field_out", POST)])
    assert _grade_pick_pre_suspension(resp, 11) == "hit"


def test_grade_resumed_hit_does_not_count():
    # pre-suspension out, resumed-portion home run -> the homer is excluded -> MISS
    resp = _feed(resume_dt=RESUME, plays=[_play(11, "field_out", PRE), _play(11, "home_run", POST)])
    assert _grade_pick_pre_suspension(resp, 11) == "miss"


def test_grade_resumed_only_is_void():
    resp = _feed(resume_dt=RESUME, plays=[_play(11, "single", POST)])  # only resumed PA
    assert _grade_pick_pre_suspension(resp, 11) == "void"


def test_grade_batter_absent_returns_none():
    resp = _feed(resume_dt=RESUME, plays=[_play(99, "single", PRE)])
    assert _grade_pick_pre_suspension(resp, 11) is None  # fall back to boxscore / other games


def test_grade_non_suspended_returns_none():
    resp = _feed(resume_dt=None, plays=[_play(11, "single", PRE)])
    assert _grade_pick_pre_suspension(resp, 11) is None


def test_grade_matches_by_name_fallback():
    play = _play(11, "single", PRE, full_name="Chase DeLauter")
    play["matchup"]["batter"]["id"] = 0  # id mismatch -> must match on name
    resp = _feed(resume_dt=RESUME, plays=[play])
    assert _grade_pick_pre_suspension(resp, 999, batter_name="Chase DeLauter") == "hit"


# ---- _check_hit_in_game (suspension-aware) ----

def test_check_hit_suspended_overrides_cumulative_boxscore():
    # boxscore shows a hit (cumulative incl. resumed), but pre-suspension was an out -> NOT a hit
    resp = _feed(resume_dt=RESUME, boxscore_hits=1,
                 plays=[_play(11, "field_out", PRE), _play(11, "home_run", POST)])
    assert _check_hit_in_game(resp, 11) is False


def test_check_hit_suspended_pre_suspension_hit_is_true():
    resp = _feed(resume_dt=RESUME, boxscore_hits=1,
                 plays=[_play(11, "single", PRE), _play(11, "field_out", POST)])
    assert _check_hit_in_game(resp, 11) is True


def test_check_hit_normal_game_uses_boxscore_unchanged():
    assert _check_hit_in_game(_feed(boxscore_hits=2), 11) is True
    assert _check_hit_in_game(_feed(boxscore_hits=0), 11) is False
    assert _check_hit_in_game(_feed(), 11) is None  # batter not in boxscore


def test_boxscore_hit_is_pure_cumulative():
    # the boxscore helper itself stays cumulative (used only for normal games)
    assert _boxscore_hit(_feed(boxscore_hits=1), 11) is True
    assert _boxscore_hit(_feed(boxscore_hits=0), 11) is False
