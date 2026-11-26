"""Pure-logic tests for the post-tabulation MLB scoring-change check (season-wrap W0.6)."""
from __future__ import annotations

from datetime import datetime, timezone

from scripts.audit.bts_scoring_change_check import (
    bts_outcome,
    diff_outcomes,
    select_timecode,
    streak_changed,
)


# --- BTS outcome from an official batting line (Official Rules §6) -------------
def test_any_hit_is_hit():
    assert bts_outcome({"hits": 1, "atBats": 4, "sacFlies": 0}) == "hit"


def test_at_bat_without_hit_is_no_hit():
    assert bts_outcome({"hits": 0, "atBats": 3, "sacFlies": 0}) == "no_hit"


def test_sac_fly_alone_counts_as_an_opportunity():
    # "so long as your Pick had at least one official at-bat or one sacrifice fly"
    assert bts_outcome({"hits": 0, "atBats": 0, "sacFlies": 1}) == "no_hit"


def test_walks_hbp_sac_bunts_only_is_pass():
    assert bts_outcome({"hits": 0, "atBats": 0, "sacFlies": 0, "baseOnBalls": 2,
                        "hitByPitch": 1, "sacBunts": 1}) == "pass"


def test_did_not_play_is_pass():
    assert bts_outcome({}) == "pass"
    assert bts_outcome(None) == "pass"


# --- timecode selection -------------------------------------------------------
def test_select_timecode_takes_latest_at_or_before():
    ts = ["20260927_174931", "20260927_220913", "20260928_130000"]
    at = datetime(2026, 9, 27, 22, 51, 54, tzinfo=timezone.utc)
    assert select_timecode(ts, at) == "20260927_220913"


def test_select_timecode_inclusive_boundary():
    ts = ["20260927_174931", "20260927_225154"]
    assert select_timecode(ts, datetime(2026, 9, 27, 22, 51, 54, tzinfo=timezone.utc)) == "20260927_225154"


def test_select_timecode_none_before_first():
    assert select_timecode(["20260927_174931"], datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)) is None


# --- outcome diff --------------------------------------------------------------
def test_diff_reports_only_changed_players_and_treats_missing_as_pass():
    before = {1: "hit", 2: "no_hit", 3: "pass"}
    after = {1: "hit", 2: "hit", 4: "no_hit"}
    got = {(c["player_id"], c["before"], c["after"]) for c in diff_outcomes(before, after)}
    # 3 disappears (missing == pass == unchanged); 4 appears (pass -> no_hit)
    assert got == {(2, "no_hit", "hit"), (4, "pass", "no_hit")}


# --- correction table (initial -> revised by 08:00 ET next day) ------------------
def test_streak_changed_follows_the_official_table():
    # Hit revised to Pass still "Increase by one" -> no streak change
    assert streak_changed("hit", "pass") is False
    assert streak_changed("hit", "no_hit") is True      # increase -> End
    assert streak_changed("pass", "hit") is True        # hold -> increase
    assert streak_changed("pass", "no_hit") is True     # hold -> End
    assert streak_changed("no_hit", "pass") is True     # End -> hold
    assert streak_changed("no_hit", "hit") is True      # End -> increase
    for o in ("hit", "no_hit", "pass"):
        assert streak_changed(o, o) is False
