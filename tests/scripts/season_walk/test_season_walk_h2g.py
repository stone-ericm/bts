"""Known-answer tests for the season walk's extension on the frozen head (closing test re-run, part d).

H (the longest-streak rule from 57) and H2 (the contest-state caps raised to 400) are tested
TOGETHER: a streak reaching 57 and 58 on the automatic path, every day count, the 400 edge; G (the
monotone skip guard) as "no quality bin skips while a weaker bin plays", on the walk's own records
and exhaustively on the shipped table.
"""
from datetime import date
from unittest.mock import patch

import pytest

from scripts.audit.season_walk import walk


def test_extended_states_cover_57_58_and_far_above_on_both_paths():
    labels = {s.label() for s in walk.EXTENDED_STATES}
    assert "automatic/57/57/active/0.96" in labels
    assert "automatic/58/60/used/0.96" in labels
    assert "dated/400/400/used/0.96" in labels
    assert any(s.streak >= 57 and s.top_p == 0.72 for s in walk.EXTENDED_STATES)
    # every extended state below 57 comes in both candidate strengths (the guard pair check)
    low = [s for s in walk.EXTENDED_STATES if s.streak < walk.LONGEST_STREAK_FROM]
    keys = {(s.kind, s.streak, s.best, s.saver) for s in low}
    assert all({0.96, 0.72} <= {s.top_p for s in low if (s.kind, s.streak, s.best, s.saver) == k}
               for k in keys)
    assert not labels & {s.label() for s in walk.DEFAULT_STATES}


def test_guard_pair_check_flags_a_stronger_skip_beside_a_weaker_play():
    base = dict(day="2027-07-15", in_season=True)
    records = [base | dict(state="dated/50/50/used/0.96", action="skip"),
               base | dict(state="dated/50/50/used/0.72", action="single")]
    problems = walk.guard_pair_problems(records)
    assert len(problems) == 1 and "0.96 skips while 0.72 plays" in problems[0]
    assert walk.guard_pair_problems([records[1], base | dict(state="dated/50/50/used/0.96",
                                                            action="single")]) == []
    # after the finale every action is a skip; the pair check is in-season only
    late = [r | {"day": "2027-10-01", "in_season": False} for r in records]
    assert walk.guard_pair_problems(late) == []


def test_shipped_table_never_skips_a_bin_above_a_playing_bin():
    result = walk.table_guard_check()
    assert result["problems"] == []
    assert result["cells"] == 57 * 180 * 2 * 5
    assert result["changed_states"] == 1511          # G's documented count


def test_table_guard_check_catches_the_unguarded_lookup():
    # failure direction: the raw table action (no guard) is exactly the defect G removed
    from bts.simulate import mdp
    result = walk.table_guard_check(lookup=lambda *a: (mdp.lookup_action(*a), None))
    assert len(result["problems"]) == 1511


def test_cap_edges_accept_400_and_refuse_401_on_both_writers():
    result = walk.cap_checks()
    assert result["problems"] == []
    assert result["dated_400"] == "accepted" and result["dated_401"] == "refused"
    assert result["automatic_400"] == "accepted" and result["automatic_401"] == "refused"


def test_automatic_58_plays_a_single_with_the_longest_objective():
    rec = walk.walk_one(date(2027, 8, 3), walk.State("automatic", 58, 60, "used", 0.96))
    assert rec["error"] is None and rec["setup"] == []
    assert (rec["objective"], rec["action"], rec["chosen_on_kind"]) == ("longest_streak", "single",
                                                                         "automatic")
    assert walk.check([rec]) == []


def test_sequence_from_55_through_59_keeps_playing_and_every_reader_accepts():
    result = walk.sequence_57_58()
    assert result["problems"] == []
    days = result["days"]
    assert [d["mlb_streak"] for d in days] == [55, 56, 57, 58, 59]
    assert [d["objective"] for d in days][2:] == ["longest_streak"] * 3
    assert [d["action"] for d in days][2:] == ["single"] * 3
    assert all(d["fetch_exit"] == 0 and d["saved_pick_refusal"] is None for d in days)
    assert [d["model_streak_after_hit"] for d in days] == [56, 57, 58, 59, 60]


def test_sequence_without_raised_caps_stops_at_58():
    # failure direction: H without H2 (the caps back at 57) must show up as problems at 58
    import bts.contest_fetch as fetch_mod
    import bts.contest_state as state_mod
    import bts.picks as picks_mod
    with patch.object(state_mod, "MAX_SEASON_STREAK", 57), \
            patch.object(picks_mod, "MAX_SEASON_STREAK", 57), \
            patch.object(fetch_mod, "MAX_SEASON_STREAK", 57, create=True):
        result = walk.sequence_57_58()
    assert any("2027-08-04" in p for p in result["problems"])
