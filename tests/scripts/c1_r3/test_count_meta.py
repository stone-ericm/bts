"""T1: certified starting identities and the chronological PA list from one feed (synthetic)."""
from scripts.audit.c1_r3 import count_meta as M
from tests.scripts.c1_r3.feeds import feed, lineup, play


def test_starters_come_from_exact_hundreds_and_substitutes_are_kept_apart():
    m = M.extract(feed(away=lineup(100, subs=[(111, "101"), (112, "402")])))
    assert m.starters["away"] == {k: 100 + k for k in range(1, 10)}
    assert m.substitutes["away"] == {111: (1, 1), 112: (4, 2)}
    assert m.starters["home"][9] == 209 and m.problems == ()


def test_a_missing_or_duplicate_starting_slot_is_a_problem():
    short = M.extract(feed(away=lineup(100, n=8)))
    assert any("away" in p and "slots" in p for p in short.problems)
    dup = lineup(100)
    dup["ID199"] = {"person": {"id": 199}, "battingOrder": "300"}
    assert any("duplicate" in p for p in M.extract(feed(away=dup)).problems)


def test_the_starting_pitcher_is_the_first_boxscore_pitcher_cross_checked_with_the_plays():
    m = M.extract(feed())
    assert m.starting_pitcher == {"away": 150, "home": 250} and m.problems == ()
    plays = [play(0, "top", 101, 999), play(1, "bottom", 201, 150)]
    bad = M.extract(feed(plays=plays))
    assert any("home" in p and "starting pitcher" in p for p in bad.problems)


def test_pa_list_is_chronological_pa_ending_only_with_batting_side():
    plays = [play(0, "top", 101, 250), play(1, "top", 102, 250, event="caught_stealing_2b"),
             play(2, "bottom", 201, 150, event="walk")]
    m = M.extract(feed(plays=plays))
    assert [(p.index, p.side, p.batter, p.pitcher, p.event) for p in m.pas] == \
        [(0, "away", 101, 250, "single"), (2, "home", 201, 150, "walk")]
    rev = M.extract(feed(plays=[play(1, "top", 101, 250), play(0, "bottom", 201, 150)]))
    assert any("atBatIndex" in p for p in rev.problems)


def test_the_resumed_portion_is_flagged_as_production_flags_it():
    plays = [play(0, "top", 101, 250, start="2023-06-01T23:10:00Z"),
             play(1, "bottom", 201, 150, start="2023-06-02T20:00:00Z"),
             play(2, "top", 102, 250, start=None)]
    m = M.extract(feed(plays=plays, resume="2023-06-02T19:00:00Z"))
    assert [p.resumed for p in m.pas] == [False, True, True]          # a missing time is treated as resumed
    assert all(not p.resumed for p in M.extract(feed(plays=plays)).pas)  # no resume time: nothing is resumed


def test_identity_fields():
    m = M.extract(feed(pk=716404, date="2023-09-28"))
    assert (m.game_pk, m.official_date, m.season) == (716404, "2023-09-28", 2023)


# ---------- review r1 F4/F5: the certificates are checked, not assumed ----------
def test_slot_codes_must_be_three_digit_strings():
    for code in ("100.9", 100, "1000", "010", "abc"):
        players = lineup(100)
        players["ID101"]["battingOrder"] = code
        assert any("slot code" in p or "missing" in p for p in M.extract(feed(away=players)).problems), code


def test_the_same_person_cannot_start_twice_or_for_both_sides():
    players = lineup(100)
    players["ID102"]["person"]["id"] = 101
    assert any("two slots" in p or "duplicate" in p for p in M.extract(feed(away=players)).problems)
    home = lineup(200)
    home["ID201"]["person"]["id"] = 101
    assert any("both sides" in p for p in M.extract(feed(home=home)).problems)


def test_an_unknown_half_inning_or_backwards_time_or_a_gap_is_a_problem():
    assert any("halfInning" in p for p in M.extract(feed(plays=[play(0, "middle", 101, 250)])).problems)
    back = [play(0, "top", 101, 250, start="2023-06-01T23:30:00Z"), play(1, "bottom", 201, 150, start="2023-06-01T23:10:00Z")]
    assert any("backwards" in p for p in M.extract(feed(plays=back)).problems)
    gap = [play(0, "top", 101, 250), play(2, "bottom", 201, 150)]
    assert any("contiguous" in p for p in M.extract(feed(plays=gap)).problems)


def test_an_unfinished_game_or_a_date_outside_its_season_is_a_problem():
    for st in ("Suspended", "In Progress", "Postponed"):
        assert any("not a completed game" in p for p in M.extract(feed(status=st)).problems), st
    assert M.extract(feed(status="Completed Early: Rain")).completed
    assert not any("ISO date" in p for p in M.extract(feed(date="2023-06-01")).problems)
    bad = feed(date="2023-06-01")
    bad["gameData"]["datetime"]["officialDate"] = "2026-06-01"
    assert any("ISO date in season" in p for p in M.extract(bad).problems)


def test_the_starting_pitcher_is_checked_on_the_first_play_of_any_kind():
    """A starter who faced the first batter in a non-PA play (e.g. an inning-ending caught stealing) is confirmed."""
    plays = [play(0, "top", 101, 250, event="caught_stealing_2b", inning=1), play(1, "bottom", 201, 150, inning=1),
             play(2, "top", 101, 260, inning=2)]
    assert not any("starting pitcher" in p for p in M.extract(feed(plays=plays)).problems)


def test_a_garbled_timestamp_is_a_problem_not_an_exception():
    plays = [play(0, "top", 101, 250, start="garbled"), play(1, "bottom", 201, 150)]
    m = M.extract(feed(plays=plays, resume="2023-06-02T19:00:00Z"))
    assert any("unparseable startTime" in p for p in m.problems)


# ---------- review r2 N2/N4: totals, terminal extent, strict values ----------
def test_unsupported_values_are_problems_not_coercions():
    """r2 N4's demonstrated cases."""
    cases = {
        "boolean game id": feed(pk=True),
        "fractional season": feed(season="2023.9"),
        "garbage date": feed(date="2023-garbage"),
        "status suffix": feed(status="Finalish - In Progress"),
    }
    for label, f in cases.items():
        assert M.extract(f).problems, label
    no_innings = feed()
    for pl in no_innings["liveData"]["plays"]["allPlays"]:
        del pl["about"]["inning"]
    assert any("inning" in p for p in M.extract(no_innings).problems)
    nl = lineup(100)
    nl["ID101"]["battingOrder"] = "100\n"
    assert any("slot code" in p for p in M.extract(feed(away=nl)).problems)
    both = M.extract(feed(away=lineup(100, subs=[(111, "101")]), home=lineup(200, subs=[(111, "101")])))
    assert any("both sides" in p for p in both.problems)


def test_completed_statuses_are_bounded():
    for st in ("Final", "Game Over", "Completed Early", "Completed Early: Rain", "Final: Tied"):
        assert M.extract(feed(status=st)).completed, st
    for st in ("Final Score Pending", "Completed", "In Progress", "Suspended: Rain"):
        assert not M.extract(feed(status=st)).completed, st


def test_the_last_play_must_be_complete_and_in_the_linescore_inning():
    f = feed()
    f["liveData"]["plays"]["allPlays"][-1]["about"]["isComplete"] = False
    assert any("not complete" in p for p in M.extract(f).problems)
    assert any("currentInning" in p for p in M.extract(feed(current_inning=9)).problems)


def test_half_innings_must_move_forward():
    plays = [play(0, "top", 101, 250, inning=2), play(1, "bottom", 201, 150, inning=1)]
    assert any("backwards" in p for p in M.extract(feed(plays=plays, current_inning=1)).problems)
