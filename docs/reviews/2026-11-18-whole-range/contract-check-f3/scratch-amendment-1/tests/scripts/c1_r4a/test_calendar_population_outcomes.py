"""C1 rank 4a, T1-T3: the fixed calendar, the eligible slate population and the known outcomes
(registration docs/sota_audit/2026-10-04-prereg-c1-calibration.md §§2-3; review r1 R1, R7, R9). Synthetic fixtures."""
import hashlib
import json
from datetime import date, timedelta

import pytest

from scripts.audit.c1_r4a import calendar as C
from scripts.audit.c1_r4a import outcomes as O
from scripts.audit.c1_r4a import population as P
from tests.scripts.c1_r4a.fixtures import T, box_of, feed


# ---- T1: the calendar ---------------------------------------------------------------------------------------------

def days_from(start, n, year=2027):
    d0 = date(year, *start)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]


def cal_bytes(days, *, season=2027, end="last"):
    return json.dumps({"schema": "c1_r4a_calendar_v1", "season": season, "dates": days,
                       "contest_end": days[-1] if end == "last" and days else end}).encode()


def load(raw):
    return C.load(raw, hashlib.sha256(raw).hexdigest())


def test_numbering_follows_the_frozen_calendar_not_the_data():
    days = days_from((3, 25), 7) + days_from((4, 1), 30) + days_from((5, 1), 31) + days_from((6, 1), 30) + \
        days_from((7, 1), 20)
    cal = load(cal_bytes(days))
    assert cal.number(date(2027, 3, 25)) == 1 and cal.number(date(2027, 4, 23)) == 30
    assert cal.window(1) == "fit" and cal.window(30) == "fit" and cal.window(31) == "test" and cal.window(90) == "test"
    assert cal.window(91) is None and cal.number(date(2027, 3, 24)) is None
    assert cal.date_of(40) == date(2027, 5, 3) and cal.date_of(90) == date(2027, 6, 22)
    assert cal.contest_end == date(2027, 7, 20)


def test_a_date_without_a_slate_still_counts():
    """Missing slates never shift the windows: numbering is the calendar's, whatever data exists."""
    cal = load(cal_bytes(days_from((4, 1), 120)))
    assert [cal.number(date(2027, 4, d)) for d in (1, 2, 30)] == [1, 2, 30]
    assert cal.fit_dates() == [date(2027, 4, d) for d in range(1, 31)]
    assert len(cal.test_dates()) == 60 and cal.test_dates()[0] == date(2027, 5, 1)


@pytest.mark.parametrize("raw, why", [
    (cal_bytes(["2027-04-02", "2027-04-01"] + days_from((4, 3), 90)), "increasing"),
    (cal_bytes(["2027-04-01", "2027-04-01"] + days_from((4, 2), 90)), "increasing"),
    (cal_bytes(days_from((4, 1), 120, year=2026), season=2026), "season is not 2027"),   # another season (R7)
    (cal_bytes(days_from((4, 1), 119) + ["2028-01-01"]), "outside the season"),
    (cal_bytes(["2027-13-01"] + days_from((4, 1), 90)), "unreadable"),
    (cal_bytes([], end="2027-09-30"), "need 90"),
    (cal_bytes(days_from((4, 1), 89)), "need 90"),                                   # no date 90 (R7)
    (cal_bytes(days_from((4, 1), 39)), "need 90"),                                   # no date 40
    (cal_bytes(days_from((4, 1), 120), end="2027-09-30"), "contest end"),
    (cal_bytes(days_from((4, 1), 120), end=None), "schema"),
    (json.dumps({"schema": "c1_r4a_calendar_v1", "season": True, "dates": days_from((4, 1), 120),
                 "contest_end": "2027-07-29"}).encode(), "season is not 2027"),
])
def test_a_malformed_calendar_refuses(raw, why):
    with pytest.raises(C.CalendarError, match=why):
        load(raw)


def test_the_calendar_must_match_its_pin():
    with pytest.raises(C.CalendarError, match="pin"):
        C.load(cal_bytes(days_from((4, 1), 120)), "0" * 64)


def test_a_date_past_the_calendar_refuses_explicitly():
    cal = load(cal_bytes(days_from((4, 1), 90)))
    assert cal.date_of(90) == date(2027, 6, 29)
    with pytest.raises(C.CalendarError, match="no contest date 91"):
        cal.date_of(91)


# ---- T2: the population -------------------------------------------------------------------------------------------

def slate(rows, written_at="2027-04-01T16:00:00+00:00", schema="bts_slate_v2", d="2027-04-01"):
    return json.dumps({"schema_version": schema, "date": d, "tier": "local", "written_at": written_at,
                       "n_rows": len(rows), "rows": rows}).encode()


def row(batter, game, p, *, game_time="2027-04-01T23:10:00Z", status="Scheduled", projected=False):
    return {"batter_id": batter, "game_pk": game, "p_game_hit": p, "game_time": game_time, "status": status,
            "projected": projected}


D = date(2027, 4, 1)


def test_eligibility_and_every_exclusion_is_counted():
    rows = [row(1, 10, 0.80), row(2, 10, 0.70, status="Pre-Game"), row(3, 11, 0.60, status="Warmup"),
            row(4, 12, 0.90, game_time="2027-04-01T16:04:00Z"),          # write is not before start - 5 min
            row(5, 13, 0.75, status="Postponed"), row(6, 14, 0.75, status="In Progress"),
            row(7, 15, 0.75, status=None), row(8, 16, 0.75, game_time=None),
            row(9, 17, True), row(10, 18, float("nan")), row(11, 19, 1.5), row(12, 20, "0.7"),
            row(13, 21, 0.5), row(13, 21, 0.6),                          # a duplicated identity
            row(14, 22, 0.7, status="Delayed Start: Rain"), row(15, 0, 0.7), "not a row"]
    raw = slate(rows)
    pop = P.population(raw, d=D)
    assert [r["batter_id"] for r in pop.eligible] == [1, 2, 3, 14]
    assert pop.excluded == {"write_not_before_cutoff": 1, "status_not_pregame": 2, "missing_status": 1,
                            "missing_start": 1, "invalid_probability": 4, "duplicate_identity": 2,
                            "invalid_identity": 2}
    assert pop.slate_sha256 == hashlib.sha256(raw).hexdigest() and pop.rank1["batter_id"] == 1
    assert pop.rank1_index == 0


def test_invalid_identities_are_counted_before_duplicates_and_never_alias_valid_ones():
    """R9: True == 1 and 1.0 == 1 in Python; a typed check comes first, so the valid row survives."""
    pop = P.population(slate([row(1, 10, 0.8), row(True, 10, 0.7), row(1.0, 10, 0.6), row(1, 10.0, 0.6)]), d=D)
    assert [r["batter_id"] for r in pop.eligible] == [1]
    assert pop.excluded == {"invalid_identity": 3}


def test_an_unhashable_identity_is_counted_not_raised():
    pop = P.population(slate([row([1], 10, 0.8), row(2, {"pk": 10}, 0.7), row(3, 10, 0.6)]), d=D)
    assert [r["batter_id"] for r in pop.eligible] == [3] and pop.excluded == {"invalid_identity": 2}


def test_a_huge_or_non_finite_probability_is_counted_not_raised():
    raw = slate([row(1, 10, 0.8), row(2, 11, 0.5)]).replace(b'"p_game_hit": 0.5', b'"p_game_hit": ' + b"1" * 400)
    raw2 = slate([row(3, 12, 0.5)]).replace(b'"p_game_hit": 0.5', b'"p_game_hit": 1e400')
    assert P.population(raw, d=D).excluded == {"invalid_probability": 1}
    assert P.population(raw2, d=D).excluded == {"invalid_probability": 1}
    assert P.population(slate([row(4, 13, 1), row(5, 14, 0)]), d=D).excluded == {}       # integer 0 / 1 are valid


def test_genuine_duplicates_exclude_every_copy():
    pop = P.population(slate([row(1, 10, 0.8), row(1, 10, 0.8), row(1, 10, 0.7), row(2, 10, 0.6)]), d=D)
    assert [r["batter_id"] for r in pop.eligible] == [2] and pop.excluded == {"duplicate_identity": 3}


def test_rank1_ties_keep_stored_order_and_never_look_at_outcomes():
    pop = P.population(slate([row(5, 10, 0.70), row(6, 11, 0.80), row(7, 12, 0.80)]), d=D)
    assert pop.rank1["batter_id"] == 6 and pop.rank1_index == 1


@pytest.mark.parametrize("raw", [
    slate([row(1, 10, 0.8)], schema="bts_slate_v1"), slate([row(1, 10, 0.8)], d="2027-04-02"), b"{not json",
    slate([row(1, 10, 0.8)], written_at="2027-04-01T16:00:00"), b"[1, 2]",
    json.dumps({"schema_version": "bts_slate_v2", "date": "2027-04-01", "written_at": "2027-04-01T16:00:00+00:00",
                "rows": {"a": 1}}).encode(),
    b"[" * 100000,
])
def test_an_unsupported_slate_refuses(raw):
    with pytest.raises(P.PopulationError):
        P.population(raw, d=D)


def test_states_are_counted_for_the_eligible_rows_and_every_exclusion():
    rows = [row(1, 10, 0.8, projected=True), row(2, 11, 0.7), row(3, 12, 0.7, status="Postponed", projected=True),
            {**row(4, 13, 0.6), "projected": None}, row(5, 14, True)]
    pop = P.population(slate(rows), d=D)
    assert pop.projected == {"projected": 1, "confirmed": 1, "unknown": 1}
    assert pop.excluded_by_state == {"status_not_pregame|projected": 1, "invalid_probability|confirmed": 1}


def test_the_envelope_is_kept_without_the_rows():
    pop = P.population(slate([row(1, 10, 0.8)]), d=D)
    assert pop.envelope["tier"] == "local" and "rows" not in pop.envelope


# ---- T3: outcomes -------------------------------------------------------------------------------------------------

def test_hit_no_hit_and_no_pa_in_a_complete_game():
    raw = feed(10, [(1, "single", T), (2, "strikeout", T), (2, "field_out", T)], bench=[(3, "away")])
    g = O.game_outcomes(raw, game_pk=10)
    assert g.complete and (g.outcome(1), g.outcome(2), g.outcome(3), g.outcome(4)) == ("hit", "no_hit", "no_pa", "no_pa")


def test_a_truncated_play_list_is_not_a_complete_source():
    """R1 (review counterexample): batter 1 strikes out then singles, batter 2 singles; the archive keeps only the
    first play. Every affected label is unknown, never no_hit or no_pa."""
    full = [(1, "strikeout", T, True), (1, "single", T, True), (2, "single", T, True)]
    whole = O.game_outcomes(feed(10, full), game_pk=10)
    assert (whole.outcome(1), whole.outcome(2)) == ("hit", "hit")
    g = O.game_outcomes(feed(10, full[:1], box=box_of(full)), game_pk=10)
    assert not g.complete and "reconcile" in g.reason
    assert (g.outcome(1), g.outcome(2)) == ("unknown", "unknown")


def test_a_truncation_that_hides_an_absent_batters_only_pa_leaves_him_unknown():
    full = [(1, "single", T, True), (3, "field_out", T, True)]
    g = O.game_outcomes(feed(10, full[:1], box=box_of(full)), game_pk=10)
    assert g.complete                                   # every hit is present, so labels with a PA are exact
    assert (g.outcome(1), g.outcome(3)) == ("hit", "unknown")    # but batter 3's boxscore PA forbids no_pa


def test_a_dropped_out_changes_no_known_label():
    """The accounting: a suffix with no hit cannot change hit/no_hit; a retained PA is real; no_pa needs the box."""
    full = [(1, "strikeout", T, True), (2, "single", T, True), (1, "field_out", T, True)]
    g = O.game_outcomes(feed(10, full[:2], box=box_of(full)), game_pk=10)
    assert (g.outcome(1), g.outcome(2)) == ("no_hit", "hit")


@pytest.mark.parametrize("kw, plays", [
    ({"box": {1: ("away", 0, 1), 2: ("away", 1, 1)}}, [(1, "single", T, True), (2, "strikeout", T, True)]),  # swapped
    ({"line": {"away": 2, "home": 0}}, [(1, "single", T, True)]),
    ({"team": {"away": 0, "home": 0}}, [(1, "single", T, True)]),
    ({"box": {1: ("home", 1, 1)}}, [(1, "single", T, True)]),           # the hit is on the other side's line
    ({"box": {}}, [(1, "strikeout", T, True)]),                         # a PA with no boxscore batting line
])
def test_unreconciled_hits_make_the_game_unknown(kw, plays):
    g = O.game_outcomes(feed(10, plays, **kw), game_pk=10)
    assert not g.complete and g.outcome(1) == "unknown"


def test_an_incomplete_game_makes_every_outcome_unknown():
    for raw in (feed(10, [(1, "single", T)], status="In Progress"), feed(10, [(1, "single", T)], complete=False),
                feed(10, [(1, "strikeout", T), (1, "single", T)], gap=True), feed(10, [], status="Final")):
        g = O.game_outcomes(raw, game_pk=10)
        assert g.outcome(1) == "unknown" and g.outcome(2) == "unknown"


def test_the_resumed_portion_is_excluded_after_the_whole_game_reconciles():
    """Only pre-suspension PAs count. A batter seen only after the resume has no pre-resume PA, but the boxscore
    counts his resumed PA, so he is unknown rather than no_pa."""
    raw = feed(10, [(1, "strikeout", "2027-04-01T23:30:00Z"), (1, "single", "2027-04-03T17:00:00Z"),
                    (2, "single", "2027-04-03T17:05:00Z")], resume="2027-04-03T16:00:00Z")
    g = O.game_outcomes(raw, game_pk=10)
    assert g.complete and g.outcome(1) == "no_hit" and g.outcome(2) == "unknown"


def test_an_intentional_walk_alone_is_unknown_not_no_pa():
    raw = feed(10, [(1, "single", T), (2, "intent_walk", T)])
    g = O.game_outcomes(raw, game_pk=10)
    assert g.complete and g.outcome(2) == "unknown"


def test_completed_early_and_game_over_are_complete_statuses():
    for st in ("Game Over", "Completed Early: Rain", "Final"):
        assert O.game_outcomes(feed(10, [(1, "single", T)], status=st), game_pk=10).outcome(1) == "hit"


def test_a_feed_for_another_game_is_refused():
    with pytest.raises(O.OutcomeError):
        O.game_outcomes(feed(11, [(1, "single", T)]), game_pk=10)


def test_an_unattributable_pa_makes_the_game_unknown():
    for plays in ([(1, "single", T), (None, "strikeout", T)], [(1, "single", T), (True, "strikeout", T)]):
        assert O.game_outcomes(feed(10, plays, box={1: ("away", 1, 1)}), game_pk=10).outcome(1) == "unknown"
    raw = json.loads(feed(10, [(1, "single", T)]))
    del raw["liveData"]["plays"]["allPlays"][0]["about"]["isTopInning"]
    assert O.game_outcomes(json.dumps(raw).encode(), game_pk=10).outcome(1) == "unknown"


def _mutate(path, value):
    raw = json.loads(feed(10, [(1, "single", T)]))
    node = raw
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return json.dumps(raw).encode()


@pytest.mark.parametrize("raw", [
    b"[1]", b"not json", b"[" * 100000, json.dumps({"gamePk": 10, "gameData": [], "liveData": {}}).encode(),
    _mutate(["liveData"], "x"), _mutate(["liveData", "plays"], []), _mutate(["liveData", "plays", "allPlays"], {}),
    _mutate(["liveData", "plays", "allPlays", 0], 5), _mutate(["liveData", "plays", "allPlays", 0, "result"], []),
    _mutate(["liveData", "plays", "allPlays", 0, "result", "eventType"], ["single"]),
    _mutate(["liveData", "plays", "allPlays", 0, "matchup"], None),
    _mutate(["liveData", "plays", "allPlays", 0, "about", "atBatIndex"], False),
    _mutate(["liveData", "boxscore"], None), _mutate(["liveData", "boxscore", "teams", "away", "players"], []),
    _mutate(["liveData", "boxscore", "teams", "away", "players", "ID1", "person", "id"], 2),
    _mutate(["liveData", "boxscore", "teams", "away", "players", "ID1", "stats", "batting", "hits"], "1"),
    _mutate(["liveData", "boxscore", "teams", "home", "players"], {"ID1": {"person": {"id": 1}, "stats": {}}}),
    _mutate(["liveData", "linescore"], {"teams": {"away": {"hits": True}, "home": {"hits": 0}}}),
    _mutate(["gameData", "status"], "Final"), _mutate(["gameData", "datetime"], [1]),
])
def test_a_malformed_feed_is_unknown_and_never_raises_anything_else(raw):
    g = O.game_outcomes(raw, game_pk=10)
    assert not g.complete and g.outcome(1) == "unknown"


def test_a_non_integer_batter_lookup_is_unknown():
    g = O.game_outcomes(feed(10, [(1, "single", T)]), game_pk=10)
    assert g.outcome(True) == "unknown" and g.outcome("1") == "unknown" and g.outcome(1) == "hit"


@pytest.mark.parametrize("value", ["drop", None, 0, "false"])
def test_a_pa_without_a_boolean_batting_side_is_unknown_even_when_totals_would_agree(value):
    """A home batter's missing or non-boolean isTopInning would fall on the home side by luck; it is still refused."""
    raw = json.loads(feed(10, [(1, "single", T, False)]))
    about = raw["liveData"]["plays"]["allPlays"][0]["about"]
    if value == "drop":
        del about["isTopInning"]
    else:
        about["isTopInning"] = value
    g = O.game_outcomes(json.dumps(raw).encode(), game_pk=10)
    assert not g.complete and g.reason == "a PA without a batting side" and g.outcome(1) == "unknown"
