"""C1 rank 4a, T1-T3: the fixed calendar, the eligible slate population and the known outcomes
(registration docs/sota_audit/2026-10-04-prereg-c1-calibration.md §§2-3). Synthetic fixtures only."""
import hashlib
import json
from datetime import date

import pytest

from scripts.audit.c1_r4a import calendar as C
from scripts.audit.c1_r4a import outcomes as O
from scripts.audit.c1_r4a import population as P


# ---- T1: the calendar ---------------------------------------------------------------------------------------------

def cal_bytes(days):
    return json.dumps({"schema": "c1_r4a_calendar_v1", "season": 2027, "dates": days}).encode()


def test_numbering_follows_the_frozen_calendar_not_the_data():
    days = [f"2027-03-{d:02d}" for d in range(25, 32)] + [f"2027-04-{d:02d}" for d in range(1, 31)] + \
           [f"2027-05-{d:02d}" for d in range(1, 32)] + [f"2027-06-{d:02d}" for d in range(1, 31)]
    raw = cal_bytes(days)
    cal = C.load(raw, hashlib.sha256(raw).hexdigest())
    assert cal.number(date(2027, 3, 25)) == 1 and cal.number(date(2027, 4, 23)) == 30
    assert cal.window(1) == "fit" and cal.window(30) == "fit" and cal.window(31) == "test" and cal.window(90) == "test"
    assert cal.window(91) is None and cal.number(date(2027, 3, 24)) is None
    assert cal.date_of(40) == date(2027, 5, 3) and cal.date_of(90) == date(2027, 6, 22)


def test_a_date_without_a_slate_still_counts():
    """Missing slates never shift the windows: numbering is the calendar's, whatever data exists."""
    days = [f"2027-04-{d:02d}" for d in range(1, 31)] + [f"2027-05-{d:02d}" for d in range(1, 32)] + \
           [f"2027-06-{d:02d}" for d in range(1, 31)]
    raw = cal_bytes(days)
    cal = C.load(raw, hashlib.sha256(raw).hexdigest())
    assert [cal.number(date(2027, 4, d)) for d in (1, 2, 30)] == [1, 2, 30]
    assert cal.fit_dates() == [date(2027, 4, d) for d in range(1, 31)]
    assert len(cal.test_dates()) == 60 and cal.test_dates()[0] == date(2027, 5, 1)


@pytest.mark.parametrize("days, why", [
    (["2027-04-02", "2027-04-01"], "sorted"),
    (["2027-04-01", "2027-04-01"], "duplicate"),
    (["2026-04-01"], "season"),
    (["2027-13-01"], "date"),
    ([], "empty"),
])
def test_a_malformed_calendar_refuses(days, why):
    raw = cal_bytes(days)
    with pytest.raises(C.CalendarError):
        C.load(raw, hashlib.sha256(raw).hexdigest())


def test_the_calendar_must_match_its_pin():
    raw = cal_bytes(["2027-04-01"])
    with pytest.raises(C.CalendarError, match="pin"):
        C.load(raw, "0" * 64)


# ---- T2: the population -------------------------------------------------------------------------------------------

def slate(rows, written_at="2027-04-01T16:00:00+00:00", schema="bts_slate_v2", d="2027-04-01"):
    return json.dumps({"schema_version": schema, "date": d, "tier": "local", "written_at": written_at,
                       "n_rows": len(rows), "rows": rows}).encode()


def row(batter, game, p, *, game_time="2027-04-01T23:10:00Z", status="Scheduled", projected=False):
    return {"batter_id": batter, "game_pk": game, "p_game_hit": p, "game_time": game_time, "status": status,
            "projected": projected}


def test_eligibility_and_every_exclusion_is_counted():
    rows = [row(1, 10, 0.80), row(2, 10, 0.70, status="Pre-Game"), row(3, 11, 0.60, status="Warmup"),
            row(4, 12, 0.90, game_time="2027-04-01T16:04:00Z"),          # write is not before start - 5 min
            row(5, 13, 0.75, status="Postponed"), row(6, 14, 0.75, status="In Progress"),
            row(7, 15, 0.75, status=None), row(8, 16, 0.75, game_time=None),
            row(9, 17, True), row(10, 18, float("nan")), row(11, 19, 1.5), row(12, 20, "0.7"),
            row(13, 21, 0.5), row(13, 21, 0.6)]                          # a duplicated identity
    raw = slate(rows)
    pop = P.population(raw, d=date(2027, 4, 1))
    assert [r["batter_id"] for r in pop.eligible] == [1, 2, 3]
    assert pop.excluded == {"write_not_before_cutoff": 1, "status_not_pregame": 2, "missing_status": 1,
                            "missing_start": 1, "invalid_probability": 4, "duplicate_identity": 2}
    assert pop.slate_sha256 == hashlib.sha256(raw).hexdigest() and pop.rank1["batter_id"] == 1


def test_rank1_ties_keep_stored_order_and_never_look_at_outcomes():
    pop = P.population(slate([row(5, 10, 0.70), row(6, 11, 0.80), row(7, 12, 0.80)]), d=date(2027, 4, 1))
    assert pop.rank1["batter_id"] == 6


def test_a_wrong_schema_date_or_unparseable_slate_refuses():
    with pytest.raises(P.PopulationError):
        P.population(slate([row(1, 10, 0.8)], schema="bts_slate_v1"), d=date(2027, 4, 1))
    with pytest.raises(P.PopulationError):
        P.population(slate([row(1, 10, 0.8)], d="2027-04-02"), d=date(2027, 4, 1))
    with pytest.raises(P.PopulationError):
        P.population(b"{not json", d=date(2027, 4, 1))
    with pytest.raises(P.PopulationError):
        P.population(slate([row(1, 10, 0.8)], written_at="2027-04-01T16:00:00"), d=date(2027, 4, 1))   # naive


def test_projected_and_confirmed_states_are_counted():
    pop = P.population(slate([row(1, 10, 0.8, projected=True), row(2, 11, 0.7)]), d=date(2027, 4, 1))
    assert pop.projected == {"projected": 1, "confirmed": 1}


# ---- T3: outcomes -------------------------------------------------------------------------------------------------

def feed(pk, plays, *, status="Final", resume=None, complete=True, gap=False):
    ps = []
    for i, (batter, event, start) in enumerate(plays):
        ps.append({"about": {"atBatIndex": i + (1 if gap and i else 0), "isComplete": True, "startTime": start},
                   "matchup": {"batter": {"id": batter}}, "result": {"eventType": event}})
    if ps and not complete:
        ps[-1]["about"]["isComplete"] = False
    dt = {"officialDate": "2027-04-01"}
    if resume:
        dt["resumeDateTime"] = resume
    return json.dumps({"gamePk": pk, "gameData": {"game": {"pk": pk}, "status": {"detailedState": status},
                                                  "datetime": dt},
                       "liveData": {"plays": {"allPlays": ps}}}).encode()


T = "2027-04-01T23:30:00Z"


def test_hit_no_hit_and_no_pa_in_a_complete_game():
    raw = feed(10, [(1, "single", T), (2, "strikeout", T), (2, "field_out", T)])
    g = O.game_outcomes(raw, game_pk=10)
    assert (g.outcome(1), g.outcome(2), g.outcome(3)) == ("hit", "no_hit", "no_pa")


def test_an_incomplete_game_makes_every_outcome_unknown():
    for raw in (feed(10, [(1, "single", T)], status="In Progress"), feed(10, [(1, "single", T)], complete=False),
                feed(10, [(1, "strikeout", T), (1, "single", T)], gap=True), feed(10, [], status="Final")):
        g = O.game_outcomes(raw, game_pk=10)
        assert g.outcome(1) == "unknown" and g.outcome(2) == "unknown"


def test_the_resumed_portion_is_excluded():
    """A suspended game resumed later: only pre-suspension PAs count. A batter seen only after the resume has no PA."""
    raw = feed(10, [(1, "strikeout", "2027-04-01T23:30:00Z"), (1, "single", "2027-04-03T17:00:00Z"),
                    (2, "single", "2027-04-03T17:05:00Z")], resume="2027-04-03T16:00:00Z")
    g = O.game_outcomes(raw, game_pk=10)
    assert g.outcome(1) == "no_hit" and g.outcome(2) == "no_pa"


def test_completed_early_and_game_over_are_complete_statuses():
    for st in ("Game Over", "Completed Early: Rain", "Final"):
        assert O.game_outcomes(feed(10, [(1, "single", T)], status=st), game_pk=10).outcome(1) == "hit"


def test_a_feed_for_another_game_is_refused():
    with pytest.raises(O.OutcomeError):
        O.game_outcomes(feed(11, [(1, "single", T)]), game_pk=10)


def test_an_unattributable_pa_makes_the_game_unknown():
    raw = feed(10, [(1, "single", T), (None, "strikeout", T)])
    assert O.game_outcomes(raw, game_pk=10).outcome(1) == "unknown"
