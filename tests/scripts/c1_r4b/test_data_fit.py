"""T1–T3: calendars, profile validation/pairing/coverage, classifier and availability-conditioned fitting."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts.audit.c1_r4b import data as D
from scripts.audit.c1_r4b import fit as F
from scripts.audit.c1_r4b import solvers as S


def sched(*rows):
    out = {}
    for d, pk, st in rows:
        out.setdefault(d, []).append({"gamePk": pk, "officialDate": d, "status": {"detailedState": st}})
    return {"dates": [{"date": d, "games": g} for d, g in sorted(out.items())]}


CAL = D.calendar_from_schedule(sched(("2023-09-25", 1, "Final"), ("2023-09-26", 2, "Postponed"),
                                     ("2023-09-28", 3, "Final"), ("2023-09-28", 4, "Final")), 2023)


# ---------- T1 calendars ----------
def test_calendar_counts_calendar_days_not_rows():
    assert CAL.opening == date(2023, 9, 25) and CAL.final == date(2023, 9, 28)
    assert CAL.exclusive_end == date(2023, 9, 29) and CAL.horizon == 4
    assert CAL.no_opportunity == frozenset({date(2023, 9, 26), date(2023, 9, 27)})
    # the Codex r1 F4 example: rows on 9/25 and 9/28 count down [4, 1] by calendar, not [2, 1] by row
    assert [(d, opp, left) for d, opp, left in CAL.days()] == [
        (date(2023, 9, 25), True, 4), (date(2023, 9, 26), False, 3),
        (date(2023, 9, 27), False, 2), (date(2023, 9, 28), True, 1)]


# ---------- T2 profiles ----------
def prof(rows):
    cols = ["date", "rank", "batter_id", "game_pk", "p_game_hit", "actual_hit"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


GOOD = prof([("2023-09-25", 1, 10, 1, 0.80, 1), ("2023-09-25", 2, 11, 1, 0.78, 0), ("2023-09-25", 3, 12, 9, 0.77, 1),
             ("2023-09-28", 1, 20, 3, 0.70, 0), ("2023-09-28", 2, 21, 3, 0.69, 1)])


def test_pairing_takes_the_first_lower_ranked_different_game_and_classifies_coverage():
    sd = D.season_days(D.validate_profile(GOOD), CAL)
    assert list(sd["opp"]) == [True, False, False, True] and list(sd["known"]) == [True, False, False, True]
    assert sd["p1"][0] == 0.80 and sd["hit1"][0] and sd["partner"][0] and sd["hit2"][0] and sd["partner_rank"][0] == 3
    assert not sd["partner"][3] and sd["partner_rank"][3] == 0        # only a same-game rank 2: partnerless
    assert sd["unknown_dates"] == []


def test_an_uncovered_game_day_is_unknown_coverage():
    sd = D.season_days(D.validate_profile(GOOD[GOOD["date"] != date(2023, 9, 28)]), CAL)
    assert list(sd["known"]) == [True, False, False, False] and sd["unknown_dates"] == [date(2023, 9, 28)]


@pytest.mark.parametrize("bad", [
    GOOD.assign(rank=[1, 1, 3, 1, 2]),                                          # duplicate rank
    GOOD[GOOD["rank"] != 1],                                                    # no rank 1 on a date
    GOOD.assign(p_game_hit=[0.8, 1.2, 0.7, 0.7, 0.6]),                          # p out of range
    GOOD.assign(actual_hit=[1, 2, 1, 0, 1]),                                    # label not 0/1
    pd.concat([GOOD, GOOD.iloc[[0]].assign(rank=4)]),                           # duplicate (date, batter, game)
])
def test_malformed_profiles_are_refused(bad):
    with pytest.raises(D.ProfileError):
        D.season_days(D.validate_profile(bad), CAL)


def test_a_profile_row_on_a_no_opportunity_date_is_refused():
    bad = pd.concat([GOOD, prof([("2023-09-26", 1, 30, 2, 0.7, 1)])])
    with pytest.raises(D.ProfileError):
        D.season_days(D.validate_profile(bad), CAL)


# ---------- T3 classifier and fitting ----------
def test_classifier_puts_equality_in_the_upper_bin():
    cuts = np.array([0.7, 0.75, 0.8, 0.85])
    assert list(F.classify(np.array([0.69, 0.7, 0.749, 0.85, 0.9]), cuts)) == [0, 1, 1, 4, 4]


def days(rows, late):
    """Synthetic season-days: rows = (p1, hit1, partner, hit2)."""
    n = len(rows)
    return {"opp": np.ones(n, bool), "known": np.ones(n, bool), "d_raw": np.full(n, 10 if late else 100),
            "p1": np.array([r[0] for r in rows]), "hit1": np.array([r[1] for r in rows], bool),
            "partner": np.array([r[2] for r in rows], bool), "hit2": np.array([r[3] for r in rows], bool)}


def test_rates_condition_primary_and_joint_on_the_same_availability_stratum():
    """Codex r2 R2-1: one partner-eligible joint hit and one partnerless primary miss in the same bin give
    coherent types (1.0/1.0 with a partner, 0.0/0 without), never p_hit 0.5 with p_both 1.0."""
    cuts = np.array([0.9, 0.91, 0.92, 0.93])
    early = days([(0.5, True, True, True), (0.5, False, False, False)] + [(0.95, True, True, True)] * 2
                 + [(0.905, True, True, False), (0.915, False, True, False), (0.925, True, False, False)], late=False)
    late = days([(0.5, True, True, True), (0.905, True, False, False), (0.915, True, True, True),
                 (0.925, False, True, False), (0.95, True, True, True)], late=True)
    env, stats = F.fit_environment([early, late], cuts, late_days=30)
    S.validate(env)
    b0 = {t.partner: t for t in env.early if t.q == 0}
    assert (b0[True].p_hit, b0[True].p_both, b0[False].p_hit, b0[False].p_both) == (1.0, 1.0, 0.0, 0.0)
    assert sum(t.freq for t in env.early) == pytest.approx(1.0) and sum(t.freq for t in env.late) == pytest.approx(1.0)


def test_an_empty_phase_bin_stops_the_fit():
    cuts = np.array([0.9, 0.91, 0.92, 0.93])
    early = days([(0.5, True, True, True)] * 5, late=False)        # only bin 0 present
    with pytest.raises(ValueError):
        F.fit_environment([early], cuts, late_days=30)


def test_r_bar_is_the_partner_eligible_leg_hit_rate():
    d = days([(0.5, True, True, True), (0.5, True, True, False), (0.5, True, False, True)], late=False)
    assert F.r_bar([d]) == pytest.approx(0.5)
