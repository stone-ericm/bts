"""W2.1 item 2 / W2.2 streak rules (design B-E3; code review r1 F3, r2 R2-1): no exact window maximum or run dates
without an entered-round completeness witness (none is stored); observed-segment lower bounds only, from qualified
COMPLETE rounds only, never merged through a miss, an ambiguous round or inconsistent reported values; carried-in
streak excluded; max(streak_after) never used. Informative tests give each round a synthetic validated slot-set
witness (as a verified final-grab raw response would); without one no round contributes."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from bts.leaderboard.models import PickRow
from scripts.audit.field_products import picks as P
from scripts.audit.field_products import streaks as S

D0 = date(2026, 4, 25)
CAP = datetime(2026, 7, 5, 12)


def _witness(obs):
    """A synthetic validated slot-set witness for each round's (single) batch."""
    return {(int(u), int(r)): {"file": g["file"].iloc[-1], "captured_at": g["captured_at"].iloc[-1],
                               "slot_numbers": frozenset(int(x) for x in g["pick_number"])}
            for (u, r), g in obs.groupby(["user_id", "round_id"])}


def _resolve(rows, witness=True, user_id=7):
    obs = P.with_provenance(pd.DataFrame(rows), user_id=user_id, source="daily", file="x.parquet", file_rank=0)
    return P.resolve(obs, witness=_witness(obs) if witness else None).rounds


def _rounds(spec, user_id=7, witness=True):
    """spec: list of (day offset from D0, labels tuple, streak_after) -> resolved rounds (witnessed by default)."""
    rows = []
    for k, (off, labels, streak) in enumerate(spec):
        for pn, lab in enumerate(labels, start=1):
            rows.append(PickRow(captured_at=CAP, round_id=100 + k, pick_date=D0 + timedelta(days=off), pick_number=pn,
                                unit_id=10 + pn, bts_player_id=20 + pn, result=lab, at_bats=4, hits=1,
                                streak_after=streak).model_dump())
    return _resolve(rows, witness=witness, user_id=user_id)


H, HH, M, MIX, V, PEND = ("hit",), ("hit", "hit"), ("not_hit",), ("hit", "not_hit"), ("void",), ("",)
W0, W1 = date(2026, 5, 1), date(2026, 7, 3)
NO_WITNESS = "no_entered_round_completeness_witness"


def _win(spec):
    return S.window_summary(_rounds(spec), W0, W1)


def test_review_probe_miss_reported_zero_is_never_bridged_into_a_saver_continuation():
    """F3 probe 1: May 1 hit/1, May 2 miss/0, May 5 hit/2. Endpoints add up, but the May 2 row reports a reset and a
    hidden May 4 hit explains May 5; the segments stay split and nothing is exact or recoverable."""
    spec = [(6, H, 1), (7, M, 0), (10, H, 2)]
    w = _win(spec)
    assert w["status"] == "lower_bound_only" and w["longest_exact"] is None and w["lower_bound"] == 1
    assert NO_WITNESS in w["reasons"] and w["splits"]["after_miss"] == 1
    r = S.attaining_runs(_rounds(spec), best=2)
    assert r["status"] == "dates_unavailable" and r["runs"][0]["recoverable"] is False
    assert r["runs"][0]["start_date"] is None and r["runs"][0]["observed_segment_lower_bound"] == 1


def test_review_probe_a_single_retained_round_gives_only_its_own_contribution():
    """F3 probe 2: only June 1 hit/20. A run begun May 13 and one carried in fit the same row: no exact maximum."""
    w = _win([(37, H, 20)])
    assert w["status"] == "lower_bound_only" and w["longest_exact"] is None and w["lower_bound"] == 1


def test_streak_carried_into_the_window_is_excluded_and_max_streak_after_is_never_the_window_run():
    w = _win([(4, H, 19), (5, H, 20), (6, HH, 22)])
    assert w["lower_bound"] == 2 and w["longest_exact"] is None


def test_adjacent_rounds_with_consistent_reported_values_form_one_observed_segment():
    w = _win([(6, H, 1), (7, H, 2), (9, HH, 4)])
    assert w["lower_bound"] == 4 and w["segments"] == 1 and w["status"] == "lower_bound_only"


def test_inconsistent_reported_values_split_the_segment():
    w = _win([(6, H, 1), (7, H, 3), (8, H, 4)])
    assert w["lower_bound"] == 2 and w["splits"]["inconsistent_values"] == 1


def test_pass_round_is_not_absorbed_even_when_the_endpoints_add_up():
    w = _win([(6, H, 1), (7, V, 1), (8, H, 2)])
    assert w["lower_bound"] == 1 and w["splits"]["after_ambiguous"] == 1


def test_saver_consistent_values_through_a_miss_are_still_split():
    w = _win([(6, H, 11), (7, M, 11), (8, H, 12)])
    assert w["lower_bound"] == 1 and w["splits"]["after_miss"] == 1


def test_null_coerced_streak_round_is_ambiguous_not_a_reset():
    w = _win([(6, H, 1), (7, H, 0), (8, H, 3)])
    assert w["lower_bound"] == 1 and w["kinds"] == {"H": 2, "A": 1}


def test_mixed_dd_is_a_miss_round():
    w = _win([(6, H, 1), (7, H, 2), (8, MIX, 0), (9, H, 1)])
    assert w["lower_bound"] == 2 and w["kinds"] == {"H": 3, "M": 1}


def test_all_miss_window_has_a_zero_lower_bound_and_no_rounds_is_unavailable():
    w = _win([(6, M, 0), (7, PEND, 0)])
    assert w["lower_bound"] == 0 and w["longest_exact"] is None and w["incomplete_rounds"] == 1
    assert S.window_summary(_rounds([(70, H, 1)]), W0, W1)["status"] == "no_window_rounds"


def test_a_conflicting_slot_makes_the_round_ambiguous():
    rows = [PickRow(captured_at=CAP, round_id=100, pick_date=date(2026, 5, 2), pick_number=1, unit_id=1,
                    bts_player_id=p, result="hit", at_bats=4, hits=1, streak_after=1).model_dump() for p in (20, 21)]
    rows += [PickRow(captured_at=CAP, round_id=101, pick_date=date(2026, 5, 3), pick_number=pn, unit_id=u,
                     bts_player_id=u, result="hit", at_bats=4, hits=1, streak_after=2).model_dump()
             for pn, u in ((1, 30), (2, 0))]             # a DD whose second leg's identity is null-coerced
    rows.append(PickRow(captured_at=CAP, round_id=102, pick_date=date(2026, 5, 4), pick_number=1, unit_id=40,
                        bts_player_id=40, result="not_hit", at_bats=4, hits=0, streak_after=0).model_dump())
    w = S.window_summary(_resolve(rows), W0, W1)
    assert w["kinds"] == {"A": 2, "M": 1} and w["lower_bound"] == 0   # conflict; unresolved (null-coerced) identity


def test_attaining_run_reports_attainment_and_a_lower_bound_but_never_dates():
    r = S.attaining_runs(_rounds([(0, H, 1), (1, H, 2), (3, HH, 4), (4, M, 0)]), best=4)
    assert r["status"] == "dates_unavailable" and len(r["runs"]) == 1
    run = r["runs"][0]
    assert run["start_date"] is None and run["end_date"] is None and run["recoverable"] is False
    assert run["reported_attainment_date"] == "2026-04-28" and run["observed_segment_lower_bound"] == 4
    assert run["n_rounds"] == 3 and run["n_dd_rounds"] == 1 and NO_WITNESS in r["reason"]


def test_every_round_reporting_the_best_is_listed():
    r = S.attaining_runs(_rounds([(0, H, 1), (1, H, 2), (2, M, 0), (3, H, 1), (4, H, 2)]), best=2)
    assert [x["reported_attainment_date"] for x in r["runs"]] == ["2026-04-26", "2026-04-29"]


def test_no_round_reports_the_best_gives_only_the_longest_observed_segment():
    r = S.attaining_runs(_rounds([(0, H, 1), (1, H, 2), (2, M, 0), (3, H, 1)]), best=10)
    assert r["status"] == "no_settled_round_reports_best" and r["runs"] == []
    assert r["longest_observed_segment"] == 2


def test_missing_or_zero_best():
    assert S.attaining_runs(_rounds([(0, H, 1)]), best=None)["status"] == "best_unavailable"
    assert S.attaining_runs(_rounds([(0, M, 0)]), best=0)["status"] == "best_is_zero"


def _one_slot(pick_number, streak=10):
    return [PickRow(captured_at=CAP, round_id=100, pick_date=date(2026, 5, 1), pick_number=pick_number, unit_id=11,
                    bts_player_id=21, result="hit", at_bats=4, hits=1, streak_after=streak).model_dump()]


def test_review_probe_primary_only_hit_without_a_witness_contributes_nothing():
    """R2-1: a retained May 1 hit at reported streak 10 is compatible with a mixed DD whose unretained leg missed
    while a saver kept a carried-in 10 — no winning-round increment at all. No positive contribution."""
    rounds = _resolve(_one_slot(1), witness=False)
    assert not rounds.iloc[0]["complete"]
    w = S.window_summary(rounds, W0, W1)
    assert w["status"] == "no_complete_rounds" and w["lower_bound"] is None and w["longest_exact"] is None
    assert w["segments"] == 0 and w["splits"] == {} and w["kinds"] == {"A": 1} and w["ambiguous_rounds"] == 1
    r = S.attaining_runs(rounds, best=10)
    assert r["runs"] == [] and r["longest_observed_segment"] is None


def test_review_probe_secondary_only_hit_contributes_nothing_even_with_a_slot_set_witness():
    rounds = _resolve(_one_slot(2), witness=True)
    assert rounds.iloc[0]["incomplete_reason"] == "missing_primary_slot"
    w = S.window_summary(rounds, W0, W1)
    assert w["status"] == "no_complete_rounds" and w["lower_bound"] is None
    assert S.attaining_runs(rounds, best=10)["runs"] == []


def test_unwitnessed_daily_rounds_give_no_positive_bound_or_dd_count():
    rounds = _rounds([(6, H, 1), (7, H, 2), (8, HH, 4)], witness=False)
    w = S.window_summary(rounds, W0, W1)
    assert w["status"] == "no_complete_rounds" and w["lower_bound"] is None and w["incomplete_rounds"] == 3
    r = S.attaining_runs(rounds, best=4)
    assert r["status"] == "no_settled_round_reports_best" and r["runs"] == []
