"""W2.1 item 2 / W2.2 streak rules (design B-E3): only recoverable runs; carried-in streak excluded from the window;
max(streak_after) never used; null streaks, missing rounds, mixed DD, Pass and saver never fabricated."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from bts.leaderboard.models import PickRow
from scripts.audit.field_products import picks as P
from scripts.audit.field_products import streaks as S

D0 = date(2026, 4, 25)
CAP = datetime(2026, 7, 5, 12)


def _rounds(spec, user_id=7):
    """spec: list of (day offset from D0, labels tuple, streak_after) -> resolved rounds via the data contract."""
    rows = []
    for k, (off, labels, streak) in enumerate(spec):
        for pn, lab in enumerate(labels, start=1):
            rows.append(PickRow(captured_at=CAP, round_id=100 + k, pick_date=D0 + timedelta(days=off), pick_number=pn,
                                unit_id=10 + pn, bts_player_id=20 + pn, result=lab, at_bats=4, hits=1,
                                streak_after=streak).model_dump())
    obs = P.with_provenance(pd.DataFrame(rows), user_id=user_id, source="final_grab", file="x.parquet", file_rank=0)
    return P.resolve(obs).rounds


H, HH, M, MIX, V, PEND = ("hit",), ("hit", "hit"), ("not_hit",), ("hit", "not_hit"), ("void",), ("",)
W0, W1 = date(2026, 5, 1), date(2026, 7, 3)


def _win(spec):
    return S.window_summary(_rounds(spec), W0, W1)


def test_streak_carried_into_the_window_is_excluded_and_max_streak_after_is_not_the_window_run():
    """r1 B3: enters May 1 with 20, hits both DD legs -> streak_after 22, window contribution 2."""
    w = _win([(4, H, 19), (5, H, 20), (6, HH, 22)])
    assert w["status"] == "exact" and w["longest_exact"] == 2 and w["lower_bound"] == 2


def test_observed_miss_then_entry_evidence_resets_the_window_run():
    w = _win([(6, H, 1), (7, H, 2), (8, M, 0), (9, H, 1), (10, H, 2), (11, H, 3)])
    assert w["status"] == "exact" and w["longest_exact"] == 3
    assert w["links"] == {"continuation": 3, "new_run_after_miss": 1}


def test_saver_consistent_continuation_through_a_miss_is_evidenced_by_reported_values():
    w = _win([(6, H, 11), (7, M, 11), (8, H, 12)])
    assert w["status"] == "exact" and w["longest_exact"] == 2 and w["links"] == {"continuation_through_miss": 1}


def test_mixed_dd_result_is_a_miss_round_not_a_fabricated_hit():
    w = _win([(6, H, 1), (7, H, 2), (8, H, 3), (9, MIX, 0), (10, H, 1)])
    assert w["status"] == "exact" and w["longest_exact"] == 3


def test_pass_absorbed_between_evidenced_values_keeps_the_run():
    w = _win([(6, H, 1), (7, V, 1), (8, H, 2)])
    assert w["status"] == "exact" and w["longest_exact"] == 2 and w["links"] == {"continuation_absorbing": 1}


def test_pass_followed_by_an_unexplained_reset_is_unavailable_not_a_reset():
    w = _win([(6, H, 1), (7, H, 2), (8, V, 2), (9, H, 1)])
    assert w["status"] == "lower_bound_only" and w["longest_exact"] is None and w["lower_bound"] == 2
    assert "unabsorbed_ambiguous_round" in w["reasons"]


def test_null_coerced_streak_is_not_a_fabricated_reset_or_continuation():
    w = _win([(6, H, 1), (7, H, 0), (8, H, 3)])
    assert w["status"] == "lower_bound_only" and w["longest_exact"] is None and w["lower_bound"] == 1
    assert w["kinds"] == {"H": 2, "A": 1}


def test_reset_without_an_observed_miss_means_missing_rounds():
    w = _win([(6, H, 1), (7, H, 2), (8, H, 3), (9, H, 4), (12, H, 1)])
    assert w["status"] == "lower_bound_only" and w["lower_bound"] == 4 and "new_run_unexplained" in w["reasons"]


def test_unresolvable_jump_after_a_miss_is_unavailable():
    w = _win([(6, H, 3), (7, M, 0), (8, H, 7)])
    assert w["status"] == "lower_bound_only" and "unresolved" in w["reasons"]


def test_trailing_pending_round_is_unavailable_and_counted():
    w = _win([(6, H, 1), (7, H, 2), (8, PEND, 0)])
    assert w["status"] == "lower_bound_only" and w["lower_bound"] == 2 and w["incomplete_rounds"] == 1


def test_all_miss_window_is_an_exact_zero_and_no_rounds_is_unavailable():
    assert _win([(6, M, 0), (7, M, 0)])["longest_exact"] == 0
    assert S.window_summary(_rounds([(70, H, 1)]), W0, W1)["status"] == "no_window_rounds"


def test_recoverable_attaining_run_reports_start_and_end_dates():
    r = S.attaining_runs(_rounds([(0, H, 1), (1, H, 2), (3, HH, 4), (4, M, 0)]), best=4)
    assert r["status"] == "recoverable" and len(r["runs"]) == 1
    run = r["runs"][0]
    assert run["start_date"] == "2026-04-25" and run["end_date"] == "2026-04-28" and run["n_rounds"] == 3
    assert run["n_dd_rounds"] == 1 and run["recoverable"]


def test_unevidenced_start_reports_dates_unavailable_with_a_lower_bound():
    r = S.attaining_runs(_rounds([(0, H, 5), (1, H, 6)]), best=6)
    run = r["runs"][0]
    assert not run["recoverable"] and run["start_date"] is None and run["end_date"] is None
    assert run["observed_segment_lower_bound"] == 2 and run["reported_attainment_date"] == "2026-04-26"
    assert r["status"] == "dates_unavailable"


def test_every_recoverable_run_attaining_the_best_is_reported():
    r = S.attaining_runs(_rounds([(0, H, 1), (1, H, 2), (2, M, 0), (3, H, 1), (4, H, 2)]), best=2)
    assert [x["end_date"] for x in r["runs"]] == ["2026-04-26", "2026-04-29"] and r["status"] == "recoverable"


def test_saver_run_is_recoverable_through_the_miss():
    spec = [(k, H, k + 1) for k in range(11)] + [(11, M, 11), (12, H, 12)]
    r = S.attaining_runs(_rounds(spec), best=12)
    assert r["status"] == "recoverable" and r["runs"][0]["start_date"] == "2026-04-25"
    assert r["runs"][0]["links"] == {"continuation": 10, "continuation_through_miss": 1}


def test_no_round_reports_the_best_gives_only_a_segment_lower_bound():
    r = S.attaining_runs(_rounds([(0, H, 1), (1, H, 2), (2, M, 0), (3, H, 1)]), best=10)
    assert r["status"] == "no_settled_round_reports_best" and r["runs"] == []
    assert r["longest_evidenced_segment"] == 2


def test_missing_best_is_unavailable():
    assert S.attaining_runs(_rounds([(0, H, 1)]), best=None)["status"] == "best_unavailable"
