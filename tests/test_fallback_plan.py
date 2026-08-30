"""plan_fallback_action — the deadline-aware replacement for the 12:50-snapshot boolean.

2026-08-30: the in-loop fallback deferred Kwan at 13:20 (cutoff 13:35) on a
`has_pending_future_window` boolean computed at 12:50 that counted the overdue
13:10 check and three post-pitch runs. The planner re-decides after the refresh,
against the live clock, and only defers for a window that (a) can change THIS
decision and (b) can finish before this pick must be delivered.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def T(h, m):
    return datetime(2026, 8, 30, h, m, tzinfo=ET)


def _runs():
    return [{"time_et": T(13, 10), "game_pks": [823662, 823740, 823010]},
            {"time_et": T(14, 10), "game_pks": [823580]},
            {"time_et": T(15, 5), "game_pks": [824959, 823987]},
            {"time_et": T(18, 20), "game_pks": [824636]}]


def _confirmed(*pks):
    return {(pk, side) for pk in pks for side in ("away", "home")}


def _plan(**kw):
    from bts.scheduler import plan_fallback_action
    base = dict(now=T(13, 20), cutoff=T(13, 35), should_post=False, should_post_ungated=False,
                block_reason="gap", contender_game_pk=823987, remaining_runs=_runs(),
                confirmed_sides=_confirmed(823662, 823740, 823010, 823580, 824959),
                budget_min=20, operator_reserve_min=10)
    base.update(kw)
    return plan_fallback_action(**base)


def test_incident_2026_08_30_delivers_kwan_at_13_20():
    plan = _plan()
    assert plan.action == "deliver" and plan.reason == "gap_no_feasible_window"


def test_feasible_contender_window_defers():
    # contender's run at 12:30, now 12:00, cutoff 15:35 → 12:30 + 20 = 12:50 ≤ 15:25
    runs = [{"time_et": T(12, 30), "game_pks": [823987]}]
    plan = _plan(now=T(12, 0), cutoff=T(15, 35), remaining_runs=runs, confirmed_sides=set())
    assert plan.action == "defer"
    assert plan.reason == "gap_contender_window_feasible"
    assert plan.window_time_et == T(12, 30)


def test_overrun_run_uses_now_as_start():
    # scheduled 13:10, now 13:20, budget 20 → finishes 13:40; cutoff 14:00 → deliver_by 13:50 → feasible
    runs = [{"time_et": T(13, 10), "game_pks": [823987]}]
    assert _plan(cutoff=T(14, 0), remaining_runs=runs, confirmed_sides=set()).action == "defer"
    # cutoff 13:45 → deliver_by 13:35 < 13:40 → infeasible → deliver
    assert _plan(cutoff=T(13, 45), remaining_runs=runs, confirmed_sides=set()).action == "deliver"


def test_unrelated_pending_windows_do_not_defer_a_gap_block():
    # contender game 823987 already confirmed; 824636 pending but irrelevant
    plan = _plan(now=T(12, 0), cutoff=T(15, 35),
                 confirmed_sides=_confirmed(823662, 823740, 823010, 823580, 824959, 823987))
    assert plan.action == "deliver" and plan.reason == "gap_no_feasible_window"


def test_gap_block_without_contender_game_falls_back_to_any_feasible_window():
    runs = [{"time_et": T(12, 30), "game_pks": [824636]}]
    plan = _plan(now=T(12, 0), cutoff=T(15, 35), contender_game_pk=None,
                 remaining_runs=runs, confirmed_sides=set())
    assert plan.action == "defer"


def test_primary_projected_defers_even_when_infeasible():
    plan = _plan(block_reason="primary_projected", contender_game_pk=None)
    assert plan.action == "defer" and plan.reason == "primary_projected_pending_window"


def test_primary_projected_with_no_pending_window_delivers():
    plan = _plan(block_reason="primary_projected", contender_game_pk=None,
                 confirmed_sides=_confirmed(823662, 823740, 823010, 823580, 824959, 823987, 824636))
    assert plan.action == "deliver" and plan.reason == "primary_projected_no_window"


def test_gate_only_and_unknown_and_true_deliver():
    assert _plan(should_post_ungated=True, block_reason="dd_projected").reason == "dd_gate_only"
    assert _plan(should_post=None, should_post_ungated=None).reason == "lock_decision_unknown"
    assert _plan(should_post=True).reason == "should_lock_true"


def test_legacy_block_reasons_keep_the_old_rule():
    # status_failure / slot_unavailable: any pending window → defer, else deliver
    assert _plan(block_reason="status_failure", contender_game_pk=None).action == "defer"
    assert _plan(block_reason="slot_unavailable", contender_game_pk=None,
                 confirmed_sides=_confirmed(823662, 823740, 823010, 823580, 824959, 823987, 824636)
                 ).action == "deliver"


def test_effective_budget_uses_last_two_measured_runs():
    from bts.scheduler import effective_cascade_budget_min
    assert effective_cascade_budget_min(12, []) == 12
    # last two: 930 s, 320 s → ceil(15.5) = 16 → +2 = 18
    assert effective_cascade_budget_min(12, [300.0, 930.0, 320.0]) == 18
    # last two: 300 s, 320 s → 6 + 2 = 8 → floor 12
    assert effective_cascade_budget_min(12, [930.0, 300.0, 320.0]) == 12
    assert effective_cascade_budget_min(12, [None, 930.0]) == 18


def test_fallback_min_floor():
    from bts.scheduler import _fallback_min_with_floor
    # cutoff 5 + budget 12 + reserve 10 = 27 ≤ 35 → unchanged
    assert _fallback_min_with_floor(35, 12, 10) == 35
    # morning 25 < 27 → raised
    assert _fallback_min_with_floor(25, 12, 10) == 27
    # measured budget 18 → 33
    assert _fallback_min_with_floor(25, 18, 10) == 33
