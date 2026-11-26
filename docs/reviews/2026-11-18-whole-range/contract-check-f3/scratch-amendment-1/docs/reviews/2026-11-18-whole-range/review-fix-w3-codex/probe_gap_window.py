"""Check gap eligibility when the added status lookup consumes the last feasible second."""

import json
import sys
from threading import current_thread, main_thread

from bts import daily_decision, scheduler

from review_world import DAY, at, events, health, offline


IS_BASE = "base/src" in scheduler.__file__
EXPECT_DELIVERY_GUARDS = IS_BASE or "scratch/repair/src" in scheduler.__file__


def run(correction):
    with offline(at(18, 25)) as (world, mp):
        world.dated(30)
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": at(18, 25), "game_pks": [101]},
            {"time_et": at(18, 42), "game_pks": [102]},
        ])
        real_check = scheduler.run_single_check
        real_plan = scheduler.plan_fallback_action
        real_status = scheduler._delivery_game_statuses
        forecasts = []
        plans = []
        in_check = False
        did_extra_status = False

        def check(*args, **kwargs):
            nonlocal in_check
            in_check = True
            try:
                return real_check(*args, **kwargs)
            finally:
                in_check = False

        def forecast():
            forecasts.append((world.now, in_check))
            frame = world.frame((0.99, 0.98), (False, True))
            if len(forecasts) == 1:
                world.advance(17 * 60 + 59)
                return frame
            if IS_BASE and len(forecasts) == 2 and not in_check:
                world.advance(2)
                return frame
            return None

        def statuses(date):
            nonlocal did_extra_status
            response = real_status(date)
            if current_thread() is not main_thread():
                assert world.now == at(18, 42, 59)
                did_extra_status = True
                world.advance(2)
                if correction:
                    world.dated(30, best=31)
            return response

        def plan(**kwargs):
            decision = real_plan(**kwargs)
            plans.append((kwargs["now"].isoformat(), decision.action, decision.reason,
                          kwargs["budget_min"], kwargs["operator_reserve_min"],
                          kwargs["contender_game_pks"]))
            if IS_BASE and correction and len(plans) == 1:
                world.dated(30, best=31)
            return decision

        world.predictions = forecast
        mp.setattr(scheduler, "run_single_check", check)
        mp.setattr(scheduler, "plan_fallback_action", plan)
        mp.setattr(scheduler, "_delivery_game_statuses", statuses)
        mp.setattr(scheduler, "check_confirmed_lineups", lambda game_pks: {
            pk: {"away", "home"} for pk in game_pks if pk == 101})
        scheduler.run_day(DAY, world.config)
        state = scheduler.load_state(DAY, world.picks)
        record = daily_decision.load_decision(DAY, world.picks)
        facts = {"base": IS_BASE, "correction": correction, "extra_status": did_extra_status,
                 "forecast": [(instant.isoformat(), scheduled) for instant, scheduled in forecasts],
                 "plans": plans, "health": health(world), "pick_dm": events(world, "dm"),
                 "record": record and record["primary"], "clock": world.now.isoformat(),
                 "history": state.fallback_refreshes,
                 "archive_count": len(list((world.picks / DAY).glob("deferred_fallback*.json")))}
        print("GAP_WINDOW", json.dumps(facts), flush=True)
        assert plans[0][3:] == (12.0, 10.0, (102,))
        if EXPECT_DELIVERY_GUARDS:
            assert plans[-1][0] == at(18, 43, 1).isoformat() and plans[-1][2] == "gap_no_feasible_window"
            if correction:
                assert not facts["pick_dm"] and record is None
                assert any("[contest_state]" in alert for alert in facts["health"])
                assert any("[missed_pick]" in alert for alert in facts["health"])
            else:
                assert len(facts["pick_dm"]) == 1 and record["scoreable"]
        else:
            assert did_extra_status and plans[-1][0] == at(18, 42, 59).isoformat()
            assert plans[-1][2] == "gap_contender_window_feasible"
            assert not facts["pick_dm"] and not facts["health"] and record is None
            assert facts["archive_count"] == 1


if __name__ == "__main__":
    run(False)
    run(True)
