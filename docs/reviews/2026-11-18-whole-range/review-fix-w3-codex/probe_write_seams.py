"""Locate the known non-reuse gate failures at the real candidate write boundary."""

import json
import sys

from bts import contest_state, picks, scheduler

from review_world import DAY, at, events, game, health, offline


IS_BASE = "base/src" in scheduler.__file__
IS_PATCHED = hasattr(picks, "capture_pick_write")


def run(case, seam):
    start = at(23, 58) if case == "midnight" else at(18, 25)
    with offline(start) as (world, mp):
        if case == "midnight":
            world.schedule[DAY] = [game(101, at(0, 15, day="2027-03-25")),
                                   game(102, at(0, 20, day="2027-03-25"))]
            world.automatic(0)
        world.dated(0)
        mode = case if case in {"public", "private"} else "dm"
        world.config["scheduler"]["pick_delivery"] = mode
        if case == "optional":
            world.config["scheduler"]["contest_state_required"] = False
            world.config["health_checks"]["contest_state_expected"] = False
        if case == "private":
            world.config["scheduler"]["entry_intent"] = "research"
        world.predictions = lambda: world.frame((0.99, 0.98))
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": start, "game_pks": [101, 102]},
        ])
        changed = False
        entering_delivery = False
        trace = []
        save = picks.save_pick
        append = picks.append_lineup_evolution
        deliver = scheduler._deliver_and_lock_pick
        read = contest_state.load_decision_streak_state

        def change(daily):
            nonlocal changed
            if changed:
                return
            trace.append(["fixture change", entering_delivery, daily.delivery_attempted, world.now.isoformat()])
            changed = True
            if case == "midnight":
                world.now = at(0, 1, day="2027-03-25")
            else:
                world.dated(30)

        def public_wrapper(daily, directory):
            path = save(daily, directory)
            change(daily)
            return path

        def actual_write_boundary(daily, directory):
            append(daily, directory)
            change(daily)

        def delivery(daily, config, directory, state, date, label, **kwargs):
            nonlocal entering_delivery
            assert not kwargs.get("reused_check_cascade")
            entering_delivery = True
            trace.append(["entered delivery", label, world.now.isoformat()])
            try:
                result = deliver(daily, config, directory, state, date, label, **kwargs)
                trace.append(["delivery returned", result])
                return result
            finally:
                entering_delivery = False

        def reading(*args, **kwargs):
            result = read(*args, **kwargs)
            if entering_delivery:
                trace.append(["final read", result.streak, world.now.isoformat()])
            return result

        if seam == "public-save":
            mp.setattr(picks, "save_pick", public_wrapper)
        else:
            mp.setattr(picks, "append_lineup_evolution", actual_write_boundary)
        mp.setattr(scheduler, "_deliver_and_lock_pick", delivery)
        mp.setattr(contest_state, "load_decision_streak_state", reading)
        if case == "midnight":
            result = scheduler.run_single_check(DAY, [101, 102], set(), world.config, .03)
            state = scheduler.SchedulerState(DAY, world.now.isoformat(), [], [], [], False, None, None, None)
            delivery(result["pick_result"].daily, world.config, world.picks, state, DAY, "lineup",
                     selection=result["selection"])
        else:
            scheduler.run_day(DAY, world.config)
        print("WRITE_SEAM", json.dumps({"base": IS_BASE, "patched": IS_PATCHED,
                                        "case": case, "seam": seam, "trace": trace,
                                        "dm": events(world, "dm"), "public": events(world, "post"),
                                        "health": health(world)}), flush=True)
        early = IS_BASE or IS_PATCHED or seam == "actual-write"
        assert changed and (trace[0][0] == "fixture change") is early
        returned = [step[1] for step in trace if step[0] == "delivery returned"]
        if case in {"optional", "private"}:
            assert returned == [True]
            if case == "optional":
                assert [step[1] for step in trace if step[0] == "final read"] == [30 if early else 0]
        else:
            assert returned[0] is not early
            assert any("[contest_state]" in msg for msg in health(world)) is early


if __name__ == "__main__":
    for path in ["dm", "public", "midnight", "optional", "private"]:
        for injection in ["public-save", "actual-write"]:
            run(path, injection)
