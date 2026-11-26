"""Exercise changes to the gap planner around its final elapsed-time boundary."""

import json
from pathlib import Path
import sys
from threading import current_thread, main_thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import review_world
from bts import daily_decision, scheduler


review_world.fixture.HERE = Path(__file__).resolve().parent
DAY, at, offline = review_world.DAY, review_world.at, review_world.offline
OLD = "/patched/src/" in scheduler.__file__
BASE = "/base/src/" in scheduler.__file__
REPAIR = "/planner-repair/src/" in scheduler.__file__


def run(case):
    with offline(at(18, 25)) as (world, mp):
        world.dated(30)
        world.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": at(18, 25), "game_pks": [101]},
            {"time_et": at(18, 42), "game_pks": [102]},
        ])
        check, save_state = scheduler.run_single_check, scheduler.save_state
        statuses, plan = scheduler._delivery_game_statuses, scheduler.plan_fallback_action
        forecasts, observations, read_events = [], [], []
        in_check = False
        advanced = False

        def forecast():
            forecasts.append((world.now.isoformat(), in_check))
            frame = world.frame((.99, .98), (False, True))
            if len(forecasts) == 1:
                world.advance(17 * 60 + 59)
                return frame
            if BASE and len(forecasts) == 2:
                world.advance(2)
                return frame
            return None

        def observed_check(*args, **kwargs):
            nonlocal in_check
            in_check = True
            try:
                result = check(*args, **kwargs)
            finally:
                in_check = False
            if case == "known-void" and len(forecasts) == 1:
                world.schedule[DAY][0]["status"].update(detailedState="Postponed", statusCode="PO")
            return result

        def observed_statuses(date):
            if current_thread() is not main_thread():
                result = statuses(date)
                read_events.append(["defer-only", result[101]["detailed"]])
                if case == "known-void":
                    world.advance(2)
                return result
            if case == "known-void":
                read_events.append(["transport guard", "OSError"])
                raise OSError("synthetic temporary schedule failure")
            return statuses(date)

        def observe_save(state, directory):
            nonlocal advanced
            result = save_state(state, directory)
            if case != "known-void" and not BASE and not advanced and state.fallback_refreshes:
                assert world.now == at(18, 42, 59)
                advanced = True
                world.advance(2)
                if case == "save-state-correction":
                    world.dated(30, best=31)
            return result

        def observed_plan(**kwargs):
            result = plan(**kwargs)
            observations.append([kwargs["now"].isoformat(), result.action, result.reason])
            if BASE and case == "save-state-correction" and len(observations) == 1:
                world.dated(30, best=31)
            return result

        world.predictions = forecast
        mp.setattr(scheduler, "run_single_check", observed_check)
        mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {
            pk: {"home", "away"} for pk in pks if pk == 101})
        mp.setattr(scheduler, "_delivery_game_statuses", observed_statuses)
        mp.setattr(scheduler, "save_state", observe_save)
        mp.setattr(scheduler, "plan_fallback_action", observed_plan)
        scheduler.run_day(DAY, world.config)
        record = daily_decision.load_decision(DAY, world.picks)
        state = scheduler.load_state(DAY, world.picks)
        alerts = review_world.health(world)
        sent = review_world.events(world, "dm")
        assert observations[0][-1] == ("gap_no_feasible_window" if BASE else "gap_contender_window_feasible")
        facts = {"case": case, "base": BASE, "old": OLD, "clock": world.now.isoformat(),
                 "plans": observations, "forecasts": forecasts, "status_reads": read_events,
                 "history": state.fallback_refreshes, "alerts": alerts, "pick_dm": sent,
                 "committed": record and record["primary"],
                 "archived": len(list((world.picks / DAY).glob("deferred_fallback*.json")))}
        print("EDGE", json.dumps(facts), flush=True)
        if case == "known-void":
            assert read_events[0] == ["defer-only", "Postponed"]
            if OLD or REPAIR:
                assert not sent and not record and any("[postponed_pick]" in msg for msg in alerts)
            else:
                assert read_events == [["defer-only", "Postponed"], ["transport guard", "OSError"]]
                assert len(sent) == 1 and record["primary"]["game_pk"] == 101 and not alerts
        elif (BASE or REPAIR) and case == "save-state-correction":
            assert not record and not sent and any("[missed_pick]" in msg for msg in alerts)
            assert any("[contest_state]" in msg for msg in alerts)
        elif BASE or REPAIR:
            assert len(sent) == 1 and record["primary"]["game_pk"] == 101
        else:
            assert advanced and facts["archived"] == 1 and not sent and not alerts and not record
            assert all(row[0] == at(18, 42, 59).isoformat() for row in observations)


if __name__ == "__main__":
    for case in sys.argv[1:]:
        run(case)
