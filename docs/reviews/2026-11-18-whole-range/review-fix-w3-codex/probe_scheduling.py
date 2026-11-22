"""Check cutoff at reload and the reuse planner's live scheduled-run inputs."""

import json
import sys

from bts import daily_decision, picks, scheduler

from review_world import DAY, at, game, health, offline, pending_window


def reload_cutoff():
    with offline(at(18, 25)) as (world, mp):
        world.dated(30)
        pending_window(world, mp, scheduler)
        starts = []
        crossed = False
        load = picks.load_pick_bytes

        def read(*a, **kw):
            nonlocal crossed
            value = load(*a, **kw)
            if not crossed and world.now == at(18, 47) and not kw.get("regular_only"):
                crossed = True
                world.advance(18 * 60)
            return value

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                world.advance(22 * 60)
                return world.frame((.99, .40), (True, True))
            return world.frame((.40, .99), (True, False)).iloc[[1, 0]]

        world.predictions = forecast
        mp.setattr(picks, "load_pick_bytes", read)
        mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {
            pk: {"away", "home"} for pk in pks if pk == 101})
        scheduler.run_day(DAY, world.config)
        record = daily_decision.load_decision(DAY, world.picks)
        state = scheduler.load_state(DAY, world.picks)
        print("RELOAD_CUTOFF", json.dumps({"crossed": crossed,
                                          "starts": [t.isoformat() for t in starts],
                                          "history": state.fallback_refreshes,
                                          "refusals": state.delivery_refusals,
                                          "health": health(world), "commit": record["primary"]}), flush=True)
        assert crossed and starts == [at(18, 25), at(19, 5)]
        assert sum("[late_delivery]" in msg for msg in health(world)) == 1
        assert record["primary"]["game_pk"] == 102
        assert len(list((world.picks / DAY).glob("refused_delivery*.json"))) == 1
        assert not list((world.picks / DAY).glob("deferred_fallback*.json"))


def coalescing():
    with offline(at(18, 20)) as (world, mp):
        world.dated(30)
        world.schedule[DAY].extend([game(103, at(20, 20)), game(104, at(20, 30))])
        times = [(18, 20, 101), (18, 42, 102), (18, 45, 103), (19, 0, 104)]
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": at(h, m), "game_pks": [pk]} for h, m, pk in times])
        mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {
            pk: {"home", "away"} for pk in pks if pk in {101, 103}})
        starts = []
        traces = []
        planner = scheduler.plan_fallback_action

        def forecast():
            starts.append(world.now)
            if world.now == at(18, 35):
                world.advance(11 * 60)
            if world.now == at(19):
                return world.frame((.40, .40, .40, .99), (True, True, False, False)).iloc[[3, 2, 1, 0]]
            return world.frame((.99, .40, .40, .40), (True, True, False, True))

        def plan(**kwargs):
            result = planner(**kwargs)
            traces.append(([run["game_pks"] for run in kwargs["remaining_runs"]], result.action,
                           kwargs["budget_min"], kwargs["operator_reserve_min"]))
            return result

        world.predictions = forecast
        mp.setattr(scheduler, "plan_fallback_action", plan)
        scheduler.run_day(DAY, world.config)
        record = daily_decision.load_decision(DAY, world.picks)
        state = scheduler.load_state(DAY, world.picks)
        print("COALESCING", json.dumps({"starts": [t.isoformat() for t in starts], "planner": traces,
                                       "runs": state.runs_completed, "history": state.fallback_refreshes,
                                       "commit": record["primary"], "health": health(world)}), flush=True)
        assert starts == [at(18, 20), at(18, 35), at(18, 46), at(19)]
        assert traces[0] == ([[102], [103], [104]], "defer", 12.0, 10.0)
        assert all(trace == ([[104]], "defer", 12.0, 10.0) for trace in traces[1:])
        old_refresh, reuse = state.fallback_refreshes
        assert not old_refresh["reused_check_cascade"] and reuse["reused_check_cascade"]
        assert reuse["duration_sec"] == 0.0 and record["primary"]["game_pk"] == 104
        assert sum(row.get("reason") == "coalesced_after_fallback" for row in state.runs_completed) == 1


if __name__ == "__main__":
    {"reload": reload_cutoff, "coalescing": coalescing}[sys.argv[1]]()
