"""Check standing skips, early return shapes, selection metadata, and planner inputs."""

from dataclasses import replace
import json
import sys

from bts import contest_state, daily_decision, orchestrator, picks, scheduler, strategy

from review_world import DAY, at, game, health, offline


IS_BASE = "base/src" in scheduler.__file__


def early(case):
    with offline(at(18, 25)) as (world, mp):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.dated(30 if case == "skip" else 0)
        world.predictions = lambda: world.frame((0.99,), (True,))
        required = {"skipped", "new_lineups", "should_post", "pick_result", "pick_name", "pick_p", "selection"}
        if case == "locked":
            daily = scheduler.run_single_check(DAY, [101], set(), world.config, 0.03)["pick_result"].daily
            picks.save_pick(replace(daily, delivery_attempted=True), world.picks)
        elif case == "empty":
            world.predictions = lambda: None
            required.add("skip_summary")
        elif case == "skip":
            world.predictions = lambda: world.frame((0.10,), (True,))
            required.add("skip_summary")
        elif case == "contest":
            def error(*a, **kw):
                raise contest_state.ContestStateError("offline contest-state fixture")

            mp.setattr(orchestrator, "run_and_pick", error)
        else:
            def fail(*a, **kw):
                raise OSError("simulated atomic pick write failure")

            mp.setattr(picks, "atomic_write_text", fail)
        try:
            result = scheduler.run_single_check(DAY, [101], set(), world.config, 0.03)
        except OSError as exc:
            assert case == "write-error" and "atomic pick write" in str(exc)
            assert not (world.picks / f"{DAY}.json").exists()
            print("EARLY", json.dumps({"case": case, "base": IS_BASE, "error": str(exc)}), flush=True)
            return
        assert case != "write-error"
        assert set(result) == required and not result["should_post"]
        selection = result.get("selection")
        pick_result = result["pick_result"]
        print("EARLY", json.dumps({"case": case, "base": IS_BASE, "keys": sorted(result),
                                   "locked": bool(pick_result and pick_result.locked),
                                   "skip": selection and selection.action, "health": health(world)}), flush=True)
        assert bool(pick_result and pick_result.locked) == (case == "locked")
        if case == "skip":
            assert selection.action == "skip" and result["skip_summary"] is not None


def standing(cached):
    with offline(at(17, 40)) as (world, mp):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.dated(30)
        check = scheduler.run_single_check
        world.predictions = lambda: world.frame((0.99,), (True,))
        assert check(DAY, [101], set(), world.config, 0.03)["pick_result"] is not None
        actions = []
        decide = strategy.resolve_policy_decision

        def policy(*args, **kwargs):
            choice = decide(*args, **kwargs)
            return replace(choice, source="mdp") if choice.action == "skip" else choice

        def forecast():
            if world.now == at(17, 40):
                return world.frame((0.10,), (True,))
            if world.now == at(18, 25):
                world.advance(35 * 60)
                return world.frame((0.99,), (True,))
            return None

        def observe(*args, **kwargs):
            result = check(*args, **kwargs)
            actions.append(result["selection"].action)
            if actions == ["skip"] and cached:
                world.schedule[DAY][0]["status"].update(detailedState="Warmup", statusCode="PW")
            return result

        world.predictions = forecast
        mp.setattr(strategy, "resolve_policy_decision", policy)
        mp.setattr(scheduler, "run_single_check", observe)
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": t, "game_pks": [101]} for t in (at(17, 40), at(18, 25))])
        scheduler.run_day(DAY, world.config)
        record = daily_decision.load_decision(DAY, world.picks)
        sent = [msg for msg in world.sends("dm") if msg.startswith("Today's pick")]
        print("SKIP", json.dumps({"cached": cached, "base": IS_BASE, "actions": actions,
                                   "outcome": record["action"], "scoreable": record["scoreable"],
                                   "sent": sent, "health": health(world)}), flush=True)
        assert actions == (["skip", None] if cached else ["skip", "single"])
        assert bool(sent) is not cached and record["scoreable"] is not cached
        assert not health(world)


def metadata(case):
    with offline(at(18, 25)) as (world, mp):
        if case in {"warmup", "mismatched"}:
            world.schedule[DAY] = world.schedule[DAY][:1]
        world.dated(0 if case in {"warmup", "mismatched", "partial-void"} else 30)
        real = scheduler.run_single_check
        seen = []
        if case == "warmup":
            world.predictions = lambda: world.frame((0.99,), (True,))
            assert real(DAY, [101], set(), world.config, .03)["pick_result"] is not None
            world.schedule[DAY][0]["status"].update(detailedState="Warmup", statusCode="PW")

        def forecast():
            world.advance(35 * 60)
            return world.frame((0.99, 0.98), (True, True))

        def check(*args, **kwargs):
            result = real(*args, **kwargs)
            selection = result["selection"]
            seen.append(selection)
            if case == "partial-void":
                assert result["pick_result"].daily.double_down
                world.schedule[DAY][0]["status"].update(detailedState="Postponed", statusCode="PO")
            if case == "mismatched":
                assert selection.action == "single"
                result["selection"] = replace(selection, source="fixture-wrong-source",
                                              primary_candidate=dict(selection.primary_candidate, p_game_hit=.10))
            return result

        world.predictions = forecast
        mp.setattr(scheduler, "run_single_check", check)
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": at(18, 25), "game_pks": [101]},
        ])
        scheduler.run_day(DAY, world.config)
        record = daily_decision.load_decision(DAY, world.picks)
        daily = picks.load_pick(DAY, world.picks)
        selection, = seen
        print("METADATA", json.dumps({"case": case, "action": selection.action,
                                       "primary": record["primary"], "secondary": record["double_down"],
                                       "policy": daily.policy_decision, "source": record["source"],
                                       "dm": world.sends("dm"), "health": health(world)}), flush=True)
        assert len(world.sends("dm")) == 1 and record["primary"]["p_game_hit"] == .99
        if case in {"warmup", "mismatched"}:
            assert record["primary"] == scheduler._row_from_daily(daily.pick)
            assert record["objective"] == daily.policy_decision["objective"]
            assert record["source"] == daily.policy_decision["source"]
        if case == "single-runner-up":
            assert daily.double_down is None and record["action"] == "single"
            assert record["double_down"] == selection.double_candidate and record["double_down"] is not None
        if case == "partial-void":
            assert record["action"] == "double" and daily.double_down.game_pk == 102
            assert "Synthetic Batter 101" in world.sends("dm")[0] and "Synthetic Batter 102" in world.sends("dm")[0]
            assert not health(world)


def multi_contender():
    with offline(at(18, 25)) as (world, mp):
        world.dated(30)
        world.schedule[DAY].append(game(103, at(19, 45)))
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": at(h, m), "game_pks": [pk]}
            for h, m, pk in [(18, 25, 101), (18, 42, 103), (18, 50, 102)]
        ])
        starts = []
        inputs = []
        planner = scheduler.plan_fallback_action

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                world.advance(17 * 60 + 30)
                return world.frame((.99, .985, .975), (False, True, True))
            return world.frame((.40, .40, .99), (False, True, False)).iloc[[2, 1, 0]]

        def plan(**kwargs):
            result = planner(**kwargs)
            inputs.append((kwargs["contender_game_pk"], kwargs["contender_game_pks"],
                           kwargs["should_post"], kwargs["should_post_ungated"], result.action,
                           result.window_time_et.isoformat() if result.window_time_et else None))
            return result

        world.predictions = forecast
        mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {
            pk: {"home", "away"} for pk in pks if pk == 101})
        mp.setattr(scheduler, "plan_fallback_action", plan)
        scheduler.run_day(DAY, world.config)
        state = scheduler.load_state(DAY, world.picks)
        record = daily_decision.load_decision(DAY, world.picks)
        print("CONTENDERS", json.dumps({"inputs": inputs, "runs": state.runs_completed,
                                       "refresh": state.fallback_refreshes, "record": record["primary"]}), flush=True)
        assert len(starts) == 2
        assert inputs[0] == (102, (102, 103), False, False, "defer", at(18, 42).isoformat())
        assert state.fallback_refreshes[0]["reused_check_cascade"] and record["primary"]["game_pk"] == 103


if __name__ == "__main__":
    if sys.argv[1] == "early":
        for name in ["locked", "empty", "skip", "contest", "write-error"]:
            early(name)
        standing(True)
        standing(False)
    else:
        for name in ["warmup", "mismatched", "single-runner-up", "partial-void"]:
            metadata(name)
        multi_contender()
