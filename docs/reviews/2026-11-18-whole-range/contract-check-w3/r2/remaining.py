"""Finish the calendar fixture, and isolate the new defer-only status read."""

from contextlib import contextmanager
import json
import sys
from unittest.mock import patch

import probe
from bts import daily_decision, picks, scheduler


def boundary():
    original = probe.offline.offline

    @contextmanager
    def with_standard_margin(at):
        with original(at) as (world, mp):
            world.config["scheduler"]["fallback_deadline_min"] = 40
            yield world, mp

    with patch.object(probe.offline, "offline", with_standard_margin):
        for day in ("2027-06-18", "2026-12-31"):
            baseline = probe.run("midnight", "legacy", day)
            reused = probe.run("midnight", "regular", day)
            assert not baseline[1] and any("[contest_state]" in alert for alert in baseline[2])
            assert len(reused[1]) == 1


def status_delay(reuse):
    day = probe.DAY
    at = probe.offline.instant
    with probe.offline.offline(at(day, 18, 25)) as (world, mp):
        world.dated(30)
        world.schedule[day][1]["gameDate"] = at(day, 19, 20).isoformat()
        world.config["scheduler"]["fallback_deadline_min"] = 35
        plan_times = [{"time_et": at(day, hh, mm), "game_pks": [pk]}
                      for hh, mm, pk in [(18, 25, 101), (18, 50, 102)]]
        regular_check = scheduler.run_single_check
        decide = scheduler._lock_decision_from_predictions
        regular_plan = scheduler.plan_fallback_action
        status_read = scheduler._delivery_game_statuses
        real_write = picks.atomic_write_text
        starts = []
        written = []
        started_defer_only_status = False

        def serialize(path, text):
            real_write(path, text)
            written.append(text.encode("ascii"))

        def forecast():
            starts.append(world.now.strftime("%H:%M"))
            if len(starts) == 1:
                world.advance(22 * 60)
                return world.frame((0.99, 0.70), (True, True))
            if world.now >= at(day, 18, 50):
                frame = world.frame((0.70, 0.99), (True, False))
                return frame.iloc[[1, 0]]
            return world.frame((0.99, 0.70), (True, True))

        def game_status(date):
            if started_defer_only_status:
                world.advance(30 * 60)
                raise TimeoutError("fixture status retry exhausted")
            return status_read(date)

        def plan(**kwargs):
            nonlocal started_defer_only_status
            result = regular_plan(**kwargs)
            if result.action == "defer" and reuse and not started_defer_only_status:
                started_defer_only_status = True
                daily = picks.load_pick(day, world.picks)
                assert not scheduler._void_committed_games(daily, day)
                # The proposed clock check now refuses rather than defers the
                # early pick. Either way the later check starts at 19:17.
            return result

        def check(*a, **kw):
            decisions = []

            def capture(*args, **kwargs):
                result = decide(*args, **kwargs)
                decisions.append(result)
                return result

            with patch.object(scheduler, "_lock_decision_from_predictions", capture):
                result = regular_check(*a, **kw)
            if reuse and len(starts) == 1:
                _, raw = picks.load_pick_bytes(day, world.picks)
                assert raw == written[-1]
                result["lock_decision"], = decisions
                result["saved_pick_bytes"] = raw
            return result

        world.predictions = forecast
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: plan_times)
        mp.setattr(scheduler, "check_confirmed_lineups",
                   lambda pks: {pk: {"away", "home"} for pk in pks if pk == 101})
        mp.setattr(scheduler, "_delivery_game_statuses", game_status)
        mp.setattr(scheduler, "plan_fallback_action", plan)
        mp.setattr(picks, "atomic_write_text", serialize)
        mp.setattr(scheduler, "run_single_check", check)
        scheduler.run_day(day, world.config)
        record = daily_decision.load_decision(day, world.picks)
        print("STATUS", json.dumps({"reuse": reuse, "starts": starts,
                                    "did_defer_only_status": started_defer_only_status,
                                    "sent": bool(world.sends("dm")),
                                    "committed": (record or {}).get("primary")}), flush=True)
        assert ("19:17" if reuse else "18:50") in starts
        assert bool(world.sends("dm")) is not reuse


if __name__ == "__main__":
    if sys.argv[1] == "calendar":
        boundary()
    else:
        for reuse in (False, True):
            status_delay(reuse)
