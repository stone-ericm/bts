"""Replay candidate binding, deadline equality, sleep, and saved metadata."""

from copy import deepcopy
from dataclasses import replace
import json
import sys

from bts import daily_decision, picks, scheduler

from review_world import DAY, at, events, health, offline


IS_BASE = "base/src" in scheduler.__file__


def run(case):
    with offline(at(18, 25)) as (world, mp):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.dated(0)
        path = world.picks / f"{DAY}.json"
        mp.setattr(scheduler, "plan_run_times", lambda *_args, **_kwargs: [
            {"time_et": at(18, 25), "game_pks": [101]},
        ])
        starts = []
        checks = []
        writes = []
        sleeps = []
        check = scheduler.run_single_check
        append = picks.append_lineup_evolution
        write = picks.atomic_write_text
        was_swapped = False

        def swap(daily):
            nonlocal was_swapped
            assert not was_swapped
            was_swapped = True
            other = deepcopy(daily)
            other.pick = replace(daily.pick, batter_name="Replacement Batter", batter_id=999)
            picks.save_pick(other, world.picks)

        def capture_write(file, data):
            write(file, data)
            if file == path:
                writes.append(data.encode("ascii"))

        def audit(daily, directory):
            append(daily, directory)
            if not was_swapped and not checks and case == "audit-swap":
                swap(daily)

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                world.advance({"sleep": 0, "equality": 10 * 60}.get(case, 35 * 60))
                frame = world.frame((0.99,), (True,))
                frame.loc[0, "batter_name"] = "José Fixture"
                return frame
            if case == "late-second-cascade":
                world.advance(6 * 60)
            return None

        def capture_check(*args, **kwargs):
            first_write = len(writes)
            result = check(*args, **kwargs)
            checks.append(result)
            if not IS_BASE:
                assert result["saved_pick_bytes"] == writes[first_write]
                assert isinstance(result["lock_decision"], scheduler.LockDecision)
                assert b"\\u00e9" in writes[first_write]
            if case == "swap":
                swap(result["pick_result"].daily)
            if case == "whitespace":
                path.write_bytes(path.read_bytes() + b"\n")
            if case == "decision-only":
                result.pop("saved_pick_bytes", None)
            return result

        def sleep(duration):
            sleeps.append((world.now.isoformat(), duration))
            world.advance(duration)

        world.predictions = forecast
        mp.setattr(scheduler, "run_single_check", capture_check)
        mp.setattr(scheduler, "_watchdog_ping_sleep", sleep)
        mp.setattr(picks, "atomic_write_text", capture_write)
        mp.setattr(picks, "append_lineup_evolution", audit)
        scheduler.run_day(DAY, world.config)
        record = daily_decision.load_decision(DAY, world.picks)
        state = scheduler.load_state(DAY, world.picks)
        sent = events(world, "dm")
        refresh, = state.fallback_refreshes
        reused = not IS_BASE and case in {"late-second-cascade", "equality", "overrun"}
        print("BINDING", json.dumps({"case": case, "base": IS_BASE,
                                     "starts": [start.isoformat() for start in starts],
                                     "sleeps": sleeps, "refresh": refresh,
                                     "send": sent, "record": record and record["primary"],
                                     "health": health(world)}), flush=True)
        assert refresh["reused_check_cascade"] is reused
        assert len(starts) == (1 if reused else 2)
        assert any(start == at(18, 25).isoformat() and duration == 600
                   for start, duration in sleeps) is (case == "sleep")
        if reused:
            assert refresh["duration_sec"] == 0.0
        if IS_BASE and case == "late-second-cascade":
            assert not sent and record is None
            assert sum("[late_delivery]" in line for line in health(world)) == 1
            assert len(list((world.picks / DAY).glob("refused_delivery*.json"))) == 1
        else:
            assert len(sent) == 1 and record["primary"]["batter_name"] in sent[0][1]
            assert record["primary"]["batter_id"] == (999 if was_swapped else 601)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["late-second-cascade", "swap", "audit-swap", "whitespace",
                                "sleep", "equality", "decision-only", "overrun"]:
        run(name)
