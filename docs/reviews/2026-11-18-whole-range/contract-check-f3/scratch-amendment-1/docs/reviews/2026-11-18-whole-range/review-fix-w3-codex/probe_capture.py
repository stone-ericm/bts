"""Verify the patched writer's exact bytes, first-write rule and observer scope."""

from copy import deepcopy
from dataclasses import replace
import json
from threading import Thread

from bts import picks, scheduler

from review_world import DAY, at, offline


with offline(at(18, 25)) as (world, mp):
    world.dated(0)
    world.schedule[DAY] = world.schedule[DAY][:1]
    world.predictions = lambda: world.frame((.99,), (True,))
    daily = scheduler.run_single_check(DAY, [101], set(), world.config, .03)["pick_result"].daily
    first = deepcopy(daily)
    first.pick = replace(first.pick, batter_name="José First")
    other = deepcopy(first)
    other.pick = replace(other.pick, batter_id=999, batter_name="Another Candidate")
    source = world.picks / f"{DAY}.json"

    with picks.capture_pick_write(first) as outer:
        results = []

        def thread_write():
            results.append(picks.save_pick(first, world.picks))

        thread = Thread(target=thread_write, daemon=True)
        thread.start()
        thread.join(2)
        assert not thread.is_alive() and results == [source] and not outer

        with picks.capture_pick_write(other) as inner:
            assert picks.save_pick(other, world.picks) == source
            assert inner == [source.read_bytes()] and not outer
        assert len(inner) == 1

        assert picks.save_pick(first, world.picks) == source
        expected = source.read_bytes()
        assert b"Jos\\u00e9 First" in expected and outer == [expected]
        first.pick = replace(first.pick, batter_name="Same object on later save")
        assert picks.save_pick(first, world.picks) == source
        assert source.read_bytes() != expected and outer == [expected]

    assert picks._PICK_WRITE_OBSERVER.get() is None
    real_write = picks.atomic_write_text
    for fail_class in [OSError, KeyboardInterrupt, SystemExit]:
        def fail(*_args, **_kwargs):
            raise fail_class("fixture failure before replace")

        with mp.context() as fault:
            fault.setattr(picks, "atomic_write_text", fail)
            try:
                with picks.capture_pick_write(first) as failed:
                    picks.save_pick(first, world.picks)
            except fail_class:
                assert not failed and picks._PICK_WRITE_OBSERVER.get() is None
            else:
                raise AssertionError("writer did not propagate fixture failure")
        assert picks.atomic_write_text is real_write
    print("CAPTURE", json.dumps({"saved": json.loads(expected)["pick"]["batter_name"],
                                 "path_returned": results[0] == source, "captures": len(outer),
                                 "nested_isolated": len(inner) == 1, "thread_isolated": True,
                                 "observer_restored_on_errors": True}), flush=True)
