"""Run the independent sequence checks, with a separate retry countdown check."""
import sys

import sequences
from support import ALL_CLEAR, Arena, MATCH, RIGHT, WRONG, print_source


def remaining_tier_orders():
    for first in ("success", "none"):
        world = sequences.new_world()
        opened = sequences.observation(world, "18:30", dm=first)
        sequences.check_marker(opened, "alerted", "wrong_game", ["wrong_game"])
        removal = sequences.observation(world, "18:45", (), dm="failure")
        sequences.check_marker(removal, "alerted", "no_pick", ["wrong_game"])
        removed = sequences.observation(world, "18:46", ())
        sequences.check_marker(removed, "alerted", "no_pick", ["initial", "wrong_game"])
        assert removed["dm"] == [text.replace("(50 min", "(49 min") for text in removal["attempted_dm"]]
        world.capture([RIGHT])
        fixed = sequences.observation(world, "18:50")
        sequences.check_marker(fixed, "confirmed", "match", ["initial", "wrong_game"], 0)
        assert fixed["dm"] == [ALL_CLEAR]
        world.capture([WRONG])
        terminal = sequences.observation(world, "18:55")
        assert terminal["fetches"] == [] and terminal["dm"] == []
        sequences.ARENA.clear(world)

    world = sequences.new_world()
    late = sequences.observation(world, "19:21")
    sequences.check_marker(late, "alerted", "wrong_game", ["t15", "t30", "wrong_game"], 1)
    world.capture([RIGHT])
    failed_clear = sequences.observation(world, "19:22", dm="failure")
    assert failed_clear["attempted_dm"] == [ALL_CLEAR]
    assert sequences.observation(world, "19:23")["attempted_dm"] == []
    sequences.ARENA.clear(world)


def proof_of_every_leg():
    from support import DOUBLE, RIGHT_DD, WRONG_DD
    world = sequences.new_world(double=True)
    world.capture([RIGHT, WRONG_DD])
    sequences.observation(world, "18:30", DOUBLE)
    world.capture([RIGHT, {**RIGHT_DD, "feedId": None}])
    incomplete = sequences.observation(world, "18:45", DOUBLE)
    sequences.check_marker(incomplete, "alerted", "wrong_game", ["wrong_game"], 1)
    assert not incomplete["dm"]
    world.capture([RIGHT, RIGHT_DD])
    complete = sequences.observation(world, "18:46", DOUBLE)
    sequences.check_marker(complete, "confirmed", "match", ["wrong_game"], 0)
    assert complete["dm"] == [ALL_CLEAR]
    sequences.ARENA.clear(world)


def failed_first_without_other_alert():
    world = sequences.new_world()
    first = sequences.observation(world, "18:30", dm="failure")
    sequences.check_marker(first, "dm_failed", "wrong_game", [], 1)
    gap = sequences.observation(world, "18:45", crosswalk={})
    sequences.check_marker(gap, "present_unverified", "present_unverified", [], 0)
    world.capture([RIGHT], "2027-03-24T221000Z")
    lost = sequences.observation(world, "19:00", MATCH)
    sequences.dump("A1b met without earlier missing DM", lost["exit"] == 1)
    sequences.ARENA.clear(world)


if __name__ == "__main__":
    print_source()
    sequences.ARENA = Arena(f"sequence-run-{sys.argv[2]}")
    for name in sys.argv[2:]:
        (globals().get(name) or getattr(sequences, name))()
