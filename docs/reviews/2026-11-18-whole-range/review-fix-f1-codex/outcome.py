"""Read real, sealed receipts through day-outcome and replay a rollback."""
from pathlib import Path
import sys

from bts import entry_receipt
from support import (ALL_CLEAR, Arena, DAY, HERE, MATCH, PITCH, RIGHT, WRONG, dump, normalized,
                     print_source)


def outcome(world, expected, health):
    morning = world.morning()
    alerts = world.health()
    dump(expected, normalized({"morning": morning, "health": alerts}, world.root))
    assert morning["outcome"]["state"] == expected
    assert bool(morning["dm"]) == (not morning["outcome"]["ok"])
    assert [item["level"] for item in alerts] == health


def write_world(arena, bound):
    world = arena.world()
    world.commit(pitch=PITCH)
    if bound:
        world.capture([bound])
    return world


def states(side):
    candidate = side == "candidate"
    arena = Arena(f"morning-{side}")
    cases = (
        ("right", RIGHT, MATCH, 0, "success", "entered", []),
        ("unbound", {**RIGHT, "feedId": None}, MATCH, 0, "success", "entered_game_unverified", []),
        ("late proof", RIGHT, MATCH, 61, "success", "entered_late", []),
        ("absent", RIGHT, (), 0, "success", "not_entered", ["WARN"]),
        ("wrong only", WRONG, MATCH, 0, "success", "entry_unverifiable", ["WARN"] if candidate else []),
        ("wrong DM failed", WRONG, MATCH, 0, "failure", "entry_unverifiable", ["CRITICAL"] if candidate else []),
        ("present unverified", RIGHT, MATCH, 0, "success", "entry_unverifiable", []),
        ("unreadable capture", WRONG, MATCH, 0, "success", "entry_unverifiable", []),
        ("no receipt", None, (), 0, "success", "entry_unverifiable", []),
    )
    for label, bound, pending, elapsed, dm, expected, health in cases:
        print(label)
        world = write_world(arena, bound)
        if label == "unreadable capture":
            world.capture([]).write_bytes(b"unreadable archive")
        if label != "no receipt":
            world.check("19:34" if elapsed else "18:30", pending,
                        dm=dm, elapsed=elapsed, crosswalk={} if label == "present unverified" else None)
        outcome(world, expected, health)
        arena.clear(world)

    world = write_world(arena, WRONG)
    world.check("18:30")
    world.check("18:45", MATCH)
    outcome(world, "not_entered", ["WARN"] if candidate else [])
    arena.clear(world)
    if candidate:
        world = write_world(arena, WRONG)
        world.check("18:30", MATCH)
        world.check("18:45", MATCH, crosswalk={})
        outcome(world, "entry_unverifiable", [])
        arena.clear(world)
        world = write_world(arena, WRONG)
        world.check("18:30", MATCH)
        world.capture([RIGHT], "2027-03-24T230000Z")
        incomplete = world.check("19:05", MATCH)
        assert incomplete["marker"]["status"] == "alerted" and incomplete["exit"] == 1
        assert [r["qualification"]["slots"][0]["state"] for r in incomplete["receipts"]] == [
            "wrong_game", "unit_unverified"]
        outcome(world, "entered_game_unverified", ["WARN"])
        arena.clear(world)


def changing_captures():
    arena = Arena("receipt-independent")
    for on_run, at_publish, expected, health in (
            (WRONG, RIGHT, "entered", ["WARN"]), (RIGHT, WRONG, "entry_unverifiable", [])):
        world = write_world(arena, on_run)
        old_reader = entry_receipt.load_units
        calls = []

        def move_capture(picks, year):
            calls.append(year)
            answer = old_reader(picks, year)
            if len(calls) == 1:
                world.capture([at_publish])
            return answer

        result = world.check("18:30", MATCH,
                             more_patches=[("bts.entry_receipt.load_units", move_capture)])
        assert calls == [2027, 2027]
        assert result["receipts"][0]["verifier"]["reason"] == "match"
        outcome(world, expected, health)
        if on_run is WRONG:
            world.capture([WRONG])
            later = world.check("18:45", MATCH)
            assert later["marker"]["reason"] == "wrong_game"
            outcome(world, "entered", ["WARN"])
        arena.clear(world)


def rollback(side):
    arena = Arena("rollback")
    if side == "candidate":
        world = write_world(arena, WRONG)
        assert world.check("18:30", MATCH)["marker"]["reason"] == "wrong_game"
    else:
        world = arena.world()
        assert world.read_marker()["reason"] == "wrong_game"
        result = world.check("18:45", MATCH)
        dump("rollback", normalized(result, world.root))
        assert result["dm"] == [ALL_CLEAR] and result["marker"]["status"] == "confirmed"
        assert result["receipts"][-1]["qualification"]["slots"][0]["state"] == "wrong_game"
        again = world.check("19:00", MATCH)
        assert again["fetches"] == [] and again["dm"] == []
        arena.clear(world)


if __name__ == "__main__":
    print_source()
    side, check = sys.argv[1:]
    if check == "changing_captures":
        changing_captures()
    else:
        globals()[check](side)
