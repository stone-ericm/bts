"""Drive the actual watchdog through alert history and all contract texts."""
from shlex import quote
import sys

from support import (ALL_CLEAR, Arena, DAY, DOUBLE, FIRST, MATCH, PITCH, REMINDER, RIGHT, RIGHT_DD,
                     WRONG, WRONG_DD, dump, normalized, print_source, row)

ARENA = Arena("sequences")
PICK = "pick Batter 1010: recommended game 101 (first pitch 7:40 PM ET)"
DD = "double-down Batter 1020: recommended game 201 (first pitch 7:40 PM ET)"
VERIFY = (" Check the delivered pick and the game shown in MLB for each listed leg. Change only a leg whose "
          "correct game you can verify and that MLB still lets you edit. Keep any correct leg as entered.")
COMPARE = " Check the delivered pick against the game shown in MLB."


def wrong_legs(legs):
    return f"{FIRST} {legs}.{VERIFY}"


def remind(legs):
    return f"{REMINDER} {legs}.{COMPARE}"


def observation(world, at, rows=MATCH, **kwargs):
    result = world.check(at, rows, **kwargs)
    result = {key: value for key, value in result.items() if key != "receipts"}
    dump(at, normalized(result, world.root))
    return result


def check_marker(result, status, reason, tiers=None, code=None):
    marker = result["marker"]
    assert (marker["status"], marker["reason"]) == (status, reason), result
    if tiers is not None:
        assert marker["escalations"] == tiers, result
    if code is not None:
        assert result["exit"] == code, result


def new_world(*, double=False, **kwargs):
    world = ARENA.world()
    world.commit(pitch=PITCH, dd=(1020, 201, PITCH) if double else None, **kwargs)
    world.capture([WRONG, RIGHT_DD] if double else [WRONG])
    return world


def tier_orders():
    world = new_world()
    missing = observation(world, "18:26", (), dm="failure")
    check_marker(missing, "dm_failed", "no_pick")
    initial = observation(world, "18:30", ())
    check_marker(initial, "alerted", "no_pick", ["initial"], 1)
    one = observation(world, "18:45", dm="failure")
    check_marker(one, "alerted", "wrong_game", ["initial"], 1)
    two = observation(world, "18:46")
    check_marker(two, "alerted", "wrong_game", ["initial", "wrong_game"], 1)
    assert two["dm"] == one["attempted_dm"] == [wrong_legs(f"{PICK}, the entry is recorded in game 102")]
    gap = observation(world, "19:00", crosswalk={})
    check_marker(gap, "present_unverified", "present_unverified", ["initial", "wrong_game"], 0)
    for time, retry, tiers in (("19:05", "19:06", ["initial", "t30", "wrong_game"]),
                               ("19:20", "19:21", ["initial", "t15", "t30", "wrong_game"])):
        failed = observation(world, time, dm="failure")
        sent = observation(world, retry)
        check_marker(sent, "alerted", "wrong_game", tiers, 1)
        assert sent["dm"] == failed["attempted_dm"] == [remind(f"{PICK}, the entry is recorded in game 102")]
    repeated = observation(world, "19:30")
    assert repeated["dm"] == []
    assert repeated["stdout"] == (
        f"check-pick-entered: {DAY} game verification unresolved (pick); re-verifying each run until cutoff\n")
    cutoff = observation(world, "19:34", elapsed=61)
    check_marker(cutoff, "alerted", "wrong_game", ["initial", "t15", "t30", "wrong_game"], 1)
    assert not cutoff["attempted_dm"]
    assert observation(world, "19:35")["fetches"] == []
    ARENA.clear(world)

    for first in ("success", "none"):
        world = new_world()
        opened = observation(world, "18:30", dm=first)
        check_marker(opened, "alerted", "wrong_game", ["wrong_game"])
        removal = observation(world, "18:45", (), dm="failure")
        check_marker(removal, "alerted", "no_pick", ["wrong_game"])
        removed = observation(world, "18:46", ())
        check_marker(removed, "alerted", "no_pick", ["initial", "wrong_game"])
        assert removed["dm"] == removal["attempted_dm"]
        world.capture([RIGHT])
        fixed = observation(world, "18:50")
        check_marker(fixed, "confirmed", "match", ["initial", "wrong_game"], 0)
        assert fixed["dm"] == [ALL_CLEAR]
        world.capture([WRONG])
        terminal = observation(world, "18:55")
        assert terminal["fetches"] == [] and terminal["dm"] == []
        ARENA.clear(world)

    world = new_world()
    late = observation(world, "19:21")
    check_marker(late, "alerted", "wrong_game", ["t15", "t30", "wrong_game"], 1)
    world.capture([RIGHT])
    failed_clear = observation(world, "19:22", dm="failure")
    assert failed_clear["attempted_dm"] == [ALL_CLEAR]
    assert observation(world, "19:23")["attempted_dm"] == []
    ARENA.clear(world)


def forgotten_observations():
    for bridge, rows, kwargs in (
            ("crosswalk", MATCH, {"crosswalk": {}}), ("no-pick", (), {}),
            ("mismatch", [row(player=3)], {})):
        print(f"BRIDGE {bridge}")
        world = new_world()
        observation(world, "18:20", ())
        observation(world, "18:25", dm="failure")
        observation(world, "18:30", rows, **kwargs)
        world.capture([RIGHT], "2027-03-24T221000Z")
        result = observation(world, "18:35")
        dump("A1b expected", {"reason": "wrong_game", "exit": 1, "first_wrong_game_dm": True})
        dump("A1b met", result["exit"] == 1 and result["marker"]["reason"] == "wrong_game")
        ARENA.clear(world)


def exact_texts():
    world = new_world(double=True)
    world.capture([RIGHT, WRONG_DD])
    result = observation(world, "18:30", DOUBLE)
    assert result["dm"] == [wrong_legs(f"{DD}, the entry is recorded in game 202")]
    world.capture([RIGHT, RIGHT_DD], "2027-03-24T221000Z")
    reminder = observation(world, "19:05", DOUBLE)
    assert reminder["dm"] == [remind(DD)]
    assert "entry is recorded" not in reminder["dm"][0]
    ARENA.clear(world)

    for name in (None, "", 42):
        world = new_world(name=name)
        assert observation(world, "18:30")["dm"] == [
            wrong_legs(f"{PICK.replace('Batter 1010', 'batter 1010')}, the entry is recorded in game 102")]
        ARENA.clear(world)

    world = new_world(double=True, delivery="locked_unconfirmed", name=None)
    world.capture([WRONG, WRONG_DD])
    detail = f"{PICK.replace('Batter 1010', 'batter 1010')}, the entry is recorded in game 102"
    detail += f"; {DD}, the entry is recorded in game 202"
    for time in ("18:30", "19:05", "19:20"):
        alert = observation(world, time, DOUBLE)
        assert alert["dm"] == [
            "⚠️ Delivery of this pick was NOT confirmed. The scheduler tried to send None + DD Batter 1020. "
            f"MLB game evidence needs verification. {detail}. "
            "Check your DMs and the MLB entry before making any change. "
            f"Run `bts season-contract status --picks-dir {quote(str(world.picks))}`."]
    ARENA.clear(world)

    world = new_world(double=True)
    world.commit(pitch=PITCH, dd=(1020, 201, "not-a-time"))
    bad = observation(world, "18:30", DOUBLE)
    assert bad["exception"] == "ValueError" and bad["fetches"] == []
    ARENA.clear(world)

    world = new_world(double=True)
    observation(world, "18:30", DOUBLE)
    world.commit(game=None, pitch=PITCH, dd=(1020, 201, PITCH))
    world.capture([RIGHT_DD])
    no_pk = observation(world, "19:05", DOUBLE)
    assert no_pk["dm"] == [remind(PICK.replace("game 101", "game None"))]
    ARENA.clear(world)


def saved_game_ids():
    for saved in (None, "101", 101.0, True, 101):
        world = new_world(game=saved)
        result = observation(world, "18:30")
        if sys.argv[1] == "candidate":
            assert result["exit"] == (1 if type(saved) is int else 0)
        ARENA.clear(world)
    world = new_world(double=True)
    world.commit(pitch=PITCH, dd=(1010, 201, PITCH))
    result = observation(world, "18:30")
    assert result["exit"] == 0
    ARENA.clear(world)


if __name__ == "__main__":
    print_source()
    for name in sys.argv[2:]:
        globals()[name]()
