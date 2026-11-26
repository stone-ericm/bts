"""Check window boundaries and both real response shapes with archived sources."""
from datetime import date
from itertools import product
import sys

from bts import picks
from support import Arena, DAY, DOUBLE, MATCH, PITCH, RIGHT, RIGHT_DD, WRONG, WRONG_DD, dump, row
from support import normalized, print_source


def events(pending, sources):
    profile = []
    flat = []
    for entry, source in zip(pending, sources, strict=True):
        if "profile" in source:
            profile.append({"roundId": entry["roundId"], "roundPredictions": [entry]})
        if "pending" in source:
            flat.append(entry)
    return {"predictions": profile}, flat


def check_pair(world, side, bound, sources):
    world.commit(pitch=PITCH, dd=(1020, 201, PITCH))
    world.capture(bound[:1], f"{DAY}T210000Z")
    world.capture(bound[1:], f"{DAY}T220000Z")
    profile, pending = events(DOUBLE, sources)
    result = world.check("18:25", pending, profile=profile)
    receipt, = result["receipts"]
    states = tuple(slot["state"] for slot in receipt["qualification"]["slots"])
    expected = tuple("wrong_game" if unit["feedId"] in (102, 202) else "confirmed" for unit in bound)
    assert states == expected
    assert result["exit"] == (1 if side == "candidate" and "wrong_game" in expected else 0)
    if side == "candidate" and expected.count("wrong_game") == 1:
        assert ("double-down " in result["dm"][0]) == (expected[1] == "wrong_game")
        assert ("pick Batter " in result["dm"][0]) == (expected[0] == "wrong_game")


def main():
    side = sys.argv[1]
    arena = Arena(f"rows-{side}")
    source = ("profile", "pending", "profile+pending")
    for modes, units in product(product(source, repeat=2), product((RIGHT, WRONG), (RIGHT_DD, WRONG_DD))):
        world = arena.world()
        check_pair(world, side, units, modes)
        arena.clear(world)
    print("36 single-source and mixed-source doubles matched the file-backed expected states")

    for time in ("18:24", "18:25", "19:34", "19:35"):
        for legacy in (False, True):
            world = arena.world()
            daily = world.commit(pitch=PITCH, delivery=None if legacy else "delivered")
            if legacy:
                daily.notification_sent = True
                picks.save_pick(daily, world.picks)
            world.capture([WRONG])
            result = world.check(time, [*MATCH, row(88, 3, 2, 2)],
                                 rounds={1: date.fromisoformat(DAY), 2: date(2027, 3, 23)})
            dump(time, normalized({key: result[key] for key in ("exit", "fetches", "marker")}, world.root))
            assert bool(result["fetches"]) == (time in ("18:25", "19:34"))
            if result["fetches"]:
                assert result["exit"] == (1 if side == "candidate" else 0)
            arena.clear(world)


if __name__ == "__main__":
    print_source()
    main()
