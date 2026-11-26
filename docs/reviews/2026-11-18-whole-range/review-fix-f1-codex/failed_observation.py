"""Regression check for a wrong-game observation followed by a verifier gap."""
import sys

from support import ALL_CLEAR, Arena, MATCH, PITCH, RIGHT, WRONG, dump, normalized, print_source, row


def main():
    side = sys.argv[1]
    arena = Arena(f"failed-observation-{side}")
    missed = []
    for bridge, entries, options in (
            ("crosswalk", MATCH, {"crosswalk": {}}),
            ("removal", (), {}),
            ("mismatch", [row(player=3)], {})):
        world = arena.world()
        world.commit(pitch=PITCH)
        world.capture([WRONG])
        results = [world.check("18:25", ()), world.check("18:35", MATCH, dm="failure")]
        results.append(world.check("18:45", entries, **options))
        world.capture([RIGHT], "2027-03-24T230000Z")
        results.append(world.check("19:00", MATCH))
        for result in results:
            result["receipt_slots"] = [
                item["qualification"]["slots"][0]["state"]
                for item in result.pop("receipts") if item["qualification"]]
        dump(bridge, normalized(results, world.root))
        assert results[0]["marker"]["reason"] == "no_pick" and results[0]["dm"]
        if side == "candidate":
            assert results[1]["marker"]["reason"] == "wrong_game"
            assert results[1]["marker"]["escalations"] == ["initial"]
            last = results[-1]
            if last["exit"] != 1 or last["marker"]["reason"] != "wrong_game" or ALL_CLEAR in last["dm"]:
                missed.append(bridge)
        arena.clear(world)
    assert not missed, f"False all-clear after these intermediate results: {missed}"


if __name__ == "__main__":
    print_source()
    main()
