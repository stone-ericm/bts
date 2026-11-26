"""Tell elapsed fetch time apart from elapsed local capture-read time."""
import sys

from bts import entry_receipt
from support import Arena, MATCH, PITCH, WRONG, dump, normalized, print_source


def main():
    side = sys.argv[1]
    arena = Arena(f"capture-cutoff-{side}")
    world = arena.world()
    world.commit(pitch=PITCH)
    world.capture([WRONG])
    elapsed = 0
    original = entry_receipt.load_units

    def time_now():
        return 1000 + elapsed

    def read_units(*args):
        nonlocal elapsed
        result = original(*args)
        elapsed = 61
        return result

    result = world.check("19:34", MATCH, more_patches=[
        ("time.monotonic", time_now),
        ("bts.entry_receipt.load_units", read_units),
    ])
    observation = result["receipts"][0]["observation"]
    assert observation["before_cutoff"] is True
    assert observation["response_completed_at"].endswith("19:34:00-04:00")
    assert result["dm"] == []
    if side == "candidate":
        assert result["exit"] == 1 and result["marker"]["reason"] == "wrong_game"
        assert "cutoff passed during fetch" in result["stdout"]
    else:
        assert result["exit"] == 0 and result["marker"]["reason"] == "match"
    dump("cutoff during capture read", {"stdout": result["stdout"], "before_cutoff": observation["before_cutoff"],
                                       "response_completed_at": observation["response_completed_at"],
                                       "published_at": result["receipts"][0]["published_at"]})
    arena.clear(world)


if __name__ == "__main__":
    print_source()
    main()
