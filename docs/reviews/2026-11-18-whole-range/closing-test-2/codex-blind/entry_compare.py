"""Feed archived watchdog the same committed doubleheader selections."""
import gzip
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from harness import ACCOUNT, IS_BASE, RECIPIENT, World
from opening import restore


def main(folder):
    assert IS_BASE
    previous = json.loads((folder / "season-head.json").read_text())
    day = "2027-07-15"
    rows = []
    for source in previous["doubleheader"]:
        with TemporaryDirectory(prefix="wrblind-entry-base-") as temp:
            world = World(Path(temp), day, "12:05", health=False, required=False)
            restore(world, source["snapshot"])
            # Re-run the entry timeline with the same HEAD commit but without
            # inheriting the HEAD watchdog's result marker.
            marker = world.picks.parent / "health_state/pick_entry_check.json"
            marker.unlink(missing_ok=True)
            with world.switched_on():
                world.slate(day, ("13:05", "20:05"), (.98,), doubleheader=True)
                world.crosswalk = {1: 100}
                world.pending = [{"roundId": 1, "playerId": 1,
                                  "unitId": 77 if source["entered"] == 700 else 88}]
                from bts.entry_receipt import units_snapshot_dir
                target = units_snapshot_dir(world.picks)
                target.mkdir(parents=True)
                (target / f"{day}T130000Z.json.gz").write_bytes(gzip.compress(json.dumps({
                    "units": [{"id": 77, "feedId": 700, "roundId": 1},
                              {"id": 88, "feedId": 701, "roundId": 1}],
                }).encode()))
                world.invoke("check-pick-entered", "--picks-dir", world.picks,
                             "--expected-username", ACCOUNT, "--dm-recipient", RECIPIENT)
            rows.append({"entered": source["entered"], "events": world.events,
                         "snapshot": world.snapshot()})
    (folder / "entry-rollback-results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
