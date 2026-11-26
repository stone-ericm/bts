"""Replay the three actual closing-test pick/decision snapshots through the CLI."""
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from harness import DAY, World

HERE = Path(__file__).resolve().parent
FIXTURES = (HERE.parents[1] / "docs" / "reviews" / "2026-11-18-whole-range" /
            "closing-test" / "codex-blind" / "evidence")
results = {}
with TemporaryDirectory(dir=HERE) as scratch:
    for label, feed in (("wrong_game", 102), ("right_game", 101), ("unbound", None)):
        with gzip.open(FIXTURES / f"interactions-entry-{label}.json.gz", "rt") as file:
            events = json.load(file)
        snapshot = next(e[3] for e in reversed(events) if e[1] == "snapshot")
        world = World(Path(scratch) / label)
        pick_file = f"{DAY}.json"
        decision = world.picks / DAY / "decision.json"
        (world.picks / pick_file).write_text(json.dumps(snapshot[f"data/picks/{pick_file}"]))
        decision.parent.mkdir()
        decision.write_text(json.dumps(snapshot[f"data/picks/{DAY}/decision.json"]))
        batter = snapshot[f"data/picks/{pick_file}"]["pick"]["batter_id"]
        world.capture([{"id": 77, "roundId": 1, "feedId": feed}], "2027-03-24T210000Z")
        actual = []
        for at in ("18:30", "18:45", "19:00"):
            pending = [] if at == "18:30" else [{"roundId": 1, "unitId": 77, "playerId": 1, "number": 1}]
            actual.append(world.check(at, pending, crosswalk={1: batter}))
        results[label] = {"runs": actual, "morning": world.morning(), "health": world.health()}
print(json.dumps(results, indent=2))
