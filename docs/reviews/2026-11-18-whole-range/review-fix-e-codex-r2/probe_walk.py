"""Compare the fresh walk to both references; then plant a bad comparison."""
import gzip
import json
from collections import Counter

from fixtures import HERE, ROOT
from scripts.audit.season_walk import walk

results = ROOT / "scripts/audit/season_walk/results"
with gzip.open(results / "walk-bc143131.json.gz", "rt") as stream:
    base = json.load(stream)
with gzip.open(results / "walk-e.json.gz", "rt") as stream:
    saved_head = json.load(stream)
head = json.loads((HERE / "walk-head.json").read_text())
assert head == saved_head
assert base["runs"] == head["runs"] == 4599
assert all(base[key] == head[key] for key in ("by_day", "boundary"))
assert head["problems"] == [] and walk.check(head["records"]) == []
old_flags = {line.split(": skip message", 1)[0] for line in base["problems"] if ": skip message" in line}
fields = ("day", "state", "action", "pick_bar", "expected_days_left", "days_left", "days_effective",
          "batter", "double", "saver_available", "chosen_on_kind", "objective", "source", "reason", "no_pick_reason")
counts = Counter()
for old, new in zip(base["records"], head["records"], strict=True):
    assert all(old[key] == new[key] for key in fields)
    if f"{old['day']} {old['state']}" in old_flags:
        assert "below the" not in new["skip_dm"]
        prefix = "at or above the" if old["skip_bar"] is not None else "No probability comparison"
        assert prefix in new["skip_dm"]
        counts[prefix] += 1
assert counts == {"at or above the": 229, "No probability comparison": 612}
example = next(record for record in head["records"] if "at or above the" in (record.get("skip_dm") or ""))
wrong = example | {"skip_dm": example["skip_dm"].replace("at or above the", "below the", 1)}
assert walk.check([example]) == []
failure = walk.check([wrong])
assert any("skip message" in problem for problem in failure)
print("4599 same actions, bars, day counts and records; 229 K3, 612 K4;", failure)
