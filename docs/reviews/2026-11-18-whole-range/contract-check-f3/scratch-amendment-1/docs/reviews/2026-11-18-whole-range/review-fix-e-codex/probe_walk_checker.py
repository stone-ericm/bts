"""Check the actual walk's text checker with a real in-season K3 record."""
import gzip
import json
from collections import Counter
from pathlib import Path

from scripts.audit.season_walk import walk

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "scripts/audit/season_walk/results"


with gzip.open(RESULTS / "walk-bc143131.json.gz", "rt") as file:
    before = json.load(file)
with gzip.open(RESULTS / "walk-e.json.gz", "rt") as file:
    committed = json.load(file)
after = json.loads((HERE / "walk.json").read_text())
assert before["runs"] == after["runs"] == 4599 and committed == after
assert not after["problems"] and not walk.check(after["records"])
assert before["by_day"] == after["by_day"] and before["boundary"] == after["boundary"]
changed_text = {"best_p", "skip_dm"}
for left, right in zip(before["records"], after["records"], strict=True):
    assert {k: v for k, v in left.items() if k not in changed_text} == {
        k: v for k, v in right.items() if k not in changed_text
    }
flags = {line.split(": skip message", 1)[0] for line in before["problems"] if ": skip message" in line}
counts = Counter()
for row in after["records"]:
    if f"{row['day']} {row['state']}" not in flags:
        continue
    dm = row["skip_dm"]
    assert "below the" not in dm
    reason = "K3" if "at or above the" in dm else "K4"
    if reason == "K4":
        assert "No probability comparison is available for this check." in dm
    counts[reason] += 1
assert counts == {"K3": 229, "K4": 612}, counts
print("4,599 fresh records checked:", counts)

point = next(row for row in after["records"] if row["in_season"] and "at or above" in (row["skip_dm"] or ""))
planted = point | {"skip_dm": point["skip_dm"].replace("at or above", "below", 1)}
assert not walk.check([point])
problems = walk.check([planted])
assert any("skip message" in problem for problem in problems), problems
print("In-season planted result:", problems)
