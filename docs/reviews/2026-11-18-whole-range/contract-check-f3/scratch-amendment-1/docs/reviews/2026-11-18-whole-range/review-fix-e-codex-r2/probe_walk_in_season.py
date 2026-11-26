"""Plant the false comparison inside the season where the checker applies."""
import json

from fixtures import HERE
from scripts.audit.season_walk import walk

records = json.loads((HERE / "walk-head.json").read_text())["records"]
example = next(record for record in records if record["in_season"]
               and "at or above the" in (record.get("skip_dm") or ""))
wrong = example | {"skip_dm": example["skip_dm"].replace("at or above the", "below the", 1)}
assert walk.check([example]) == []
failure = walk.check([wrong])
assert any("skip message" in problem for problem in failure), (wrong, failure)
print("Clean record passed; planted comparison failed:", failure)
