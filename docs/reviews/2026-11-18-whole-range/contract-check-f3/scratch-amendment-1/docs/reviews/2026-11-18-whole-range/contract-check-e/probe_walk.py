"""Compare the offline base walk with the committed closing-test fixture."""
import gzip
import json
from collections import Counter
from pathlib import Path

from scripts.audit.season_walk import walk

saved = Path("scripts/audit/season_walk/results/walk-bc143131.json.gz")
with gzip.open(saved, "rt") as file:
    expected = json.load(file)
records = walk.walk()
keys = ("day", "state", "action", "pick_bar", "days_left", "days_effective")
actual_fields = [tuple(r.get(k) for k in keys) for r in records]
saved_fields = [tuple(r.get(k) for k in keys) for r in expected["records"]]
problems = walk.check(records) + walk.boundary_checks()["problems"]
flagged = {(p.split(": ", 1)[0]) for p in problems if "skip message" in p}
print(len(records), "records, parity", actual_fields == saved_fields, len(problems), "problems",
      Counter((r.get("skip_bar") is None, bool(r.get("reason"))) for r in records
              if f"{r['day']} {r['state']}" in flagged))
