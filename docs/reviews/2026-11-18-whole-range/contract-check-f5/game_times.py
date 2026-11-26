"""Inspect only the authorized raw seasons and checked-in processed schemas."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RAW = Path("/Users/eric/projects/bts/data/raw")
ET = ZoneInfo("America/New_York")

def game_metadata(path):
    with path.open(encoding="utf-8") as stream:
        prefix = stream.read(262144)
        start = re.search(r'"gameData"\s*:\s*(\{)', prefix)
        if not start:
            raise ValueError("gameData missing")
        try:
            obj, end = json.JSONDecoder().raw_decode(prefix, start.start(1))
        except json.JSONDecodeError:
            prefix += stream.read()
            obj, end = json.JSONDecoder().raw_decode(prefix, start.start(1))
        return obj

for season in range(2021, 2026):
    root = RAW / str(season)
    if not root.is_dir():
        print(season, "raw directory absent")
        continue
    types, date_relations, statuses = Counter(), Counter(), Counter()
    examined, errors, rows = 0, [], []
    for file in sorted(root.glob("*.json")):
        examined += 1
        try:
            data = game_metadata(file)
            game, time = data["game"], data["datetime"]
            types[game["type"]] += 1
            if game["type"] != "R":
                continue
            start = datetime.fromisoformat(time["dateTime"].replace("Z", "+00:00")).astimezone(ET)
            official = time["officialDate"]
            status = data.get("status", {}).get("detailedState")
            statuses[status] += 1
            relation = (start.date() - datetime.fromisoformat(official).date()).days
            date_relations[relation] += 1
            rows.append((start.strftime("%H:%M:%S"), start.isoformat(), official, game["pk"], status, file.name))
        except (KeyError, ValueError, UnicodeError, OSError) as exc:
            errors.append((file.name, type(exc).__name__, str(exc)))
    print(season, "files", examined, "types", types, "errors", errors[:10], "total errors", len(errors),
          "date differences", date_relations, "statuses", statuses)
    print("latest", sorted(rows, reverse=True)[:20])
    for name, matches in (
        ("late", [r for r in rows if r[0] > "22:40:00"]),
        ("next_et_date", [r for r in rows if r[1][:10] > r[2]]),
        ("before_06", [r for r in rows if r[0] < "06:00:00"]),
    ):
        print(name, len(matches), sorted(matches))
    (HERE / f"game-times-{season}.json").write_text(json.dumps(sorted(rows), indent=2) + "\n")

for path in sorted((REPO / "data").rglob("*.parquet")):
    names = pq.read_schema(path).names
    times = [name for name in names if re.search(r"time|pitch|start", name, re.I)]
    plausible = [name for name in times if re.search(r"game.*(time|start)|first.*pitch|date.?time|scheduled", name, re.I)]
    if plausible:
        print("processed candidate", path.relative_to(REPO), plausible)
print("Processed schema scan complete.")
