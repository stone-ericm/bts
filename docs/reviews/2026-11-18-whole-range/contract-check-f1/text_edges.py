"""Check wrong-game text with stale pitch times and incomplete saved metadata."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from harness import RIGHT, WRONG, World, row

results = {}
with TemporaryDirectory(dir=Path(__file__).parent) as temp:
    root = Path(temp)
    for label, names, elapsed, delivery in (("stale_time", ..., 0, "delivered"),
                                             ("rounds_to_zero", ..., 50, "delivered"),
                                             ("null_name", None, 0, "delivered"),
                                             ("private", ..., 0, "private_locked"),
                                             ("legacy", ..., 0, None)):
        w = World(root / label)
        later = (1020, 102, f"{w.day}T20:10:00-04:00") if label == "stale_time" else None
        w.commit(dd=later, delivery=delivery)
        if names is not ...:
            path = w.picks / f"{w.day}.json"
            pick = json.loads(path.read_text())
            pick["pick"]["batter_name"] = names
            path.write_text(json.dumps(pick))
        if later:
            w.capture([{**RIGHT, "startDate": f"{w.day}T19:10:00-04:00"},
                       {**WRONG, "id": 78, "startDate": f"{w.day}T18:25:00-04:00"}])
        else:
            w.capture([WRONG])
        results[label] = w.check("19:04" if label == "rounds_to_zero" else "18:30",
                                 [row(), row(player=2, number=2)] if later else [row()], elapsed=elapsed)
print(json.dumps(results, indent=2))
