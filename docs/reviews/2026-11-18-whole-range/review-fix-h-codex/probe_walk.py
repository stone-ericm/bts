"""Check the full season walk's output and plant wrong actions in real rows."""
from collections import Counter
import gzip
import json
from pathlib import Path

from scripts.audit.season_walk import walk

HERE = Path(__file__).resolve().parent
RESULTS = Path("scripts/audit/season_walk/results")


def saved(name):
    with gzip.open(RESULTS / f"walk-{name}.json.gz", "rt") as stream:
        return json.load(stream)


def main():
    current = json.loads((HERE / "review-walk.json").read_text())
    base = saved("g")
    assert current == saved("h")
    extra, = set(current["states"]) - set(base["states"])
    assert extra == f"dated/{walk.LONGEST_STREAK_FROM}/{walk.LONGEST_STREAK_FROM}/used/0.96"
    original = [r for r in current["records"] if r["state"] != extra]
    added = [r for r in current["records"] if r["state"] == extra]
    assert len(original) == 4599 and original == base["records"]
    assert len(current["problems"]) == 612 and current["problems"] == base["problems"]
    assert Counter(int(p.split("/")[1]) for p in current["problems"]) == {50: 141, 55: 157, 56: 314}
    assert current["boundary"] == base["boundary"]
    assert len(added) == len({r["day"] for r in added}) == 219
    assert Counter(r["action"] for r in added) == {"single": 207, "double": 3, "skip": 9}
    assert sum(not r["in_season"] and r["action"] == "single" for r in added) == 23
    assert [r["day"] for r in added if r["action"] == "double"] == [
        "2027-09-24", "2027-09-25", "2027-09-26"]
    for r in added:
        positive = r["days_left"] > 0
        assert r["objective"] == ("longest_streak" if positive else "reach57")
        if positive:
            assert r["chosen_on_valid"] and r["batter"] == "Batter 0"
            assert r["double"] == ("Batter 1" if r["action"] == "double" else None)
    assert walk.check(added) == []
    early = next(r for r in added if r["day"] == "2027-07-15")
    finale = next(r for r in added if r["day"] == "2027-09-26")
    for planted, expected in ((early | {"action": "skip", "named": True}, "single"),
                              (finale | {"action": "single", "double": None}, "double")):
        planted["objective"] = "reach57"
        messages = walk.check([planted])
        assert any(f"action {planted['action']!r}, expected {expected!r}" in m for m in messages), messages
        print("plant", planted["day"], messages)
    missing = finale | {"days_effective": None, "source": None, "artifacts": None}
    print("checker with missing effective days, source and artifact provenance:", walk.check([missing]))
    print("4,599 original rows and 612 warnings exactly match G, "
          "219 added rows with 210 valid plays and nine post-finale skips")


if __name__ == "__main__":
    main()
