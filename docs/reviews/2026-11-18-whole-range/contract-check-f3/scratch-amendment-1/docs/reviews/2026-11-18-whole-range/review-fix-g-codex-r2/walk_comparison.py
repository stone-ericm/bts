"""Account for every changed record and warning in the full fresh season walk."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "scripts/audit/season_walk/results"


def by_day_and_state(rows):
    result = {(row["day"], row["state"]): row for row in rows}
    assert len(result) == len(rows)
    return result


def warning_by_state(problems):
    result = {tuple(problem.split(":", 1)[0].split()): problem for problem in problems}
    assert len(result) == len(problems)
    return result


if __name__ == "__main__":
    fresh_json = (HERE / "walk.json").read_bytes()
    assert fresh_json == gzip.decompress((RESULTS / "walk-g.json.gz").read_bytes())
    before = json.loads(gzip.decompress((RESULTS / "walk-bc143131.json.gz").read_bytes()))
    after = json.loads(fresh_json)
    saved_body, saved_summary = (RESULTS / "walk-g.txt").read_bytes().split(b"\n\nBaseline comparison")
    assert saved_body + b"\n" == (HERE / "walk.txt").read_bytes()
    assert b"252 skip-to-single" in saved_summary and b"612 warnings" in saved_summary
    assert before["boundary"] == after["boundary"] and not after["boundary"]["problems"]
    raw_before, raw_after = by_day_and_state(before["records"]), by_day_and_state(after["records"])
    assert raw_before.keys() == raw_after.keys() and len(raw_before) == 4599
    before_warnings = warning_by_state(before["problems"])
    after_warnings = warning_by_state(after["problems"])
    assert len(before_warnings) == 841 and len(after_warnings) == 612
    assert all(before_warnings[key] == warning for key, warning in after_warnings.items())
    with (HERE / "affected-states.csv").open() as source:
        affected = {tuple(int(row[field]) for field in ("streak", "days_left", "saver", "bin")):
                    row["guarded_action"] for row in csv.DictReader(source)}
    with np.load(ROOT / "data/models/mdp_policy.npz") as artifact:
        table, edges = artifact["policy_table"], artifact["boundaries"]
    counts = Counter()
    for key, record in raw_after.items():
        original = raw_before[key]
        kind, streak, _, _, probability = key[1].split("/")
        s, d, v = int(streak), record["days_effective"], int(record["saver_available"])
        q = sum(float(probability) >= edge for edge in edges)
        action = affected.get((s, d, v, q)) if record["objective"] == "reach57" else None
        if action:
            assert original["action"] == "skip" and original["named"] and original["skip_dm"]
            assert record == original | {
                "action": "single", "named": False, "batter": "Batter 0", "chosen_on_kind": kind,
                "skip_dm": None, "skip_bar": None, "pick_bar": None, "best_p": None,
            }
            assert action == "single"
            assert (key in before_warnings) == record["in_season"] and key not in after_warnings
            counts["flagged" if key in before_warnings else "preseason"] += 1
        else:
            assert record == original, key
        if key in after_warnings:
            assert record["in_season"] and record["action"] == "skip" and record["objective"] == "reach57"
            assert (table[s, d, v] == 0).all()
            counts["all-skip warnings"] += 1
    assert counts == {"flagged": 229, "preseason": 23, "all-skip warnings": 612}
    print("4599 rows, 252 skip-to-single, 229 previously warned, 612 all-skip warnings, zero other changes")
    print("Same boundary checks, exact decompressed JSON, exact stdout before saved summary")
