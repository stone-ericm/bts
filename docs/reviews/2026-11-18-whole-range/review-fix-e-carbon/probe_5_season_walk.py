"""Probe 5: Run the 2027 season walk end-to-end on HEAD (4,599 runs), verify equivalence to
`results/walk-bc143131.json.gz` and `results/walk-e.json.gz`, verify 229 K3 and 612 K4,
and verify the planted-false-text check fails when false text is planted.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from bts import scheduler
from scripts.audit.season_walk import walk


def main() -> None:
    out_path = REPO / ".carbon-review/wr-e/walk-head-run.json"
    rc = walk.main(["--out", str(out_path)])
    assert rc == 0, f"walk.main returned non-zero exit code: {rc}"

    live = json.loads(out_path.read_text())
    results_dir = REPO / "scripts/audit/season_walk/results"
    with gzip.open(results_dir / "walk-bc143131.json.gz", "rt") as fh:
        baseline = json.load(fh)
    with gzip.open(results_dir / "walk-e.json.gz", "rt") as fh:
        saved_e = json.load(fh)

    assert live["runs"] == baseline["runs"] == saved_e["runs"] == 4599
    assert live["problems"] == [] and walk.check(live["records"]) == []
    assert live["by_day"] == baseline["by_day"] == saved_e["by_day"]
    assert live["boundary"] == baseline["boundary"] == saved_e["boundary"]
    assert live == saved_e, "Live walk output differs from committed walk-e.json.gz!"

    former_flags = {
        entry.split(": skip message", 1)[0]
        for entry in baseline["problems"]
        if ": skip message" in entry
    }
    assert len(former_flags) == 841

    counts = Counter()
    fields = ("day", "state", "action", "pick_bar", "expected_days_left", "days_left", "days_effective")
    for b_rec, l_rec in zip(baseline["records"], live["records"], strict=True):
        assert [b_rec[k] for k in fields] == [l_rec[k] for k in fields]
        key = f"{b_rec['day']} {b_rec['state']}"
        if key in former_flags:
            dm = l_rec["skip_dm"]
            assert "below the" not in dm
            if "at or above the" in dm:
                counts["K3"] += 1
            elif "No probability comparison is available for this check." in dm:
                counts["K4"] += 1
            else:
                raise AssertionError(f"Unexpected DM on former flag {key}: {dm}")
    assert counts == {"K3": 229, "K4": 612}, f"Unexpected K3/K4 counts: {counts}"

    # Verify planted false text fails both via walk_one + walk.check AND via the known-answer test logic
    with patch.object(
        scheduler,
        "format_skip_dm",
        lambda d, s: f"BTS {d}: No pick yet — model's best is {s['best_batter']} ({s['best_team']}) {s['best_p']:.1%}, below the streak-{s['streak']} bar of 82.5%.",
    ):
        planted_k3 = walk.walk_one(date(2027, 8, 30), walk.State("dated", 56, 56, "used", 0.99))
        planted_k4 = walk.walk_one(date(2027, 7, 15), walk.State("dated", 56, 56, "used", 0.96))
    probs_k3 = walk.check([planted_k3])
    probs_k4 = walk.check([planted_k4])
    assert any("skip message" in p for p in probs_k3), f"Planted K3 false text not caught: {probs_k3}"
    assert any("skip message" in p for p in probs_k4), f"Planted K4 false text not caught: {probs_k4}"

    print(json.dumps({
        "runs": live["runs"],
        "problems": len(live["problems"]),
        "matches_saved_walk_e_exactly": live == saved_e,
        "former_flagged_breakdown": dict(counts),
        "planted_false_text_caught": True,
    }, indent=2))


if __name__ == "__main__":
    main()
