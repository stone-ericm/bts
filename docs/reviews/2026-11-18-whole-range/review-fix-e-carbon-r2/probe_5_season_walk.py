"""Probe for Item 5 (The season walk):
Runs scripts.audit.season_walk.walk end to end on HEAD, verifies 4,599 runs, 0 problems,
equality of actions/bars/day counts/by_day/boundary against walk-bc143131.json.gz and walk-e.json.gz,
229 K3 and 612 K4 among the 841 formerly flagged runs, and verifies planted false 'below the' text
causes walk.check to fail on both K3 and K4 in-season records.
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT_DIR = Path(__file__).resolve().parent

from scripts.audit.season_walk import walk
from tests.scripts.season_walk.test_season_walk import (
    test_check_flags_a_skip_message_that_calls_a_clearing_best_below_its_bar,
)


def main() -> None:
    walk_out = OUT_DIR / "walk-head.json"
    rc = walk.main(["--out", str(walk_out)])
    assert rc == 0, f"walk.main returned {rc}"

    fresh = json.loads(walk_out.read_text())
    results_dir = REPO / "scripts/audit/season_walk/results"
    with gzip.open(results_dir / "walk-bc143131.json.gz", "rt") as f:
        baseline = json.load(f)
    with gzip.open(results_dir / "walk-e.json.gz", "rt") as f:
        committed_e = json.load(f)

    assert fresh["runs"] == baseline["runs"] == committed_e["runs"] == 4599
    assert fresh["problems"] == [] and walk.check(fresh["records"]) == []
    assert fresh["by_day"] == baseline["by_day"] == committed_e["by_day"]
    assert fresh["boundary"] == baseline["boundary"] == committed_e["boundary"]
    assert fresh == committed_e

    former_flags = {
        entry.split(": skip message", 1)[0]
        for entry in baseline["problems"]
        if ": skip message" in entry
    }
    assert len(former_flags) == 841

    compare_fields = (
        "day", "state", "in_season", "expected_days_left", "action", "source",
        "objective", "named", "days_left", "days_effective", "batter", "double",
        "no_pick_reason", "saver_available", "chosen_on_kind", "reason", "pick_bar",
    )
    former_counts = Counter()
    sample_k3 = None
    sample_k4 = None
    for b_rec, f_rec in zip(baseline["records"], fresh["records"], strict=True):
        for k in compare_fields:
            assert b_rec[k] == f_rec[k], f"Mismatch on {k} at {b_rec['day']} {b_rec['state']}: {b_rec[k]} vs {f_rec[k]}"
        key = f"{b_rec['day']} {b_rec['state']}"
        if key in former_flags:
            dm_txt = f_rec["skip_dm"]
            assert "below the" not in dm_txt
            if b_rec["skip_bar"] is None:
                assert "No probability comparison is available for this check." in dm_txt
                former_counts["K4"] += 1
                if sample_k4 is None and f_rec["in_season"]:
                    sample_k4 = dict(f_rec)
            else:
                assert "at or above the" in dm_txt
                former_counts["K3"] += 1
                if sample_k3 is None and f_rec["in_season"]:
                    sample_k3 = dict(f_rec)

    assert dict(former_counts) == {"K3": 229, "K4": 612}

    # Planted false-text checks
    test_check_flags_a_skip_message_that_calls_a_clearing_best_below_its_bar()
    assert sample_k3 is not None and sample_k4 is not None
    planted_k3 = sample_k3 | {
        "skip_dm": sample_k3["skip_dm"].replace("at or above the", "below the")
    }
    planted_k4 = sample_k4 | {
        "skip_dm": "BTS 2027-07-15: No pick yet — model's best is Batter 0 (T0) 96%, below the ~80% bar."
    }
    probs_k3 = walk.check([planted_k3])
    probs_k4 = walk.check([planted_k4])
    assert any("skip message" in p for p in probs_k3), probs_k3
    assert any("skip message" in p for p in probs_k4), probs_k4

    summary = {
        "runs": fresh["runs"],
        "problems": len(fresh["problems"]),
        "matches_walk_bc143131_actions_bars_days": True,
        "matches_walk_e_exact": True,
        "former_flagged_total": len(former_flags),
        "former_flagged_k3": former_counts["K3"],
        "former_flagged_k4": former_counts["K4"],
        "planted_false_k3_caught": probs_k3,
        "planted_false_k4_caught": probs_k4,
    }
    out_path = OUT_DIR / "probe_5_results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
