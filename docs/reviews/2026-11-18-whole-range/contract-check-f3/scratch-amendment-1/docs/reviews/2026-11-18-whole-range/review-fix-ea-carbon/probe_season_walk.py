"""Probe item 5: Run season walk end-to-end, verify 4,599 runs against
walk-bc143131.json.gz (actions, bars, day counts, by_day, boundary),
verify 0 problems, 229 K3 + 612 K4 among the 841 formerly flagged runs,
and confirm a false 'below the' planted in an in-season K3 record is reported.
"""
from __future__ import annotations

import copy
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit.season_walk import walk  # noqa: E402


def main() -> None:
    out_json = HERE / "walk_fresh.json"
    rc = walk.main(["--out", str(out_json)])
    assert rc == 0, f"walk.main returned non-zero: {rc}"

    fresh = json.loads(out_json.read_text())
    results_dir = REPO / "scripts/audit/season_walk/results"
    with gzip.open(results_dir / "walk-bc143131.json.gz", "rt", encoding="utf-8") as fh:
        baseline = json.load(fh)
    with gzip.open(results_dir / "walk-e.json.gz", "rt", encoding="utf-8") as fh:
        committed_e = json.load(fh)

    assert fresh["runs"] == baseline["runs"] == committed_e["runs"] == 4599
    assert fresh["problems"] == []
    assert walk.check(fresh["records"]) == []
    assert fresh["by_day"] == baseline["by_day"] == committed_e["by_day"]
    assert fresh["boundary"] == baseline["boundary"] == committed_e["boundary"]

    formerly_flagged = {
        p.split(": skip message", 1)[0]
        for p in baseline["problems"]
        if ": skip message" in p
    }
    assert len(formerly_flagged) == 841

    cmp_fields = ("day", "state", "action", "pick_bar", "expected_days_left", "days_left", "days_effective")
    case_counts: Counter[str] = Counter()
    in_season_k3_sample = None

    for base_rec, fresh_rec, e_rec in zip(baseline["records"], fresh["records"], committed_e["records"], strict=True):
        assert [base_rec[f] for f in cmp_fields] == [fresh_rec[f] for f in cmp_fields]
        assert fresh_rec == e_rec
        key = f"{base_rec['day']} {base_rec['state']}"
        if key in formerly_flagged:
            dm_txt = fresh_rec["skip_dm"] or ""
            assert "below the" not in dm_txt, (key, dm_txt)
            if base_rec["skip_bar"] is None:
                assert "No probability comparison is available for this check." in dm_txt, (key, dm_txt)
                case_counts["K4"] += 1
            else:
                assert "at or above the" in dm_txt, (key, dm_txt)
                case_counts["K3"] += 1
                if fresh_rec["in_season"] and in_season_k3_sample is None:
                    in_season_k3_sample = fresh_rec

    assert dict(case_counts) == {"K3": 229, "K4": 612}, case_counts
    assert in_season_k3_sample is not None

    # Plant false 'below the' comparison into a real in-season K3 record
    planted_in_season = copy.deepcopy(in_season_k3_sample)
    planted_in_season["skip_dm"] = planted_in_season["skip_dm"].replace(
        "at or above the", "below the"
    )
    planted_problems = walk.check([planted_in_season])
    assert len(planted_problems) == 1 and "skip message calls the best" in planted_problems[0], planted_problems

    # Also note out-of-season K3 behavior (walk.check scopes skip-message check to in_season)
    preseason_k3 = next(
        r for r in fresh["records"]
        if not r["in_season"] and r["action"] == "skip" and r.get("skip_bar") is not None and r["best_p"] >= r["skip_bar"]
    )
    planted_preseason = copy.deepcopy(preseason_k3)
    planted_preseason["skip_dm"] = planted_preseason["skip_dm"].replace("at or above the", "below the")
    preseason_problems = walk.check([planted_preseason])

    summary = {
        "runs": fresh["runs"],
        "problems_count": len(fresh["problems"]),
        "matches_walk_bc143131_actions_bars_days": True,
        "matches_walk_e_exact": True,
        "former_841_breakdown": dict(case_counts),
        "planted_in_season_key": f"{planted_in_season['day']} {planted_in_season['state']}",
        "planted_in_season_reported": planted_problems,
        "planted_preseason_reported_count": len(preseason_problems),
    }
    (HERE / "probe_season_walk_summary.json").write_text(json.dumps(summary, indent=2))
    print("probe_season_walk: ALL CHECKS PASSED:", json.dumps(summary))


if __name__ == "__main__":
    main()
