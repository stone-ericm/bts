"""Item 8 regression probe: run the full 2027 season walk and verify record-by-record against walk-bc143131 and walk-g."""
from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import numpy as np

REVIEW_DIR = REPO / ".codex-review" / "wr-g"
WALK_RESULTS = REPO / "scripts" / "audit" / "season_walk" / "results"
BASE_WALK_GZ = WALK_RESULTS / "walk-bc143131.json.gz"
AUTHOR_WALK_GZ = WALK_RESULTS / "walk-g.json.gz"
CSV_PATH = REPO / "docs" / "reviews" / "2026-11-18-whole-range" / "contract-g" / "affected-states.csv"


def main() -> None:
    fresh_json = REVIEW_DIR / "fresh_walk.json"
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.audit.season_walk.walk", "--out", str(fresh_json)],
        cwd=REPO, capture_output=True, text=True, timeout=110,
    )
    # Exit code 1 with the 612 remaining all-skip warnings is expected per contract
    assert proc.returncode == 1, f"unexpected walk exit code {proc.returncode}: {proc.stdout[:500]}"

    fresh = json.loads(fresh_json.read_text())
    base = json.loads(gzip.decompress(BASE_WALK_GZ.read_bytes()))
    author = json.loads(gzip.decompress(AUTHOR_WALK_GZ.read_bytes()))

    # 1. Exact equality with author's saved walk-g.json.gz
    assert fresh == author, "fresh season walk differs from committed results/walk-g.json.gz"

    # 2. Load CSV & table for independent verification
    affected = {}
    with CSV_PATH.open() as fh:
        for row in csv.DictReader(fh):
            affected[(int(row["streak"]), int(row["days_left"]), bool(int(row["saver"])), int(row["bin"]))] = row["guarded_action"]

    with np.load(REPO / "data" / "models" / "mdp_policy.npz") as data:
        table = data["policy_table"]
        bounds = data["boundaries"].tolist()

    def bin_of(p: float) -> int:
        q = 0
        for idx, edge in enumerate(bounds):
            if p >= edge:
                q = idx + 1
        return min(q, len(bounds))

    assert fresh["runs"] == base["runs"] == 4599
    assert fresh["boundary"] == base["boundary"]
    assert fresh["boundary"]["problems"] == []

    base_map = {(r["day"], r["state"]): r for r in base["records"]}
    fresh_map = {(r["day"], r["state"]): r for r in fresh["records"]}
    assert set(base_map.keys()) == set(fresh_map.keys()) and len(fresh_map) == 4599

    base_problem_keys = {p.split(": ", 1)[0] for p in base["problems"]}
    fresh_problem_keys = {p.split(": ", 1)[0] for p in fresh["problems"]}
    assert len(base["problems"]) == 841
    assert len(fresh["problems"]) == 612

    changed_total = 0
    changed_flagged_in_season = 0
    changed_unflagged_preseason = 0
    other_diffs = []

    for key, b_rec in base_map.items():
        f_rec = fresh_map[key]
        day_s, state_s = key
        kind, s_str, _best_str, _saver_str, p_str = state_s.split("/")
        s = int(s_str)
        p = float(p_str)
        q = bin_of(p)
        d_eff = b_rec["days_effective"]
        v = bool(b_rec["saver_available"])

        csv_hit = affected.get((s, d_eff, v, q)) if b_rec["objective"] == "reach57" else None

        if b_rec == f_rec:
            assert csv_hit is None, f"missed CSV match on {key}"
            continue

        if csv_hit is not None:
            assert csv_hit == "single", (key, csv_hit)
            assert b_rec["action"] == "skip" and f_rec["action"] == "single"
            # Check exact changed fields per contract
            expected_f = dict(b_rec)
            expected_f.update(
                action="single",
                named=False,
                batter="Batter 0",
                double=None,
                chosen_on_kind=kind,
                pick_bar=None,
                skip_dm=None,
                skip_bar=None,
                best_p=None,
            )
            assert f_rec == expected_f, f"unexpected field diff on changed record {key}: {f_rec} vs {expected_f}"
            changed_total += 1
            where = f"{day_s} {state_s}"
            if where in base_problem_keys:
                assert b_rec["in_season"] is True
                assert where not in fresh_problem_keys
                changed_flagged_in_season += 1
            else:
                assert b_rec["in_season"] is False
                changed_unflagged_preseason += 1
        else:
            other_diffs.append(key)

    assert other_diffs == [], f"unexpected record diffs: {other_diffs}"
    assert (changed_total, changed_flagged_in_season, changed_unflagged_preseason) == (252, 229, 23)

    # Verify all 612 remaining warnings are in-season states where EVERY bin skips in the table
    for prob in fresh["problems"]:
        where = prob.split(": ", 1)[0]
        day_s, state_s = where.split(" ", 1)
        f_rec = fresh_map[(day_s, state_s)]
        assert f_rec["in_season"] is True
        assert f_rec["action"] == "skip" and f_rec["objective"] == "reach57"
        s = int(state_s.split("/")[1])
        d_eff = f_rec["days_effective"]
        v = int(bool(f_rec["saver_available"]))
        row_actions = [int(x) for x in table[min(s, 56), d_eff, v]]
        assert row_actions == [0, 0, 0, 0, 0], f"warning in non-all-skip state {where}: {row_actions}"

    summary = {
        "status": "PASS",
        "exit_code": proc.returncode,
        "runs": fresh["runs"],
        "equals_walk_g_json_gz": True,
        "changed_skip_to_single": changed_total,
        "changed_formerly_flagged_in_season": changed_flagged_in_season,
        "changed_unflagged_preseason": changed_unflagged_preseason,
        "remaining_warnings": len(fresh["problems"]),
        "all_remaining_warnings_all_skip": True,
        "other_record_diffs": len(other_diffs),
    }
    (REVIEW_DIR / "probe_08_season_walk.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
