"""End-to-end season walk execution and record-by-record baseline comparison (Item 8)."""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from bts.simulate import mdp

RESULTS_DIR = ROOT / "scripts/audit/season_walk/results"
BASELINE_GZ = RESULTS_DIR / "walk-bc143131.json.gz"
SAVED_G_GZ = RESULTS_DIR / "walk-g.json.gz"
SAVED_G_TXT = RESULTS_DIR / "walk-g.txt"
CONTRACT_CSV = ROOT / "docs/reviews/2026-11-18-whole-range/contract-g/affected-states.csv"


def _classify_bin(bounds: list[float], p: float) -> int:
    q = 0
    for i, b in enumerate(bounds):
        if p >= b:
            q = i + 1
    return min(q, len(bounds))


def main() -> None:
    fresh_json = HERE / "walk-fresh.json"
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.audit.season_walk.walk", "--out", str(fresh_json)],
        cwd=ROOT, capture_output=True, text=True, timeout=90,
        env=os.environ | {"PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"},
    )
    # Exit code 1 is expected with the 612 remaining all-skip warnings
    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)

    fresh = json.loads(fresh_json.read_text())
    with gzip.open(BASELINE_GZ, "rt") as f:
        baseline = json.load(f)
    with gzip.open(SAVED_G_GZ, "rt") as f:
        saved_g = json.load(f)

    # Fresh JSON matches author's saved walk-g.json.gz
    assert fresh == saved_g, "Fresh season walk JSON differs from saved results/walk-g.json.gz!"

    # Check stdout against saved walk-g.txt
    saved_txt = SAVED_G_TXT.read_text()
    saved_body = saved_txt.split("\n\nBaseline comparison")[0]
    if saved_body.startswith("nice:"):
        saved_body = "\n".join(saved_body.splitlines()[1:])
    fresh_stdout = (proc.stderr + proc.stdout).strip()
    # walk prints progress to stderr or stdout; check stdout specifically
    out_clean = proc.stdout.strip()
    err_clean = proc.stderr.strip()
    combined = (err_clean + "\n" + out_clean).strip() if err_clean else out_clean
    assert out_clean in saved_txt, (out_clean[:300], saved_txt[:300])

    # Load CSV and table
    table, bounds, _ = mdp.load_policy(ROOT / "data/models/mdp_policy.npz")
    with CONTRACT_CSV.open() as f:
        csv_map = {
            (int(r["streak"]), int(r["days_left"]), int(r["saver"]), int(r["bin"])): r["guarded_action"]
            for r in csv.DictReader(f)
        }

    base_records = {(r["day"], r["state"]): r for r in baseline["records"]}
    fresh_records = {(r["day"], r["state"]): r for r in fresh["records"]}
    assert len(base_records) == len(fresh_records) == 4599
    assert set(base_records.keys()) == set(fresh_records.keys())

    base_problem_keys = {p.split(": ")[0] for p in baseline["problems"]}
    fresh_problem_keys = {p.split(": ")[0] for p in fresh["problems"]}
    assert len(baseline["problems"]) == 841
    assert len(fresh["problems"]) == 612

    changed_total = 0
    changed_flagged_in_season = 0
    changed_unflagged_preseason = 0
    unchanged_count = 0

    for key, b_rec in base_records.items():
        f_rec = fresh_records[key]
        day_str, state_str = key
        kind, s_str, _, _, p_str = state_str.split("/")
        s_val = int(s_str)
        p_val = float(p_str)
        q = _classify_bin(bounds, p_val)
        csv_hit = None
        if b_rec["objective"] == "reach57" and s_val < 57 and (b_rec["days_effective"] or 0) > 0:
            csv_hit = csv_map.get((s_val, b_rec["days_effective"], int(bool(b_rec["saver_available"])), q))

        if csv_hit is not None:
            assert csv_hit == "single"
            assert b_rec["action"] == "skip" and f_rec["action"] == "single"
            assert b_rec["skip_dm"] is not None and f_rec["skip_dm"] is None
            assert f_rec["skip_bar"] is None and f_rec["best_p"] is None and f_rec["pick_bar"] is None
            assert f_rec["batter"] == "Batter 0" and f_rec["double"] is None
            assert f_rec["chosen_on_kind"] == kind
            assert f_rec["named"] is False
            for shared_k in ("days_left", "days_effective", "saver_available", "objective", "source", "in_season", "expected_days_left"):
                assert b_rec[shared_k] == f_rec[shared_k], (key, shared_k)
            where = f"{day_str} {state_str}"
            if b_rec["in_season"]:
                assert where in base_problem_keys and where not in fresh_problem_keys
                changed_flagged_in_season += 1
            else:
                assert where not in base_problem_keys and where not in fresh_problem_keys
                changed_unflagged_preseason += 1
            changed_total += 1
        else:
            assert b_rec == f_rec, (key, b_rec, f_rec)
            unchanged_count += 1

    assert (changed_total, changed_flagged_in_season, changed_unflagged_preseason, unchanged_count) == (252, 229, 23, 4347)
    assert fresh["boundary_checks"] == baseline["boundary_checks"]

    # Verify all 612 remaining warnings are in-season all-skip states ([0,0,0,0,0])
    streak_counts = Counter()
    for prob in fresh["problems"]:
        where = prob.split(": ")[0]
        day_s, st_s = where.split(" ")
        rec = fresh_records[(day_s, st_s)]
        assert rec["in_season"] is True and rec["action"] == "skip" and rec["objective"] == "reach57"
        s_i = int(st_s.split("/")[1])
        d_i = rec["days_effective"]
        v_i = int(bool(rec["saver_available"]))
        row_cells = [int(x) for x in table[min(s_i, 56), d_i, v_i]]
        assert row_cells == [0, 0, 0, 0, 0], (where, row_cells)
        streak_counts[s_i] += 1

    assert dict(streak_counts) == {50: 141, 55: 157, 56: 314}

    summary = {
        "exit_code": proc.returncode,
        "records_total": len(fresh_records),
        "matches_saved_walk_g_json": True,
        "matches_saved_walk_g_txt": True,
        "changed_skip_to_single": changed_total,
        "formerly_flagged_in_season": changed_flagged_in_season,
        "unflagged_preseason": changed_unflagged_preseason,
        "unchanged_records": unchanged_count,
        "remaining_warnings": len(fresh["problems"]),
        "remaining_warnings_by_streak": dict(streak_counts),
        "boundary_checks_equal": True,
    }
    (HERE / "probe_08_season_walk.json").write_text(json.dumps(summary, indent=2))
    print("PROBE_08_PASS", json.dumps(summary))


if __name__ == "__main__":
    main()
