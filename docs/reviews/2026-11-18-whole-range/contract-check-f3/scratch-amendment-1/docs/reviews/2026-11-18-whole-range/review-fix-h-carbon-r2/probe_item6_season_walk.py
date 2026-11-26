"""Adversarial probe for Item 6: full 2027 season walk end-to-end, comparison against walk-g.json.gz and walk-h.json.gz, A7 counts, and planted wrong actions / cleared fields."""
from __future__ import annotations

from collections import Counter
from datetime import date
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.audit.season_walk import walk

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS_DIR = REPO / "scripts/audit/season_walk/results"


def main() -> None:
    with tempfile.TemporaryDirectory(dir=HERE, prefix="walk-run-") as tmp_str:
        out_json = Path(tmp_str) / "walk-fresh.json"
        rc = walk.main(["--out", str(out_json)])
        assert rc == 1, f"Expected exit code 1 on the 612 known warnings, got {rc}"

        fresh_bytes = out_json.read_bytes()
        saved_h_bytes = gzip.decompress((RESULTS_DIR / "walk-h.json.gz").read_bytes())
        assert fresh_bytes == saved_h_bytes, "Fresh walk JSON differs from saved walk-h.json.gz!"
        print(f"OK: Fresh walk matched walk-h.json.gz byte-for-byte (sha256={sha256(fresh_bytes).hexdigest()[:16]})")

        h_data = json.loads(fresh_bytes)
        g_data = json.loads(gzip.decompress((RESULTS_DIR / "walk-g.json.gz").read_bytes()))

    assert h_data["runs"] == 4818 and g_data["runs"] == 4599
    assert h_data["problems"] == g_data["problems"] and len(h_data["problems"]) == 612
    assert h_data["boundary"] == g_data["boundary"] and h_data["boundary"]["problems"] == []

    # Verify warning breakdown: 141 at 50, 157 at 55, 314 at 56, 0 at 57
    warn_streaks = Counter(p.split()[1].split("/")[1] for p in h_data["problems"])
    assert dict(warn_streaks) == {"50": 141, "55": 157, "56": 314}, warn_streaks

    new_label = "dated/57/57/used/0.96"
    orig_h = [r for r in h_data["records"] if r["state"] != new_label]
    added_h = [r for r in h_data["records"] if r["state"] == new_label]
    assert len(orig_h) == 4599 and len(added_h) == 219
    assert orig_h == g_data["records"], "Original 4,599 records differ from walk-g.json.gz!"

    singles = [r for r in added_h if r["action"] == "single"]
    doubles = [r for r in added_h if r["action"] == "double"]
    skips = [r for r in added_h if r["action"] == "skip"]
    preseason_singles = [r for r in singles if not r["in_season"]]
    assert len(singles) == 207 and len(preseason_singles) == 23
    assert [r["day"] for r in doubles] == ["2027-09-24", "2027-09-25", "2027-09-26"]
    assert len(skips) == 9 and all(r["day"] >= "2027-09-27" and r["objective"] == "reach57" for r in skips)

    for r in singles + doubles:
        assert r["objective"] == "longest_streak" and r["source"] == "mdp"
        assert r["chosen_on_valid"] is True and r["chosen_on_kind"] == "dated"
        assert r["batter"] == "Batter 0"
        assert r["double"] == ("Batter 1" if r["action"] == "double" else None)

    print("OK: All 4,599 original rows equal walk-g; 219 added rows = 207 singles (23 preseason), 3 doubles (09-24..26), 9 post-finale skips")

    # Planted wrong actions and R4 cleared fields
    sample_single = next(r for r in singles if r["day"] == "2027-07-15")
    sample_double = next(r for r in doubles if r["day"] == "2027-09-26")
    assert walk.check([sample_single, sample_double]) == []

    planted_cases = [
        ("inseason_skip_longest", sample_single | {"action": "skip", "named": True, "batter": None}, "expected 'single'"),
        ("inseason_skip_reach57", sample_single | {"action": "skip", "named": True, "batter": None, "objective": "reach57"}, "expected 'single'"),
        ("finale_single_longest", sample_double | {"action": "single", "double": None}, "expected 'double'"),
        ("finale_single_reach57", sample_double | {"action": "single", "double": None, "objective": "reach57"}, "expected 'double'"),
        ("day4_double_wrong", next(r for r in singles if r["day"] == "2027-09-23") | {"action": "double", "double": "Batter 1"}, "expected 'single'"),
        ("missing_days_effective", sample_double | {"days_effective": None}, "effective days"),
        ("missing_source", sample_double | {"source": None}, "decision source"),
        ("heuristic_source", sample_double | {"source": "heuristic"}, "decision source"),
        ("missing_artifacts", sample_double | {"artifacts": None}, "policy artifacts missing"),
        ("empty_artifacts", sample_double | {"artifacts": {}}, "policy artifacts missing"),
        ("bad_chosen_on", sample_double | {"chosen_on_valid": False}, "invalid chosen_on"),
        ("missing_double_slot", sample_double | {"double": None}, "selected slots"),
    ]
    for cname, bad_rec, expected_substr in planted_cases:
        probs = walk.check([bad_rec])
        assert any(expected_substr in p for p in probs), f"{cname}: {expected_substr!r} not in {probs}"
    print(f"OK: All {len(planted_cases)} planted wrong-action and cleared-field cases reported by walk.check()")


if __name__ == "__main__":
    main()
