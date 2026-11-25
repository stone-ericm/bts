"""Adversarial probe 6: compare live season walk run with walk-g.json.gz / walk-h.json.gz and test planted faults."""
from __future__ import annotations

from collections import Counter
from datetime import date
import gzip
import json
from pathlib import Path
import sys
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bts import strategy
from bts.simulate import tail_policy
from scripts.audit.season_walk import walk


def verify_walk_outputs(live_walk_path: Path) -> dict:
    walk_g = json.loads(gzip.decompress((REPO / "scripts/audit/season_walk/results/walk-g.json.gz").read_bytes()))
    walk_h = json.loads(gzip.decompress((REPO / "scripts/audit/season_walk/results/walk-h.json.gz").read_bytes()))
    live = json.loads(live_walk_path.read_text())

    # 1. Exact agreement between fresh live run and committed walk-h.json.gz
    assert live == walk_h, "Fresh walk output differs from committed walk-h.json.gz!"
    assert live["runs"] == 4818
    assert len(live["problems"]) == 612

    # 2. Original 4,599 records & 612 findings equal walk-g.json.gz
    added_label = f"dated/{tail_policy.LONGEST_STREAK_FROM}/{tail_policy.LONGEST_STREAK_FROM}/used/0.96"
    assert live["states"] == [*walk_g["states"], added_label]
    orig_records = [r for r in live["records"] if r["state"] != added_label]
    added_records = [r for r in live["records"] if r["state"] == added_label]
    assert len(orig_records) == 4599
    assert len(added_records) == 219
    assert orig_records == walk_g["records"]
    assert live["problems"] == walk_g["problems"]
    assert live["boundary"] == walk_g["boundary"] and live["boundary"]["problems"] == []

    # Verify exact warning counts by streak (50: 141, 55: 157, 56: 314)
    streak_problem_counts: Counter[str] = Counter()
    for p in live["problems"]:
        state_tok = p.split()[1].split(":")[0]
        streak_tok = state_tok.split("/")[1]
        streak_problem_counts[streak_tok] += 1
    assert dict(streak_problem_counts) == {"50": 141, "55": 157, "56": 314}, streak_problem_counts

    # 3. Verify the 219 added records against Amendment A7
    assert walk.check(added_records) == []
    preseason_singles = [r["day"] for r in added_records if r["action"] == "single" and not r["in_season"]]
    inseason_singles = [r["day"] for r in added_records if r["action"] == "single" and r["in_season"]]
    doubles = [r["day"] for r in added_records if r["action"] == "double"]
    skips = [r["day"] for r in added_records if r["action"] == "skip"]

    assert len(preseason_singles) == 23
    assert len(inseason_singles) == 184
    assert len(preseason_singles) + len(inseason_singles) == 207
    assert doubles == ["2027-09-24", "2027-09-25", "2027-09-26"]
    assert len(skips) == 9 and skips[0] == "2027-09-27" and skips[-1] == "2027-10-05"

    for r in added_records:
        if r["day"] <= "2027-09-26":
            assert r["action"] in ("single", "double")
            assert r["objective"] == "longest_streak"
            assert r["source"] == "mdp"
            assert r["chosen_on_valid"] is True
            assert r["chosen_on_kind"] == "dated"
            assert r["batter"] == "Batter 0"
            assert (r["double"] == "Batter 1") if r["action"] == "double" else (r["double"] is None)
            assert r["days_effective"] == min(r["expected_days_left"], 180)
        else:
            assert r["action"] == "skip" and r["named"] is True
            assert r["objective"] == "reach57" and r["days_effective"] == 0

    return {
        "orig_records": len(orig_records),
        "added_records": len(added_records),
        "singles_total": 207,
        "singles_preseason": len(preseason_singles),
        "singles_inseason": len(inseason_singles),
        "doubles": doubles,
        "post_finale_skips": len(skips),
        "problem_counts_by_streak": dict(streak_problem_counts),
    }


def check_planted_faults() -> dict:
    b = tail_policy.LONGEST_STREAK_FROM
    st = walk.State("dated", b, b, "used", 0.96)

    # 1. Real walk_one with planted wrong rule outputs ("single" on finale, "skip" in July, "skip" in preseason)
    with patch.object(strategy, "longest_streak_action", lambda *_a: "single"):
        bad_finale_single = walk.walk_one(date(2027, 9, 26), st)
    probs_1 = walk.check([bad_finale_single])
    assert any("longest-streak action 'single', expected 'double'" in p for p in probs_1), probs_1

    with patch.object(strategy, "longest_streak_action", lambda *_a: "skip"):
        bad_july_skip = walk.walk_one(date(2027, 7, 15), st)
        bad_pre_skip = walk.walk_one(date(2027, 3, 10), st)
    probs_2 = walk.check([bad_july_skip])
    probs_3 = walk.check([bad_pre_skip])
    assert any("longest-streak action 'skip', expected 'single'" in p for p in probs_2), probs_2
    assert any("longest-streak action 'skip', expected 'single'" in p for p in probs_3), probs_3

    # 2. Planted faults even when recorded objective is wrong ("reach57") across all 3 double dates & positive skip
    for d_str in ("2027-09-24", "2027-09-25", "2027-09-26"):
        good = walk.walk_one(date.fromisoformat(d_str), st)
        planted_single = dict(good, action="single", double=None, objective="reach57")
        p_single = walk.check([planted_single])
        assert any("longest-streak action 'single', expected 'double'" in x for x in p_single), p_single
        assert any("longest-streak objective 'reach57'" in x for x in p_single), p_single

        planted_skip = dict(good, action="skip", named=True, batter=None, double=None, objective="reach57")
        p_skip = walk.check([planted_skip])
        assert any("longest-streak action 'skip', expected 'double'" in x for x in p_skip), p_skip

    # 3. Planted invalid chosen_on and missing double slot
    good_fin = walk.walk_one(date(2027, 9, 26), st)
    bad_chosen = dict(good_fin, chosen_on_valid=False)
    bad_slots = dict(good_fin, double=None)
    assert any("invalid chosen_on or selected slots" in x for x in walk.check([bad_chosen]))
    assert any("invalid chosen_on or selected slots" in x for x in walk.check([bad_slots]))

    return {"planted_fault_cases": 11}


def test_probe_6_faults():
    res = check_planted_faults()
    assert res["planted_fault_cases"] == 11


def main() -> None:
    live_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (OUT_DIR / "walk-run.json")
    summary = {
        "walk_comparison": verify_walk_outputs(live_path),
        "planted_faults": check_planted_faults(),
    }
    out_log = OUT_DIR / "probe_6_walk_compare_and_faults.log"
    out_log.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
