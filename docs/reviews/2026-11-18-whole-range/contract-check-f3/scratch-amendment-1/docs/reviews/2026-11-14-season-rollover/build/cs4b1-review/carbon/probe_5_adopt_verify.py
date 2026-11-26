"""Probe 5: Verify adopt/test_season_contract_cs4b1_review.py on base (8421e3cd), unpatched HEAD (28ae3aa9), and scratch-fixed src/."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

from probe_4_tests_audit import NEW_TESTS, TOUCHED_TESTS, run_pytest  # noqa: E402

ADOPT_FILE = str(HERE / "adopt/test_season_contract_cs4b1_review.py")


def main() -> None:
    # 1. Run adopt file against archived base 8421e3cd -> ALL 7 tests must FAIL!
    r_base = run_pytest(["-v", "--tb=short", ADOPT_FILE], src_path=HERE / "base_tree/src")
    base_summary = r_base.stdout.strip().splitlines()[-1]
    print("ADOPT ON BASE:", base_summary, flush=True)
    assert "7 failed" in base_summary, r_base.stdout

    # 2. Run adopt file against unpatched HEAD 28ae3aa9 -> B1 (1) + B2 (2) = 3 failed, 4 passed!
    r_head = run_pytest(["-v", "--tb=short", ADOPT_FILE], src_path=ROOT / "src")
    head_summary = r_head.stdout.strip().splitlines()[-1]
    print("ADOPT ON HEAD (unpatched):", head_summary, flush=True)
    assert "3 failed, 4 passed" in head_summary, r_head.stdout

    # 3. Apply smallest fixes for B1 and B2 in a temporary scratch copy of src/ (NEVER touching tracked files!)
    fix_dir = HERE / "_fixed_src"
    if fix_dir.exists():
        shutil.rmtree(fix_dir)
    shutil.copytree(ROOT / "src", fix_dir / "src")
    sc_path = fix_dir / "src/bts/season_contract.py"
    sc_txt = sc_path.read_text(encoding="utf-8")

    # Smallest fix for B2 (line 157):
    old_b2 = "and isinstance(raw.get(\"season\"), int))"
    new_b2 = "and type(raw.get(\"season\")) is int)"
    assert old_b2 in sc_txt
    sc_txt = sc_txt.replace(old_b2, new_b2, 1)

    # Smallest fix for B1 (lines 168-180):
    old_b1 = """            rows, bad = [], 0
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("not an object")
                    rows.append(row)
                except (ValueError, RecursionError):
                    bad += 1
            lines.append(f"ledger: readable; rows: {len(rows)}; bad lines: {bad}")
            keys = ["recorded_at", "active_streak", "best_streak", "source_date", "season", "switched",
                    "activation_id", "settled_by_year", "settled_unmapped", "unsettled"]
            lines.extend("  " + "; ".join(f"{key}={row.get(key, '—')}" for key in keys) for row in rows[-10:])"""
    new_b1 = """            keys = ["recorded_at", "active_streak", "best_streak", "source_date", "season", "switched",
                    "activation_id", "settled_by_year", "settled_unmapped", "unsettled"]
            rows, bad = [], 0
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("not an object")
                    rows.append("  " + "; ".join(f"{key}={row.get(key, '—')}" for key in keys))
                except (ValueError, RecursionError):
                    bad += 1
            lines.append(f"ledger: readable; rows: {len(rows)}; bad lines: {bad}")
            lines.extend(rows[-10:])"""
    assert old_b1 in sc_txt
    sc_txt = sc_txt.replace(old_b1, new_b1, 1)
    sc_path.write_text(sc_txt, encoding="utf-8")

    # Run adopt + all 7 touched files against scratch-fixed src under both 2027-07-15 and 2028-07-15!
    fixed_runs = {}
    for clk in ["2027-07-15T12:00:00-04:00", "2028-07-15T12:00:00-04:00"]:
        rf = run_pytest(["-q", ADOPT_FILE, *TOUCHED_TESTS], src_path=fix_dir / "src", clock_iso=clk)
        f_tail = rf.stdout.strip().splitlines()[-1]
        print(f"ADOPT + TOUCHED ON SCRATCH-FIXED ({clk}): {f_tail}", flush=True)
        assert rf.returncode == 0, rf.stdout + rf.stderr
        fixed_runs[clk] = f_tail

    # Also verify M1, M2, M6 are NOW KILLED when adopt file is included!
    killed_check = {}
    for m_name, rel_file, t_s, r_s in [
        (
            "M1",
            "src/bts/cli.py",
            '    if season != year:\n        raise click.UsageError(f"--season must equal the current ET year {year}")\n    picks = Path(picks_dir)\n    state = switch_state(picks, now)',
            '    picks = Path(picks_dir)\n    state = switch_state(picks, now)\n    if season != year:\n        raise click.UsageError(f"--season must equal the current ET year {year}")',
        ),
        (
            "M2",
            "src/bts/season_contract.py",
            "except (ValueError, RecursionError):",
            "except ValueError:",
        ),
        (
            "M6",
            "src/bts/contest_state.py",
            '        season = data.get("season")\n        if type(season) is not int:\n            raise ContestStateError(f"automatic contest season invalid in {path}: must be an integer")\n        activation_id = data.get("activation_id")\n        if not valid_activation_id(activation_id):\n            raise ContestStateError(f"automatic contest activation_id invalid in {path}: must be 32 lowercase hexadecimal characters")',
            '        activation_id = data.get("activation_id")\n        if not valid_activation_id(activation_id):\n            raise ContestStateError(f"automatic contest activation_id invalid in {path}: must be 32 lowercase hexadecimal characters")\n        season = data.get("season")\n        if type(season) is not int:\n            raise ContestStateError(f"automatic contest season invalid in {path}: must be an integer")',
        ),
    ]:
        mdir = HERE / "_m_verify"
        if mdir.exists():
            shutil.rmtree(mdir)
        shutil.copytree(fix_dir / "src", mdir / "src")
        mp = mdir / rel_file
        mt = mp.read_text(encoding="utf-8")
        if m_name == "M2":
            mt = mt.replace("from bts.contest_state import _UNREADABLE", "_UNREADABLE = (OSError, ValueError)")
        mp.write_text(mt.replace(t_s, r_s, 1), encoding="utf-8")
        rm = run_pytest(["-q", "--tb=no", ADOPT_FILE], src_path=mdir / "src")
        killed_check[m_name] = rm.returncode != 0
        shutil.rmtree(mdir)
        assert killed_check[m_name], f"{m_name} was not killed by adopt file!"

    shutil.rmtree(fix_dir)
    out = {
        "adopt_on_base": base_summary,
        "adopt_on_head_unpatched": head_summary,
        "adopt_plus_touched_on_scratch_fixed": fixed_runs,
        "previously_surviving_mutations_killed_by_adopt": killed_check,
    }
    (HERE / "probe5_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("PROBE 5 SUMMARY:", json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
