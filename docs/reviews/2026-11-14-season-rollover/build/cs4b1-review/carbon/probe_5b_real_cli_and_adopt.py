"""Probe 5b: Demonstrate B1 & B2 via real CLI subprocess AND verify adopt/test_season_contract_cs4b1_adopt.py across base, HEAD, and scratch-fixed src/."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

from probe_4_tests_audit import TOUCHED_TESTS, run_pytest  # noqa: E402

ADOPT_FILE = str(HERE / "adopt/test_season_contract_cs4b1_adopt.py")


def main() -> None:
    # Remove draft review file so adopt/ contains only the single clean adoptable pytest file
    draft = HERE / "adopt/test_season_contract_cs4b1_review.py"
    if draft.exists():
        draft.unlink()

    # 0. Demonstrate B1 & B2 via a clean `python -m bts.cli` subprocess (real production CLI invocation!)
    demo_dir = HERE / "_cli_demo/picks/account_state"
    if demo_dir.parents[1].exists():
        shutil.rmtree(demo_dir.parents[1])
    demo_dir.mkdir(parents=True)
    (demo_dir / "saver_state.json").write_text('{"state": "used", "season": true}', encoding="utf-8")
    good_row = '{"recorded_at": "2027-07-15T12:00:00Z", "active_streak": 7}'

    # Find exact depth for real CLI subprocess (`python -c "from bts.cli import cli; cli()"`)
    cli_b1_depth = None
    cli_b1_output = None
    env_cli = dict(os.environ, TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT / "src"))
    for d in range(9985, 9998):
        (demo_dir / "contest_ledger.jsonl").write_text(
            good_row + "\n" + '{"active_streak": ' + "[" * d + "0" + "]" * d + "}\n",
            encoding="utf-8",
        )
        p = subprocess.run(
            [sys.executable, "-c", "from bts.cli import cli; cli()", "season-contract", "status", "--picks-dir", str(demo_dir.parent)],
            cwd=ROOT,
            env=env_cli,
            capture_output=True,
            text=True,
        )
        l_hdrs = [ln for ln in p.stdout.splitlines() if ln.startswith("ledger:")]
        if len(l_hdrs) > 1:
            cli_b1_depth = d
            cli_b1_output = p.stdout
            break
    shutil.rmtree(demo_dir.parents[1])
    assert cli_b1_depth is not None, "Expected to reproduce B1 in real CLI subprocess!"
    print(f"REAL CLI REPRODUCED B1 AT DEPTH={cli_b1_depth}:\n{cli_b1_output}", flush=True)

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

    # 3. Apply smallest fixes for B1 and B2 in a temporary scratch copy of src/
    fix_dir = HERE / "_fixed_src"
    if fix_dir.exists():
        shutil.rmtree(fix_dir)
    shutil.copytree(ROOT / "src", fix_dir / "src")
    sc_path = fix_dir / "src/bts/season_contract.py"
    sc_txt = sc_path.read_text(encoding="utf-8")

    sc_txt = sc_txt.replace('and isinstance(raw.get("season"), int))', 'and type(raw.get("season")) is int)', 1)
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

    fixed_runs = {}
    for clk in ["2027-07-15T12:00:00-04:00", "2028-07-15T12:00:00-04:00"]:
        rf = run_pytest(["-q", ADOPT_FILE, *TOUCHED_TESTS], src_path=fix_dir / "src", clock_iso=clk)
        f_tail = rf.stdout.strip().splitlines()[-1]
        print(f"ADOPT + TOUCHED ON SCRATCH-FIXED ({clk}): {f_tail}", flush=True)
        assert rf.returncode == 0, rf.stdout + rf.stderr
        fixed_runs[clk] = f_tail

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
        "real_cli_b1_reproduced_depth": cli_b1_depth,
        "real_cli_b1_and_b2_output": cli_b1_output,
        "adopt_on_base": base_summary,
        "adopt_on_head_unpatched": head_summary,
        "adopt_plus_touched_on_scratch_fixed": fixed_runs,
        "previously_surviving_mutations_killed_by_adopt": killed_check,
    }
    (HERE / "probe5b_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("PROBE 5B SUMMARY:", json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
