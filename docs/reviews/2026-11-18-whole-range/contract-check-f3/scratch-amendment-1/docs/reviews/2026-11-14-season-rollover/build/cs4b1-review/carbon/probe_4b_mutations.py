"""Probe 4b: Verify untouched related suites (974 items), classify all 29 base-GREEN nodes, run 9 scratch mutations, and verify smallest fixes."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

from probe_4_tests_audit import NEW_TESTS, TOUCHED_TESTS, RELATED_SUITES, run_pytest  # noqa: E402


def main() -> None:
    # 1. Parse the 29 base-GREEN nodes from probe4_base_new_tests.log
    base_log = (HERE / "probe4_base_new_tests.log").read_text(encoding="utf-8")
    passed_on_base = [
        line.split(" PASSED")[0].strip()
        for line in base_log.splitlines()
        if " PASSED" in line and "::" in line
    ]
    failed_on_base = [
        line.split(" FAILED")[0].split(" ERROR")[0].strip()
        for line in base_log.splitlines()
        if (" FAILED" in line or " ERROR" in line) and "::" in line
    ]

    # 2. Run untouched RELATED_SUITES without fake clock (network audit hook still active via _clock_plugin!)
    existing_related = [p for p in RELATED_SUITES if (ROOT / p).exists()]
    r_rel = run_pytest(["-q", *existing_related], src_path=ROOT / "src", clock_iso=None)
    rel_tail = r_rel.stdout.strip().splitlines()[-1] if r_rel.stdout.strip() else r_rel.stderr.strip()
    print(f"RELATED SUITES (unpatched datetime): exit={r_rel.returncode} ({rel_tail})", flush=True)
    assert r_rel.returncode == 0, r_rel.stdout + r_rel.stderr

    # 3. Run 9 scratch mutations
    mut_dir = HERE / "_mut_src"
    mutations = [
        (
            "M1_enable_reads_switch_before_year_check",
            "src/bts/cli.py",
            '    if season != year:\n        raise click.UsageError(f"--season must equal the current ET year {year}")\n    picks = Path(picks_dir)\n    state = switch_state(picks, now)',
            '    picks = Path(picks_dir)\n    state = switch_state(picks, now)\n    if season != year:\n        raise click.UsageError(f"--season must equal the current ET year {year}")',
            NEW_TESTS,
        ),
        (
            "M2_drop_RecursionError_in_season_contract",
            "src/bts/season_contract.py",
            "except (ValueError, RecursionError):",
            "except ValueError:",
            NEW_TESTS,
        ),
        (
            "M3_unconditional_obs_write_in_fetch",
            "src/bts/cli.py",
            "    if switched:\n        _atomic_write_json(out_path, observation)",
            "    _atomic_write_json(out_path, observation)",
            NEW_TESTS,
        ),
        (
            "M4_unconditional_saver_call_in_fetch",
            "src/bts/cli.py",
            '    if not switched:\n        click.echo(f"season {season} is not switched to automatic; reading recorded in the ledger and the fetch status: {summary}")\n        return',
            '    if not switched:\n        click.echo(f"season {season} is not switched to automatic; reading recorded in the ledger and the fetch status: {summary}")',
            NEW_TESTS,
        ),
        (
            "M5_second_clock_read_for_saver_year",
            "src/bts/cli.py",
            '    if source_date is not None and source_date.year == season:\n        maybe_auto_earn_saver(picks, best_streak=observation["best_streak"], season=season)',
            '    _late_y = datetime.now(ET).year\n    if source_date is not None and source_date.year == _late_y:\n        maybe_auto_earn_saver(picks, best_streak=observation["best_streak"], season=_late_y)',
            NEW_TESTS,
        ),
        (
            "M6_swap_v2_season_and_id_order",
            "src/bts/contest_state.py",
            '        season = data.get("season")\n        if type(season) is not int:\n            raise ContestStateError(f"automatic contest season invalid in {path}: must be an integer")\n        activation_id = data.get("activation_id")\n        if not valid_activation_id(activation_id):\n            raise ContestStateError(f"automatic contest activation_id invalid in {path}: must be 32 lowercase hexadecimal characters")',
            '        activation_id = data.get("activation_id")\n        if not valid_activation_id(activation_id):\n            raise ContestStateError(f"automatic contest activation_id invalid in {path}: must be 32 lowercase hexadecimal characters")\n        season = data.get("season")\n        if type(season) is not int:\n            raise ContestStateError(f"automatic contest season invalid in {path}: must be an integer")',
            NEW_TESTS,
        ),
        (
            "M7_allow_best_58_in_validate_fetch",
            "src/bts/contest_fetch.py",
            '    if best_streak > 57:\n        raise ContestFetchError("seasonBestStreak must be <= 57")',
            "    pass",
            NEW_TESTS + ["tests/test_contest_state_cs1_review.py"],
        ),
        (
            "M8_migrated_cold_10_saver_promotes_bug",
            "src/bts/saver_state.py",
            '    if current == "uninitialized" and best_streak < 10:',
            '    if current == "uninitialized":',
            ["tests/test_contest_fetch.py"],
        ),
        (
            "M9_migrated_other_year_saver_gate_dropped",
            "src/bts/cli.py",
            "    if source_date is not None and source_date.year == season:",
            "    if source_date is not None:",
            ["tests/test_saver_cs3.py"],
        ),
    ]

    mut_results = {}
    for m_name, rel_file, target_s, repl_s, test_targets in mutations:
        if mut_dir.exists():
            shutil.rmtree(mut_dir)
        shutil.copytree(ROOT / "src", mut_dir / "src")
        f_path = mut_dir / rel_file
        txt = f_path.read_text(encoding="utf-8")
        assert target_s in txt, f"Target string not found for {m_name}"
        if m_name == "M2_drop_RecursionError_in_season_contract":
            txt = txt.replace("from bts.contest_state import _UNREADABLE", "_UNREADABLE = (OSError, ValueError)")
        f_path.write_text(txt.replace(target_s, repl_s, 1), encoding="utf-8")
        rm = run_pytest(["-q", "--tb=no", *test_targets], src_path=mut_dir / "src")
        tail = rm.stdout.strip().splitlines()[-1] if rm.stdout.strip() else rm.stderr.strip()
        killed = rm.returncode != 0
        mut_results[m_name] = {"killed": killed, "exit": rm.returncode, "summary": tail}
        print(f"MUTATION {m_name}: killed={killed} ({tail})", flush=True)
    if mut_dir.exists():
        shutil.rmtree(mut_dir)

    summary = {
        "base_new_tests": {
            "passed_count": len(passed_on_base),
            "failed_count": len(failed_on_base),
            "total": len(passed_on_base) + len(failed_on_base),
            "passed_nodes": passed_on_base,
        },
        "related_suites": {"exit": r_rel.returncode, "summary": rel_tail},
        "mutations": mut_results,
    }
    (HERE / "probe4b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("PROBE 4B COMPLETE", flush=True)


if __name__ == "__main__":
    main()
