"""Probe 4: Audit new and migrated tests — RED/GREEN on base, 4 fake clocks (2027/2028), related suites, and 9 scratch mutations."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

NEW_TESTS = [
    "tests/test_season_contract_cs4b1.py",
    "tests/test_season_contract_cs4b1_edges.py",
]

TOUCHED_TESTS = NEW_TESTS + [
    "tests/test_contest_fetch.py",
    "tests/test_cli_integration.py",
    "tests/test_saver_cs3.py",
    "tests/test_contest_dated_cs4a_review.py",
    "tests/test_contest_state_cs1_review.py",
]

RELATED_SUITES = [
    "tests/test_contest_ledger.py",
    "tests/test_contest_state.py",
    "tests/test_contest_state_defensive_reads.py",
    "tests/test_contest_dated_cs4a.py",
    "tests/health/test_contest_state.py",
    "tests/test_backup.py",
    "tests/scripts/season_ledger",
]


def run_pytest(args: list[str], *, src_path: Path, clock_iso: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = f"{src_path}:{HERE}:{ROOT}"
    if clock_iso:
        env["FAKE_CLOCK_ISO"] = clock_iso
    else:
        env.pop("FAKE_CLOCK_ISO", None)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-p",
        "_clock_plugin",
        *args,
    ]
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


def main() -> None:
    results: dict[str, object] = {}

    # 1. Run NEW_TESTS against archived base src (`8421e3cd`) with -v to inspect every PASSED and FAILED node!
    r_base = run_pytest(["-v", "--tb=no", *NEW_TESTS], src_path=HERE / "base_tree/src")
    (HERE / "probe4_base_new_tests.log").write_text(r_base.stdout + r_base.stderr, encoding="utf-8")
    passed_on_base = []
    failed_on_base = []
    for line in r_base.stdout.splitlines():
        if " PASSED" in line and "::" in line:
            passed_on_base.append(line.split(" PASSED")[0].strip())
        elif (" FAILED" in line or " ERROR" in line) and "::" in line:
            failed_on_base.append(line.split(" FAILED")[0].split(" ERROR")[0].strip())
    results["base_new_tests"] = {
        "passed_count": len(passed_on_base),
        "failed_count": len(failed_on_base),
        "total": len(passed_on_base) + len(failed_on_base),
        "passed_nodes": passed_on_base,
    }
    print(f"BASE NEW TESTS: {len(failed_on_base)} failed, {len(passed_on_base)} passed", flush=True)

    # 2. Run ALL 7 TOUCHED_TESTS under 4 fake clocks (including 2027-07-15 and 2028-07-15)!
    clock_runs = {}
    for clk in [
        "2027-07-15T12:00:00-04:00",
        "2028-07-15T12:00:00-04:00",
        "2027-12-31T23:59:59-05:00",
        "2028-02-29T12:00:00-05:00",
    ]:
        rc = run_pytest(["-q", *TOUCHED_TESTS], src_path=ROOT / "src", clock_iso=clk)
        tail = rc.stdout.strip().splitlines()[-1] if rc.stdout.strip() else rc.stderr.strip()
        clock_runs[clk] = {"exit": rc.returncode, "summary": tail}
        print(f"CLOCK {clk}: exit={rc.returncode} ({tail})", flush=True)
        assert rc.returncode == 0, f"Clock {clk} failed:\n{rc.stdout}\n{rc.stderr}"
    results["frozen_clock_runs"] = clock_runs

    # 3. Run RELATED_SUITES on HEAD under 2027-07-15
    existing_related = [p for p in RELATED_SUITES if (ROOT / p).exists()]
    r_rel = run_pytest(["-q", *existing_related], src_path=ROOT / "src", clock_iso="2027-07-15T12:00:00-04:00")
    rel_tail = r_rel.stdout.strip().splitlines()[-1] if r_rel.stdout.strip() else r_rel.stderr.strip()
    results["related_suites"] = {"files": existing_related, "exit": r_rel.returncode, "summary": rel_tail}
    print(f"RELATED SUITES: exit={r_rel.returncode} ({rel_tail})", flush=True)
    assert r_rel.returncode == 0, f"Related suites failed:\n{r_rel.stdout}\n{r_rel.stderr}"

    # 4. Scratch mutation testing (NEVER modifies tracked src/!)
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
            "from bts.contest_state import _UNREADABLE",
            "_UNREADABLE = (OSError, ValueError)",
            NEW_TESTS,
        ),
        (
            "M2b_drop_RecursionError_everywhere_in_season_contract",
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
            "    if source_date is not None and source_date.year == season:\n        maybe_auto_earn_saver(picks, best_streak=observation[\"best_streak\"], season=season)",
            "    _late_y = datetime.now(ET).year\n    if source_date is not None and source_date.year == _late_y:\n        maybe_auto_earn_saver(picks, best_streak=observation[\"best_streak\"], season=_late_y)",
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
    ]

    mut_results = {}
    for m_name, rel_file, target_s, repl_s, test_targets in mutations:
        if mut_dir.exists():
            shutil.rmtree(mut_dir)
        shutil.copytree(ROOT / "src", mut_dir / "src")
        f_path = mut_dir / rel_file
        txt = f_path.read_text(encoding="utf-8")
        assert target_s in txt, f"Target string not found for {m_name}"
        # For M2b also replace _UNREADABLE in season_contract.py so neither site catches RecursionError!
        if m_name == "M2b_drop_RecursionError_everywhere_in_season_contract":
            txt = txt.replace("from bts.contest_state import _UNREADABLE", "_UNREADABLE = (OSError, ValueError)")
        f_path.write_text(txt.replace(target_s, repl_s, 1), encoding="utf-8")
        rm = run_pytest(["-q", "--tb=no", *test_targets], src_path=mut_dir / "src")
        tail = rm.stdout.strip().splitlines()[-1] if rm.stdout.strip() else rm.stderr.strip()
        killed = rm.returncode != 0
        mut_results[m_name] = {"killed": killed, "exit": rm.returncode, "summary": tail}
        print(f"MUTATION {m_name}: killed={killed} ({tail})", flush=True)
    if mut_dir.exists():
        shutil.rmtree(mut_dir)
    results["mutations"] = mut_results

    (HERE / "probe4_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("PROBE 4 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
