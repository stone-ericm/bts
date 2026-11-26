"""Adversarial Probe 4: Item 8 Test Audit —
1. Base (09d9b51f) RED vs GREEN classification across `test_live_gate_cs6a.py` and migrated tests.
2. Scratch mutation testing across all 16 specified production/example mutations (comparing CS6a's own tests vs pre-existing suite vs our `adopt/` tests).
3. Wall-clock runs in 2027-07-15, 2028-07-15, and 2027-12-31T22:30-05:00 (UTC/ET year split) across all added/touched test files + day_outcome + adopt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_SRC = REVIEW_DIR / "base_tree" / "src"
BASE_EX = REVIEW_DIR / "base_tree" / "config" / "orchestrator.example.toml"
BRANCH_SRC = REPO / "src"
BRANCH_EX = REPO / "config" / "orchestrator.example.toml"
WRAPPER_PY = REVIEW_DIR / "wrapper_python.sh"
HARNESS_CFG = REVIEW_DIR / "harness_config.json"
MUTANTS_ROOT = REVIEW_DIR / "mutants"
ADOPT_TEST = REVIEW_DIR / "adopt" / "test_cs6a_adversarial_adopt.py"


def ensure_wrapper_executable() -> None:
    WRAPPER_PY.chmod(WRAPPER_PY.stat().st_mode | stat.S_IXUSR)


def run_pytest_under_harness(
    test_targets: list[str],
    *,
    override_src: Path | None = None,
    override_example: Path | None = None,
    fake_clock_iso: str | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    cfg = {
        "wrapper_python": str(WRAPPER_PY),
        "override_src": str(override_src) if override_src else None,
        "override_example_toml": str(override_example) if override_example else None,
        "fake_clock_iso": fake_clock_iso,
    }
    HARNESS_CFG.write_text(json.dumps(cfg))
    env = dict(
        os.environ,
        TZ="America/New_York",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(REVIEW_DIR / "harness_site"),
    )
    cmd = [
        str(WRAPPER_PY), "-m", "pytest", "-p", "no:cacheprovider", "-v", "--tb=no",
        *(extra_args or []),
        *test_targets,
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=180)
    passed, failed, errors = [], [], []
    for line in proc.stdout.splitlines():
        m = re.match(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR)\b", line)
        if m:
            node, status = m.group(1), m.group(2)
            if status == "PASSED":
                passed.append(node)
            elif status == "FAILED":
                failed.append(node)
            else:
                errors.append(node)
    return {
        "returncode": proc.returncode,
        "passed_count": len(passed),
        "failed_count": len(failed),
        "error_count": len(errors),
        "passed": passed,
        "failed": failed,
        "errors": errors,
    }


def make_mutant_src(name: str, rel_file: str, old: str, new: str) -> Path:
    dest = MUTANTS_ROOT / name / "src"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(BRANCH_SRC, dest)
    target = dest / rel_file
    text = target.read_text()
    assert old in text, f"Target snippet not found in {rel_file} for {name}:\n{old}"
    target.write_text(text.replace(old, new, 1))
    return dest


def make_mutant_example(name: str, old: str, new: str) -> Path:
    dest_dir = MUTANTS_ROOT / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "orchestrator.example.toml"
    text = BRANCH_EX.read_text()
    assert old in text, f"Target snippet not found in example TOML for {name}: {old}"
    dest.write_text(text.replace(old, new, 1))
    return dest


def main() -> None:
    ensure_wrapper_executable()
    MUTANTS_ROOT.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    try:
        # -------------------------------------------------------------------
        # 1. Base (09d9b51f) RED / GREEN audit
        # -------------------------------------------------------------------
        base_cs6a = run_pytest_under_harness(
            [
                "tests/test_live_gate_cs6a.py",
                "tests/test_entry_intent.py",
                "tests/test_cli_schedule_date.py",
                "tests/test_season_contract_cs4b2_edges.py",
                "tests/test_cli_integration.py",
            ],
            override_src=BASE_SRC,
            override_example=BASE_EX,
            extra_args=["-k", "not TestBtsPreview and not TestBtsCheckResults"],
        )
        # Filter specifically to test_live_gate_cs6a.py passed nodes to verify ONLY named GREEN controls pass!
        cs6a_passed_on_base = [n for n in base_cs6a["passed"] if n.startswith("tests/test_live_gate_cs6a.py::")]
        cs6a_failed_on_base = [n for n in base_cs6a["failed"] + base_cs6a["errors"] if n.startswith("tests/test_live_gate_cs6a.py::")]
        summary["base_09d9b51f_audit"] = {
            "cs6a_passed_on_base": cs6a_passed_on_base,
            "cs6a_failed_on_base_count": len(cs6a_failed_on_base),
            "cs6a_failed_on_base": cs6a_failed_on_base,
            "other_failed_on_base": [n for n in base_cs6a["failed"] + base_cs6a["errors"] if not n.startswith("tests/test_live_gate_cs6a.py::")],
        }

        # -------------------------------------------------------------------
        # 2. Scratch Mutant Suite
        # -------------------------------------------------------------------
        mutants_spec = [
            (
                "m01_no_gate_schedule",
                "bts/cli.py",
                '        if not dry_run:\n            click.echo(f"bts schedule: refusing to start: {problem}", err=True)\n            raise SystemExit(2)',
                "        if False:\n            pass",
                ["tests/test_live_gate_cs6a.py"],
            ),
            (
                "m02_no_gate_schedule_dry",
                "bts/cli.py",
                '        click.echo(f"bts schedule: WARNING (a real run would refuse to start): {problem}", err=True)',
                "        pass",
                ["tests/test_live_gate_cs6a.py", "tests/test_cli_schedule_date.py"],
            ),
            (
                "m03_no_gate_orchestrate",
                "bts/cli.py",
                '    if problem is not None:\n        click.echo(f"bts orchestrate: refusing to run: {problem}", err=True)\n        raise SystemExit(2)',
                "    if False:\n        pass",
                ["tests/test_live_gate_cs6a.py"],
            ),
            (
                "m04_no_gate_entry_intent",
                "bts/entry_intent.py",
                "    if include_live_gate:\n        problem = live_gate_problem(config)\n        if problem is not None:\n            problems.append(problem)",
                "    pass",
                ["tests/test_live_gate_cs6a.py", "tests/test_entry_intent.py"],
            ),
            (
                "m05_gate_ignore_intent",
                "bts/entry_intent.py",
                '    if isinstance(intent, str) and intent == "enter":\n        conditions.append("scheduler.entry_intent = \'enter\'")',
                "    pass",
                ["tests/test_live_gate_cs6a.py"],
            ),
            (
                "m06_gate_ignore_delivery",
                "bts/entry_intent.py",
                '    if delivery in ("dm", "public"):\n        label = "scheduler delivery" if delivery_override is None else "delivery"\n        conditions.append(f"{label} {delivery!r}")',
                "    pass",
                ["tests/test_live_gate_cs6a.py"],
            ),
            (
                "m07_gate_ignore_override",
                "bts/entry_intent.py",
                "    delivery = delivery_override",
                "    delivery = None",
                ["tests/test_live_gate_cs6a.py"],
            ),
            (
                "m08_gate_malformed_sched_ok",
                "bts/entry_intent.py",
                '    sched = config.get("scheduler", {})\n    if not isinstance(sched, dict):\n        return "[scheduler] is not a table"',
                '    sched = config.get("scheduler", {})\n    if not isinstance(sched, dict):\n        sched = {}\n        config = dict(config, scheduler={})',
                ["tests/test_live_gate_cs6a.py"],
            ),
            (
                "m09_gate_malformed_health_ok",
                "bts/entry_intent.py",
                '    if not isinstance(config.get("health_checks", {}), dict):\n        return "[health_checks] is not a table"',
                '    if not isinstance(config.get("health_checks", {}), dict):\n        config = dict(config, health_checks={})',
                ["tests/test_live_gate_cs6a.py"],
            ),
            (
                "m10_run_read1_optional",
                "bts/cli.py",
                "    # Step 2: Apply strategy (streak-aware thresholds)\n    try:\n        decision_state = load_decision_streak_state(picks_path, require_contest_state=True)",
                "    # Step 2: Apply strategy (streak-aware thresholds)\n    try:\n        decision_state = load_decision_streak_state(picks_path)",
                ["tests/test_live_gate_cs6a.py"],
            ),
            (
                "m11_run_read2_optional",
                "bts/cli.py",
                "        if not result.daily.bluesky_posted:\n            try:\n                decision_state = load_decision_streak_state(picks_path, require_contest_state=True)",
                "        if not result.daily.bluesky_posted:\n            try:\n                decision_state = load_decision_streak_state(picks_path)",
                ["tests/test_live_gate_cs6a.py"],
            ),
            (
                "m12_run_read3_optional",
                "bts/cli.py",
                "    # Post to Bluesky if appropriate\n    try:\n        decision_state = load_decision_streak_state(picks_path, require_contest_state=True)",
                "    # Post to Bluesky if appropriate\n    try:\n        decision_state = load_decision_streak_state(picks_path)",
                ["tests/test_live_gate_cs6a.py", "tests/test_season_contract_cs4b2_edges.py"],
            ),
            (
                "m13a_orchestrate_reload_load_config",
                "bts/orchestrator.py",
                "    if config is None:\n        config = load_config(config_path)",
                "    config = load_config(config_path)",
                ["tests/test_live_gate_cs6a.py"],
            ),
        ]

        mutant_results: dict[str, dict] = {}
        for mname, rel_file, old, new, targets in mutants_spec:
            msrc = make_mutant_src(mname, rel_file, old, new)
            r = run_pytest_under_harness(targets, override_src=msrc)
            mutant_results[mname] = {
                "killed": r["failed_count"] + r["error_count"] > 0,
                "failed_count": r["failed_count"] + r["error_count"],
                "sample_failures": (r["failed"] + r["errors"])[:5],
            }

        # Special Mutant 13b: orchestrator reloads config_path via direct open() bypassing load_config()
        msrc_13b = make_mutant_src(
            "m13b_orchestrate_reload_direct_open",
            "bts/orchestrator.py",
            "    if config is None:\n        config = load_config(config_path)",
            '    with open(config_path, "rb") as _f:\n        config = tomllib.load(_f)',
        )
        r_13b_author = run_pytest_under_harness(["tests/test_live_gate_cs6a.py"], override_src=msrc_13b)
        r_13b_adopt = run_pytest_under_harness([str(ADOPT_TEST)], override_src=msrc_13b)
        mutant_results["m13b_orchestrate_reload_direct_open"] = {
            "killed_by_test_live_gate_cs6a": r_13b_author["failed_count"] + r_13b_author["error_count"] > 0,
            "author_failed_count": r_13b_author["failed_count"] + r_13b_author["error_count"],
            "killed_by_adopt_test": r_13b_adopt["failed_count"] + r_13b_adopt["error_count"] > 0,
            "adopt_failures": r_13b_adopt["failed"] + r_13b_adopt["errors"],
        }

        # Special Mutant 14: drop include_live_gate=False in day_outcome.gather
        msrc_14 = make_mutant_src(
            "m14_day_outcome_no_opt_out",
            "bts/day_outcome.py",
            "    intent = check_entry_intent(config, include_live_gate=False)",
            "    intent = check_entry_intent(config)",
        )
        r_14_cs6a_only = run_pytest_under_harness(["tests/test_live_gate_cs6a.py", "tests/test_entry_intent.py"], override_src=msrc_14)
        r_14_day_and_adopt = run_pytest_under_harness(
            ["tests/test_day_outcome.py", "tests/test_day_outcome_entry_sequence.py", str(ADOPT_TEST)],
            override_src=msrc_14,
        )
        mutant_results["m14_day_outcome_no_opt_out"] = {
            "killed_by_cs6a_and_entry_intent_alone": r_14_cs6a_only["failed_count"] + r_14_cs6a_only["error_count"] > 0,
            "killed_by_existing_day_outcome_and_adopt": r_14_day_and_adopt["failed_count"] + r_14_day_and_adopt["error_count"] > 0,
            "day_outcome_and_adopt_failed_count": r_14_day_and_adopt["failed_count"] + r_14_day_and_adopt["error_count"],
            "sample_failures": (r_14_day_and_adopt["failed"] + r_14_day_and_adopt["errors"])[:8],
        }

        # Mutants 15 & 16: turn one example setting off independently
        for mname, old, new in [
            ("m15_example_sched_false", "contest_state_required = true", "contest_state_required = false"),
            ("m16_example_health_false", "contest_state_expected = true", "contest_state_expected = false"),
        ]:
            mex = make_mutant_example(mname, old, new)
            r_ex = run_pytest_under_harness(["tests/test_live_gate_cs6a.py", "-k", "example"], override_example=mex)
            mutant_results[mname] = {
                "killed": r_ex["failed_count"] + r_ex["error_count"] > 0,
                "failed": r_ex["failed"] + r_ex["errors"],
            }

        summary["mutants"] = mutant_results

        # -------------------------------------------------------------------
        # 3. Wall-clock runs in 2027, 2028, and New Year's Eve ET/UTC split
        # -------------------------------------------------------------------
        clock_targets = [
            "tests/test_live_gate_cs6a.py",
            "tests/test_entry_intent.py",
            "tests/test_cli_integration.py",
            "tests/test_cli_schedule_date.py",
            "tests/test_season_contract_cs4b2_edges.py",
            "tests/test_day_outcome.py",
            "tests/test_day_outcome_entry_sequence.py",
            "tests/test_orchestrator.py",
            str(ADOPT_TEST),
        ]
        clock_runs = {}
        for label, iso in [
            ("clock_2027_07_15", "2027-07-15T12:00:00-04:00"),
            ("clock_2028_07_15", "2028-07-15T12:00:00-04:00"),
            ("clock_2027_12_31_utc_2028", "2027-12-31T22:30:00-05:00"),
        ]:
            rc = run_pytest_under_harness(clock_targets, fake_clock_iso=iso)
            clock_runs[label] = {
                "returncode": rc["returncode"],
                "passed_count": rc["passed_count"],
                "failed_count": rc["failed_count"],
                "error_count": rc["error_count"],
                "failures": rc["failed"] + rc["errors"],
            }
        summary["clock_runs"] = clock_runs

    finally:
        if HARNESS_CFG.exists():
            HARNESS_CFG.unlink()

    out_path = REVIEW_DIR / "mutations_and_clocks_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
