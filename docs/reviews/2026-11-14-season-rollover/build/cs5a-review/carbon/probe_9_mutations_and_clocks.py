"""Adversarial probe for CS5a item 9: base GREEN controls, 16-mutant kill matrix, and 2027/2028 clock runs."""

from __future__ import annotations

import datetime as _real_dt_mod
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

NEW_TESTS = [
    "tests/test_pick_chosen_on_cs5a.py",
    "tests/test_saved_pick_paths_cs5a.py",
]

ADOPT_TEST = ".codex-review/cs5a-review/adopt/test_cs5a_adversarial_review.py"

ALL_TOUCHED_TESTS = [
    "tests/test_pick_chosen_on_cs5a.py",
    "tests/test_saved_pick_paths_cs5a.py",
    "tests/test_live_gate_cs6a.py",
    "tests/test_live_gate_cs6a_review.py",
    "tests/test_season_contract_cs4b2_edges.py",
    ADOPT_TEST,
]


def run_pytest(repo_root: Path, pythonpath: str, args: list[str], extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["TZ"] = "America/New_York"
    env["MPLCONFIGDIR"] = "/tmp/mpl"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = pythonpath
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [str(repo_root / ".venv" / "bin" / "python"), "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )


def check_base_green_controls(repo_root: Path) -> dict:
    base_dir = repo_root / ".codex-review" / "cs5a-review" / "scratch" / "base_4cab4212"
    py_path = f"{base_dir / 'src'}:{base_dir}:{repo_root}"
    proc = run_pytest(repo_root, py_path, ["-v", "--tb=no", *NEW_TESTS])
    passed = []
    failed = []
    skipped = []
    for line in proc.stdout.splitlines():
        if " PASSED" in line:
            passed.append(line.split(" PASSED")[0].strip())
        elif " FAILED" in line:
            failed.append(line.split(" FAILED")[0].strip())
        elif " SKIPPED" in line:
            skipped.append(line.split(" SKIPPED")[0].strip())
    assert len(failed) > 0, "Expected new tests to fail on base!"
    return {
        "passed_count": len(passed),
        "failed_count": len(failed),
        "skipped_count": len(skipped),
        "passed_ids": passed,
    }


MUTANTS: list[tuple[str, str, str, str]] = [
    (
        "m01_remove_scheduler_judge",
        "src/bts/scheduler.py",
        "refusal = None if fresh_choice else saved_pick_refusal(daily, decision_state)",
        "refusal = None",
    ),
    (
        "m02_remove_run_catchup_judge",
        "src/bts/cli.py",
        "refusal = saved_pick_refusal(result.daily, decision_state)",
        "refusal = None",
    ),
    (
        "m03_remove_run_unlocked_judge",
        "src/bts/cli.py",
        "refusal = saved_pick_refusal(daily, decision_state)",
        "refusal = None",
    ),
    (
        "m04_remove_orchestrate_catchup_judge",
        "src/bts/orchestrator.py",
        "refusal = saved_pick_refusal(result.daily, decision_state)",
        "refusal = None",
    ),
    (
        "m05_remove_orchestrate_unlocked_judge",
        "src/bts/orchestrator.py",
        "refusal = saved_pick_refusal(daily, decision_state)",
        "refusal = None",
    ),
    (
        "m06_drop_decision_not_none",
        "src/bts/scheduler.py",
        "selection is not None and selection.decision is not None",
        "selection is not None",
    ),
    (
        "m07_drop_identity_part",
        "src/bts/scheduler.py",
        "and selection.pick_result is not None and selection.pick_result.daily is daily",
        "and selection.pick_result is not None",
    ),
    (
        "m08_judge_accept_required_false",
        "src/bts/picks.py",
        '    elif not record["required"]:\n        chosen = "it was chosen without the live check (for example by the preview)"\n    else:',
        "    else:",
    ),
    (
        "m09_dated_compare_kind_only",
        "src/bts/picks.py",
        'live_kind == "automatic" or (live_kind == "dated" and record["dated"] == live_dated)',
        'live_kind in {"automatic", "dated"}',
    ),
    (
        "m10_auto_compare_streak",
        "src/bts/picks.py",
        'live_kind == "automatic"',
        '(live_kind == "automatic" and record["streak"] == decision_state.streak)',
    ),
    (
        "m11_stamp_reused_in_run_and_pick",
        "src/bts/orchestrator.py",
        "if sel.decision is not None and sel.pick_result is not None and sel.pick_result.daily is not None:",
        "if sel.pick_result is not None and sel.pick_result.daily is not None:",
    ),
    (
        "m12_stamp_reused_in_run",
        "src/bts/cli.py",
        "    if sel.decision is not None:\n        daily.chosen_on = chosen_on_record(decision_state, required=True)",
        "    daily.chosen_on = chosen_on_record(decision_state, required=True)",
    ),
    (
        "m13_preview_overwrite_record",
        "src/bts/cli.py",
        "    if sel.decision is not None:\n        daily.chosen_on = chosen_on_record(decision_state, required=False)",
        "    daily.chosen_on = chosen_on_record(decision_state, required=False)",
    ),
    (
        "m14a_validator_outer_streak_bool_for_int",
        "src/bts/picks.py",
        "or type(streak) is not int or streak < 0",
        "or not isinstance(streak, int) or streak < 0",
    ),
    (
        "m14b_validator_active_streak_bool_for_int",
        "src/bts/picks.py",
        "or type(active) is not int or type(best) is not int",
        "or not isinstance(active, int) or type(best) is not int",
    ),
    (
        "m14c_validator_best_streak_bool_for_int",
        "src/bts/picks.py",
        "or type(active) is not int or type(best) is not int",
        "or type(active) is not int or not isinstance(best, int)",
    ),
    (
        "m15_raising_alert_propagates",
        "src/bts/scheduler.py",
        '            try:\n                _alert_contest_state_failure(config, ContestStateError(refusal))\n            except Exception as alert_error:\n                print(f"  Contest-state alert failed: {alert_error}", file=sys.stderr)',
        "            _alert_contest_state_failure(config, ContestStateError(refusal))",
    ),
    (
        "m16_restore_old_missed_pick_text",
        "src/bts/scheduler.py",
        "    if _contest_state_required(config):",
        "    if False:",
    ),
    (
        "m17_saver_consistency_streak_ignored",
        "src/bts/picks.py",
        "policy_saver = policy_saver_available(SaverState(state=saver, season=effective_date.year), streak)",
        "policy_saver = policy_saver_available(SaverState(state=saver, season=effective_date.year), 0)",
    ),
]


def run_mutations(repo_root: Path) -> dict:
    results = {}
    for name, rel_path, target, replacement in MUTANTS:
        with tempfile.TemporaryDirectory() as td:
            mut_root = Path(td)
            shutil.copytree(repo_root / "src", mut_root / "src")
            file_path = mut_root / rel_path
            src_text = file_path.read_text()
            assert src_text.count(target) == 1, f"Mutant {name}: expected 1 match in {rel_path}, found {src_text.count(target)}"
            file_path.write_text(src_text.replace(target, replacement, 1))

            py_path = f"{mut_root / 'src'}:{repo_root}"
            proc = run_pytest(repo_root, py_path, ["-q", "--tb=no", *NEW_TESTS])
            summary_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            m = re.search(r"(\d+)\s+failed", summary_line)
            fail_count = int(m.group(1)) if m else 0
            killed_author = proc.returncode != 0 and fail_count > 0

            killed_adopt = None
            adopt_summary = None
            if not killed_author:
                proc_ad = run_pytest(repo_root, py_path, ["-q", "--tb=no", ADOPT_TEST])
                adopt_summary = proc_ad.stdout.strip().splitlines()[-1] if proc_ad.stdout.strip() else ""
                m_ad = re.search(r"(\d+)\s+failed", adopt_summary)
                killed_adopt = proc_ad.returncode != 0 and bool(m_ad and int(m_ad.group(1)) > 0)
                assert killed_adopt, f"MUTANT SURVIVED BOTH SUITES: {name} -> {adopt_summary}"

            results[name] = {
                "killed_by_author_tests": killed_author,
                "author_failed_count": fail_count,
                "author_summary": summary_line,
                "killed_by_adopt_tests": killed_adopt,
                "adopt_summary": adopt_summary,
            }
    return results


def run_clock_worker(fake_iso: str) -> int:
    # Initialize Cython C-API PyDateTime_IMPORT in pandas before replacing Python's datetime attributes.
    import pandas as _pd  # noqa: F401
    import pytest

    real_dt = _real_dt_mod.datetime
    real_date = _real_dt_mod.date
    pinned = real_dt.fromisoformat(fake_iso)

    class FakeDate(real_date):
        @classmethod
        def today(cls):
            return pinned.date()

    class FakeDatetime(real_dt):
        @classmethod
        def now(cls, tz=None):
            return pinned.astimezone(tz) if tz is not None else pinned.replace(tzinfo=None)

        @classmethod
        def utcnow(cls):
            return pinned.astimezone(_real_dt_mod.timezone.utc).replace(tzinfo=None)

    _real_dt_mod.date = FakeDate
    _real_dt_mod.datetime = FakeDatetime

    return pytest.main(["-p", "no:cacheprovider", *ALL_TOUCHED_TESTS, "-q"])


def run_clocks(repo_root: Path) -> dict:
    out = {}
    for label, iso in (
        ("clock_2027_07_15", "2027-07-15T14:00:00-04:00"),
        ("clock_2028_07_15", "2028-07-15T14:00:00-04:00"),
    ):
        env = os.environ.copy()
        env["TZ"] = "America/New_York"
        env["MPLCONFIGDIR"] = "/tmp/mpl"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = f"{repo_root / 'src'}:{repo_root}"
        proc = subprocess.run(
            [str(repo_root / ".venv" / "bin" / "python"), __file__, "--clock-worker", iso],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
        )
        summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        assert proc.returncode == 0, f"{label} failed:\n{proc.stdout}\n{proc.stderr}"
        out[label] = {"returncode": proc.returncode, "summary": summary}
    return out


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--clock-worker":
        raise SystemExit(run_clock_worker(sys.argv[2]))

    repo_root = Path(__file__).resolve().parents[2]
    base_res = check_base_green_controls(repo_root)
    mut_res = run_mutations(repo_root)
    clk_res = run_clocks(repo_root)

    report_data = {
        "base_on_new_tests": base_res,
        "mutants": mut_res,
        "clocks": clk_res,
        "status": "ALL_ASSERTIONS_PASSED",
    }
    Path(__file__).with_name("probe_9_results.json").write_text(json.dumps(report_data, indent=2))
    author_kills = sum(1 for v in mut_res.values() if v["killed_by_author_tests"])
    survived_author = [k for k, v in mut_res.items() if not v["killed_by_author_tests"]]
    print(
        f"probe_9 PASSED: base had {base_res['passed_count']} green controls & {base_res['failed_count']} RED failures; "
        f"author tests killed {author_kills}/{len(mut_res)} mutants (survived author tests, killed by adopt: {survived_author}); "
        f"2027 ({clk_res['clock_2027_07_15']['summary']}) & "
        f"2028 ({clk_res['clock_2028_07_15']['summary']}) passed."
    )


if __name__ == "__main__":
    main()
