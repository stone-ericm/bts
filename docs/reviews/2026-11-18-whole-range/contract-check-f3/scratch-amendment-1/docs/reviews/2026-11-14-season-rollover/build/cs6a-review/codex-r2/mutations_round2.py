"""Run the round-two cases on HEAD, round one and targeted scratch regressions."""

import local_guard

from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from schedule_round2 import HERE, ROOT, archive


LEAD = "tests/test_live_gate_cs6a.py::"
ADOPTED = "tests/test_live_gate_cs6a_review.py"
LATE_DRY = LEAD + "test_dry_run_default_date_is_resolved_before_the_configuration_is_loaded"
EARLY_REAL = LEAD + "test_real_run_refusal_still_computes_no_default_date"
OUTCOME = ADOPTED + "::test_day_outcome_ignores_live_gate_problem_in_gather_and_cli"
LOAD_SNAPSHOT = ADOPTED + "::test_orchestrate_cli_never_rereads_config_file_even_via_direct_open"
TRUTHY = ADOPTED + "::test_accepted_outward_truthy_nonbool_configs_still_block_uncovered_state_in_workers"
PRIVATE = ADOPTED + "::test_accepted_private_schedule_locks_locally_without_dm_or_post"
RUN = ADOPTED + "::test_bts_run_locked_already_posted_and_dated_corrupt_switch_and_read3_outside_window"

MUTATIONS = (
    ("late-dry-date", "cli.py", "    if dry_run and date is None:\n        date = _today_et()\n\n", "", LATE_DRY, 2),
    ("early-real-date", "cli.py", "    if dry_run and date is None:\n", "    if date is None:\n", EARLY_REAL, 1),
    ("classify-with-gate", "day_outcome.py", "check_entry_intent(config, include_live_gate=False)",
     "check_entry_intent(config)", OUTCOME, 3),
    ("reopen-checked-config", "orchestrator.py", "    if config is None:\n        config = load_config(config_path)\n",
     "    if config is None:\n        config = load_config(config_path)\n"
     "    else:\n        with open(config_path, 'rb') as handle:\n            config = tomllib.load(handle)\n", LOAD_SNAPSHOT, 1),
    ("optional-selection", "orchestrator.py",
     '    progress.mark("loading_decision_state")\n'
     "    decision_state = load_decision_streak_state(\n        picks_dir,\n        require_contest_state=_contest_state_required(config),",
     '    progress.mark("loading_decision_state")\n'
     "    decision_state = load_decision_streak_state(\n        picks_dir,\n        require_contest_state=False,", TRUTHY, 2),
    ("optional-cached-delivery", "scheduler.py", "            require_contest_state=_contest_state_required(config),",
     "            require_contest_state=False,", TRUTHY, 2),
    ("private-post", "scheduler.py", '    if mode == "private":\n        save_pick(daily, picks_dir)\n',
     '    if mode == "private":\n        from bts.posting import post_to_bluesky\n'
     '        post_to_bluesky("accidental private post")\n        save_pick(daily, picks_dir)\n', PRIVATE, 1),
    ("optional-read-3", "cli.py", "    # Post to Bluesky if appropriate\n    try:\n"
     "        decision_state = load_decision_streak_state(picks_path, require_contest_state=True)",
     "    # Post to Bluesky if appropriate\n    try:\n"
     "        decision_state = load_decision_streak_state(picks_path, require_contest_state=False)", RUN, 1),
    ("already-posted-read-2", "cli.py", "        # Catch-up posting if needed\n        if not result.daily.bluesky_posted:\n",
     "        # Catch-up posting if needed\n        if True:\n", RUN, 1),
    ("dated-manual-reads-switch", "contest_state.py", "    if require_contest_state and not (\n"
     "            contest is not None and dated_observation_in_effect(contest, now_dt)):\n",
     "    if require_contest_state:\n", RUN, 1),
)


def child(src, temp, fail_count, *nodes):
    sys.path.insert(0, src)
    from bts import cli
    import pytest

    assert Path(cli.__file__).is_relative_to(Path(src)), cli.__file__

    class Checks:
        passed = []
        failed = []
        invalid = []

        def pytest_runtest_logreport(self, report):
            if report.when == "call" and report.passed:
                self.passed.append(report.nodeid)
            elif report.when == "call" and report.failed:
                self.failed.append(report.nodeid)
            elif report.failed or report.skipped:
                self.invalid.append((report.when, report.nodeid))

    counts = Checks()
    result = pytest.main(["-q", "-o", "addopts=", "-p", "no:cacheprovider", "--tb=short",
                          "--basetemp", temp, *nodes], plugins=[counts])
    assert not local_guard.network, local_guard.network
    assert not counts.invalid, counts.invalid
    expected = pytest.ExitCode.TESTS_FAILED if fail_count else pytest.ExitCode.OK
    assert result == expected, (result, expected)
    assert len(counts.failed) == fail_count and len(counts.passed) + fail_count > 0, counts.failed
    print(f"Observed {len(counts.passed)} pass, {len(counts.failed)} fail as expected", flush=True)
    print(*counts.passed, sep="\n")


def run(label, src, scratch, expected, *nodes):
    print(f"=== {label} ===", flush=True)
    args = [sys.executable, str(__file__), "child", str(src), str(scratch / (label + "-tmp")), str(expected), *nodes]
    subprocess.run(args, cwd=ROOT, check=True)


def replace_once(path, old, new):
    source = path.read_text()
    assert source.count(old) == 1, (path, old, source.count(old))
    path.write_text(source.replace(old, new, 1))


def main():
    src = ROOT / "src"
    with TemporaryDirectory(prefix="mutant-runs-", dir=HERE) as folder:
        scratch = Path(folder)
        run("head", src, scratch, 0, LATE_DRY, EARLY_REAL, ADOPTED)
        round_one = archive("cad7c5ec", scratch / "round-one")
        run("round-one", round_one, scratch, 2, LATE_DRY, EARLY_REAL, ADOPTED)
        for label, file, old, new, node, failures in MUTATIONS:
            mutant = scratch / label / "src"
            shutil.copytree(src, mutant, ignore=shutil.ignore_patterns("__pycache__"))
            replace_once(mutant / "bts" / file, old, new)
            run(label, mutant, scratch, failures, node)

        # Reuse the adopted assertions with the legacy input its module docstring names.
        replacement = scratch / "test_legacy_private.py"
        replacement.write_text((ROOT / ADOPTED).read_text())
        replace_once(replacement, "'[scheduler]\\npick_delivery = \"private\"\\nentry_intent = \"research\"\\n'",
                     "'[scheduler]\\nprivate_mode = \"false\"\\nentry_intent = \"research\"\\n'")
        run("legacy-private-control", src, scratch, 0, str(replacement) + "::" + PRIVATE.split("::")[1])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]), *sys.argv[5:])
    else:
        main()
