"""Run the current assertions against separate scratch workflow edits."""
from pathlib import Path
from shutil import copy2
from subprocess import run
from tempfile import TemporaryDirectory
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKFLOW = Path(".github/workflows/deploy.yml")
TEST = Path("tests/test_deploy_workflow.py")

CASES = [
    ("forget-legacy-group", 'elif (( legacy_group_status == 0 )) && post_group=$(scheduler_group system_unit "$scheduler") &&',
     'elif true; then\n                                    system_proof=0\n                                elif (( legacy_group_status == 0 )) && post_group=$(scheduler_group system_unit "$scheduler") &&',
     "d3b_recorded_legacy_group"),
    ("drop-residual-window", 'deadline=$((containment_deadline - 75000))',
     'deadline=$((containment_deadline - 60000))', "d3b_containment_windows"),
    ("unreadable-means-unpublished", 'elif (( rc == 3 )) || {',
     'elif (( rc == 3 || rc == 126 )) || {', "d3b_unreadable_record_while_beginning"),
    ("beginning-commit-success", 'elif (( rc == 3 )) || {',
     'elif (( rc == 0 )) && same_record && [[ $r_state == committed ]]; then\n                    answer=success\n                elif (( rc == 3 )) || {',
     "d3b_same_ids_while_beginning"),
    ("curl-7-200", '[[ $2 == 0 && $3 =~ ^2[0-9][0-9]$ ]]',
     '[[ ( $2 == 0 || $2 == 7 ) && $3 =~ ^2[0-9][0-9]$ ]]', "post_requires_exit_zero_and_2xx"),
    ("report-outside-prove", 'prove report\n                prove recheck',
     "prove report\n                log 'Scheduler stopped: verified.'\n                prove recheck", "only_the_proof_function"),
    ("drop-link-check", '[[ ! -e $legacy_dir/$scheduler && ! -L $legacy_dir/$scheduler ]]',
     '[[ ! -e $legacy_dir/$scheduler ]]', "d3b_unproved_legacy_absence"),
    ("drop-pre-stop-status", 'uid_status == 0 && user_group_status == 0',
     'uid_status == 0', "d3b_failed_user_group_query"),
    ("forget-run-id", '[[ $r_run == "$RUN_ID" && $r_try == "$RUN_ATTEMPT" ]]',
     '[[ $r_try == "$RUN_ATTEMPT" ]]', "round2_different_run"),
    ("follow-redirect", 'curl --disable --globoff --noproxy',
     'curl --disable --globoff -L --noproxy', "round2_failure_post_never_follows"),
    ("break-lockfile", 'if op == "lockfile":\n',
     'if op == "lockfile":\n                    raise ValueError()\n', "round2_recovery_creates_real_lockfile"),
    ("break-mark", 'elif op == "mark":\n',
     'elif op == "mark":\n                    raise ValueError()\n', "round2_recovery_creates_real_lockfile"),
    ("do-not-wait-for-lock", 'flock -w 1500 9', 'flock -n 9', "round2_recovery_waits_for_a_held_lock"),
    ("omit-recheck-evidence", '[[ $repeat == yes && $stop$marker$page == verifiedverifiedverified ]]',
     '[[ $repeat == yes ]]', "d3b_recovery_reproves_stop_and_marker"),
    ("loose-commit-match", '&& same_record && [[ $r_state == committed ]]',
     '&& [[ $r_state == committed ]]', "d3b_success_needs_this_attempts_exact_commit"),
    ("wrong-pid-proof", 'pid=$(field MainPID "$output") && [[ $pid == 0 ]]',
     'pid=$(field MainPID "$output") && [[ $pid =~ ^[0-9]+$ ]]', "stop_verification_requires_all_evidence"),
    ("extra-report-time", 'timeout -k 1 1 /bin/bash -c',
     'timeout -k 5 1 /bin/bash -c', "d3b_containment_windows or only_the_proof_function or d3b_recovery_reproves_stop_and_marker"),
]

with TemporaryDirectory(prefix="mutations-", dir=HERE) as name:
    scratch = Path(name)
    for path in (WORKFLOW, TEST):
        (scratch / path).parent.mkdir(parents=True, exist_ok=True)
        copy2(ROOT / path, scratch / path)
    original = (scratch / WORKFLOW).read_text()
    for label, old, new, selection in CASES:
        if old not in original:
            raise AssertionError((label, old))
        (scratch / WORKFLOW).write_text(original.replace(old, new))
        result = run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", str(TEST),
                      "-k", selection, "-q", "--tb=short"], cwd=scratch,
                     capture_output=True, text=True, timeout=90)
        print(f"{label}: rc={result.returncode} ({selection})\n{result.stdout}{result.stderr}", flush=True)
        if result.returncode not in (0, 1) or " passed" not in result.stdout and " failed" not in result.stdout:
            raise SystemExit(f"{label}: test selection did not run")
