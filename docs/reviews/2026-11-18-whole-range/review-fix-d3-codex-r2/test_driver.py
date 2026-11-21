"""Run the focused set or sequential scratch mutations under the shared wrapper."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from harness import REPO


WORKFLOW = ".github/workflows/deploy.yml"
WORKFLOW_TEST = "tests/test_deploy_workflow.py"
CANARY_TEST = "tests/test_deploy_canary.py"
HEARTBEAT_TEST = "tests/test_heartbeat_staleness.py"
ARGS = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-m", "slow or not slow", "-q", "--tb=short"]


def replace(source, old, new, count=1):
    assert source.count(old) == count, (old, source.count(old))
    return source.replace(old, new)


def commit_before_stamp(source):
    start = source.index('            as_bts 10 mutation /usr/bin/python3 -c "$fs_code" stamp ')
    end = source.index("            trap '' HUP INT TERM PIPE", start)
    lines = source[start:end].splitlines(keepends=True)
    assert len(lines) == 7 and "publish committed" in lines[4]
    return source[:start] + "".join(lines[3:] + lines[:3]) + source[end:]


def heartbeat_after_timestamp(source):
    start = source.index('    if raw.get("state") == "deploy_failed":')
    end = source.index('    ts_str = raw.get("timestamp")', start)
    later = source.index('    if (now.utcoffset() is None)', end)
    return source[:start] + source[end:later] + source[start:end] + source[later:]


def not_started_after_begun(source):
    source = replace(source, '                    begun) contain interrupted 130 ;;', '                    begun) ;;')
    return replace(source, '                if [[ $publication == beginning ]] && (( rc == 3 )); then',
                   '                if [[ $publication == begun ]] && (( rc != 0 )); then\n'
                   '                    unknown_begin\n'
                   '                fi\n'
                   '                if [[ $publication == beginning ]] && (( rc == 3 )); then')


@dataclass(frozen=True)
class Mutation:
    changed: str
    apply: object
    tests: tuple[str, ...]


def nodes(*names):
    return tuple(f"{WORKFLOW_TEST}::{name}" for name in names)


MUTATIONS = {
    "skip-contain-after-reset": Mutation(WORKFLOW,
        lambda s: replace(s, 'step_status reset "$?"', '(( $? == 0 )) || exit 1'),
        (WORKFLOW_TEST, "-k", "failing_phase_command_contains_first_failure and reset")),
    "unconditional-stop-verified": Mutation(WORKFLOW,
        lambda s: replace(s, 'if [[ $user_disabled$user_stopped$legacy_disabled$legacy_stopped$user_proof$system_proof$residual == 0000001 ]]; then', 'if true; then', 2),
        nodes("test_stop_verification_requires_all_evidence[pending-job]")),
    "recovery-ignores-attempt-ids": Mutation(WORKFLOW,
        lambda s: replace(s, 'if (( rc == 3 )) || ! same_ids; then', 'if (( rc == 3 )); then'),
        nodes("test_older_recovery_does_not_mutate_newer_commit", "test_attempt_record_controls_authorization")),
    "commit-before-stamp": Mutation(WORKFLOW, commit_before_stamp,
        (WORKFLOW_TEST, "-k", "stamp_precedes_pass or (owner_signal_at_each_phase_boundary and TERM and stamp and not stage)")),
    "unbounded-wait": Mutation(WORKFLOW,
        lambda s: replace(s, 'run 35 2 wait sleep 30 >/dev/null', 'sleep 30 >/dev/null'),
        nodes("test_healthy_wait_deadline")),
    "redirect-as-page": Mutation(WORKFLOW,
        lambda s: replace(s, '$post_status == 0 && $output =~ ^2[0-9][0-9]$', '$post_status == 0 && $output =~ ^[23][0-9][0-9]$', 2),
        nodes("test_post_requires_exit_zero_and_2xx[post-0-301]")),
    "public-skips-state": Mutation("src/bts/cli.py",
        lambda s: replace(s, 'if delivery not in ("dm", "public") or not _contest_state_required(config):',
                          'if delivery != "dm" or not _contest_state_required(config):'),
        (CANARY_TEST, "-k", "live_modes_use_either_contest_flag and public")),
    "timestamp-before-deploy-failure": Mutation("scripts/check_heartbeat.py", heartbeat_after_timestamp,
        (HEARTBEAT_TEST + "::test_deploy_failure_precedes_timestamp",)),
    "not-started-on-read-error-after-begun": Mutation(WORKFLOW, not_started_after_begun,
        nodes("test_round1_late_signal_uses_confirmed_begin[record-read-126]")),
    "drop-containment-clock-fallback": Mutation(WORKFLOW,
        lambda s: replace(s, '[[ $budget_origin =~ ^[0-9]+$ ]] || budget_origin=0',
                          '[[ $budget_origin =~ ^[0-9]+$ ]] || { budget_origin=0; budget_start=; }', 2),
        nodes("test_round1_clock_failure_does_not_skip_containment")),
    "blank-control-group-counts-as-absent": Mutation(WORKFLOW,
        lambda s: replace(s, '[[ -n $previous_group || -n $group ]] || return 1', '[[ -n $group ]] || return 0', 2),
        nodes("test_round1_unknown_control_group_cannot_verify_stop", "test_round1_blank_control_group_retains_populated_subtree")),
    "drop-globoff": Mutation(WORKFLOW,
        lambda s: replace(s, 'curl --disable --globoff ', 'curl --disable ', 3),
        nodes("test_round1_curl_disables_globs")),
    "collapse-commit-record-status": Mutation(WORKFLOW,
        lambda s: replace(s, 'step_status commit-record "$?"', '(( $? == 0 )) || contain commit-record 1'),
        nodes("test_round1_commit_status_survives_failed_publication", "test_round2_all_commit_timeouts_keep_timeout_status")),
    "carbon-ignore-run-id": Mutation(WORKFLOW,
        lambda s: replace(s, '[[ $r_run == "$RUN_ID" && $r_try == "$RUN_ATTEMPT" ]]', '[[ $r_try == "$RUN_ATTEMPT" ]]', 2),
        nodes("test_round2_different_run_with_same_attempt_cannot_recover")),
    "carbon-follow-redirects": Mutation(WORKFLOW,
        lambda s: replace(s, 'run 20 5 normal curl --disable --globoff', 'run 20 5 normal curl --disable --globoff -L', 2),
        nodes("test_round2_failure_post_never_follows_redirects")),
    "carbon-break-lockfile-helper": Mutation(WORKFLOW,
        lambda s: replace(s, 'elif op == "lockfile":\n                    p, = args',
                          'elif op == "lockfile":\n                    raise ValueError()\n                    p, = args', 2),
        nodes("test_round2_recovery_creates_real_lockfile_and_mark")),
    "carbon-break-mark-helper": Mutation(WORKFLOW,
        lambda s: replace(s, 'elif op == "mark":\n                    p, = args',
                          'elif op == "mark":\n                    raise ValueError()\n                    p, = args', 2),
        nodes("test_round2_recovery_creates_real_lockfile_and_mark")),
    "carbon-recovery-no-lock-wait": Mutation(WORKFLOW,
        lambda s: replace(s, 'flock -w 1500 9', 'flock -n 9', 2),
        nodes("test_round2_recovery_waits_for_a_held_lock")),
}


def run_pytest(options, *, cwd, env, timeout, capture=False):
    print("pytest", json.dumps(options), flush=True)
    output = subprocess.run(ARGS + list(options), cwd=cwd, env=env, timeout=timeout,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None)
    if capture:
        data = output.stdout[:2000] + b"\n[output abbreviated]\n" + output.stdout[-3400:] if len(output.stdout) > 5400 else output.stdout
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    return output.returncode


def mutation_tests(scratch):
    for folder in ("src", "tests", "scripts", ".github", "docs/ops"):
        shutil.copytree(REPO / folder, scratch / folder, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for path in ("pyproject.toml", "INCIDENT.md", "CLAUDE.md"):
        shutil.copy2(REPO / path, scratch / path)
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join(map(str, (scratch / "src", scratch))), GIT_DIR=str(REPO / ".git"))
    for name, spec in MUTATIONS.items():
        print(f"=== {name} ===", flush=True)
        control = run_pytest(spec.tests, cwd=scratch, env=env, timeout=180, capture=True)
        assert control == 0, (name, "unmodified control", control)
        source = scratch / spec.changed
        before = source.read_bytes()
        after = spec.apply(before.decode("utf-8")).encode("utf-8")
        assert before != after
        try:
            source.write_bytes(after)
            outcome = run_pytest(spec.tests, cwd=scratch, env=env, timeout=180, capture=True)
            assert outcome in (0, 1), (name, outcome)
            if outcome == 0:
                family = spec.tests[0].split("::")[0]
                outcome = run_pytest((family,), cwd=scratch, env=env, timeout=6 * 3600, capture=True)
                assert outcome in (0, 1), (name, outcome)
            print(f"RESULT {name} {'DETECTED' if outcome == 1 else 'SURVIVED'}", flush=True)
        finally:
            source.write_bytes(before)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("focus", "mutations"))
    if parser.parse_args().mode == "focus":
        sys.exit(run_pytest((WORKFLOW_TEST, CANARY_TEST, HEARTBEAT_TEST, "tests/test_heartbeat_churn.py",
                             "tests/test_lightgbm_optional.py", "tests/test_web_saver.py"),
                            cwd=REPO, env=os.environ, timeout=8 * 3600))
    with TemporaryDirectory(prefix="wrd3-r2-mutations-", dir="/tmp") as scratch:
        mutation_tests(Path(scratch))
