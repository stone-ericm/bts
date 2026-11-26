"""Apply each fault to a scratch checkout and run the named detecting tests."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WORKFLOW = ".github/workflows/deploy.yml"
WORKFLOW_TEST = "tests/test_deploy_workflow.py"
CANARY_TEST = "tests/test_deploy_canary.py"
HEARTBEAT_TEST = "tests/test_heartbeat_staleness.py"
PYTEST_ARGS = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-m", "slow or not slow", "-q", "--tb=short"]


def replace(source, old, new, count=1):
    assert source.count(old) == count, (old, source.count(old))
    return source.replace(old, new)


def commit_before_stamp(source):
    start = source.index('            as_bts 10 mutation /usr/bin/python3 -c "$fs_code" stamp ')
    final = source.index("            trap '' HUP INT TERM PIPE", start)
    lines = source[start:final].splitlines(keepends=True)
    assert len(lines) == 5 and 'publish committed' in lines[3]
    return source[:start] + "".join(lines[3:] + lines[:3]) + source[final:]


def heartbeat_after_timestamp(source):
    start = source.index('    if raw.get("state") == "deploy_failed":')
    end = source.index('    ts_str = raw.get("timestamp")', start)
    after_timestamp = source.index('    if (now.utcoffset() is None)', end)
    return source[:start] + source[end:after_timestamp] + source[start:end] + source[after_timestamp:]


@dataclass(frozen=True)
class Mutation:
    changed: str
    apply: object
    tests: tuple[str, ...]


MUTATIONS = {
    "skip-contain-after-reset": Mutation(WORKFLOW, lambda s: replace(s, 'step_status reset "$?"', '(( $? == 0 )) || exit 1'),
        (WORKFLOW_TEST, "-k", "failing_phase_command_contains_first_failure and reset")),
    "unconditional-stop-verified": Mutation(WORKFLOW,
        lambda s: replace(s, 'if [[ $user_disabled$user_stopped$legacy_disabled$legacy_stopped$user_proof$system_proof$residual == 0000001 ]]; then', 'if true; then', 2),
        (WORKFLOW_TEST + "::test_stop_verification_requires_all_evidence[pending-job]",)),
    "recovery-ignores-attempt-ids": Mutation(WORKFLOW,
        lambda s: replace(s, 'if (( rc == 3 )) || ! same_ids; then', 'if (( rc == 3 )); then'),
        (WORKFLOW_TEST + "::test_older_recovery_does_not_mutate_newer_commit",
         WORKFLOW_TEST + "::test_attempt_record_controls_authorization")),
    "commit-before-stamp": Mutation(WORKFLOW, commit_before_stamp,
        (WORKFLOW_TEST, "-k", "stamp_precedes_pass or (owner_signal_at_each_phase_boundary and TERM and stamp and not stage)")),
    "unbounded-wait": Mutation(WORKFLOW,
        lambda s: replace(s, 'run 35 2 wait sleep 30 >/dev/null', 'sleep 30 >/dev/null'),
        (WORKFLOW_TEST + "::test_healthy_wait_deadline",)),
    "redirect-as-page": Mutation(WORKFLOW,
        lambda s: replace(s, '$post_status == 0 && $output =~ ^2[0-9][0-9]$', '$post_status == 0 && $output =~ ^[23][0-9][0-9]$', 2),
        (WORKFLOW_TEST + "::test_post_requires_exit_zero_and_2xx[post-0-301]",)),
    "public-skips-state": Mutation("src/bts/cli.py",
        lambda s: replace(s, 'if delivery not in ("dm", "public") or not _contest_state_required(config):',
                          'if delivery != "dm" or not _contest_state_required(config):'),
        (CANARY_TEST, "-k", "live_modes_use_either_contest_flag and public")),
    "timestamp-before-deploy-failure": Mutation("scripts/check_heartbeat.py", heartbeat_after_timestamp,
        (HEARTBEAT_TEST + "::test_deploy_failure_precedes_timestamp",)),
}


def run_pytest(options, *, cwd, env, limit):
    print("pytest", json.dumps(options), flush=True)
    return subprocess.run(PYTEST_ARGS + list(options), cwd=cwd, env=env, timeout=limit).returncode


def test_mutations(scratch):
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join(map(str, (scratch / "src", scratch))), GIT_DIR=str(REPO / ".git"))
    for name, mutation in MUTATIONS.items():
        print(f"=== {name} ===", flush=True)
        control = run_pytest(mutation.tests, cwd=scratch, env=env, limit=180)
        assert control == 0, (name, "unmodified control", control)
        path = scratch / mutation.changed
        original = path.read_bytes()
        try:
            changed = mutation.apply(original.decode("utf-8"))
            assert changed.encode() != original
            path.write_text(changed, encoding="utf-8")
            outcome = run_pytest(mutation.tests, cwd=scratch, env=env, limit=180)
            assert outcome in (0, 1), (name, "unexpected pytest exit", outcome)
            if outcome == 0:
                family = mutation.tests[0].split("::")[0]
                full_control = run_pytest((family,), cwd=scratch, env=env, limit=6 * 3600)
                path.write_bytes(original)
                if full_control != 0:
                    baseline = run_pytest((family,), cwd=scratch, env=env, limit=6 * 3600)
                    assert baseline == 0, (name, "unmodified family", baseline)
                outcome = full_control
            print(f"RESULT {name} {'DETECTED' if outcome == 1 else 'SURVIVED'}", flush=True)
        finally:
            path.write_bytes(original)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("focus", "mutations"))
    args = parser.parse_args()
    if args.mode == "focus":
        sys.exit(run_pytest((WORKFLOW_TEST, CANARY_TEST, HEARTBEAT_TEST, "tests/test_heartbeat_churn.py",
                             "tests/test_lightgbm_optional.py", "tests/test_web_saver.py"),
                            cwd=REPO, env=os.environ, limit=8 * 3600))
    test_mutations(HERE / "mutation-copy")
