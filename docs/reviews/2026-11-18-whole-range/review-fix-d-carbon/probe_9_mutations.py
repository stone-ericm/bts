#!/usr/bin/env python3
"""Item 9 mutation sweep against an isolated scratch copy under .codex-review/wr-d/mut_scratch/.
NEVER run before 16:45 ET, and ONLY invoke via ~/bts-runs/yield_run.sh!"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

DEPLOY_TESTS = [
    "tests/test_deploy_canary.py",
    "tests/test_deploy_hold_monitor.py",
    "tests/test_deploy_hold_tool.py",
    "tests/test_deploy_workflow.py",
    "tests/test_entry_intent.py",
    "tests/test_install_deploy_guard.py",
    "tests/test_web_saver.py",
    "tests/test_heartbeat_staleness.py",
    "tests/test_heartbeat_churn.py",
]


def exact_replace(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    assert count == 1, f"Expected 1 occurrence of {old!r} in {path}, found {count}"
    path.write_text(text.replace(old, new, 1))


MUTANTS = [
    (
        "1_check_clear_accepts_pending",
        "scripts/install-deploy-guard-hetzner.sh",
        'return record is None or (record["state"] == "activating" and owner_alive(root, record))',
        'return record is None or record["state"] == "pending" or (record["state"] == "activating" and owner_alive(root, record))',
    ),
    (
        "2_drop_writer_lock_around_arm",
        "scripts/install-deploy-guard-hetzner.sh",
        'with lock(root, "writers.lock", fcntl.LOCK_EX):\n                    armed = int(time.time())',
        'if True:\n                    armed = int(time.time())',
    ),
    (
        "3_release_before_activation_verified",
        "scripts/install-deploy-guard-hetzner.sh",
        '                record = transition(root, record, "activating", "activation")\n'
        '                step = "activation"\n'
        '                activate_scheduler(uid, authorized_at)\n'
        '                publish_json(root, run_name(run_id, "evidence"), {\n'
        '                    "activation": "verified", "stop": "not-attempted", "page": "not-attempted",\n'
        '                })\n'
        '                step = "release"\n'
        '                release(root, record, "deploy")',
        '                record = transition(root, record, "activating", "activation")\n'
        '                publish_json(root, run_name(run_id, "evidence"), {\n'
        '                    "activation": "verified", "stop": "not-attempted", "page": "not-attempted",\n'
        '                })\n'
        '                release(root, record, "deploy")\n'
        '                step = "activation"\n'
        '                activate_scheduler(uid, authorized_at)',
    ),
    (
        "4_skip_guard_stale",
        "scripts/install-deploy-guard-hetzner.sh",
        '    if hashlib.sha256(script).hexdigest().encode() + b"\\n" != fingerprint:\n'
        '        raise Refused("guard-stale", previous)',
        '    if False:\n'
        '        raise Refused("guard-stale", previous)',
    ),
    (
        "5_monitor_pings_success_with_hold_present",
        "scripts/check_heartbeat.py",
        '        if decision == "fail":\n'
        '            print(reason)\n'
        '            if args.ping_url:\n'
        '                ping_deploy_failure(args.ping_url, reason)\n'
        '            raise SystemExit(1)',
        '        if decision == "fail":\n'
        '            print(reason)\n'
        '            if args.ping_url:\n'
        '                ping(args.ping_url)\n'
        '            raise SystemExit(0)',
    ),
    (
        "6_deploy_canary_skips_contest_read_for_dm",
        "src/bts/cli.py",
        '    if intent.intent == "research":',
        '    if intent.intent == "research" or intent.delivery == "dm":',
    ),
    (
        "7_hook_ignores_incomplete_failed_record",
        "scripts/install-deploy-guard-hetzner.sh",
        '        if record["state"] != "failed":\n'
        '            record = transition(root, record, "failed", "owner-ended")\n'
        '        contained = containment(root, record, bts_uid())',
        '        if record["state"] == "failed":\n'
        '            return 0\n'
        '        record = transition(root, record, "failed", "owner-ended")\n'
        '        contained = containment(root, record, bts_uid())',
    ),
    (
        "8_accept_saver_post_when_guard_unavailable",
        "src/bts/web.py",
        '            except OSError:\n                clear = False',
        '            except OSError:\n                clear = True',
    ),
]


def main() -> int:
    here = Path(__file__).resolve().parent
    scratch_base = Path(tempfile.mkdtemp(prefix="mut_scratch_", dir=here))
    try:
        archive = subprocess.Popen(
            ["git", "archive", "HEAD", "src", "scripts", "tests", ".github", "pyproject.toml"],
            cwd=REPO, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        )
        untar = subprocess.run(
            ["tar", "-x", "-C", str(scratch_base)],
            stdin=archive.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30,
        )
        archive.stdout.close()
        assert archive.wait(timeout=10) == 0 and untar.returncode == 0, untar.stderr

        results = {}
        survivors = []
        for name, rel_path, old_s, new_s in MUTANTS:
            target = scratch_base / rel_path
            backup = target.read_bytes()
            try:
                exact_replace(target, old_s, new_s)
                env = dict(os.environ)
                env["PYTHONPATH"] = f"{scratch_base / 'src'}:{scratch_base}"
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                res = subprocess.run(
                    [PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", *DEPLOY_TESTS],
                    cwd=scratch_base, env=env, stdin=subprocess.DEVNULL,
                    capture_output=True, text=True, timeout=120,
                )
                killed = res.returncode != 0
                if not killed:
                    survivors.append(name)
                results[name] = {
                    "killed": killed,
                    "returncode": res.returncode,
                    "summary": res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "",
                }
            finally:
                target.write_bytes(backup)

        report = {"total_mutants": len(MUTANTS), "survivors": survivors, "mutants": results}
        (here / "probe_9_mutations.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        shutil.rmtree(scratch_base, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
