#!/usr/bin/env python3
"""Round-2 Item 4 mutation sweep:
Re-runs all 8 Round-1 mutants on HEAD (confirming the pending mutant is now caught)
PLUS 16 new mutants (one per F item, F1 through F16) against an isolated scratch copy."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

R2_DIR = Path(__file__).resolve().parent
REPO = R2_DIR.parents[2]
PYTHON = sys.executable

FOCUSED_TESTS = [
    "tests/test_deploy_amendment3.py",
    "tests/test_deploy_canary.py",
    "tests/test_deploy_hold_monitor.py",
    "tests/test_deploy_hold_tool.py",
    "tests/test_deploy_workflow.py",
    "tests/test_entry_intent.py",
    "tests/test_install_deploy_guard.py",
    "tests/test_heartbeat_staleness.py",
    "tests/test_heartbeat_churn.py",
    "tests/test_lightgbm_optional.py",
]


def exact_replace(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    assert count == 1, f"Expected 1 occurrence of {old!r} in {path.name}, found {count}"
    path.write_text(text.replace(old, new, 1))


# (name, rel_path, primary_pytest_args, old_str, new_str)
MUTANTS = [
    # --- Round-1 Mutation Set (8 mutants) ---
    (
        "R1_1_check_clear_accepts_pending",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_hold_tool.py", "tests/test_deploy_amendment3.py"],
        'return record is None or (record["state"] == "activating" and owner_alive(root, record))',
        'return record is None or record["state"] == "pending" or (record["state"] == "activating" and owner_alive(root, record))',
    ),
    (
        "R1_2_drop_writer_lock_around_arm",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_amendment3.py", "tests/test_deploy_hold_tool.py"],
        'with lock(root, "writers.lock", fcntl.LOCK_EX):\n                    armed = int(time.time())',
        'if True:\n                    armed = int(time.time())',
    ),
    (
        "R1_3_release_before_activation_verified",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_hold_tool.py"],
        '                step = "activation"\n'
        '                activate_scheduler(uid, authorized_at)\n'
        '                try:\n'
        '                    release_date = canary()\n'
        '                except Refused:\n'
        '                    raise Refused("activation") from None\n'
        '                if release_date != certified_date:\n'
        '                    raise Refused("activation")\n'
        '                publish_json(root, run_name(run_id, "evidence"), {\n'
        '                    "activation": "verified", "stop": "not-attempted", "page": "not-attempted",\n'
        '                })\n'
        '                step = "release"\n'
        '                release(root, record, "deploy", authorized_at=authorized_at, certified_date=certified_date)',
        '                publish_json(root, run_name(run_id, "evidence"), {\n'
        '                    "activation": "verified", "stop": "not-attempted", "page": "not-attempted",\n'
        '                })\n'
        '                release(root, record, "deploy", authorized_at=authorized_at, certified_date=certified_date)\n'
        '                step = "activation"\n'
        '                activate_scheduler(uid, authorized_at)',
    ),
    (
        "R1_4_skip_guard_stale",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_hold_tool.py"],
        '    if hashlib.sha256(script).hexdigest().encode() + b"\\n" != fingerprint:\n'
        '        raise Refused("guard-stale", previous)',
        '    if False:\n'
        '        raise Refused("guard-stale", previous)',
    ),
    (
        "R1_5_monitor_pings_success_with_hold_present",
        "scripts/check_heartbeat.py",
        ["tests/test_deploy_hold_monitor.py"],
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
        "R1_6_deploy_canary_skips_contest_read_for_dm",
        "src/bts/cli.py",
        ["tests/test_deploy_canary.py"],
        '    if intent.intent == "research":',
        '    if intent.intent == "research" or intent.delivery == "dm":',
    ),
    (
        "R1_7_hook_ignores_incomplete_failed_record",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_hold_tool.py"],
        '        if record["state"] != "failed":\n'
        '            record = transition(root, record, "failed", "owner-ended")\n'
        '        return 0 if finish_incident(root, record) else 1',
        '        if record["state"] == "failed":\n'
        '            return 0\n'
        '        record = transition(root, record, "failed", "owner-ended")\n'
        '        return 0 if finish_incident(root, record) else 1',
    ),
    (
        "R1_8_accept_saver_post_when_guard_unavailable",
        "src/bts/web.py",
        ["tests/test_web_saver.py", "-k", "not socket_wiring"],
        '            except OSError:\n                clear = False',
        '            except OSError:\n                clear = True',
    ),
    # --- One Mutant per F Item (F1..F16) ---
    (
        "F1_crontab_skip_quote_stripping",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_amendment3.py"],
        'invoked = set(WRITER_INVOKES.findall(line.translate({ord("\'"): None, ord(\'"\'): None})))',
        'invoked = set(WRITER_INVOKES.findall(line))',
    ),
    (
        "F2_owner_alive_accepts_deactivating",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_amendment3.py"],
        'if fields.get("ActiveState") != "active" or not re.fullmatch(r"[1-9][0-9]*", pid):',
        'if fields.get("ActiveState") not in ("active", "deactivating") or not re.fullmatch(r"[1-9][0-9]*", pid):',
    ),
    (
        "F3_unit_allows_exit_1_in_success_status",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_amendment3.py", "tests/test_install_deploy_guard.py"],
        'or "1" in fields["SuccessExitStatus"].split()',
        'or False',
    ),
    (
        "F4_skip_stop_dashboard_under_writer_lock",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_amendment3.py"],
        'if not verified_stop_dashboard(uid):\n                        raise Refused(step)',
        'if False:\n                        raise Refused(step)',
    ),
    (
        "F5_canary_skips_validate_runtime_config",
        "src/bts/cli.py",
        ["tests/test_deploy_canary.py"],
        'if validate_runtime_config(config) or not intent.ok:',
        'if not intent.ok:',
    ),
    (
        "F6_skip_post_activation_date_check",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_amendment3.py"],
        'if release_date != certified_date:\n                    raise Refused("activation")',
        'if False:\n                    raise Refused("activation")',
    ),
    (
        "F7_revert_incident_contest_recovery_command",
        "INCIDENT.md",
        ["tests/test_deploy_amendment3.py"],
        'uv run --no-sync bts season-contract status --picks-dir <the configured picks_dir>',
        'bts season-contract status',
    ),
    (
        "F8_deploy_release_uses_wall_clock_instead_of_authorized_at",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_amendment3.py"],
        '"released_at": int(time.time()) if authorized_at is None else authorized_at,',
        '"released_at": int(time.time()),',
    ),
    (
        "F9_hand_release_reverts_to_1800s_no_recovery",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_amendment3.py"],
        'window(target, target, time.time() + 30, minimum=30)',
        'window("", target, time.time() + 30)',
    ),
    (
        "F10_absent_guard_dir_still_pages_lookup_failed",
        "scripts/check_heartbeat.py",
        ["tests/test_deploy_hold_monitor.py"],
        '            except FileNotFoundError:\n                return _legacy_check(args)',
        '            except FileNotFoundError:\n                pass',
    ),
    (
        "F11_do_not_retry_claimed_page",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_amendment3.py"],
        'if claim in ("not-attempted", "claimed") and type(attempts) is int and 0 <= attempts < 2:',
        'if claim == "not-attempted" and type(attempts) is int and 0 <= attempts < 2:',
    ),
    (
        "F12_monitor_suppresses_dead_owner_pending",
        "scripts/check_heartbeat.py",
        ["tests/test_deploy_hold_monitor.py"],
        'if state in ("pending", "activating") and age <= 1380 and _owner_alive(hold, record["owner_unit"]):',
        'if state in ("pending", "activating") and age <= 1380:',
    ),
    (
        "F13_monitor_always_invokes_production_guard_tool",
        "scripts/check_heartbeat.py",
        ["tests/test_deploy_hold_monitor.py"],
        '    if hold == HOLD_PATH:\n        result = subprocess.run(',
        '    if True:\n        result = subprocess.run(',
    ),
    (
        "F14_drop_canary_classified_line_print",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_amendment3.py"],
        '    print(line, flush=True)\n    if code != 0:',
        '    if code != 0:',
    ),
    (
        "F15_tailscale_failure_refuses_instead_of_loopback",
        "scripts/install-deploy-guard-hetzner.sh",
        ["tests/test_deploy_hold_tool.py"],
        '    except Refused:\n        address = ""',
        '    except Refused:\n        raise',
    ),
    (
        "F16_remove_box_canary_step_from_a7",
        "docs/ops/2027-season-start.md",
        ["tests/test_deploy_amendment3.py"],
        'uv run --no-sync bts deploy-canary --config ~/.bts-orchestrator.toml',
        'bts deploy-canary',
    ),
]


def main() -> int:
    scratch_base = Path(tempfile.mkdtemp(prefix="r2_mut_scratch_", dir=R2_DIR))
    try:
        archive = subprocess.Popen(
            [
                "git", "archive", "HEAD",
                "src", "scripts", "tests", ".github", "config", "docs/ops/2027-season-start.md",
                "CLAUDE.md", "INCIDENT.md", "pyproject.toml",
            ],
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
        for name, rel_path, primary_args, old_s, new_s in MUTANTS:
            target = scratch_base / rel_path
            backup = target.read_bytes()
            try:
                exact_replace(target, old_s, new_s)
                env = dict(os.environ)
                env["PYTHONPATH"] = f"{scratch_base / 'src'}:{scratch_base}"
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                env["MPLCONFIGDIR"] = str(scratch_base / ".mpl")
                res = subprocess.run(
                    [PYTHON, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *primary_args],
                    cwd=scratch_base, env=env, stdin=subprocess.DEVNULL,
                    capture_output=True, text=True, timeout=180,
                )
                if res.returncode == 0:
                    res = subprocess.run(
                        [PYTHON, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *FOCUSED_TESTS],
                        cwd=scratch_base, env=env, stdin=subprocess.DEVNULL,
                        capture_output=True, text=True, timeout=180,
                    )
                killed = res.returncode != 0
                if not killed:
                    survivors.append(name)
                lines = [ln for ln in res.stdout.splitlines() if "FAILED" in ln or "passed" in ln]
                results[name] = {
                    "killed": killed,
                    "returncode": res.returncode,
                    "detail": lines[-2:] if lines else res.stdout[-240:],
                }
                print(f"{name}: killed={killed} rc={res.returncode}", flush=True)
            finally:
                target.write_bytes(backup)

        report = {"total_mutants": len(MUTANTS), "survivors": survivors, "mutants": results}
        (R2_DIR / "r2_probe_mutants.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 0
    finally:
        shutil.rmtree(scratch_base, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
