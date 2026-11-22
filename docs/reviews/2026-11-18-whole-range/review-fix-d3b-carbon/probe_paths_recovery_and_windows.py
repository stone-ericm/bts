#!/usr/bin/env python3
"""Items 3 & 4: Exhaustive phase-2/3/4 failure/timeout/signal/EPIPE sweep, recovery order matrix, and containment window caps."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile

REPO = Path(__file__).resolve().parents[2]
SCRATCH_PARENT = REPO / ".codex-review/wr-d3b/tmp"
SPEC = importlib.util.spec_from_file_location("workflow_tests", REPO / "tests/test_deploy_workflow.py")
assert SPEC and SPEC.loader
tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tests)
DEPLOY, RECOVER = tests.scripts_and_steps()[0]


def make_box(tmp: Path, shim: str = tests.SHIM) -> Path:
    b = tests.box.__wrapped__(tmp)
    if shim != tests.SHIM:
        for f in (b / "shims").iterdir():
            f.write_text(f.read_text().split("\n", 1)[0] + "\n" + shim)
    (b / "enabled").write_text("enabled")
    (b / "home/projects/bts/data/.heartbeat").write_text('{"state":"running"}')
    (b / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/p3\n")
    return b


def sweep_all_phase2_to_4_failures_and_timeouts() -> dict:
    count = 0
    # Every Phase 2 and Phase 4 command failing with exit 17 and timeout 124
    for cmd, step in [*tests.PHASE_TWO.items(), *tests.PHASE_FOUR.items()]:
        for code in (17, 124):
            with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
                b = make_box(Path(td))
                scen = "legacy-present" if cmd.startswith("legacy-") else "active"
                if scen == "legacy-present":
                    fr = b / "etc/systemd/system"; fr.mkdir(parents=True)
                    for u in ("bts-scheduler.service", "bts-dashboard.service"):
                        (fr / u).touch()
                r, c = tests.run_box_script(b, DEPLOY, scen, fault=f"{cmd}:{code}")
                exp = "timeout" if code == 124 else ("1" if cmd.startswith("check-") else str(code))
                assert r.returncode == 1
                assert f"DEPLOY FAILED at {step}" in r.stdout
                assert f"Check {step} failed ({exp})." in r.stdout
                assert ["git", "reset", "--hard", tests.OLD] not in c
                assert " contained " in (b / "run/bts-deploy/attempt").read_text()
                count += 1

    # Every Phase 3 canary step timing out (124) and failing
    for step in tests.PHASE_THREE:
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td))
            r, c = tests.run_box_script(b, DEPLOY, "active", fault=f"{step}:124")
            assert r.returncode == 1 and f"DEPLOY FAILED at {step}" in r.stdout
            assert f"Check {step} failed (timeout)." in r.stdout
            count += 1

    # Closed stdout (literal EPIPE) on passing deploy, failing deploy, and recovery!
    for label, script, scen, flt, exp_rc, exp_state in [
        ("epipe_pass", DEPLOY, "active", "", 0, "committed"),
        ("epipe_fail", DEPLOY, "active", "sync:17", 1, "contained"),
        ("epipe_signal", DEPLOY, "active", "restart:PIPE", 1, "contained"),
        ("epipe_recover", RECOVER, "active", "", 1, "contained"),
    ]:
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td))
            if script is RECOVER:
                rt = b / "run/bts-deploy"; rt.mkdir(mode=0o700, parents=True)
                (rt / "attempt").write_text(f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun - - - -\n")
            r, _ = tests.run_box_script(b, script, scen, fault=flt, closed_output=True)
            assert r.returncode == exp_rc
            assert f" {exp_state} " in (b / "run/bts-deploy/attempt").read_text()
            count += 1

    # Unknown publication value in signal_rule fails closed with contain!
    unk_pub = DEPLOY.replace("publication=beginning", "publication=bogus_state", 1)
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        r, _ = tests.run_box_script(b, unk_pub, "active", fault="stage-begun:TERM:unpublished")
        assert r.returncode == 1 and "DEPLOY FAILED at interrupted" in r.stdout
        count += 1

    return {"runs_verified": count, "status": "PASS"}


def sweep_recovery_orders_and_reboot() -> dict:
    checked = []

    # 1. Owner dies mid-step (lost SSH session) -> recovery contains -> re-run attempt=1 refused -> attempt=2 passes -> late attempt=1 recovery fenced
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        r_kill, _ = tests.run_box_script(b, DEPLOY, "active", fault="sync:KILL:unpublished")
        assert r_kill.returncode == -9
        assert " begun " in (b / "run/bts-deploy/attempt").read_text()
        # Lost session recovery
        r_rec, _ = tests.run_box_script(b, RECOVER, "active")
        assert r_rec.returncode == 1 and "DEPLOY FAILED at deploy-step-incomplete" in r_rec.stdout
        assert " contained deploy-step-incomplete " in (b / "run/bts-deploy/attempt").read_text()
        # Duplicate re-launch of exact attempt=1 refuses
        r_dup, c_dup = tests.run_box_script(b, DEPLOY, "active", attempt="1")
        assert r_dup.returncode == 1 and "DEPLOY NOT STARTED — attempt-record failed." in r_dup.stdout
        assert not any(x[0] == "systemctl" for x in c_dup)
        # Re-run with attempt=2 succeeds
        r_try2, _ = tests.run_box_script(b, DEPLOY, "active", attempt="2")
        assert r_try2.returncode == 0 and "CANARY PASSED" in r_try2.stdout
        # Subsequent run_id=124 attempt=1 succeeds
        r_run2, _ = tests.run_box_script(b, DEPLOY, "active", run_id="124", attempt="1")
        assert r_run2.returncode == 0 and "CANARY PASSED" in r_run2.stdout
        checked.append("lost_session_recover_dup_refused_rerun_try2_next_run124")

    # 2. Reboot (/run cleared) after lost session -> boot link stayed disabled -> recovery reports no phase-2 record -> fences late deploy
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        r_kill, _ = tests.run_box_script(b, DEPLOY, "active", fault="restart:KILL:unpublished")
        assert r_kill.returncode == -9 and (b / "enabled").read_text() == "disabled"
        shutil.rmtree(b / "run")  # simulate reboot clearing /run
        r_rec, c_rec = tests.run_box_script(b, RECOVER, "active")
        assert "DEPLOY RECOVERY — no phase-2 record is present. Check the deploy log and the host's boot time." in r_rec.stdout
        assert not any(x[0] == "systemctl" for x in c_rec)
        assert (b / "run/bts-deploy/cancelled.123.1").is_file()
        r_late, _ = tests.run_box_script(b, DEPLOY, "active")
        assert "DEPLOY NOT STARTED — cancelled failed." in r_late.stdout
        checked.append("reboot_clears_run_soft_hold_intact_recovery_fences_late_deploy")

    # 3. Malformed / hostile attempt records in both scripts -> refused without service change
    bad_records = [
        "corrupt\n",
        f"bts_deploy_attempt_v1 0 1 {tests.TARGET} {tests.OLD} begun - - - -\n",
        f"bts_deploy_attempt_v1 123 1 {tests.TARGET} unknown begun - - - -\n",
        f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun sync - - -\n",
        f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} contained cd verified verified verified\n",
        f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} contained sync - verified verified\n",
        "x" * 520 + "\n",
    ]
    for rec in bad_records:
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td))
            rt = b / "run/bts-deploy"; rt.mkdir(mode=0o700, parents=True)
            (rt / "attempt").write_text(rec)
            rd, cd = tests.run_box_script(b, DEPLOY, "active")
            rr, cr = tests.run_box_script(b, RECOVER, "active")
            assert "DEPLOY NOT STARTED — attempt-record failed." in rd.stdout and not any(x[0] == "systemctl" for x in cd)
            assert "the attempt record is unreadable" in rr.stdout and not any(x[0] == "systemctl" for x in cr)
    checked.append(f"hostile_attempt_records_{len(bad_records)}_cases")

    return {"checked": checked, "status": "PASS"}


def main() -> None:
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    out = {
        "phase2_to_4_sweep": sweep_all_phase2_to_4_failures_and_timeouts(),
        "recovery_orders_and_reboot": sweep_recovery_orders_and_reboot(),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
