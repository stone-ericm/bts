#!/usr/bin/env python3
"""Item 2 (The Class): Concrete false-green counterexamples in D3b's stop proof (90a9b6f0), plus in-memory verification of smallest fix."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

REPO = Path(__file__).resolve().parents[2]
SCRATCH_PARENT = REPO / ".codex-review/wr-d3b/tmp"
SPEC = importlib.util.spec_from_file_location("workflow_tests", REPO / "tests/test_deploy_workflow.py")
assert SPEC and SPEC.loader
tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tests)
DEPLOY, RECOVER = tests.scripts_and_steps()[0]


def replace_one(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, f"expected 1 occurrence of {old!r}, got {source.count(old)}"
    return source.replace(old, new)


def build_counterexample_shim() -> str:
    daemon_reload_hook = '''    if not user and args[0] == "daemon-reload":
        (root / "daemon-reloaded").touch()
    if user and args[0] in {"disable", "enable"}:'''
    shim = replace_one(tests.SHIM, '    if user and args[0] in {"disable", "enable"}:', daemon_reload_hook)

    show_hook = '''        if not user and args[1] == "bts-scheduler.service":
            if scenario == "ce1-absent-with-cgroup":
                states.update(LoadState="not-found", ActiveState="inactive", MainPID="0", ControlGroup="/legacy-orphan")
            elif scenario == "ce1b-absent-then-post-cgroup":
                seen = root / "seen-first-legacy-show"
                cg = "/legacy-orphan" if seen.exists() else ""
                seen.touch()
                states.update(LoadState="not-found", ActiveState="inactive", MainPID="0", ControlGroup=cg)
            elif scenario in {"ce2-phase2-unload-populated", "ce2b-phase2-unload-nogroup"}:
                frag = root / "etc/systemd/system/bts-scheduler.service"
                unloaded = (root / "legacy-stopped").exists() and not frag.exists() and (root / "daemon-reloaded").exists()
                if not unloaded:
                    cg = "" if scenario.endswith("nogroup") else "/legacy-orphan"
                    active_st = "inactive" if (root / "legacy-stopped").exists() else "active"
                    pid_st = "0" if (root / "legacy-stopped").exists() else "1234"
                    states.update(LoadState="loaded", ActiveState=active_st, MainPID=pid_st, ControlGroup=cg)
                else:
                    states.update(LoadState="not-found", ActiveState="inactive", MainPID="0", ControlGroup="")
        if "ControlGroup" in args and user and scenario == "main-pid":'''
    shim = replace_one(
        shim,
        '        if "ControlGroup" in args and user and scenario == "main-pid":',
        show_hook,
    )
    return shim


def make_box(tmp: Path, shim: str, *, populated: int | None = 1, with_fragment: bool = False) -> Path:
    box = tests.box.__wrapped__(tmp)
    for file in (box / "shims").iterdir():
        file.write_text(file.read_text().split("\n", 1)[0] + "\n" + shim)
    runtime = box / "run/bts-deploy"
    runtime.mkdir(mode=0o700, parents=True)
    (box / "enabled").write_text("enabled")
    (box / "home/projects/bts/data/.heartbeat").write_text('{"state":"running"}')
    (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/ce\n")
    if populated is not None:
        group = box / "sys/fs/cgroup/legacy-orphan"
        group.mkdir()
        (group / "cgroup.procs").write_text("9999\n" if populated else "")
        (group / "cgroup.events").write_text(f"populated {populated}\n")
    if with_fragment:
        frags = box / "etc/systemd/system"
        frags.mkdir(parents=True, exist_ok=True)
        (frags / "bts-scheduler.service").write_text("[Service]\nExecStart=/bin/true\n")
    return box


def cgroup_checks(calls: list[list[str]]) -> list[str]:
    return [c[-1] for c in calls if c[0] == "python3" and len(c) >= 5 and c[3] == "cgroup"]


def apply_smallest_fix(script: str, *, is_deploy: bool) -> str:
    # Note: scripts_and_steps() strips the 12-space YAML block indentation, so top-level shell statements have 0 indent.
    script = replace_one(
        script,
        "\nstaged=\n",
        "\nstaged=\nphase_legacy_seen=\nphase_legacy_group=\nphase_legacy_group_status=1\n",
    )
    script = replace_one(
        script,
        'legacy_state() {\n'
        '    local cap=$1 name=$2 output id active pid\n'
        '    output=$(system_unit "$cap" normal show "$name" -p Id -p LoadState -p ActiveState -p MainPID) || return 1\n'
        '    id=$(field Id "$output") && [[ $id == "$name" ]] || return 1\n'
        '    active=$(field ActiveState "$output") || return 1\n'
        '    pid=$(field MainPID "$output") && [[ $pid =~ ^[0-9]+$ ]] || return 1\n'
        '    legacy=$(field LoadState "$output") || return 1',
        'legacy_state() {\n'
        '    local cap=$1 name=$2 output id active pid\n'
        '    output=$(system_unit "$cap" normal show "$name" -p Id -p LoadState -p ActiveState -p MainPID -p ControlGroup) || return 1\n'
        '    id=$(field Id "$output") && [[ $id == "$name" ]] || return 1\n'
        '    active=$(field ActiveState "$output") || return 1\n'
        '    pid=$(field MainPID "$output") && [[ $pid =~ ^[0-9]+$ ]] || return 1\n'
        '    legacy=$(field LoadState "$output") || return 1\n'
        '    legacy_cgroup=$(field ControlGroup "$output" yes) || return 1',
    )
    script = replace_one(
        script,
        '    if legacy_state 10 "$scheduler" && [[ $legacy == not-found ]] && no_fragment; then\n'
        '        legacy_absent=0\n'
        '    else',
        '    if [[ -z $phase_legacy_seen ]] && legacy_state 10 "$scheduler" && [[ $legacy == not-found ]] && no_fragment; then\n'
        '        legacy_absent=0\n'
        '        legacy_group=$legacy_cgroup\n'
        '        legacy_group_status=0\n'
        '    elif [[ -n $phase_legacy_seen ]] && legacy_state 10 "$scheduler" && [[ $legacy == not-found ]] && no_fragment; then\n'
        '        legacy_group=${phase_legacy_group:-$legacy_cgroup}\n'
        '        legacy_group_status=$phase_legacy_group_status\n'
        '        legacy_disabled=0\n'
        '        legacy_stopped=0\n'
        '    else',
    )
    script = replace_one(
        script,
        '                if [[ $legacy == not-found ]] && no_fragment; then\n'
        '                    if [[ $legacy_absent == 0 ]]; then\n'
        '                        # Absent at the containment\'s first query and now: no path needed.\n'
        '                        system_proof=0\n'
        '                    elif (( legacy_group_status == 0 )) && post_group=$(scheduler_group system_unit "$scheduler") &&\n'
        '                         [[ -n $legacy_group || -n $post_group ]]; then\n'
        '                        # Seen loaded, or not seen absent: every known path must be empty.\n'
        '                        system_proof=0\n'
        '                        [[ -z $legacy_group ]] || fs 20 normal cgroup /sys/fs/cgroup "$legacy_group" || system_proof=1\n'
        '                        [[ -z $post_group || $post_group == "$legacy_group" ]] ||\n'
        '                            fs 20 normal cgroup /sys/fs/cgroup "$post_group" || system_proof=1\n'
        '                    fi',
        '                if [[ $legacy == not-found ]] && no_fragment; then\n'
        '                    post_group=$legacy_cgroup\n'
        '                    if [[ $legacy_absent == 0 ]] || { (( legacy_group_status == 0 )) && [[ -n $legacy_group || -n $post_group ]]; }; then\n'
        '                        system_proof=0\n'
        '                        [[ -z $legacy_group ]] || fs 20 normal cgroup /sys/fs/cgroup "$legacy_group" || system_proof=1\n'
        '                        [[ -z $post_group || $post_group == "$legacy_group" ]] ||\n'
        '                            fs 20 normal cgroup /sys/fs/cgroup "$post_group" || system_proof=1\n'
        '                    fi',
    )
    if is_deploy:
        script = replace_one(
            script,
            '    for name in "${legacy_names[@]}"; do\n'
            '        system_unit 20 mutation disable "$name" >/dev/null',
            '    for name in "${legacy_names[@]}"; do\n'
            '        if [[ $name == "$scheduler" ]]; then\n'
            '            phase_legacy_seen=1\n'
            '            phase_legacy_group=$(scheduler_group system_unit "$scheduler"); phase_legacy_group_status=$?\n'
            '        fi\n'
            '        system_unit 20 mutation disable "$name" >/dev/null',
        )
    return script


def main() -> None:
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    shim = build_counterexample_shim()
    report: dict[str, object] = {}

    # --- CE1: Absent-unit result (LoadState=not-found, no fragment) with ControlGroup=/legacy-orphan (populated 1) ---
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        box = make_box(Path(td), shim, populated=1, with_fragment=False)
        dep_res, dep_calls = tests.run_box_script(box, DEPLOY, "ce1-absent-with-cgroup", fault="reset:17")
        dep_checked = cgroup_checks(dep_calls)
        dep_posts = [c for c in dep_calls if c[0] == "curl" and "--data-raw" in c]
        (box / "calls").unlink()
        rec_res, rec_calls = tests.run_box_script(box, RECOVER, "ce1-absent-with-cgroup")
        rec_checked = cgroup_checks(rec_calls)
        (box / "run/bts-deploy/attempt").write_text(
            f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun - - - -\n"
        )
        (box / "calls").unlink()
        rec_begun_res, rec_begun_calls = tests.run_box_script(box, RECOVER, "ce1-absent-with-cgroup")
        report["CE1_absent_unit_with_populated_cgroup"] = {
            "deploy_stdout": dep_res.stdout.splitlines(),
            "deploy_checked_cgroups": dep_checked,
            "deploy_post_body": dep_posts[0][dep_posts[0].index("--data-raw") + 1] if dep_posts else None,
            "deploy_record": (box / "run/bts-deploy/attempt").read_text().strip(),
            "recovery_contained_stdout": rec_res.stdout.splitlines(),
            "recovery_contained_checked_cgroups": rec_checked,
            "recovery_begun_stdout": rec_begun_res.stdout.splitlines(),
            "recovery_begun_checked_cgroups": cgroup_checks(rec_begun_calls),
        }
        assert "Scheduler stopped: verified." in dep_res.stdout
        assert "/legacy-orphan" not in dep_checked
        assert "Scheduler stopped: verified." in rec_res.stdout
        assert "DEPLOY RECOVERY — containment re-checked." in rec_res.stdout
        assert "Scheduler stopped: verified." in rec_begun_res.stdout

    # --- CE1b: Absent at pre-stop (ControlGroup=""), absent at post-stop with ControlGroup=/legacy-orphan (populated 1) ---
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        box = make_box(Path(td), shim, populated=1, with_fragment=False)
        res_1b, calls_1b = tests.run_box_script(box, DEPLOY, "ce1b-absent-then-post-cgroup", fault="reset:17")
        report["CE1b_absent_then_post_stop_populated_cgroup"] = {
            "deploy_stdout": res_1b.stdout.splitlines(),
            "checked_cgroups": cgroup_checks(calls_1b),
        }
        assert "Scheduler stopped: verified." in res_1b.stdout
        assert "/legacy-orphan" not in cgroup_checks(calls_1b)

    # --- CE2: Legacy unit loaded & active at /legacy-orphan (populated 1); Phase 2 legacy-units stops, deletes fragment,
    #     and daemon-reloads it to not-found + ControlGroup=""; deploy then fails at contest-state! ---
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        box_ctrl = make_box(Path(td), shim, populated=1, with_fragment=True)
        ctrl_res, ctrl_calls = tests.run_box_script(box_ctrl, DEPLOY, "ce2-phase2-unload-populated", fault="reset:17")
        assert "Scheduler stopped: unverified." in ctrl_res.stdout
        assert "/legacy-orphan" in cgroup_checks(ctrl_calls)

    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        box_ce2 = make_box(Path(td), shim, populated=1, with_fragment=True)
        ce2_res, ce2_calls = tests.run_box_script(
            box_ce2, DEPLOY, "ce2-phase2-unload-populated", fault="contest-state:17"
        )
        ce2_checked = cgroup_checks(ce2_calls)
        (box_ce2 / "calls").unlink()
        ce2_rec_res, _ = tests.run_box_script(box_ce2, RECOVER, "ce2-phase2-unload-populated")
        report["CE2_phase2_legacy_cleanup_unloads_populated_unit_before_canary_fail"] = {
            "control_reset_fail_stdout_stop": [l for l in ctrl_res.stdout.splitlines() if "Scheduler stopped:" in l],
            "canary_fail_stdout": ce2_res.stdout.splitlines(),
            "canary_fail_checked_cgroups": ce2_checked,
            "canary_fail_record": (box_ce2 / "run/bts-deploy/attempt").read_text().strip(),
            "subsequent_recovery_stdout": ce2_rec_res.stdout.splitlines(),
        }
        assert "Scheduler stopped: verified." in ce2_res.stdout
        assert "/legacy-orphan" not in ce2_checked
        assert "DEPLOY RECOVERY — containment re-checked." in ce2_rec_res.stdout

    # --- CE2b: Same as CE2, with ControlGroup="" while loaded (violates K12: loaded unit with no known path -> unverified) ---
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        box_ce2b = make_box(Path(td), shim, populated=None, with_fragment=True)
        ce2b_res, _ = tests.run_box_script(
            box_ce2b, DEPLOY, "ce2b-phase2-unload-nogroup", fault="contest-state:17"
        )
        report["CE2b_phase2_legacy_cleanup_unloads_nogroup_unit_before_canary_fail"] = {
            "canary_fail_stdout": ce2b_res.stdout.splitlines(),
        }
        assert "Scheduler stopped: verified." in ce2b_res.stdout

    # --- Verify our smallest fix in-memory on all counterexamples + controls! ---
    fixed_deploy = apply_smallest_fix(DEPLOY, is_deploy=True)
    fixed_recover = apply_smallest_fix(RECOVER, is_deploy=False)
    assert fixed_deploy.split("\nnot_started() {", 1)[0] == fixed_recover.split("\nrecovery_exit() {", 1)[0]

    for label, script, scen, frag, pop, flt, expected_verified in [
        ("fixed_CE1_deploy", fixed_deploy, "ce1-absent-with-cgroup", False, 1, "reset:17", False),
        ("fixed_CE1_recover", fixed_recover, "ce1-absent-with-cgroup", False, 1, "", False),
        ("fixed_CE1_empty_control", fixed_deploy, "ce1-absent-with-cgroup", False, 0, "reset:17", True),
        ("fixed_CE1b_deploy", fixed_deploy, "ce1b-absent-then-post-cgroup", False, 1, "reset:17", False),
        ("fixed_CE2_deploy", fixed_deploy, "ce2-phase2-unload-populated", True, 1, "contest-state:17", False),
        ("fixed_CE2_empty_control", fixed_deploy, "ce2-phase2-unload-populated", True, 0, "contest-state:17", True),
        ("fixed_CE2b_nogroup", fixed_deploy, "ce2b-phase2-unload-nogroup", True, None, "contest-state:17", False),
        ("fixed_truly_absent_control", fixed_deploy, "active", False, None, "reset:17", True),
    ]:
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td), shim, populated=pop, with_fragment=frag)
            if script is fixed_recover:
                (b / "run/bts-deploy/attempt").write_text(
                    f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun - - - -\n"
                )
            r, _ = tests.run_box_script(b, script, scen, fault=flt)
            expected_line = f"Scheduler stopped: {'verified' if expected_verified else 'unverified'}."
            assert expected_line in r.stdout, (label, r.stdout)

    report["smallest_fix_verification"] = "ALL 8 CASES PASSED (RD8 shared prefix & prove byte-identical)"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
