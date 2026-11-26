#!/usr/bin/env python3
"""Item 7: Targeted scratch mutation testing on D3b (8 required + 9 reviewer mutations, bounded under 10 minutes)."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
SCRATCH_PARENT = REPO / ".codex-review/wr-d3b/tmp"
PYTEST = [str(REPO / ".venv/bin/python"), "-m", "pytest", "-v", "-x", "--tb=no"]


def make_scratch_repo(tmp: Path) -> Path:
    root = tmp / "scratch"
    root.mkdir()
    for rel in (
        "pyproject.toml",
        ".github/workflows/deploy.yml",
        "CLAUDE.md",
        "INCIDENT.md",
        "docs/ops/2027-season-start.md",
        "scripts/check_heartbeat.py",
        "scripts/cron-setup-hetzner.sh",
    ):
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
    shutil.copytree(REPO / "src", root / "src")
    (root / "tests").mkdir()
    for tf in (
        "test_deploy_workflow.py",
        "test_deploy_canary.py",
        "test_heartbeat_staleness.py",
        "test_heartbeat_churn.py",
        "test_season_contract_cs4b1.py",
        "test_season_contract_cs4b2.py",
    ):
        shutil.copy2(REPO / "tests" / tf, root / "tests" / tf)

    base_wf = REPO / ".codex-review/wr-d3b/base/.github/workflows/deploy.yml"
    base_hb = REPO / ".codex-review/wr-d3b/base/scripts/check_heartbeat.py"

    tdw = (root / "tests/test_deploy_workflow.py").read_text()
    old_wf_snippet = (
        '    baseline = subprocess.run(\n'
        '        ["git", "show", "24019a9a:.github/workflows/deploy.yml"],\n'
        '        cwd=WORKFLOW.parents[2], capture_output=True, text=True, check=True, timeout=5,\n'
        '    ).stdout'
    )
    assert old_wf_snippet in tdw
    tdw = tdw.replace(old_wf_snippet, f"    baseline = Path({str(base_wf)!r}).read_text()")
    (root / "tests/test_deploy_workflow.py").write_text(tdw)

    ths = (root / "tests/test_heartbeat_staleness.py").read_text()
    old_hb_snippet = (
        '    return subprocess.run(\n'
        '        ["git", "show", "24019a9a:scripts/check_heartbeat.py"], cwd=root,\n'
        '        capture_output=True, text=True, check=True, timeout=5,\n'
        '    ).stdout'
    )
    assert old_hb_snippet in ths
    ths = ths.replace(old_hb_snippet, f"    return Path({str(base_hb)!r}).read_text()")
    (root / "tests/test_heartbeat_staleness.py").write_text(ths)
    return root


def apply_replacements(root: Path, edits: list[tuple[str, str, str, int]]) -> None:
    for rel, old, new, expected_count in edits:
        path = root / rel
        txt = path.read_text()
        actual = txt.count(old)
        assert actual == expected_count, (
            f"target count mismatch in {rel}: expected {expected_count}, got {actual} for {old!r}"
        )
        path.write_text(txt.replace(old, new))


TARGETED_MUTATIONS: list[tuple[str, list[tuple[str, str, str, int]], str, str]] = [
    (
        "req1_prove_stop_skip_recorded_legacy_group_after_not_found",
        [(
            ".github/workflows/deploy.yml",
            '                                elif (( legacy_group_status == 0 )) && post_group=$(scheduler_group system_unit "$scheduler") &&\n'
            '                                     [[ -n $legacy_group || -n $post_group ]]; then\n'
            '                                    # Seen loaded, or not seen absent: every known path must be empty.\n'
            '                                    system_proof=0\n'
            '                                    [[ -z $legacy_group ]] || fs 20 normal cgroup /sys/fs/cgroup "$legacy_group" || system_proof=1\n'
            '                                    [[ -z $post_group || $post_group == "$legacy_group" ]] ||\n'
            '                                        fs 20 normal cgroup /sys/fs/cgroup "$post_group" || system_proof=1\n'
            '                                fi',
            '                                else\n'
            '                                    system_proof=0\n'
            '                                fi',
            2,
        )],
        "tests/test_deploy_workflow.py",
        "test_d3b_recorded_legacy_group_stays_in_the_proof or test_d3b_loaded_legacy_unit_without_a_known_group_is_unverified",
    ),
    (
        "req2_drop_residual_window_proof_to_220s",
        [(
            ".github/workflows/deploy.yml",
            "                        # Proof queries: to 205 s of the handler's 280 s.\n"
            "                        deadline=$((containment_deadline - 75000))",
            "                        # Proof queries: to 220 s of the handler's 280 s.\n"
            "                        deadline=$((containment_deadline - 60000))",
            2,
        )],
        "tests/test_deploy_workflow.py",
        "test_d3b_containment_windows_reserve_the_residual_check",
    ),
    (
        "req3_signal_rule_not_started_on_unreadable",
        [(
            ".github/workflows/deploy.yml",
            "                elif (( rc == 3 )) || { (( rc == 0 )) && ! same_ids; }; then\n"
            "                    answer=not-started                                        # S5, S6",
            "                elif (( rc != 0 )) || ! same_ids; then\n"
            "                    answer=not-started                                        # S5, S6",
            1,
        )],
        "tests/test_deploy_workflow.py",
        "test_d3b_unreadable_record_while_beginning_contains",
    ),
    (
        "req4_signal_rule_success_for_committed_while_beginning",
        [(
            ".github/workflows/deploy.yml",
            "                if [[ $publication == committing ]]; then\n"
            "                    if (( rc == 0 )) && same_record && [[ $r_state == committed ]]; then\n"
            "                        answer=success                                        # S3; else S4\n"
            "                    fi\n"
            "                elif (( rc == 3 )) || { (( rc == 0 )) && ! same_ids; }; then",
            "                if (( rc == 0 )) && same_record && [[ $r_state == committed ]]; then\n"
            "                    answer=success                                            # S3; else S4\n"
            "                elif [[ $publication == beginning ]] && { (( rc == 3 )) || { (( rc == 0 )) && ! same_ids; }; }; then",
            1,
        )],
        "tests/test_deploy_workflow.py",
        "test_d3b_same_ids_while_beginning_contains or test_d3b_signal_before_begin_is_published_never_contains",
    ),
    (
        "req5_prove_page_accept_curl_exit_7_with_200",
        [(
            ".github/workflows/deploy.yml",
            "                        if [[ $2 == carried ]] || [[ $2 == 0 && $3 =~ ^2[0-9][0-9]$ ]]; then",
            "                        if [[ $2 == carried ]] || [[ ( $2 == 0 || $2 == 7 ) && $3 =~ ^2[0-9][0-9]$ ]]; then",
            2,
        )],
        "tests/test_deploy_workflow.py",
        "test_post_requires_exit_zero_and_2xx",
    ),
    (
        "req6_print_result_line_outside_prove",
        [(
            ".github/workflows/deploy.yml",
            "                prove report\n"
            "                prove recheck",
            "                prove report\n"
            "                if [[ $repeat == yes && $stop$marker$page == verifiedverifiedverified ]]; then\n"
            "                    log 'DEPLOY RECOVERY — containment re-checked.'\n"
            "                fi",
            2,
        )],
        "tests/test_deploy_workflow.py",
        "test_d3b_only_the_proof_function_makes_or_prints_a_verified_result",
    ),
    (
        "req7_drop_no_fragment_link_test",
        [(
            ".github/workflows/deploy.yml",
            "            no_fragment() { [[ ! -e $legacy_dir/$scheduler && ! -L $legacy_dir/$scheduler ]]; }",
            "            no_fragment() { [[ ! -e $legacy_dir/$scheduler ]]; }",
            2,
        )],
        "tests/test_deploy_workflow.py",
        "test_d3b_unproved_legacy_absence_is_unverified",
    ),
    (
        "req8_drop_pre_stop_group_status_check",
        [
            (
                ".github/workflows/deploy.yml",
                "                        if (( uid_status == 0 && user_group_status == 0 )) && [[ $uid =~ ^[0-9]+$ ]]; then",
                "                        if (( uid_status == 0 )) && [[ $uid =~ ^[0-9]+$ ]]; then",
                2,
            ),
            (
                ".github/workflows/deploy.yml",
                "elif (( legacy_group_status == 0 )) && ",
                "elif ",
                4,
            ),
        ],
        "tests/test_deploy_workflow.py",
        "test_d3b_failed_user_group_query_is_unverified",
    ),
    (
        "own8_drop_same_record_in_signal_rule_committing",
        [(
            ".github/workflows/deploy.yml",
            "                    if (( rc == 0 )) && same_record && [[ $r_state == committed ]]; then",
            "                    if (( rc == 0 )) && [[ $r_state == committed ]]; then",
            1,
        )],
        "tests/test_deploy_workflow.py",
        "test_d3b_success_needs_this_attempts_exact_commit",
    ),
    (
        "own9_skip_previous_group_when_post_group_changed_nonempty",
        [(
            ".github/workflows/deploy.yml",
            '                [[ -n $previous_group || -n $group ]] || return 1\n'
            '                if [[ -n $previous_group && $previous_group != "$group" ]]; then\n'
            '                    fs 20 normal cgroup /sys/fs/cgroup "$previous_group" || return 1\n'
            '                fi\n'
            '                [[ -z $group ]] || fs 20 normal cgroup /sys/fs/cgroup "$group"',
            '                [[ -n $previous_group || -n $group ]] || return 1\n'
            '                fs 20 normal cgroup /sys/fs/cgroup "${group:-$previous_group}"',
            2,
        )],
        "tests/test_deploy_workflow.py",
        "test_d3b_every_known_cgroup_path_is_in_the_proof",
    ),
]


CANDIDATE_SURVIVORS: list[tuple[str, list[tuple[str, str, str, int]], str]] = [
    (
        "own1_drop_legacy_pre_stop_group_status_check_only",
        [(
            ".github/workflows/deploy.yml",
            "elif (( legacy_group_status == 0 )) && ",
            "elif ",
            4,
        )],
        "legacy or stop_verification or group",
    ),
    (
        "own2_drop_post_group_check_after_legacy_not_found",
        [(
            ".github/workflows/deploy.yml",
            '                                    [[ -z $post_group || $post_group == "$legacy_group" ]] ||\n'
            '                                        fs 20 normal cgroup /sys/fs/cgroup "$post_group" || system_proof=1',
            "                                    :",
            2,
        )],
        "legacy or cgroup or stop_verification",
    ),
    (
        "own3_drop_legacy_disable_stop_status_in_prove_stop",
        [(
            ".github/workflows/deploy.yml",
            "                        if (( system_proof == 0 )) && [[ $legacy_absent == 0 || $legacy_disabled$legacy_stopped == 00 ]]; then",
            "                        if (( system_proof == 0 )); then",
            2,
        )],
        "legacy or stop_verification",
    ),
    (
        "own4_drop_legacy_is_enabled_in_prove_stop",
        [(
            ".github/workflows/deploy.yml",
            'check_stopped system_unit "$scheduler" "$legacy_group" &&\n'
            '                                 enabled disabled 20 system_unit; then',
            'check_stopped system_unit "$scheduler" "$legacy_group"; then',
            2,
        )],
        "legacy or stop_verification or enabled",
    ),
    (
        "own5_drop_seconds_rounding_allowance_in_run",
        [(
            ".github/workflows/deploy.yml",
            "                [[ -z ${budget_start:-} ]] || available=$((available - 1))",
            "                :",
            2,
        )],
        "containment_windows or uncertain_commit_signal or clock_failure",
    ),
    (
        "own6_drop_st_uid_zero_in_lockfile_and_mark",
        [
            (
                ".github/workflows/deploy.yml",
                "if not stat.S_ISREG(s.st_mode) or s.st_uid != 0 or stat.S_IMODE(s.st_mode) != 0o600:",
                "if not stat.S_ISREG(s.st_mode) or stat.S_IMODE(s.st_mode) != 0o600:",
                2,
            ),
            (
                ".github/workflows/deploy.yml",
                "if s.st_uid != 0 or stat.S_IMODE(s.st_mode) != 0o600:",
                "if stat.S_IMODE(s.st_mode) != 0o600:",
                2,
            ),
        ],
        "lock or mark",
    ),
    (
        "own7_drop_contained_fields_match_in_publish_readback",
        [(
            ".github/workflows/deploy.yml",
            'if (( rc == 0 )) && same_record && [[ $r_state == "$wanted" && "$r_step $r_stop $r_marker $r_page" == "$*" ]]; then',
            'if (( rc == 0 )) && same_record && [[ $r_state == "$wanted" ]]; then',
            2,
        )],
        "reproves_stop_and_marker or rechecks_each_unverified or fresh_marker or surviving_commit or commit_status or commit_readback",
    ),
]


def run_pytest(root: Path, test_rel: str, extra_args: list[str], *, timeout: int = 360) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root}:{root / 'src'}"
    cmd = [*PYTEST, *extra_args, str(root / test_rel)]
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
    failed_line = next((line for line in proc.stdout.splitlines() if " FAILED" in line), "")
    lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
    tail = lines[-1] if lines else ""
    detail = f"{failed_line} | {tail}" if failed_line else tail
    return proc.returncode, detail


def main() -> None:
    started = time.monotonic()
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)

    # 1. Pre-validate all 17 target strings
    for _, edits, _, _ in TARGETED_MUTATIONS:
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            s = make_scratch_repo(Path(td))
            apply_replacements(s, edits)
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        s = make_scratch_repo(Path(td))
        for _, edits, _ in CANDIDATE_SURVIVORS:
            apply_replacements(s, edits)
    print("Pre-validation of all 17 mutation targets: OK", flush=True)

    # 2. Control check
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        control = make_scratch_repo(Path(td))
        rc, detail = run_pytest(
            control,
            "tests/test_deploy_workflow.py",
            ["-m", "not slow", "-k", "test_healthy_scheduler_passes or test_passing_transcript_order_delta"],
        )
        assert rc == 0, f"Control smoke failed: {detail}"
        print(f"Scratch control smoke: PASS ({detail})", flush=True)

    # 3. Run targeted -k checks on the 10 expected-detected mutants (~2s each)
    for name, edits, test_rel, k_expr in TARGETED_MUTATIONS:
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            scratch = make_scratch_repo(Path(td))
            apply_replacements(scratch, edits)
            rc, detail = run_pytest(scratch, test_rel, ["-m", "not slow", "-k", k_expr])
            status = "DETECTED" if rc != 0 else "SURVIVED_TARGETED"
            print(f"MUTATION {name}: {status} ({time.monotonic() - t0:.1f}s) [{detail}]", flush=True)

    # 4. Run targeted -k checks on each of the 7 candidate survivors (~5s each), then confirm survivors in ONE combined fast run!
    surviving_bundle: list[tuple[str, list[tuple[str, str, str, int]]]] = []
    for name, edits, k_expr in CANDIDATE_SURVIVORS:
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            scratch = make_scratch_repo(Path(td))
            apply_replacements(scratch, edits)
            rc, detail = run_pytest(scratch, "tests/test_deploy_workflow.py", ["-m", "not slow", "-k", k_expr])
            if rc != 0:
                print(f"MUTATION {name}: DETECTED ({time.monotonic() - t0:.1f}s) [{detail}]", flush=True)
            else:
                surviving_bundle.append((name, edits))
                print(f"MUTATION {name}: SURVIVED targeted -k ({time.monotonic() - t0:.1f}s) [{detail}]", flush=True)

    # Single combined confirmation across all fast tests in test_deploy_workflow.py (~45s total, once!)
    if surviving_bundle:
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            scratch = make_scratch_repo(Path(td))
            for _, edits in surviving_bundle:
                apply_replacements(scratch, edits)
            rc_all, det_all = run_pytest(scratch, "tests/test_deploy_workflow.py", ["-m", "not slow"])
            bundle_status = "ALL SURVIVED FULL FAST FILE" if rc_all == 0 else f"BUNDLE DETECTED ({det_all})"
            print(
                f"COMBINED_SURVIVOR_BUNDLE ({len(surviving_bundle)} mutants): {bundle_status} "
                f"({time.monotonic() - t0:.1f}s) [{det_all}]",
                flush=True,
            )

    print(f"Total mutation script elapsed: {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
