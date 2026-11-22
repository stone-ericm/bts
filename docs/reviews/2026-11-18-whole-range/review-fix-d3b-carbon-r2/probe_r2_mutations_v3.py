#!/usr/bin/env python3
"""Round 2 Item 5: Targeted scratch mutation testing on 428cfa93 (17 Round-1 + 6 required Round-2 + 5 reviewer Round-2 = 28 mutations)."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[3]
SCRATCH_PARENT = REPO / ".codex-review/wr-d3b/r2/tmp"
PYTEST = [str(REPO / ".venv/bin/python"), "-m", "pytest", "-p", "no:cacheprovider", "-q", "-x", "--tb=no"]


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
    shutil.copy2(REPO / "tests/test_deploy_workflow.py", root / "tests/test_deploy_workflow.py")

    base_wf = REPO / ".codex-review/wr-d3b/base/.github/workflows/deploy.yml"
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


MUTATIONS: list[tuple[str, list[tuple[str, str, str, int]], str]] = [
    # --- Round-1 mutation set adapted to 428cfa93 (17 mutants) ---
    (
        "r1_req1_prove_stop_skip_recorded_legacy_group_after_not_found",
        [(
            ".github/workflows/deploy.yml",
            '                                    elif (( legacy_group_status == 0 )) && [[ -n $legacy_group || -n $post_group ]]; then\n'
            '                                        # Seen loaded, or not seen absent: at least one path known.\n'
            '                                        system_proof=0\n'
            '                                    fi\n'
            '                                    # Every known path must be empty or absent.\n'
            '                                    [[ -z $legacy_group ]] || fs 20 normal cgroup /sys/fs/cgroup "$legacy_group" || system_proof=1\n'
            '                                    [[ -z $post_group || $post_group == "$legacy_group" ]] ||\n'
            '                                        fs 20 normal cgroup /sys/fs/cgroup "$post_group" || system_proof=1',
            '                                    else\n'
            '                                        system_proof=0\n'
            '                                    fi',
            2,
        )],
        "test_d3b_recorded_legacy_group_stays_in_the_proof or test_d3b_loaded_legacy_unit_without_a_known_group_is_unverified",
    ),
    (
        "r1_req2_drop_residual_window_proof_to_220s",
        [(
            ".github/workflows/deploy.yml",
            "                        # Proof queries: to 205 s of the handler's 280 s.\n"
            "                        deadline=$((containment_deadline - 75000))",
            "                        # Proof queries: to 220 s of the handler's 280 s.\n"
            "                        deadline=$((containment_deadline - 60000))",
            2,
        )],
        "test_d3b_containment_windows_reserve_the_residual_check",
    ),
    (
        "r1_req3_signal_rule_not_started_on_unreadable",
        [(
            ".github/workflows/deploy.yml",
            "                elif (( rc == 3 )) || { (( rc == 0 )) && ! same_ids; }; then\n"
            "                    answer=not-started                                        # S5, S6",
            "                elif (( rc != 0 )) || ! same_ids; then\n"
            "                    answer=not-started                                        # S5, S6",
            1,
        )],
        "test_d3b_unreadable_record_while_beginning_contains",
    ),
    (
        "r1_req4_signal_rule_success_for_committed_while_beginning",
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
        "test_d3b_same_ids_while_beginning_contains",
    ),
    (
        "r1_req5_prove_page_accept_curl_exit_7_with_200",
        [(
            ".github/workflows/deploy.yml",
            "                        if [[ $2 == carried ]] || [[ $2 == 0 && $3 =~ ^2[0-9][0-9]$ ]]; then",
            "                        if [[ $2 == carried ]] || [[ ( $2 == 0 || $2 == 7 ) && $3 =~ ^2[0-9][0-9]$ ]]; then",
            2,
        )],
        "test_post_requires_exit_zero_and_2xx",
    ),
    (
        "r1_req6_print_result_line_outside_prove",
        [(
            ".github/workflows/deploy.yml",
            "                prove report\n"
            "                exit 1",
            "                prove report\n"
            "                log 'Scheduler stopped: verified.'\n"
            "                exit 1",
            2,
        )],
        "test_d3b_only_the_proof_function_makes_or_prints_a_verified_result",
    ),
    (
        "r1_req7_drop_no_fragment_link_test",
        [(
            ".github/workflows/deploy.yml",
            "            no_fragment() { [[ ! -e $legacy_dir/$scheduler && ! -L $legacy_dir/$scheduler ]]; }",
            "            no_fragment() { [[ ! -e $legacy_dir/$scheduler ]]; }",
            2,
        )],
        "test_d3b_unproved_legacy_absence_is_unverified",
    ),
    (
        "r1_req8_drop_pre_stop_group_status_check",
        [
            (
                ".github/workflows/deploy.yml",
                "                        if (( uid_status == 0 && user_group_status == 0 )) && [[ $uid =~ ^[0-9]+$ ]]; then",
                "                        if (( uid_status == 0 )) && [[ $uid =~ ^[0-9]+$ ]]; then",
                2,
            ),
            (
                ".github/workflows/deploy.yml",
                "(( legacy_group_status == 0 )) &&",
                "",
                4,
            ),
        ],
        "test_d3b_failed_user_group_query_is_unverified or test_d3b_r1_each_legacy_stop_clause_is_required",
    ),
    (
        "r1_own1_drop_legacy_pre_stop_group_status_check_only",
        [(
            ".github/workflows/deploy.yml",
            "(( legacy_group_status == 0 )) &&",
            "",
            4,
        )],
        "test_d3b_r1_each_legacy_stop_clause_is_required or test_d3b_loaded_legacy_unit_without_a_known_group_is_unverified",
    ),
    (
        "r1_own2_drop_post_group_check_after_legacy_not_found",
        [(
            ".github/workflows/deploy.yml",
            '                                    [[ -z $post_group || $post_group == "$legacy_group" ]] ||\n'
            '                                        fs 20 normal cgroup /sys/fs/cgroup "$post_group" || system_proof=1',
            "                                    :",
            2,
        )],
        "test_d3b_r1_each_legacy_stop_clause_is_required or test_d3b_r1_a_cgroup_reported_by_an_absent_unit_is_checked",
    ),
    (
        "r1_own3_drop_legacy_disable_stop_status_in_prove_stop",
        [(
            ".github/workflows/deploy.yml",
            "                        if (( system_proof == 0 )) && [[ $legacy_absent == 0 || $legacy_disabled$legacy_stopped == 00 ]]; then",
            "                        if (( system_proof == 0 )); then",
            2,
        )],
        "test_d3b_r1_each_legacy_stop_clause_is_required",
    ),
    (
        "r1_own4_drop_legacy_is_enabled_in_prove_stop",
        [(
            ".github/workflows/deploy.yml",
            'check_stopped system_unit "$scheduler" "$legacy_group" && enabled disabled 20 system_unit; then',
            'check_stopped system_unit "$scheduler" "$legacy_group"; then',
            2,
        )],
        "test_d3b_r1_each_legacy_stop_clause_is_required",
    ),
    (
        "r1_own5_drop_seconds_rounding_allowance_in_run",
        [(
            ".github/workflows/deploy.yml",
            "                [[ -z ${budget_start:-} ]] || available=$((available - 1))",
            "                :",
            2,
        )],
        "test_d3b_r1_rounding_allowance_is_in_both_scripts",
    ),
    (
        "r1_own6_drop_st_uid_zero_in_lockfile_and_mark",
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
        "test_round2_recovery_creates_real_lockfile_and_mark or test_invalid_lock_paths_are_refused or test_recovery_does_not_mutate_services_when_mark_fails or test_d3b_r1_mark_failure_contains_before_the_legacy_stop",
    ),
    (
        "r1_own7_drop_contained_fields_match_in_publish_readback",
        [(
            ".github/workflows/deploy.yml",
            'if (( rc == 0 )) && same_record && [[ $r_state == "$wanted" && "$r_step $r_stop $r_marker $r_page" == "$*" ]]; then',
            'if (( rc == 0 )) && same_record && [[ $r_state == "$wanted" ]]; then',
            2,
        )],
        "test_d3b_recovery_reproves_stop_and_marker or test_recovery_rechecks_each_unverified_result or test_d3b_surviving_commit_record_is_reported_as_a_record or test_round1_commit_status_survives_failed_publication",
    ),
    (
        "r1_own8_drop_same_record_in_signal_rule_committing",
        [(
            ".github/workflows/deploy.yml",
            "                    if (( rc == 0 )) && same_record && [[ $r_state == committed ]]; then",
            "                    if (( rc == 0 )) && [[ $r_state == committed ]]; then",
            1,
        )],
        "test_d3b_success_needs_this_attempts_exact_commit",
    ),
    (
        "r1_own9_skip_previous_group_when_post_group_changed_nonempty",
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
        "test_d3b_every_known_cgroup_path_is_in_the_proof",
    ),
    # --- 6 Required Round-2 Mutations (r2/brief.md item 5) ---
    (
        "r2_req1_drop_legacy_mark_in_first_sample",
        [(
            ".github/workflows/deploy.yml",
            "                   no_fragment && ! legacy_cleaned; then",
            "                   no_fragment; then",
            2,
        )],
        "test_d3b_r1_phase_two_legacy_cleanup_marks_the_attempt or test_d3b_r1_mark_failure_contains_before_the_legacy_stop or test_d3b_r1_a_passing_deploy_with_legacy_units_still_passes or test_d3b_legacy_absent_throughout_needs_no_legacy_stop",
    ),
    (
        "r2_req2_drop_legacy_mark_in_proof",
        [(
            ".github/workflows/deploy.yml",
            '                                if ! legacy_cleaned && post_group=$(scheduler_group system_unit "$scheduler"); then',
            '                                if post_group=$(scheduler_group system_unit "$scheduler"); then',
            2,
        )],
        "test_d3b_r1_phase_two_legacy_cleanup_marks_the_attempt",
    ),
    (
        "r2_req3_write_mark_after_legacy_stop",
        [(
            ".github/workflows/deploy.yml",
            '                    if [[ $name == "$scheduler" ]]; then\n'
            '                        fs 10 mutation mark "$lock_dir/legacy.$RUN_ID.$RUN_ATTEMPT" >/dev/null\n'
            '                        step_status legacy-units "$?"\n'
            '                    fi\n'
            '                    system_unit 20 mutation disable "$name" >/dev/null\n'
            '                    step_status legacy-units "$?"\n'
            '                    system_unit 20 mutation stop "$name" >/dev/null\n'
            '                    step_status legacy-units "$?"',
            '                    system_unit 20 mutation disable "$name" >/dev/null\n'
            '                    step_status legacy-units "$?"\n'
            '                    system_unit 20 mutation stop "$name" >/dev/null\n'
            '                    step_status legacy-units "$?"\n'
            '                    if [[ $name == "$scheduler" ]]; then\n'
            '                        fs 10 mutation mark "$lock_dir/legacy.$RUN_ID.$RUN_ATTEMPT" >/dev/null\n'
            '                        step_status legacy-units "$?"\n'
            '                    fi',
            1,
        )],
        "test_d3b_r1_mark_failure_contains_before_the_legacy_stop or test_d3b_r1_phase_two_legacy_cleanup_marks_the_attempt or test_d3b_r1_a_passing_deploy_with_legacy_units_still_passes or test_passing_transcript_order_delta or test_failing_phase_command_contains_first_failure",
    ),
    (
        "r2_req4_drop_path_as_is",
        [(
            ".github/workflows/deploy.yml",
            "curl --disable --globoff --path-as-is --noproxy",
            "curl --disable --globoff --noproxy",
            2,
        )],
        "test_d3b_r1_failure_post_sends_the_path_as_written or test_d3b_r1_page_url_with_dot_segments_or_escapes_never_posts",
    ),
    (
        "r2_req5_accept_percent_in_url",
        [(
            ".github/workflows/deploy.yml",
            '                    if "%" in value or any(part in (".", "..") for part in parsed.path.split("/")):',
            '                    if any(part in (".", "..") for part in parsed.path.split("/")):',
            2,
        )],
        "test_d3b_r1_page_url_with_dot_segments_or_escapes_never_posts",
    ),
    (
        "r2_req6_drop_legacy_loaded_in_proof",
        [(
            ".github/workflows/deploy.yml",
            "                            elif [[ $legacy == loaded ]] && (( legacy_group_status == 0 )) &&",
            "                            elif (( legacy_group_status == 0 )) &&",
            2,
        )],
        "test_d3b_r1_not_found_with_a_fragment_is_not_a_loaded_proof",
    ),
    # --- 5 Reviewer New Round-2 Mutations ---
    (
        "r2_own1_drop_dot_segments_in_url",
        [(
            ".github/workflows/deploy.yml",
            '                    if "%" in value or any(part in (".", "..") for part in parsed.path.split("/")):',
            '                    if "%" in value:',
            2,
        )],
        "test_d3b_r1_page_url_with_dot_segments_or_escapes_never_posts",
    ),
    (
        "r2_own2_drop_blank_group_in_first_sample",
        [(
            ".github/workflows/deploy.yml",
            "                if (( legacy_first == 0 && legacy_group_status == 0 )) && [[ $legacy == not-found && -z $legacy_group ]] &&",
            "                if (( legacy_first == 0 && legacy_group_status == 0 )) && [[ $legacy == not-found ]] &&",
            2,
        )],
        "test_d3b_r1_a_cgroup_reported_by_an_absent_unit_is_checked or test_d3b_r1_each_legacy_stop_clause_is_required or test_d3b_legacy_absent_throughout_needs_no_legacy_stop",
    ),
    (
        "r2_own3_drop_legacy_cleaned_link_test",
        [(
            ".github/workflows/deploy.yml",
            "            legacy_cleaned() { [[ -e $lock_dir/legacy.$RUN_ID.$RUN_ATTEMPT || -L $lock_dir/legacy.$RUN_ID.$RUN_ATTEMPT ]]; }",
            "            legacy_cleaned() { [[ -e $lock_dir/legacy.$RUN_ID.$RUN_ATTEMPT ]]; }",
            2,
        )],
        "test_d3b_r1_phase_two_legacy_cleanup_marks_the_attempt or test_d3b_r1_mark_failure_contains_before_the_legacy_stop",
    ),
    (
        "r2_own4_legacy_mark_ignores_attempt_number",
        [(
            ".github/workflows/deploy.yml",
            "legacy.$RUN_ID.$RUN_ATTEMPT",
            "legacy.$RUN_ID",
            5,
        )],
        "test_d3b_r1_phase_two_legacy_cleanup_marks_the_attempt or test_new_attempt_can_deploy_after_containment",
    ),
    (
        "r2_own5_drop_staged_prefix_guard_in_contain",
        [(
            ".github/workflows/deploy.yml",
            '                if [[ $staged == "$checkout/data/.deploy."* ]]; then',
            '                if [[ -n $staged ]]; then',
            2,
        )],
        "test_stamp_failure_preserves_previous_entry or test_stamp_write_and_dirsync_faults_keep_complete_bytes or test_d3b_containment_windows_reserve_the_residual_check",
    ),
]


def run_pytest(root: Path, extra_args: list[str], *, timeout: int = 120) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root}:{root / 'src'}"
    cmd = [*PYTEST, *extra_args, str(root / "tests/test_deploy_workflow.py")]
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    tail = " | ".join(lines[-2:]) if lines else proc.stderr.strip()[-120:]
    return proc.returncode, tail


def precheck_all() -> None:
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    for name, edits, _ in MUTATIONS:
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            s = make_scratch_repo(Path(td))
            apply_replacements(s, edits)
    print(f"Pre-validation of all {len(MUTATIONS)} individual mutation targets: OK", flush=True)


def main() -> None:
    precheck_all()
    if "--precheck-only" in sys.argv:
        return

    started = time.monotonic()
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        control = make_scratch_repo(Path(td))
        rc, detail = run_pytest(control, ["-m", "not slow", "-k", "test_healthy_scheduler_passes"])
        assert rc == 0, f"Control smoke failed: {detail}"
        print(f"Scratch control smoke: PASS ({detail})", flush=True)

    targeted_survivors: list[tuple[str, list[tuple[str, str, str, int]]]] = []
    for name, edits, k_expr in MUTATIONS:
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            scratch = make_scratch_repo(Path(td))
            apply_replacements(scratch, edits)
            rc, detail = run_pytest(scratch, ["-m", "not slow", "-k", k_expr])
            if rc != 0:
                print(f"MUTATION {name}: DETECTED ({time.monotonic() - t0:.1f}s) [{detail}]", flush=True)
            else:
                targeted_survivors.append((name, edits))
                print(f"MUTATION {name}: SURVIVED_TARGETED ({time.monotonic() - t0:.1f}s) [{detail}]", flush=True)

    confirmed_survivors: list[str] = []
    caught_by_full_file: list[str] = []
    if targeted_survivors:
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            bundle_repo = make_scratch_repo(Path(td))
            can_bundle = True
            try:
                for _, edits in targeted_survivors:
                    apply_replacements(bundle_repo, edits)
            except AssertionError:
                can_bundle = False
            if can_bundle:
                t0 = time.monotonic()
                rc_b, det_b = run_pytest(bundle_repo, ["-m", "not slow"], timeout=240)
                print(f"SURVIVOR_BUNDLE ({len(targeted_survivors)} mutants): rc={rc_b} ({time.monotonic() - t0:.1f}s) [{det_b}]", flush=True)
                if rc_b == 0:
                    confirmed_survivors = [name for name, _ in targeted_survivors]

        if not confirmed_survivors:
            for name, edits in targeted_survivors:
                t0 = time.monotonic()
                with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
                    s = make_scratch_repo(Path(td))
                    apply_replacements(s, edits)
                    rc_f, det_f = run_pytest(s, ["-m", "not slow"], timeout=240)
                    if rc_f == 0:
                        confirmed_survivors.append(name)
                        print(f"FULL_FILE_CONFIRM {name}: SURVIVED ({time.monotonic() - t0:.1f}s) [{det_f}]", flush=True)
                    else:
                        caught_by_full_file.append(name)
                        print(f"FULL_FILE_CONFIRM {name}: DETECTED_OUTSIDE_TARGET ({time.monotonic() - t0:.1f}s) [{det_f}]", flush=True)

    print(f"SUMMARY total={len(MUTATIONS)} detected_targeted={len(MUTATIONS) - len(targeted_survivors)} "
          f"detected_full_file={len(caught_by_full_file)} confirmed_survivors={confirmed_survivors} "
          f"elapsed={time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
