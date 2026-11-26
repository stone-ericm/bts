#!/usr/bin/env python3
"""Round-2 Item 4 mutation testing on isolated scratch copies (run via yield_run.sh after 14:35 ET)."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[3]
PYTEST = [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", "-x", "--tb=short"]


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
    base_wf = REPO / ".codex-review/wr-d3/base/.github/workflows/deploy.yml"
    base_hb = REPO / ".codex-review/wr-d3/base/scripts/check_heartbeat.py"
    tdw = (root / "tests/test_deploy_workflow.py").read_text()
    tdw = tdw.replace(
        '    baseline = subprocess.run(\n'
        '        ["git", "show", "24019a9a:.github/workflows/deploy.yml"],\n'
        '        cwd=WORKFLOW.parents[2], capture_output=True, text=True, check=True, timeout=5,\n'
        '    ).stdout',
        f'    baseline = Path({str(base_wf)!r}).read_text()',
    )
    (root / "tests/test_deploy_workflow.py").write_text(tdw)

    ths = (root / "tests/test_heartbeat_staleness.py").read_text()
    ths = ths.replace(
        '    return subprocess.run(\n'
        '        ["git", "show", "24019a9a:scripts/check_heartbeat.py"], cwd=root,\n'
        '        capture_output=True, text=True, check=True, timeout=5,\n'
        '    ).stdout',
        f'    return Path({str(base_hb)!r}).read_text()',
    )
    (root / "tests/test_heartbeat_staleness.py").write_text(ths)
    return root


def apply_replacements(root: Path, edits: list[tuple[str, str, str]]) -> None:
    for rel, old, new in edits:
        path = root / rel
        txt = path.read_text()
        assert old in txt, f"target not found in {rel}: {old!r}"
        path.write_text(txt.replace(old, new))


INDIVIDUAL_MUTATIONS: list[tuple[str, list[tuple[str, str, str]], str]] = [
    # --- Round-1 set (1-14) ---
    (
        "1_skip_contain_after_sync",
        [(
            ".github/workflows/deploy.yml",
            '            as_bts 480 mutation env UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT="$checkout/.venv" /home/bts/.local/bin/uv sync --locked --extra model >/dev/null\n'
            '            step_status sync "$?"',
            '            as_bts 480 mutation env UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT="$checkout/.venv" /home/bts/.local/bin/uv sync --locked --extra model >/dev/null',
        )],
        "tests/test_deploy_workflow.py",
    ),
    (
        "2_print_verified_stop_unconditionally",
        [(".github/workflows/deploy.yml", 'Scheduler stopped: $stop.', 'Scheduler stopped: verified.')],
        "tests/test_deploy_workflow.py",
    ),
    (
        "3_drop_attempt_id_check_in_recovery",
        [(".github/workflows/deploy.yml", '            if (( rc == 3 )) || ! same_ids; then', '            if (( rc == 3 )); then')],
        "tests/test_deploy_workflow.py",
    ),
    (
        "4_write_committed_before_stamp",
        [(
            ".github/workflows/deploy.yml",
            '            as_bts 10 mutation /usr/bin/python3 -c "$fs_code" stamp "$staged" "$checkout/data/.last_deploy_iso" >/dev/null\n'
            '            step_status stamp "$?"\n'
            '            staged=\n'
            '            publication=committing\n'
            '            publish committed - - - -\n'
            '            step_status commit-record "$?"\n'
            '            publication=committed',
            '            publication=committing\n'
            '            publish committed - - - -\n'
            '            step_status commit-record "$?"\n'
            '            publication=committed\n'
            '            as_bts 10 mutation /usr/bin/python3 -c "$fs_code" stamp "$staged" "$checkout/data/.last_deploy_iso" >/dev/null\n'
            '            step_status stamp "$?"\n'
            '            staged=',
        )],
        "tests/test_deploy_workflow.py",
    ),
    (
        "5_plain_sleep_30_no_deadline",
        [(".github/workflows/deploy.yml", '            run 35 2 wait sleep 30 >/dev/null', '            sleep 30 >/dev/null')],
        "tests/test_deploy_workflow.py",
    ),
    (
        "6_accept_redirect_3xx_as_page",
        [(
            ".github/workflows/deploy.yml",
            'if [[ $post_status == 0 && $output =~ ^2[0-9][0-9]$ ]]; then',
            'if [[ $post_status == 0 && $output =~ ^[23][0-9][0-9]$ ]]; then',
        )],
        "tests/test_deploy_workflow.py",
    ),
    (
        "7_canary_skip_live_read_when_public",
        [(
            "src/bts/cli.py",
            '        if delivery not in ("dm", "public") or not _contest_state_required(config):',
            '        if delivery != "dm" or not _contest_state_required(config):',
        )],
        "tests/test_deploy_canary.py",
    ),
    (
        "8_heartbeat_timestamp_before_deploy_failed",
        [(
            "scripts/check_heartbeat.py",
            '    if raw.get("state") == "deploy_failed":\n'
            '        step = raw.get("step")\n'
            '        target = raw.get("target_sha")\n'
            '        if not isinstance(step, str) or step not in DEPLOY_FAILURE_STEPS:\n'
            '            step = "unknown"\n'
            '        if not isinstance(target, str) or not re.fullmatch("[0-9a-f]{7}", target):\n'
            '            target = "unknown"\n'
            '        return True, f"Deploy failed at {step} (requested target {target}). See INCIDENT.md §4."\n\n'
            '    ts_str = raw.get("timestamp")\n'
            '    state = raw.get("state", "unknown")\n'
            '    try:\n'
            '        ts = datetime.fromisoformat(ts_str)\n'
            '    except (TypeError, ValueError):\n'
            '        return True, f"heartbeat timestamp invalid: {ts_str}"',
            '    ts_str = raw.get("timestamp")\n'
            '    state = raw.get("state", "unknown")\n'
            '    try:\n'
            '        ts = datetime.fromisoformat(ts_str)\n'
            '    except (TypeError, ValueError):\n'
            '        return True, f"heartbeat timestamp invalid: {ts_str}"\n\n'
            '    if raw.get("state") == "deploy_failed":\n'
            '        step = raw.get("step")\n'
            '        target = raw.get("target_sha")\n'
            '        if not isinstance(step, str) or step not in DEPLOY_FAILURE_STEPS:\n'
            '            step = "unknown"\n'
            '        if not isinstance(target, str) or not re.fullmatch("[0-9a-f]{7}", target):\n'
            '            target = "unknown"\n'
            '        return True, f"Deploy failed at {step} (requested target {target}). See INCIDENT.md §4."',
        )],
        "tests/test_heartbeat_staleness.py",
    ),
    (
        "9_drop_run_id_check_in_same_ids",
        [(
            ".github/workflows/deploy.yml",
            '            same_ids() { [[ $r_run == "$RUN_ID" && $r_try == "$RUN_ATTEMPT" ]]; }',
            '            same_ids() { [[ $r_try == "$RUN_ATTEMPT" ]]; }',
        )],
        "tests/test_deploy_workflow.py",
    ),
    (
        "10_curl_follow_redirects_in_contain_post",
        [(
            ".github/workflows/deploy.yml",
            "output=$(run 20 5 normal curl --disable --globoff --noproxy '*' -sS -m 10",
            "output=$(run 20 5 normal curl --disable --globoff --noproxy '*' -L -sS -m 10",
        )],
        "tests/test_deploy_workflow.py",
    ),
    (
        "11_break_real_lockfile_and_mark_in_fs_code",
        [(
            ".github/workflows/deploy.yml",
            '                elif op == "lockfile":\n                    p, = args',
            '                elif op == "lockfile":\n                    raise RuntimeError("broken lockfile")\n                    p, = args',
        )],
        "tests/test_deploy_workflow.py",
    ),
    (
        "12_drop_1500s_flock_wait_in_recovery",
        [(
            ".github/workflows/deploy.yml",
            '                    run 1505 5 lock flock -w 1500 9',
            '                    run 10 5 lock flock -n 9',
        )],
        "tests/test_deploy_workflow.py",
    ),
    (
        "13_skip_contain_on_second_legacy_unit",
        [(
            ".github/workflows/deploy.yml",
            '                for name in "${legacy_names[@]}"; do\n'
            '                    system_unit 20 mutation disable "$name" >/dev/null\n'
            '                    step_status legacy-units "$?"\n'
            '                    system_unit 20 mutation stop "$name" >/dev/null\n'
            '                    step_status legacy-units "$?"\n'
            '                    run 20 5 mutation rm -f "$legacy_dir/$name" >/dev/null\n'
            '                    step_status legacy-units "$?"\n'
            '                done',
            '                for name in "${legacy_names[@]}"; do\n'
            '                    system_unit 20 mutation disable "$name" >/dev/null\n'
            '                    [[ $name != "$scheduler" ]] || step_status legacy-units "$?"\n'
            '                    system_unit 20 mutation stop "$name" >/dev/null\n'
            '                    [[ $name != "$scheduler" ]] || step_status legacy-units "$?"\n'
            '                    run 20 5 mutation rm -f "$legacy_dir/$name" >/dev/null\n'
            '                    [[ $name != "$scheduler" ]] || step_status legacy-units "$?"\n'
            '                done',
        )],
        "tests/test_deploy_workflow.py",
    ),
    (
        "14_skip_disable_inside_contain",
        [(
            ".github/workflows/deploy.yml",
            '                as_user 10 mutation disable bts-scheduler >/dev/null; user_disabled=$?',
            '                user_disabled=0',
        )],
        "tests/test_deploy_workflow.py",
    ),
    # --- Round-2 brief required additions (15-19) ---
    (
        "15_on_signal_not_started_when_read_fails_after_begun",
        [(
            ".github/workflows/deploy.yml",
            '                case $publication in\n'
            '                    begun) contain interrupted 130 ;;\n'
            '                    committed) exec 9>&-; exit 0 ;;\n'
            '                esac\n'
            '\n'
            '                # A rename may have completed before its caller observed the result.\n'
            '                local budget_start=$SECONDS budget_origin=0 deadline=15000 rc\n'
            '                read_record 15\n'
            '                rc=$?\n'
            '                if (( rc == 0 )) && same_record && [[ $r_state == committed ]]; then\n'
            '                    exec 9>&-\n'
            '                    exit 0\n'
            '                fi\n'
            '                if [[ $publication == beginning ]] && (( rc == 3 )); then\n'
            '                    not_started attempt-record\n'
            '                fi',
            '                case $publication in\n'
            '                    committed) exec 9>&-; exit 0 ;;\n'
            '                esac\n'
            '\n'
            '                # A rename may have completed before its caller observed the result.\n'
            '                local budget_start=$SECONDS budget_origin=0 deadline=15000 rc\n'
            '                read_record 15\n'
            '                rc=$?\n'
            '                if (( rc == 0 )) && same_record && [[ $r_state == committed ]]; then\n'
            '                    exec 9>&-\n'
            '                    exit 0\n'
            '                fi\n'
            '                if { [[ $publication == beginning ]] && (( rc == 3 )); } || { [[ $publication == begun ]] && (( rc != 0 )); }; then\n'
            '                    not_started attempt-record\n'
            '                fi',
        )],
        "tests/test_deploy_workflow.py",
    ),
    (
        "16_drop_fallback_clock_in_contain",
        [(
            ".github/workflows/deploy.yml",
            '                local budget_start=$SECONDS budget_origin\n'
            '                # One sample anchors the handler\'s separate budget. Bash SECONDS also\n'
            '                # keeps that budget running when the external clock cannot be sampled.\n'
            '                budget_origin=$(clock)\n'
            '                [[ $budget_origin =~ ^[0-9]+$ ]] || budget_origin=0\n'
            '                containment_deadline=$((budget_origin + 280000))',
            '                start_phase 280 || deadline=0\n'
            '                containment_deadline=$deadline',
        )],
        "tests/test_deploy_workflow.py",
    ),
    (
        "17_accept_blank_control_group_as_absence",
        [(
            ".github/workflows/deploy.yml",
            '                [[ -n $previous_group || -n $group ]] || return 1\n'
            '                if [[ -n $previous_group && $previous_group != "$group" ]]; then\n'
            '                    fs 20 normal cgroup /sys/fs/cgroup "$previous_group" || return 1\n'
            '                fi\n'
            '                [[ -z $group ]] || fs 20 normal cgroup /sys/fs/cgroup "$group"',
            '                fs 20 normal cgroup /sys/fs/cgroup "$group"',
        )],
        "tests/test_deploy_workflow.py",
    ),
    (
        "18_drop_globoff",
        [(".github/workflows/deploy.yml", "curl --disable --globoff --noproxy", "curl --disable --noproxy")],
        "tests/test_deploy_workflow.py",
    ),
    (
        "19_collapse_commit_record_status_to_1",
        [(
            ".github/workflows/deploy.yml",
            '                if [[ $wanted == committed ]]; then\n'
            '                    case $status:$rc in\n'
            '                        124:*|137:*|*:124|*:137) return 124 ;;\n'
            '                    esac\n'
            '                    (( status == 0 )) || return "$status"\n'
            '                fi',
            '                if [[ $wanted == committed ]]; then\n'
            '                    return 1\n'
            '                fi',
        )],
        "tests/test_deploy_workflow.py",
    ),
]

CANDIDATE_SURVIVOR_WORKFLOW_MUTATIONS: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "20_drop_st_uid_zero_check_in_lockfile_and_mark",
        [
            (
                ".github/workflows/deploy.yml",
                "if not stat.S_ISREG(s.st_mode) or s.st_uid != 0 or stat.S_IMODE(s.st_mode) != 0o600:",
                "if not stat.S_ISREG(s.st_mode) or stat.S_IMODE(s.st_mode) != 0o600:",
            ),
            (
                ".github/workflows/deploy.yml",
                "if s.st_uid != 0 or stat.S_IMODE(s.st_mode) != 0o600:",
                "if stat.S_IMODE(s.st_mode) != 0o600:",
            ),
        ],
    ),
    (
        "21_skip_previous_group_when_post_stop_group_changed_nonempty",
        [(
            ".github/workflows/deploy.yml",
            '                [[ -n $previous_group || -n $group ]] || return 1\n'
            '                if [[ -n $previous_group && $previous_group != "$group" ]]; then\n'
            '                    fs 20 normal cgroup /sys/fs/cgroup "$previous_group" || return 1\n'
            '                fi\n'
            '                [[ -z $group ]] || fs 20 normal cgroup /sys/fs/cgroup "$group"',
            '                [[ -n $previous_group || -n $group ]] || return 1\n'
            '                fs 20 normal cgroup /sys/fs/cgroup "${group:-$previous_group}"',
        )],
    ),
    (
        "22_drop_same_record_check_in_on_signal_committed",
        [(
            ".github/workflows/deploy.yml",
            "                if (( rc == 0 )) && same_record && [[ $r_state == committed ]]; then",
            "                if (( rc == 0 )) && [[ $r_state == committed ]]; then",
        )],
    ),
]


def run_pytest(root: Path, test_rel: str, extra_args: list[str] | None = None) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root}:{root / 'src'}"
    cmd = [*PYTEST, *(extra_args or ["-m", "not slow"]), str(root / test_rel)]
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=1500)
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    tail = " | ".join(lines[-3:]) if lines else ""
    return proc.returncode, tail


def check_workflow_mutant(root: Path, test_rel: str) -> tuple[str, str]:
    rc, summary = run_pytest(root, test_rel, ["-m", "not slow"])
    if rc != 0:
        return "DETECTED", summary
    if test_rel == "tests/test_deploy_workflow.py":
        rc_slow, sum_slow = run_pytest(root, test_rel, ["-m", "slow"])
        if rc_slow != 0:
            return "DETECTED (slow)", sum_slow
        return "SURVIVED (NOT DETECTED)", f"fast: {summary} ; slow: {sum_slow}"
    return "SURVIVED (NOT DETECTED)", summary


def main() -> None:
    # Pre-validate all target strings before running any pytest subprocess
    for name, edits, _ in INDIVIDUAL_MUTATIONS:
        with tempfile.TemporaryDirectory() as td:
            s = make_scratch_repo(Path(td))
            apply_replacements(s, edits)
    with tempfile.TemporaryDirectory() as td:
        s = make_scratch_repo(Path(td))
        for _, edits in CANDIDATE_SURVIVOR_WORKFLOW_MUTATIONS:
            apply_replacements(s, edits)
    print("Pre-validation of all 22 mutation targets: OK", flush=True)

    # Scratch control smoke check
    with tempfile.TemporaryDirectory() as td:
        control = make_scratch_repo(Path(td))
        rc, summary = run_pytest(
            control,
            "tests/test_deploy_workflow.py",
            ["-k", "test_healthy_scheduler_passes or test_passing_transcript_order_delta"],
        )
        assert rc == 0, f"Control smoke failed: {summary}"
        print(f"Scratch control smoke: PASS ({summary})", flush=True)

    for name, edits, test_rel in INDIVIDUAL_MUTATIONS:
        with tempfile.TemporaryDirectory() as td:
            scratch = make_scratch_repo(Path(td))
            apply_replacements(scratch, edits)
            status, summary = check_workflow_mutant(scratch, test_rel)
            print(f"MUTATION {name}: {status} [{summary}]", flush=True)

    # Test disjoint candidate-survivor workflow mutations together first (fast + slow); split if detected
    with tempfile.TemporaryDirectory() as td:
        scratch = make_scratch_repo(Path(td))
        for _, edits in CANDIDATE_SURVIVOR_WORKFLOW_MUTATIONS:
            apply_replacements(scratch, edits)
        status, summary = check_workflow_mutant(scratch, "tests/test_deploy_workflow.py")
        if status.startswith("SURVIVED"):
            for name, _ in CANDIDATE_SURVIVOR_WORKFLOW_MUTATIONS:
                print(f"MUTATION {name}: {status} [{summary}]", flush=True)
        else:
            print(f"Candidate-survivor bundle had a detection ({summary}); checking individually...", flush=True)
            for name, edits in CANDIDATE_SURVIVOR_WORKFLOW_MUTATIONS:
                with tempfile.TemporaryDirectory() as td2:
                    s2 = make_scratch_repo(Path(td2))
                    apply_replacements(s2, edits)
                    st2, sum2 = check_workflow_mutant(s2, "tests/test_deploy_workflow.py")
                    print(f"MUTATION {name}: {st2} [{sum2}]", flush=True)


if __name__ == "__main__":
    main()
