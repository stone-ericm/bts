#!/usr/bin/env python3
"""Item 8 mutation testing on isolated scratch copies (run under yield_run.sh)."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[2]
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


def replace_exact(path: Path, old: str, new: str) -> None:
    txt = path.read_text()
    assert old in txt, f"target not found in {path}: {old!r}"
    path.write_text(txt.replace(old, new))


REQUIRED_MUTATIONS = [
    (
        "1_skip_contain_after_sync",
        ".github/workflows/deploy.yml",
        '            as_bts 480 mutation env UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT="$checkout/.venv" /home/bts/.local/bin/uv sync --locked --extra model >/dev/null\n'
        '            step_status sync "$?"',
        '            as_bts 480 mutation env UV_CACHE_DIR=/tmp/uv-cache UV_PROJECT_ENVIRONMENT="$checkout/.venv" /home/bts/.local/bin/uv sync --locked --extra model >/dev/null',
        "tests/test_deploy_workflow.py",
    ),
    (
        "2_print_verified_stop_unconditionally",
        ".github/workflows/deploy.yml",
        'Scheduler stopped: $stop.',
        'Scheduler stopped: verified.',
        "tests/test_deploy_workflow.py",
    ),
    (
        "3_drop_attempt_id_check_in_recovery",
        ".github/workflows/deploy.yml",
        '            if (( rc == 3 )) || ! same_ids; then',
        '            if (( rc == 3 )); then',
        "tests/test_deploy_workflow.py",
    ),
    (
        "4_write_committed_before_stamp",
        ".github/workflows/deploy.yml",
        '            as_bts 10 mutation /usr/bin/python3 -c "$fs_code" stamp "$staged" "$checkout/data/.last_deploy_iso" >/dev/null\n'
        '            step_status stamp "$?"\n'
        '            staged=\n'
        '            publish committed - - - -\n'
        '            step_status commit-record "$?"',
        '            publish committed - - - -\n'
        '            step_status commit-record "$?"\n'
        '            as_bts 10 mutation /usr/bin/python3 -c "$fs_code" stamp "$staged" "$checkout/data/.last_deploy_iso" >/dev/null\n'
        '            step_status stamp "$?"\n'
        '            staged=',
        "tests/test_deploy_workflow.py",
    ),
    (
        "5_plain_sleep_30_no_deadline",
        ".github/workflows/deploy.yml",
        '            run 35 2 wait sleep 30 >/dev/null',
        '            sleep 30 >/dev/null',
        "tests/test_deploy_workflow.py",
    ),
    (
        "6_accept_redirect_3xx_as_page",
        ".github/workflows/deploy.yml",
        'if [[ $post_status == 0 && $output =~ ^2[0-9][0-9]$ ]]; then',
        'if [[ $post_status == 0 && $output =~ ^[23][0-9][0-9]$ ]]; then',
        "tests/test_deploy_workflow.py",
    ),
    (
        "7_canary_skip_live_read_when_public",
        "src/bts/cli.py",
        '        if delivery not in ("dm", "public") or not _contest_state_required(config):',
        '        if delivery != "dm" or not _contest_state_required(config):',
        "tests/test_deploy_canary.py",
    ),
    (
        "8_heartbeat_timestamp_before_deploy_failed",
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
        "tests/test_heartbeat_staleness.py",
    ),
]

EXTRA_DISJOINT_WORKFLOW_MUTATIONS = [
    (
        "9_drop_run_id_check_in_same_ids",
        '            same_ids() { [[ $r_run == "$RUN_ID" && $r_try == "$RUN_ATTEMPT" ]]; }',
        '            same_ids() { [[ $r_try == "$RUN_ATTEMPT" ]]; }',
    ),
    (
        "10_curl_follow_redirects_in_contain_post",
        "output=$(run 20 5 normal curl --disable --noproxy '*' -sS -m 10",
        "output=$(run 20 5 normal curl --disable --noproxy '*' -L -sS -m 10",
    ),
    (
        "11_break_real_lockfile_and_mark_in_fs_code",
        '                elif op == "lockfile":\n                    p, = args',
        '                elif op == "lockfile":\n                    raise RuntimeError("broken lockfile")\n                    p, = args',
    ),
    (
        "12_drop_1500s_flock_wait_in_recovery",
        '                    run 1505 5 lock flock -w 1500 9',
        '                    run 10 5 lock flock -n 9',
    ),
    (
        "13_skip_contain_on_second_legacy_unit",
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
    ),
    (
        "14_skip_disable_inside_contain",
        '                as_user 10 mutation disable bts-scheduler >/dev/null; user_disabled=$?',
        '                user_disabled=0',
    ),
]


def run_pytest(root: Path, test_rel: str, extra_args: list[str] | None = None) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root}:{root / 'src'}"
    cmd = [*PYTEST, *(extra_args or ["-m", "not slow"]), str(root / test_rel)]
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=1200)
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    tail = " | ".join(lines[-3:]) if lines else ""
    return proc.returncode, tail


def main() -> None:
    # Quick scratch smoke check
    with tempfile.TemporaryDirectory() as td:
        control = make_scratch_repo(Path(td))
        rc, summary = run_pytest(control, "tests/test_deploy_workflow.py", ["-k", "test_healthy_scheduler_passes or test_passing_transcript_order_delta"])
        assert rc == 0, f"Control smoke failed: {summary}"
        print(f"Scratch control smoke: PASS ({summary})", flush=True)

    for name, target_rel, old, new, test_rel in REQUIRED_MUTATIONS:
        with tempfile.TemporaryDirectory() as td:
            scratch = make_scratch_repo(Path(td))
            replace_exact(scratch / target_rel, old, new)
            rc, summary = run_pytest(scratch, test_rel)
            status = "DETECTED" if rc != 0 else "SURVIVED (NOT DETECTED)"
            print(f"MUTATION {name}: {status} [{summary}]", flush=True)

    # Test disjoint extra workflow mutations together first; if any fails, test individually
    with tempfile.TemporaryDirectory() as td:
        scratch = make_scratch_repo(Path(td))
        for _, old, new in EXTRA_DISJOINT_WORKFLOW_MUTATIONS:
            replace_exact(scratch / ".github/workflows/deploy.yml", old, new)
        rc, summary = run_pytest(scratch, "tests/test_deploy_workflow.py")
        if rc == 0:
            for name, _, _ in EXTRA_DISJOINT_WORKFLOW_MUTATIONS:
                print(f"MUTATION {name}: SURVIVED (NOT DETECTED) [all fast workflow tests passed: {summary}]", flush=True)
        else:
            print(f"Combined extra workflow bundle had a detection ({summary}); checking individually...", flush=True)
            for name, old, new in EXTRA_DISJOINT_WORKFLOW_MUTATIONS:
                with tempfile.TemporaryDirectory() as td2:
                    s2 = make_scratch_repo(Path(td2))
                    replace_exact(s2 / ".github/workflows/deploy.yml", old, new)
                    rc2, sum2 = run_pytest(s2, "tests/test_deploy_workflow.py")
                    status2 = "DETECTED" if rc2 != 0 else "SURVIVED (NOT DETECTED)"
                    print(f"MUTATION {name}: {status2} [{sum2}]", flush=True)


if __name__ == "__main__":
    main()
