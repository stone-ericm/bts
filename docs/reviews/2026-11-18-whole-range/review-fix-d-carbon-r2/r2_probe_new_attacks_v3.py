#!/usr/bin/env python3
"""Round-2 adversarial attacks v3 on HEAD (0645be4b) across F1..F16."""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace

from click.testing import CliRunner

R2_DIR = Path(__file__).resolve().parent
REPO = R2_DIR.parents[2]
INSTALLER = REPO / "scripts" / "install-deploy-guard-hetzner.sh"
CRON_SCRIPT = REPO / "scripts" / "cron-setup-hetzner.sh"
HEAD_MONITOR = REPO / "scripts" / "check_heartbeat.py"
BASE_MONITOR = R2_DIR.parent / "base" / "scripts" / "check_heartbeat.py"
EXAMPLE_TOML = REPO / "config" / "orchestrator.example.toml"

OLD_SHA = "a" * 40
TARGET_SHA = "b" * 40
RUN_ID = "1795197580-adv02"


def extract_guard_source() -> str:
    lines = INSTALLER.read_text().splitlines(keepends=True)
    start, = (i for i, ln in enumerate(lines) if "<<'BTS_DEPLOY_HOLD_PYTHON'" in ln)
    end, = (i for i, ln in enumerate(lines) if ln.rstrip("\n") == "BTS_DEPLOY_HOLD_PYTHON")
    return "".join(lines[start + 1 : end])


def load_guard(guard_root: Path, project_dir: Path, name: str = "guard_adv") -> ModuleType:
    mod = ModuleType(name)
    exec(compile(extract_guard_source(), str(INSTALLER), "exec"), mod.__dict__)
    mod.ROOT = guard_root
    mod.UID = os.geteuid()
    mod.PROJECT = str(project_dir)
    mod.time = SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=lambda s: None)
    return mod


def setup_guard(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o755)
    for lk in ("writers.lock", "transaction.lock", "state.lock"):
        (path / lk).touch()
        os.chmod(path / lk, 0o644)
    return path


def render_real_crontab(work: Path, intent: str) -> str:
    home = work / f"home_{intent}"
    (home / "projects" / "bts").mkdir(parents=True)
    (home / "projects" / "bts" / ".env").write_text("")
    (home / ".local" / "bin").mkdir(parents=True)
    uv = home / ".local" / "bin" / "uv"
    uv.write_text(f'#!/bin/sh\nshift 2\nexec "{sys.executable}" -c "from bts.cli import cli; cli()" "$@"\n')
    uv.chmod(0o755)
    cfg = home / ".bts-orchestrator.toml"
    if intent == "research":
        cfg.write_text('[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n')
    else:
        cfg.write_text(
            '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
    shims = work / f"shims_{intent}"
    shims.mkdir()
    c_shim = shims / "crontab"
    c_shim.write_text('#!/bin/sh\necho "no crontab for bts" >&2; exit 1\n')
    c_shim.chmod(0o755)
    res = subprocess.run(
        ["/bin/bash", str(CRON_SCRIPT), "show"],
        env={"HOME": str(home), "PATH": f"{shims}:/usr/bin:/bin", "HEALTHCHECKS_PING_URL": "https://hc-ping.com/box"},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
    )
    assert res.returncode == 0, res.stderr
    _, proposed = res.stdout.split("Would install:\n")
    return proposed.replace(str(home), "/home/bts")


def attack_f1_crontab(work: Path) -> dict:
    gdir = setup_guard(work / "f1_guard")
    pdir = work / "f1_proj"
    pdir.mkdir()
    mod = load_guard(gdir, pdir, "f1_adv")

    real_enter = render_real_crontab(work, "enter")
    real_research = render_real_crontab(work, "research")

    def check_cron(raw_str: str) -> str:
        try:
            mod.verify_crontab(raw_str.encode())
            return "ACCEPTED"
        except mod.Refused as e:
            return f"REFUSED:{e.step}"

    assert check_cron(real_enter) == "ACCEPTED"
    assert check_cron(real_research) == "ACCEPTED"

    backslash_subcmd = real_research + (
        "0 4 * * * cd /home/bts/projects/bts && /home/bts/.local/bin/uv run --no-sync bts re\\concile >> /home/bts/logs/cron.log 2>&1\n"
    )
    backslash_bts = real_research + (
        "0 4 * * * cd /home/bts/projects/bts && /home/bts/.local/bin/uv run --no-sync b\\ts check-results --date 2027-07-14\n"
    )
    shell_var_writer = real_research + (
        "0 4 * * * cd /home/bts/projects/bts && cmd=reconcile && /home/bts/.local/bin/uv run --no-sync bts $cmd\n"
    )
    python_c_entry = real_research + (
        "0 4 * * * cd /home/bts/projects/bts && /home/bts/projects/bts/.venv/bin/python -c 'from bts.cli import cli; cli()' reconcile\n"
    )

    fake_home = work / "sh_exec_demo"
    (fake_home / ".local" / "bin").mkdir(parents=True)
    ran_log = fake_home / "ran_argv.json"
    fake_uv = fake_home / ".local" / "bin" / "uv"
    fake_uv.write_text(
        f'#!{sys.executable}\nimport json, sys\n'
        f'open({str(ran_log)!r}, "w").write(json.dumps(sys.argv[1:]))\n'
    )
    fake_uv.chmod(0o755)
    now_ts = int(time.time())
    with mod.directory() as rfd:
        mod.publish_json(rfd, "hold", {
            "schema": "bts_deploy_hold_v1", "deploy_id": RUN_ID,
            "owner_unit": f"bts-deploy-{RUN_ID}.service", "state": "failed", "step": "sync",
            "target": TARGET_SHA, "previous": OLD_SHA, "armed_at": now_ts - 10, "updated_at": now_ts - 5,
        })
    sh_cmd = f"{fake_uv} run --no-sync bts re\\concile"
    subprocess.run(["/bin/sh", "-c", sh_cmd], env={"BTS_TEST_DEPLOY_GUARD_DIR": str(gdir)}, check=True, timeout=5)
    sh_executed_argv = json.loads(ran_log.read_text())

    base_no_hb = "\n".join(
        ln for ln in real_research.splitlines() if "scripts/check_heartbeat.py" not in ln
    ) + "\n"
    hb_equals_override = base_no_hb + (
        "*/5 * * * * cd /home/bts/projects/bts && /usr/bin/python3 scripts/check_heartbeat.py "
        "--heartbeat-path data/.heartbeat --deploy-hold /var/lib/bts-deploy/hold --deploy-hold=/nonexistent/hold # BTS-HETZNER\n"
    )
    hb_reboot_schedule = base_no_hb + (
        "@reboot cd /home/bts/projects/bts && set -a && . ./.env && set +a && "
        "/usr/bin/python3 scripts/check_heartbeat.py --heartbeat-path data/.heartbeat --deploy-hold /var/lib/bts-deploy/hold\n"
    )
    hb_dead_or_branch = base_no_hb + (
        "*/5 * * * * true || /usr/bin/python3 scripts/check_heartbeat.py "
        "--heartbeat-path data/.heartbeat --deploy-hold /var/lib/bts-deploy/hold\n"
    )
    hb_valid_equals = base_no_hb + (
        "*/5 * * * * cd /home/bts/projects/bts && /usr/bin/python3 scripts/check_heartbeat.py "
        "--heartbeat-path data/.heartbeat --deploy-hold=/var/lib/bts-deploy/hold\n"
    )

    hb_file = work / "fresh_hb.json"
    hb_file.write_text(json.dumps({
        "state": "sleeping",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sleeping_until": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
    }))
    absent_hold = work / "does_not_exist_dir" / "hold"
    mon_res = subprocess.run(
        [
            sys.executable, str(HEAD_MONITOR),
            "--heartbeat-path", str(hb_file),
            "--deploy-hold", str(gdir / "hold"),
            f"--deploy-hold={absent_hold}",
            "--no-churn",
        ],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
    )

    return {
        "canonical_enter": check_cron(real_enter),
        "canonical_research": check_cron(real_research),
        "writer_backslash_subcmd": {
            "verify_crontab": check_cron(backslash_subcmd),
            "bin_sh_executes_argv_while_held": sh_executed_argv,
        },
        "writer_backslash_bts": check_cron(backslash_bts),
        "writer_shell_var": check_cron(shell_var_writer),
        "writer_python_c_cli": check_cron(python_c_entry),
        "monitor_equals_override_to_nonexistent": {
            "verify_crontab": check_cron(hb_equals_override),
            "monitor_rc_while_real_guard_held": mon_res.returncode,
            "monitor_stdout": mon_res.stdout.strip(),
        },
        "monitor_at_reboot_schedule": check_cron(hb_reboot_schedule),
        "monitor_true_or_dead_branch": check_cron(hb_dead_or_branch),
        "monitor_valid_equals_syntax": check_cron(hb_valid_equals),
    }


def attack_f2_owner_and_signals(work: Path) -> dict:
    gdir = setup_guard(work / "f2_guard")
    pdir = work / "f2_proj"
    (pdir / "data").mkdir(parents=True)
    mod = load_guard(gdir, pdir, "f2_adv")
    now_ts = int(time.time())
    rec = {
        "schema": "bts_deploy_hold_v1", "deploy_id": RUN_ID,
        "owner_unit": f"bts-deploy-{RUN_ID}.service", "state": "activating", "step": "activation",
        "target": TARGET_SHA, "previous": OLD_SHA, "armed_at": now_ts - 15, "updated_at": now_ts - 5,
    }
    info = (gdir / "transaction.lock").stat()
    other_info = (gdir / "writers.lock").stat()

    proc_cases = {
        "blocked_waiter_arrow": f"8: -> FLOCK ADVISORY WRITE 555 {os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino} 0 EOF\n",
        "shared_read_flock": f"8: FLOCK ADVISORY READ 555 {os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino} 0 EOF\n",
        "posix_fcntl_lock": f"8: POSIX ADVISORY WRITE 555 {os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino} 0 EOF\n",
        "wrong_inode_writers_lock": f"8: FLOCK ADVISORY WRITE 555 {os.major(other_info.st_dev):02x}:{os.minor(other_info.st_dev):02x}:{other_info.st_ino} 0 EOF\n",
        "exact_granted_write_flock": f"8: FLOCK ADVISORY WRITE 555 {os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino} 0 EOF\n",
    }
    mod.subprocess = SimpleNamespace(
        run=lambda argv, **kw: SimpleNamespace(returncode=0, stdout=b"ActiveState=active\nMainPID=555\n"),
        Popen=subprocess.Popen, DEVNULL=subprocess.DEVNULL, PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired,
    )
    proc_results = {}
    for k, content in proc_cases.items():
        pl = work / f"pl_{k}"
        pl.write_text(content)
        mod.PROC_LOCKS = pl
        with mod.directory() as rfd:
            proc_results[k] = mod.owner_alive(rfd, rec)

    gdir_k1 = setup_guard(work / "f2_k1")
    mod_k1 = load_guard(gdir_k1, pdir, "f2_k1")
    k1_events = []
    mod_k1.bts_uid = lambda: 1000
    mod_k1.verified_stop = lambda uid: (k1_events.append("stop"), True)[1]
    mod_k1.page = lambda r, s: (k1_events.append("page"), 204)[1]
    with mod_k1.directory() as rfd:
        mod_k1.publish_json(rfd, "hold", rec)
        with redirect_stdout(io.StringIO()):
            next_rc = mod_k1.owned_deploy(rfd, "1795197599-next", TARGET_SHA, "")
        k1_hold = mod_k1.read_hold(rfd)
        k1_old_res = mod_k1.read_json(rfd, f"run-{RUN_ID}.result")

    gdir_sig = setup_guard(work / "f2_sig_guard")
    mod_sig = load_guard(gdir_sig, pdir, "f2_sig")
    mod_sig.bts_uid = lambda: 1000
    mod_sig.preflight = lambda uid, target, fetch=True: (OLD_SHA, ["git"])
    mod_sig.window = lambda *a, **kw: None
    mod_sig.verified_stop_dashboard = lambda uid: False
    mod_sig.verified_stop = lambda uid: (_ for _ in ()).throw(InterruptedError("SIGTERM during containment stop"))
    with mod_sig.directory() as rfd:
        escaped = False
        try:
            with redirect_stdout(io.StringIO()):
                mod_sig.owned_deploy(rfd, RUN_ID, TARGET_SHA, "")
        except InterruptedError:
            escaped = True
        hold_after_sig = mod_sig.read_hold(rfd)
        res_before_hook = mod_sig.read_json(rfd, f"run-{RUN_ID}.result")
        mod_sig.verified_stop = lambda uid: True
        mod_sig.page = lambda r, s: 204
        with redirect_stdout(io.StringIO()):
            hook_rc = mod_sig.reconcile(rfd, RUN_ID)
        res_after_hook = mod_sig.read_json(rfd, f"run-{RUN_ID}.result")

    return {
        "proc_locks_cases": proc_results,
        "hook_killed_before_transition_reconciled_by_next_deploy": {
            "next_rc": next_rc,
            "hold_state": k1_hold["state"],
            "hold_step": k1_hold["step"],
            "events": k1_events,
            "old_result_written": k1_old_res,
        },
        "sigterm_during_containment": {
            "escaped_owned_deploy": escaped,
            "hold_state": hold_after_sig["state"],
            "hold_step": hold_after_sig["step"],
            "result_before_execstoppost": res_before_hook,
            "execstoppost_rc": hook_rc,
            "result_after_execstoppost": res_after_hook,
        },
    }


def attack_f3_units_and_env(work: Path) -> dict:
    real_guard = setup_guard(work / "f3_real_guard")
    decoy_guard = setup_guard(work / "f3_decoy_guard")
    tool = work / "bts-deploy-hold"
    tool.write_text(extract_guard_source())
    tool.chmod(0o755)
    now_ts = int(time.time())
    (real_guard / "hold").write_text(json.dumps({
        "schema": "bts_deploy_hold_v1", "deploy_id": RUN_ID,
        "owner_unit": f"bts-deploy-{RUN_ID}.service", "state": "failed", "step": "sync",
        "target": TARGET_SHA, "previous": OLD_SHA, "armed_at": now_ts - 10, "updated_at": now_ts - 5,
    }))
    py_bin = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable
    cond_with_decoy = subprocess.run(
        [py_bin, "-I", str(tool), "check-clear", "--scheduler"],
        env={"BTS_TEST_DEPLOY_GUARD_DIR": str(decoy_guard)},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
    )
    cond_with_real = subprocess.run(
        [py_bin, "-I", str(tool), "check-clear", "--scheduler"],
        env={"BTS_TEST_DEPLOY_GUARD_DIR": str(real_guard)},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
    )

    mod = load_guard(real_guard, work, "f3_unit_adv")
    return {
        "python_minus_I_still_honors_BTS_TEST_DEPLOY_GUARD_DIR_as_nonroot": {
            "real_guard_held_rc": cond_with_real.returncode,
            "decoy_env_override_rc": cond_with_decoy.returncode,
        },
        "queried_scheduler_properties": list(mod.SCHEDULER_PROPERTIES),
        "queried_dashboard_properties": list(mod.UNIT_PROPERTIES),
    }


def attack_f5_configs(work: Path) -> dict:
    from bts import orchestrator, scheduler
    from bts.cli import cli
    from bts.picks import DailyPick, Pick
    from bts.strategy import PickResult, SelectionResult
    from tests.test_contest_dated_cs4a import observation, write

    cli_mod = import_module("bts.cli")
    orig_now = cli_mod._now_et
    now_et = datetime.fromisoformat("2027-07-15T12:00:00-04:00")
    cli_mod._now_et = lambda: now_et
    runner = CliRunner()

    try:
        valid_dir = work / "valid_cfgs"
        (valid_dir / "data" / "picks").mkdir(parents=True)
        write(valid_dir / "data" / "picks", observation("2027-07-15", recorded_at="2027-07-15T10:00:00-04:00"))
        ex_cfg = valid_dir / "example.toml"
        ex_cfg.write_bytes(EXAMPLE_TOML.read_bytes())

        hetzner_live = valid_dir / "hetzner_live.toml"
        hetzner_live.write_text(
            '[orchestrator]\npicks_dir = "data/picks"\n'
            '[bluesky]\ndm_recipient = "stonehengee.bsky.social"\n'
            '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\ncontest_state_required = true\n'
            'shadow_model = false\nlive_forward_capture_on_lock = true\n'
            '[health_checks]\nenabled = true\ncontest_state_expected = true\npooled_dir = ""\nleaderboard_dir = ""\n'
            '[[tiers]]\nname = "hetzner"\ntype = "local"\n'
        )
        hetzner_research = valid_dir / "hetzner_research.toml"
        hetzner_research.write_text(
            '[orchestrator]\npicks_dir = "data/picks"\n'
            '[scheduler]\npick_delivery = "private"\nentry_intent = "research"\n'
            '[[tiers]]\nname = "mac"\nssh_host = "mac"\nbts_dir = "/Users/stone/projects/bts"\ntimeout_min = 5\n'
        )

        sub_stdout_proc = subprocess.run(
            [sys.executable, "-c", "from bts.cli import cli; cli()", "deploy-canary", "--config", str(hetzner_live)],
            cwd=str(valid_dir),
            env={"PATH": os.environ.get("PATH", ""), "HOME": "/nonexistent", "PYTHONPATH": str(REPO / "src")},
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
        )

        old_cwd = os.getcwd()
        os.chdir(valid_dir)
        try:
            valid_outcomes = {}
            for label, p in [
                ("orchestrator_example_toml", ex_cfg),
                ("hetzner_live_local_tier_dm", hetzner_live),
                ("hetzner_research_no_bluesky", hetzner_research),
            ]:
                r = runner.invoke(cli, ["deploy-canary", "--config", str(p)])
                valid_outcomes[label] = {"rc": r.exit_code, "out": r.stdout.strip()}
        finally:
            os.chdir(old_cwd)

        adv_dir = work / "adv_f5"
        picks = adv_dir / "picks"
        write(picks, observation("2027-07-15", recorded_at="2027-07-15T10:00:00-04:00"))

        cfg_fb = adv_dir / "fb_inverted.toml"
        cfg_fb.write_text(
            f'[[tiers]]\nname = "local"\ntype = "local"\n'
            f'[orchestrator]\npicks_dir = {json.dumps(str(picks))}\nheartbeat_path = {json.dumps(str(adv_dir / ".hb1"))}\n'
            '[bluesky]\ndm_recipient = "stonehengee.bsky.social"\n'
            '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\ncontest_state_required = true\n'
            'fallback_deadline_min = 20\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
        r_fb_canary = runner.invoke(cli, ["deploy-canary", "--config", str(cfg_fb)])

        orig_fetch = scheduler.fetch_schedule
        orig_now_sched = scheduler._now_et
        orig_sleep = scheduler._watchdog_ping_sleep
        orig_single = scheduler.run_single_check
        try:
            scheduler._now_et = lambda: datetime.fromisoformat("2027-07-15T18:30:00-04:00")
            scheduler._watchdog_ping_sleep = lambda s, interval_sec=60: None
            scheduler.fetch_schedule = lambda d: [{
                "gamePk": 101, "gameDate": "2027-07-15T23:10:00Z", "gameType": "R",
                "teams": {"away": {"team": {"name": "NYY"}}, "home": {"team": {"name": "BOS"}}},
            }]
            cand_pick = Pick(
                batter_name="Test Batter", batter_id=1, team="NYY", lineup_position=1,
                pitcher_name="Test Pitcher", pitcher_id=2, p_game_hit=0.82,
                flags=["PROJECTED"], projected_lineup=True, game_pk=101,
                game_time="2027-07-15T23:10:00Z",
            )
            cand_daily = DailyPick(
                date="2027-07-15", run_time="2027-07-15T18:30:00-04:00",
                pick=cand_pick, double_down=None, runner_up=None,
            )
            pr = PickResult(daily=cand_daily, locked=False)
            sel = SelectionResult(
                pick_result=pr, action="single", source="mdp",
                primary_candidate={"batter_id": 1, "batter_name": "Test Batter"},
                double_candidate=None, no_pick_reason=None, streak=12, saver_available=True,
            )
            scheduler.run_single_check = lambda **kw: {
                "skipped": False, "new_lineups": 0, "should_post": False,
                "pick_result": pr, "pick_name": "Test Batter", "pick_p": 0.82,
                "selection": sel,
            }
            fb_crash = None
            try:
                with redirect_stderr(io.StringIO()):
                    scheduler.run_day("2027-07-15", orchestrator.load_config(cfg_fb), dry_run=False)
            except Exception as exc:
                fb_crash = f"{type(exc).__name__}: {exc}"
        finally:
            scheduler.fetch_schedule = orig_fetch
            scheduler._now_et = orig_now_sched
            scheduler._watchdog_ping_sleep = orig_sleep
            scheduler.run_single_check = orig_single

        blocker_file = adv_dir / "regular_file"
        blocker_file.write_text("not a dir")
        cfg_hb = adv_dir / "hb_under_file.toml"
        cfg_hb.write_text(
            f'[[tiers]]\nname = "local"\ntype = "local"\n'
            f'[orchestrator]\npicks_dir = {json.dumps(str(picks))}\n'
            f'heartbeat_path = {json.dumps(str(blocker_file / ".heartbeat"))}\n'
            '[scheduler]\npick_delivery = "private"\nentry_intent = "research"\n'
        )
        r_hb_canary = runner.invoke(cli, ["deploy-canary", "--config", str(cfg_hb)])
        hb_crash = None
        try:
            with redirect_stderr(io.StringIO()):
                scheduler.run_day("2027-07-15", orchestrator.load_config(cfg_hb), dry_run=True)
        except Exception as exc:
            hb_crash = f"{type(exc).__name__}: {exc}"

        cfg_neg = adv_dir / "neg_poll.toml"
        cfg_neg.write_text(
            f'[[tiers]]\nname = "local"\ntype = "local"\n'
            f'[orchestrator]\npicks_dir = {json.dumps(str(picks))}\n'
            f'heartbeat_path = {json.dumps(str(adv_dir / ".hb_neg"))}\n'
            '[scheduler]\npick_delivery = "private"\nentry_intent = "research"\n'
            'results_poll_interval_min = -5\ndoubleheader_recheck_min = -5\n'
        )
        r_neg_canary = runner.invoke(cli, ["deploy-canary", "--config", str(cfg_neg)])

        cfg_huge = adv_dir / "huge_offset.toml"
        cfg_huge.write_text(
            f'[[tiers]]\nname = "local"\ntype = "local"\n'
            f'[orchestrator]\npicks_dir = {json.dumps(str(picks))}\n'
            f'heartbeat_path = {json.dumps(str(adv_dir / ".hb2"))}\n'
            '[scheduler]\npick_delivery = "private"\nentry_intent = "research"\nlineup_check_offset_min = 1e20\n'
        )
        r_huge_canary = runner.invoke(cli, ["deploy-canary", "--config", str(cfg_huge)])
        huge_crash = None
        orig_fetch2 = scheduler.fetch_schedule
        try:
            scheduler.fetch_schedule = lambda d: [{
                "gamePk": 101, "gameDate": "2027-07-15T23:10:00Z", "gameType": "R",
                "teams": {"away": {"team": {"name": "NYY"}}, "home": {"team": {"name": "BOS"}}},
            }]
            try:
                with redirect_stderr(io.StringIO()):
                    scheduler.run_day("2027-07-15", orchestrator.load_config(cfg_huge), dry_run=True)
            except Exception as exc:
                huge_crash = f"{type(exc).__name__}: {exc}"
        finally:
            scheduler.fetch_schedule = orig_fetch2

        return {
            "valid_configs": valid_outcomes,
            "subprocess_stdout_clean_when_home_unwritable": {
                "rc": sub_stdout_proc.returncode,
                "stdout": sub_stdout_proc.stdout,
            },
            "canary_passes_then_scheduler_crashes": {
                "fallback_deadline_min_lt_morning_default": {
                    "canary_rc": r_fb_canary.exit_code,
                    "canary_out": r_fb_canary.stdout.strip(),
                    "run_day_crash": fb_crash,
                },
                "heartbeat_parent_is_regular_file": {
                    "canary_rc": r_hb_canary.exit_code,
                    "canary_out": r_hb_canary.stdout.strip(),
                    "run_day_crash_before_ready": hb_crash,
                },
                "negative_poll_or_dh_recheck_min": {
                    "canary_rc": r_neg_canary.exit_code,
                    "canary_out": r_neg_canary.stdout.strip(),
                },
                "finite_float_overflowing_timedelta": {
                    "canary_rc": r_huge_canary.exit_code,
                    "canary_out": r_huge_canary.stdout.strip(),
                    "run_day_crash": huge_crash,
                },
            },
        }
    finally:
        cli_mod._now_et = orig_now


def attack_f6_midnight_every_step(work: Path) -> dict:
    pdir = work / "f6_proj"
    (pdir / "data").mkdir(parents=True)
    before_ts = datetime.fromisoformat("2027-07-15T23:59:59.500-04:00").timestamp()
    after_ts = datetime.fromisoformat("2027-07-16T00:00:00.500-04:00").timestamp()

    steps = {}
    for step_name in ("during_activation", "at_third_canary_date_mismatch", "at_third_canary_refused", "during_last_release_write"):
        g = setup_guard(work / f"f6_{step_name}")
        mod = load_guard(g, pdir, f"f6_{step_name}")
        ev = []
        cur = [before_ts]
        mod.time = SimpleNamespace(time=lambda: cur[0], monotonic=time.monotonic, sleep=lambda s: None)
        mod.bts_uid = lambda: 1000
        mod.preflight = lambda uid, target, fetch=True: (OLD_SHA, ["git"])
        mod.window = lambda *a, **kw: None
        mod.verified_stop_dashboard = lambda uid: True
        mod.verified_stop = lambda uid: (ev.append("stop"), True)[1]
        mod.require = lambda argv, step, expected=None, limit=10: expected or b""
        mod.dashboard = lambda uid: None
        mod.page = lambda r, s: (ev.append("page"), 204)[1]
        mod.write_stamp = lambda: True

        activated = [False]

        def act(uid, auth_at, sn=step_name):
            activated[0] = True
            ev.append("activate")
            if sn == "during_activation":
                cur[0] = after_ts

        def can(sn=step_name):
            if activated[0]:
                if sn == "at_third_canary_refused":
                    raise mod.Refused("canary-contest-state")
                if sn in ("during_activation", "at_third_canary_date_mismatch"):
                    return "2027-07-16"
            return "2027-07-15"

        orig_pub = mod.publish_json

        def pub(r, name, val, sn=step_name, op=orig_pub):
            op(r, name, val)
            if sn == "during_last_release_write" and name == "last-release":
                cur[0] = after_ts

        mod.activate_scheduler = act
        mod.canary = can
        mod.publish_json = pub
        with mod.directory() as rfd:
            with redirect_stdout(io.StringIO()):
                rc = mod.owned_deploy(rfd, RUN_ID, TARGET_SHA, "")
            hold = mod.read_hold(rfd)
        steps[step_name] = {
            "rc": rc,
            "hold_state": hold["state"] if hold else None,
            "hold_step": hold["step"] if hold else None,
            "events": ev,
        }
    return steps


def attack_f8_f10_monitor_parity(work: Path) -> dict:
    now = datetime(2027, 7, 15, 18, 0, 0, tzinfo=timezone.utc)
    iso = lambda d: (now + timedelta(seconds=d)).isoformat()

    def load_mon(path: Path):
        mod = ModuleType(f"m_{path.stem}_{id(path)}")
        exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return now.astimezone(tz) if tz else now.replace(tzinfo=None)

        mod.datetime = Clock
        pings = []
        mod.ping = lambda u, s="": pings.append(("GET", u + s))
        if hasattr(mod, "ping_deploy_failure"):
            def pf(u, r):
                pings.append(("POST", u + "/fail", r))
                print("Failure ping received HTTP 2xx: verified.")
            mod.ping_deploy_failure = pf
        mod.read_nrestarts = lambda u, run=None: 2
        return mod, pings

    def run_mon(mod, argv):
        bo, be = io.StringIO(), io.StringIO()
        old = sys.argv
        os.environ.pop("BTS_TEST_DEPLOY_GUARD_DIR", None)
        sys.argv = argv
        try:
            with redirect_stdout(bo), redirect_stderr(be):
                try:
                    mod.main()
                    rc = 0
                except SystemExit as e:
                    rc = int(e.code) if e.code is not None else 0
                except Exception as exc:
                    rc = f"RAISED:{type(exc).__name__}"
        finally:
            sys.argv = old
        return rc, bo.getvalue(), be.getvalue()

    states = [
        ("missing", None),
        ("corrupt", "{bad"),
        ("running_fresh", {"state": "running", "timestamp": iso(-10)}),
        ("running_stale", {"state": "running", "timestamp": iso(-301)}),
        ("waiting_fresh", {"state": "waiting_for_games", "timestamp": iso(-10)}),
        ("waiting_stale", {"state": "waiting_for_games", "timestamp": iso(-601)}),
        ("sleeping_fresh", {"state": "sleeping", "timestamp": iso(-600), "sleeping_until": iso(3600)}),
        ("sleeping_overrun", {"state": "sleeping", "timestamp": iso(-7200), "sleeping_until": iso(-601)}),
        ("idle_fresh", {"state": "idle_end_of_day", "timestamp": iso(-60)}),
        ("idle_stale", {"state": "idle_end_of_day", "timestamp": iso(-5401)}),
        ("stalled", {"state": "stalled", "timestamp": iso(-5), "stage": "s", "stalled_for_s": 950}),
        ("unknown", {"state": "mystery", "timestamp": iso(-5)}),
        ("tz_naive_fresh", {"state": "running", "timestamp": "2027-07-15T17:59:50"}),
    ]

    absent_hold = work / "no_guard_dir" / "hold"
    clear_guard = setup_guard(work / "clear_no_release")
    post_rel_guard = setup_guard(work / "clear_with_old_release")
    (post_rel_guard / "last-release").write_text(json.dumps({
        "released_at": (now - timedelta(days=10)).timestamp(),
        "kind": "deploy",
        "deploy_id": "1790000000-old",
    }))

    mismatches = {"absent_guard": [], "clear_no_release": [], "post_release_10d_ago": []}
    for sname, payload in states:
        hb = work / f"hb_{sname}.json"
        if payload is None:
            hb.unlink(missing_ok=True)
        elif isinstance(payload, str):
            hb.write_text(payload)
        else:
            hb.write_text(json.dumps(payload))

        bm, bp = load_mon(BASE_MONITOR)
        b_res = run_mon(bm, ["check_heartbeat.py", "--heartbeat-path", str(hb), "--ping-url", "https://hc-ping.com/t", "--no-churn"])

        for gk, ghold in [
            ("absent_guard", absent_hold),
            ("clear_no_release", clear_guard / "hold"),
            ("post_release_10d_ago", post_rel_guard / "hold"),
        ]:
            hm, hp = load_mon(HEAD_MONITOR)
            h_res = run_mon(hm, [
                "check_heartbeat.py", "--deploy-hold", str(ghold),
                "--heartbeat-path", str(hb), "--ping-url", "https://hc-ping.com/t", "--no-churn",
            ])
            if (b_res, bp) != (h_res, hp):
                mismatches[gk].append({
                    "state": sname,
                    "base": {"rc": b_res[0], "out": b_res[1].strip(), "pings": bp},
                    "head": {"rc": h_res[0], "out": h_res[1].strip(), "pings": hp},
                })
    return mismatches


def attack_f9_hand_release(work: Path) -> dict:
    now_ts = int(datetime(2027, 7, 15, 18, 0, 0, tzinfo=timezone.utc).timestamp())
    iso = lambda ts: datetime.fromtimestamp(ts, timezone.utc).isoformat()
    cases = {
        "absent_hb": None,
        "stale_running_301s": {"state": "running", "timestamp": iso(now_ts - 301)},
        "fresh_running_300s": {"state": "running", "timestamp": iso(now_ts - 300)},
        "stale_waiting_601s": {"state": "waiting_for_games", "timestamp": iso(now_ts - 601)},
        "fresh_waiting_600s": {"state": "waiting_for_games", "timestamp": iso(now_ts - 600)},
        "stale_sleeping_expired_601s_ago": {"state": "sleeping", "timestamp": iso(now_ts - 3600), "sleeping_until": iso(now_ts - 601)},
        "stopped_sleeping_expired_5s_ago_within_600s_overrun": {"state": "sleeping", "timestamp": iso(now_ts - 3600), "sleeping_until": iso(now_ts - 5)},
        "fresh_sleeping_wake_in_29s": {"state": "sleeping", "timestamp": iso(now_ts - 10), "sleeping_until": iso(now_ts + 29)},
        "fresh_sleeping_wake_in_30s": {"state": "sleeping", "timestamp": iso(now_ts - 10), "sleeping_until": iso(now_ts + 30)},
        "fresh_idle_end_of_day": {"state": "idle_end_of_day", "timestamp": iso(now_ts - 10)},
    }
    results = {}
    for label, hb in cases.items():
        g = setup_guard(work / f"f9_g_{label}")
        p = work / f"f9_p_{label}"
        (p / "data").mkdir(parents=True)
        if hb is not None:
            (p / "data" / ".heartbeat").write_text(json.dumps(hb))
        mod = load_guard(g, p, f"f9_{label}")
        mod.time = SimpleNamespace(time=lambda: now_ts, monotonic=time.monotonic, sleep=lambda s: None)
        mod.bts_uid = lambda: 1000
        mod.preflight = lambda uid, target, fetch=False: (TARGET_SHA, ["git"])
        mod.canary = lambda: "2027-07-15"
        mod.call = lambda argv, *, limit=10: (0, b"")
        with mod.directory() as rfd:
            mod.publish_json(rfd, "hold", {
                "schema": "bts_deploy_hold_v1", "deploy_id": RUN_ID,
                "owner_unit": f"bts-deploy-{RUN_ID}.service", "state": "failed", "step": "sync",
                "target": TARGET_SHA, "previous": OLD_SHA, "armed_at": now_ts - 100, "updated_at": now_ts - 50,
            })
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = mod.release_by_hand(rfd, "verified recovery")
        results[label] = {"rc": rc, "out": buf.getvalue().strip().splitlines()[0]}
    return results


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="r2_adv_v3_", dir=R2_DIR))
    try:
        report = {
            "F1_attacks": attack_f1_crontab(work),
            "F2_attacks": attack_f2_owner_and_signals(work),
            "F3_attacks": attack_f3_units_and_env(work),
            "F5_attacks": attack_f5_configs(work),
            "F6_midnight_steps": attack_f6_midnight_every_step(work),
            "F8_F10_monitor_parity": attack_f8_f10_monitor_parity(work),
            "F9_hand_release_matrix": attack_f9_hand_release(work),
        }
        out_path = R2_DIR / "r2_probe_new_attacks_v3.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
