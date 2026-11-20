#!/usr/bin/env python3
"""Item 3 probe (3b): guard installer, guard-stale, unit ExecStart/ExecCondition, crontab inventory, window, busy writer, competing owner."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
INSTALLER = REPO / "scripts" / "install-deploy-guard-hetzner.sh"
CRON_SCRIPT = REPO / "scripts" / "cron-setup-hetzner.sh"
PYTHON = sys.executable

OLD_SHA = "a" * 40
TARGET_SHA = "b" * 40
SCHED_CMD = "/home/bts/.local/bin/uv run --no-sync bts schedule --config /home/bts/.bts-orchestrator.toml"
DASH_CMD = "/home/bts/.local/bin/uv run --no-sync python3 -m bts.web"
COND_CMD = "/usr/bin/python3 -I /usr/local/bin/bts-deploy-hold check-clear --scheduler"


def extract_guard_source() -> str:
    lines = INSTALLER.read_text().splitlines(keepends=True)
    start, = (i for i, ln in enumerate(lines) if "<<'BTS_DEPLOY_HOLD_PYTHON'" in ln)
    end, = (i for i, ln in enumerate(lines) if ln.rstrip("\n") == "BTS_DEPLOY_HOLD_PYTHON")
    return "".join(lines[start + 1 : end])


def load_guard(guard_root: Path, project_dir: Path) -> ModuleType:
    mod = ModuleType("guard_p3")
    exec(compile(extract_guard_source(), str(INSTALLER), "exec"), mod.__dict__)
    mod.ROOT = guard_root
    mod.UID = os.geteuid()
    mod.PROJECT = str(project_dir)
    mod.time = SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=lambda s: None)
    return mod


def render_production_crontab(work_dir: Path, intent: str = "research") -> str:
    home = work_dir / f"cron_home_{intent}"
    (home / "projects" / "bts").mkdir(parents=True, exist_ok=True)
    (home / "projects" / "bts" / ".env").write_text("")
    (home / ".local" / "bin").mkdir(parents=True, exist_ok=True)
    uv = home / ".local" / "bin" / "uv"
    uv.write_text(f'#!/bin/sh\nshift 2\nexec "{PYTHON}" -c "from bts.cli import cli; cli()" "$@"\n')
    uv.chmod(0o755)
    cfg = home / ".bts-orchestrator.toml"
    if intent == "research":
        cfg.write_text('[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n')
    else:
        cfg.write_text(
            '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'
            '[health_checks]\ncontest_state_expected = true\n'
        )
    shims = work_dir / f"cron_shims_{intent}"
    shims.mkdir(exist_ok=True)
    crontab_shim = shims / "crontab"
    crontab_shim.write_text('#!/bin/sh\necho "no crontab for bts" >&2; exit 1\n')
    crontab_shim.chmod(0o755)
    res = subprocess.run(
        ["/bin/bash", str(CRON_SCRIPT), "show"],
        env={
            "HOME": str(home),
            "PATH": f"{shims}:/usr/bin:/bin",
            "HEALTHCHECKS_PING_URL": "https://hc-ping.com/boxping",
        },
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
    )
    assert res.returncode == 0, res.stderr
    _, proposed = res.stdout.split("Would install:\n")
    return proposed.replace(str(home), "/home/bts")


def make_preflight_harness(work_dir: Path, base_cron: str):
    guard_root = work_dir / "guard"
    project_dir = work_dir / "project"
    (project_dir / "data").mkdir(parents=True, exist_ok=True)
    guard_root.mkdir(parents=True, exist_ok=True)
    os.chmod(guard_root, 0o755)
    for n in ("writers.lock", "transaction.lock", "state.lock"):
        (guard_root / n).touch()
        os.chmod(guard_root / n, 0o644)
    src_bytes = b"#!/bin/bash\n# installer v1\n"
    (guard_root / "guard-install.sha256").write_bytes(hashlib.sha256(src_bytes).hexdigest().encode() + b"\n")

    def run_preflight(
        *,
        sched_exec: bytes = SCHED_CMD.encode() + b"\n",
        dash_exec: bytes = DASH_CMD.encode() + b"\n",
        cond_exec: bytes = COND_CMD.encode() + b"\n",
        load_state: bytes = b"loaded\n",
        need_reload: bytes = b"no\n",
        sys_load_state: bytes = b"not-found\n",
        legacy_exists: bool = False,
        cron_text: str = base_cron,
        target_src: bytes = src_bytes,
    ) -> str | None:
        mod = load_guard(guard_root, project_dir)
        path_ns = SimpleNamespace(**{k: getattr(os.path, k) for k in dir(os.path) if not k.startswith("__")})
        path_ns.lexists = lambda p: legacy_exists
        os_ns = SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if not k.startswith("__")})
        os_ns.path = path_ns
        mod.os = os_ns

        def req(argv, step, expected=None, *, limit=10):
            if argv[-2:] == ["rev-parse", "HEAD"]:
                out = OLD_SHA.encode() + b"\n"
            elif "cat-file" in argv:
                out = b"commit\n"
            elif "show" in argv and f"{TARGET_SHA}:scripts/install-deploy-guard-hetzner.sh" in argv:
                out = target_src
            elif step in ("git-config", "fetch"):
                out = b""
            elif argv[:2] == ["/usr/bin/systemctl", "show"]:
                out = sys_load_state
            elif "--user" in argv and "LoadState" in argv:
                out = load_state
            elif "--user" in argv and "NeedDaemonReload" in argv:
                out = need_reload
            elif "--user" in argv and "ExecStart" in argv:
                out = sched_exec if "bts-scheduler.service" in argv else dash_exec
            elif "--user" in argv and "ExecCondition" in argv:
                out = cond_exec
            elif argv[:3] == ["/usr/bin/crontab", "-u", "bts"]:
                out = cron_text.encode()
            else:
                raise AssertionError(f"Unhandled argv: {argv}")
            if expected is not None and out != expected:
                raise mod.Refused(step)
            return out

        mod.require = req
        try:
            mod.preflight(1000, TARGET_SHA)
            return None
        except mod.Refused as exc:
            return exc.step

    return run_preflight


def probe_installer_script(work_dir: Path) -> dict:
    dest = work_dir / "destdir"
    dest.mkdir()
    shims = work_dir / "inst_shims"
    shims.mkdir()
    (shims / "id").write_text('#!/bin/sh\nprintf "1000\\n"\n')
    (shims / "chown").write_text('#!/bin/sh\nexit 0\n')
    (shims / "runuser").write_text(
        '#!/bin/sh\n'
        'case " $* " in\n'
        '  *" daemon-reload "*) exit 0;;\n'
        f'  *" ExecCondition "*) printf "%s\\n" "{COND_CMD}";;\n'
        f'  *" ExecStart "*) printf "%s\\n" "${{BTS_TEST_EXEC_START:-{SCHED_CMD}}}";;\n'
        '  *" NeedDaemonReload "*) printf "%s\\n" "${BTS_TEST_RELOAD:-no}";;\n'
        '  *) exit 99;;\n'
        'esac\n'
    )
    for p in shims.iterdir():
        p.chmod(0o755)

    env_base = {"DESTDIR": str(dest), "PATH": f"{shims}:/usr/bin:/bin"}
    ok1 = subprocess.run(["/bin/bash", str(INSTALLER)], env=env_base, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)
    ok2 = subprocess.run(["/bin/bash", str(INSTALLER)], env=env_base, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)
    nonroot = subprocess.run(["/bin/bash", str(INSTALLER)], env={"PATH": f"{shims}:/usr/bin:/bin"}, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)
    bad_dest = subprocess.run(["/bin/bash", str(INSTALLER)], env={"DESTDIR": "relative/path", "PATH": f"{shims}:/usr/bin:/bin"}, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10)
    # Note: test ExecStart with ignore_errors=yes in installer vs preflight!
    ign_yes = f"{{ path=/home/bts/.local/bin/uv ; argv[]={SCHED_CMD} ; ignore_errors=yes ; }}"
    inst_ign = subprocess.run(
        ["/bin/bash", str(INSTALLER)],
        env={**env_base, "BTS_TEST_EXEC_START": ign_yes},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
    )
    return {
        "first_install_rc": ok1.returncode,
        "reinstall_idempotent_rc": ok2.returncode,
        "nonroot_without_destdir_rc": nonroot.returncode,
        "relative_destdir_rc": bad_dest.returncode,
        "installer_accepts_execstart_ignore_errors_yes_rc": inst_ign.returncode,
    }


def main() -> int:
    work_dir = Path(tempfile.mkdtemp(prefix="p3b_", dir=Path(__file__).resolve().parent))
    try:
        inst = probe_installer_script(work_dir)
        base_cron_res = render_production_crontab(work_dir, "research")
        base_cron_ent = render_production_crontab(work_dir, "enter")
        pf = make_preflight_harness(work_dir, base_cron_res)

        assert pf(cron_text=base_cron_res) is None
        assert pf(cron_text=base_cron_ent) is None
        structured_sched = f"{{ path=/home/bts/.local/bin/uv ; argv[]={SCHED_CMD} ; ignore_errors=no ; }}\n".encode()
        structured_dash = f"{{ path=/home/bts/.local/bin/uv ; argv[]={DASH_CMD} ; ignore_errors=no ; }}\n".encode()
        structured_cond = f"{{ path=/usr/bin/python3 ; argv[]={COND_CMD} ; ignore_errors=no ; }}\n".encode()
        assert pf(sched_exec=structured_sched, dash_exec=structured_dash, cond_exec=structured_cond) is None

        unit_checks = {
            "guard_stale_source_differs": pf(target_src=b"#!/bin/bash\n# installer v2\n"),
            "user_manager_not_loaded": pf(load_state=b"not-found\n"),
            "need_daemon_reload_yes": pf(need_reload=b"yes\n"),
            "legacy_unit_file_exists": pf(legacy_exists=True),
            "legacy_system_unit_loaded": pf(sys_load_state=b"loaded\n"),
            "sched_missing_no_sync": pf(sched_exec=b"/home/bts/.local/bin/uv run bts schedule --config /home/bts/.bts-orchestrator.toml\n"),
            "dash_missing_no_sync": pf(dash_exec=b"/home/bts/.local/bin/uv run python3 -m bts.web\n"),
            "sched_ignore_errors_yes": pf(sched_exec=f"{{ path=/home/bts/.local/bin/uv ; argv[]={SCHED_CMD} ; ignore_errors=yes ; }}\n".encode()),
            "sched_two_execstart": pf(sched_exec=f"{{ argv[]={SCHED_CMD} ; }} {{ argv[]={SCHED_CMD} ; }}\n".encode()),
            "cond_missing": pf(cond_exec=b"\n"),
            "cond_ignore_errors_yes": pf(cond_exec=f"{{ path=/usr/bin/python3 ; argv[]={COND_CMD} ; ignore_errors=yes ; }}\n".encode()),
            "cond_two_entries": pf(cond_exec=f"{{ argv[]={COND_CMD} ; ignore_errors=no ; }} {{ argv[]=/bin/true ; ignore_errors=no ; }}\n".encode()),
        }

        cron_lines = base_cron_res.splitlines()
        without_one_fetch = "\n".join(ln for ln in cron_lines if "10 1 * * *" not in ln) + "\n"
        extra_gated_reconcile = base_cron_res + cron_lines[2] + "\n"
        ungated_plain = base_cron_res + "0 4 * * * cd /home/bts/projects/bts && /home/bts/.local/bin/uv run bts reconcile\n"
        commented_writer = base_cron_res + "# 0 4 * * * cd /home/bts/projects/bts && /home/bts/.local/bin/uv run bts reconcile\n"

        quoted_subcmd = base_cron_res + "0 4 * * * cd /home/bts/projects/bts && /home/bts/.local/bin/uv run bts 'reconcile' >> /home/bts/logs/cron.log 2>&1\n"
        quoted_bts = base_cron_res + '0 4 * * * cd /home/bts/projects/bts && /home/bts/.local/bin/uv run "bts" check-results --date 2027-07-01\n'
        sh_c_unquoted_end = base_cron_res + "0 4 * * * /bin/sh -c 'cd /home/bts/projects/bts && /home/bts/.local/bin/uv run bts reconcile'\n"
        subshell_end = base_cron_res + "0 4 * * * (cd /home/bts/projects/bts && /home/bts/.local/bin/uv run bts reconcile)\n"
        semicolon_end = base_cron_res + "0 4 * * * cd /home/bts/projects/bts && /home/bts/.local/bin/uv run bts reconcile; true\n"
        indirect_py_mod = base_cron_res + "0 4 * * * cd /home/bts/projects/bts && /home/bts/projects/bts/.venv/bin/python -m bts.cli reconcile\n"
        chained_same_job = "\n".join(
            ("0 2 * * * /usr/local/bin/bts-deploy-hold run reconcile -- /bin/sh -c 'true'; /home/bts/.local/bin/uv run bts reconcile"
             if ln.startswith("0 2 * * * ") else ln)
            for ln in cron_lines
        ) + "\n"
        commented_heartbeat = "\n".join(
            ("# " + ln if "scripts/check_heartbeat.py " in ln else ln)
            for ln in cron_lines
        ) + "\n"

        crontab_inventory = {
            "empty_crontab_refused": pf(cron_text=""),
            "missing_one_fetch_count_1_2_3_1": pf(cron_text=without_one_fetch),
            "extra_gated_reconcile_count_1_3_4_1": pf(cron_text=extra_gated_reconcile),
            "ungated_plain_reconcile": pf(cron_text=ungated_plain),
            "commented_ungated_writer_ignored": pf(cron_text=commented_writer),
            "ADV_quoted_subcmd_bts_singlequote_reconcile": pf(cron_text=quoted_subcmd),
            "ADV_quoted_binary_doublequote_bts": pf(cron_text=quoted_bts),
            "ADV_ungated_bin_sh_c_trailing_quote": pf(cron_text=sh_c_unquoted_end),
            "ADV_ungated_subshell_trailing_paren": pf(cron_text=subshell_end),
            "ADV_ungated_semicolon_terminated": pf(cron_text=semicolon_end),
            "ADV_indirect_python_m_bts_cli": pf(cron_text=indirect_py_mod),
            "ADV_chained_semicolon_after_gate_same_job": pf(cron_text=chained_same_job),
            "ADV_commented_out_heartbeat_monitor": pf(cron_text=commented_heartbeat),
        }

        sh_marker = work_dir / "ungated_ran_while_held"
        tool_path = work_dir / "bts-deploy-hold"
        tool_path.write_text(extract_guard_source())
        tool_path.chmod(0o755)
        (work_dir / "guard" / "hold").write_text("held")
        cron_cmd = f'"{PYTHON}" -I "{tool_path}" run reconcile -- /bin/sh -c \'true\'; : > "{sh_marker}"'
        subprocess.run(
            ["/bin/sh", "-c", cron_cmd],
            env={"BTS_TEST_DEPLOY_GUARD_DIR": str(work_dir / "guard")},
            stdin=subprocess.DEVNULL, capture_output=True, timeout=5,
        )
        crontab_inventory["chained_command_actually_executes_while_held"] = sh_marker.exists()

        w_dir = work_dir / "win_box"
        w_guard = w_dir / "guard"
        w_proj = w_dir / "project"
        (w_proj / "data").mkdir(parents=True)
        w_guard.mkdir(parents=True, mode=0o755)
        for n in ("writers.lock", "transaction.lock", "state.lock"):
            (w_guard / n).touch(mode=0o644)
        mod = load_guard(w_guard, w_proj)
        pinned = 1_800_000_000
        mod.time.time = lambda: pinned

        def test_win(hb_obj, rec_sha):
            hb_file = w_proj / "data" / ".heartbeat"
            if hb_obj is None:
                hb_file.unlink(missing_ok=True)
            else:
                hb_file.write_text(json.dumps(hb_obj))
            try:
                mod.window(rec_sha, TARGET_SHA, pinned + 1200)
                return "allowed"
            except mod.Refused as e:
                return f"refused:{e.step}"

        iso = lambda ts: datetime.fromtimestamp(ts, timezone.utc).isoformat()
        window_results = {
            "fresh_running_no_rec": test_win({"state": "running", "timestamp": iso(pinned - 300)}, ""),
            "fresh_running_with_rec": test_win({"state": "running", "timestamp": iso(pinned - 300)}, TARGET_SHA),
            "stale_running_no_rec": test_win({"state": "running", "timestamp": iso(pinned - 301)}, ""),
            "stale_running_with_rec": test_win({"state": "running", "timestamp": iso(pinned - 301)}, TARGET_SHA),
            "fresh_waiting_with_rec": test_win({"state": "waiting_for_games", "timestamp": iso(pinned - 600)}, TARGET_SHA),
            "sleep_wake_29m59s_with_rec": test_win({"state": "sleeping", "timestamp": iso(pinned - 10), "sleeping_until": iso(pinned + 1799)}, TARGET_SHA),
            "sleep_wake_30m00s_no_rec": test_win({"state": "sleeping", "timestamp": iso(pinned - 10), "sleeping_until": iso(pinned + 1800)}, ""),
            "absent_hb_no_rec": test_win(None, ""),
            "absent_hb_wrong_rec": test_win(None, OLD_SHA),
            "absent_hb_exact_rec": test_win(None, TARGET_SHA),
        }

        report = {
            "installer_checks": inst,
            "unit_and_guard_stale_checks": unit_checks,
            "crontab_inventory": crontab_inventory,
            "window_checks": window_results,
        }
        out_path = Path(__file__).resolve().parent / "probe_3b_guard_preflight.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
