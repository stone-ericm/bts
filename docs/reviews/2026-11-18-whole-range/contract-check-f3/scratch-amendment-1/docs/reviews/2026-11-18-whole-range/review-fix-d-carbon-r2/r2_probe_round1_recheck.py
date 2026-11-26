#!/usr/bin/env python3
"""Round-2 verification of all Round-1 blocking findings (both reviewers: Codex 1-7, Carbon B1-B4)
plus F8-F15, running identical protocol inputs against 46ce572d and HEAD (0645be4b)."""
from __future__ import annotations

import fcntl
import hashlib
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
from pathlib import Path
from types import ModuleType, SimpleNamespace
from zoneinfo import ZoneInfo

R2_DIR = Path(__file__).resolve().parent
REPO_HEAD = R2_DIR.parents[2]
TREE_R1 = R2_DIR / "run_46ce572d"
TREE_HEAD = R2_DIR / "run_head"
BASE_MONITOR = R2_DIR.parent / "base" / "scripts" / "check_heartbeat.py"

OLD_SHA = "a" * 40
TARGET_SHA = "b" * 40
RUN_ID = "1795197580-r2check"
GUARD_BIN = "/usr/local/bin/bts-deploy-hold"
SCHED_CMD = "/home/bts/.local/bin/uv run --no-sync bts schedule --config /home/bts/.bts-orchestrator.toml"
DASH_CMD = "/home/bts/.local/bin/uv run --no-sync python3 -m bts.web"
COND_CMD = f"/usr/bin/python3 -I {GUARD_BIN} check-clear --scheduler"


def extract_guard_source(installer: Path) -> str:
    lines = installer.read_text().splitlines(keepends=True)
    start, = (i for i, ln in enumerate(lines) if "<<'BTS_DEPLOY_HOLD_PYTHON'" in ln)
    end, = (i for i, ln in enumerate(lines) if ln.rstrip("\n") == "BTS_DEPLOY_HOLD_PYTHON")
    return "".join(lines[start + 1 : end])


def structured_exec(cmd: str, *, path: str | None = None, ignore: str = "no") -> str:
    exe = path or cmd.split()[0]
    return f"{{ path={exe} ; argv[]={cmd} ; ignore_errors={ignore} ; start_time=[n/a] ; }}"


def canonical_crontab() -> list[str]:
    lines = [
        f"0 1 * * * {GUARD_BIN} run {job} -- /bin/sh -c 'uv run bts {sub} --fixture' >> /home/bts/logs/cron.log 2>&1 # BTS-HETZNER"
        for job, sub in [
            ("check-results", "check-results"),
            ("reconcile", "reconcile"),
            ("reconcile", "reconcile"),
            *[("fetch-contest-streak", "fetch-contest-streak")] * 4,
            ("nightly-preview", "preview"),
        ]
    ]
    lines.append(
        "*/5 * * * * cd /home/bts/projects/bts && /usr/bin/python3 scripts/check_heartbeat.py "
        "--heartbeat-path data/.heartbeat --deploy-hold /var/lib/bts-deploy/hold # BTS-HETZNER"
    )
    return lines


def load_guard_mod(tree: Path, guard_root: Path, project_dir: Path, name: str) -> ModuleType:
    installer = tree / "scripts" / "install-deploy-guard-hetzner.sh"
    mod = ModuleType(name)
    exec(compile(extract_guard_source(installer), str(installer), "exec"), mod.__dict__)
    mod.ROOT = guard_root
    mod.UID = os.geteuid()
    mod.PROJECT = str(project_dir)
    mod.time = SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=lambda s: None)
    return mod


def load_monitor_mod(tree: Path, now_utc: datetime, name: str) -> tuple[ModuleType, list]:
    mon_path = tree / "scripts" / "check_heartbeat.py"
    mod = ModuleType(name)
    exec(compile(mon_path.read_text(), str(mon_path), "exec"), mod.__dict__)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now_utc.astimezone(tz) if tz else now_utc.replace(tzinfo=None)

    mod.datetime = Clock
    pings = []
    mod.ping = lambda url, suffix="": pings.append(("GET", url + suffix))
    if hasattr(mod, "ping_deploy_failure"):
        def post_fail(url, reason):
            pings.append(("POST", url + "/fail", reason))
            print("Failure ping received HTTP 2xx: verified.")
        mod.ping_deploy_failure = post_fail
    mod.read_nrestarts = lambda unit, run=None: 2
    return mod, pings


def run_monitor_main(mod: ModuleType, argv: list[str], env_guard_dir: str | None) -> tuple[int, str]:
    buf_out, buf_err = io.StringIO(), io.StringIO()
    old_argv = sys.argv
    old_env = os.environ.get("BTS_TEST_DEPLOY_GUARD_DIR")
    if env_guard_dir is None:
        os.environ.pop("BTS_TEST_DEPLOY_GUARD_DIR", None)
    else:
        os.environ["BTS_TEST_DEPLOY_GUARD_DIR"] = env_guard_dir
    sys.argv = argv
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            try:
                mod.main()
                rc = 0
            except SystemExit as exc:
                rc = int(exc.code) if exc.code is not None else 0
    finally:
        sys.argv = old_argv
        if old_env is None:
            os.environ.pop("BTS_TEST_DEPLOY_GUARD_DIR", None)
        else:
            os.environ["BTS_TEST_DEPLOY_GUARD_DIR"] = old_env
    return rc, buf_out.getvalue().strip()


def setup_guard_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o755)
    for lk in ("writers.lock", "transaction.lock", "state.lock"):
        (path / lk).touch()
        os.chmod(path / lk, 0o644)
    return path


def check_f1_crontab(tree: Path, work: Path, tag: str) -> dict:
    gdir = setup_guard_dir(work / f"f1_g_{tag}")
    pdir = work / f"f1_p_{tag}"
    pdir.mkdir(parents=True)
    installer_bytes = (tree / "scripts" / "install-deploy-guard-hetzner.sh").read_bytes()
    (gdir / "guard-install.sha256").write_text(hashlib.sha256(installer_bytes).hexdigest() + "\n")

    cases = {
        "quoted": canonical_crontab() + ["0 4 * * * uv run 'bts' 'reconcile'"],
        "double_quoted_bts": canonical_crontab() + ['0 4 * * * uv run "bts" check-results --date 2027-07-01'],
        "module": canonical_crontab() + ["0 4 * * * .venv/bin/python -m bts.cli reconcile"],
        "wrapped_sh_c": canonical_crontab() + ["0 4 * * * /bin/sh -c 'uv run bts reconcile'"],
        "subshell": canonical_crontab() + ["0 4 * * * (uv run bts reconcile)"],
        "trailing_after_gate": [
            (ln.replace(" # BTS", "; uv run bts reconcile # BTS") if i == 1 else ln)
            for i, ln in enumerate(canonical_crontab())
        ],
        "commented_monitor": canonical_crontab()[:-1] + ["# " + canonical_crontab()[-1]],
        "wrong_monitor_path": canonical_crontab()[:-1] + [
            canonical_crontab()[-1].replace("/var/lib/bts-deploy/hold", "/tmp/other")
        ],
    }
    out = {}
    for cname, lines in cases.items():
        mod = load_guard_mod(tree, gdir, pdir, f"f1_{tag}_{cname}")
        os_ns = SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if not k.startswith("__")})
        path_ns = SimpleNamespace(**{k: getattr(os.path, k) for k in dir(os.path) if not k.startswith("__")})
        path_ns.lexists = lambda p: False
        os_ns.path = path_ns
        mod.os = os_ns

        props = {
            "LoadState": "loaded",
            "NeedDaemonReload": "no",
            "ExecCondition": structured_exec(COND_CMD),
            "SuccessExitStatus": "143",
            "WorkingDirectory": str(pdir),
            "Environment": "UV_CACHE_DIR=/tmp/uv-cache PATH=/home/bts/.local/bin:/usr/local/bin:/usr/bin:/bin",
            "EnvironmentFiles": f"{pdir}/.env (ignore_errors=no)",
        }

        def call(argv, *, limit=10, cron_lines=lines):
            if "/usr/bin/crontab" in argv:
                return 0, ("\n".join(cron_lines) + "\n").encode()
            if "/usr/bin/systemctl" in argv:
                if "--user" not in argv:
                    return 0, b"not-found\n"
                svc = argv[argv.index("show") + 1]
                start = (f"{mod.UV} run --no-sync bts schedule --config {mod.CONFIG}"
                         if svc == "bts-scheduler.service" else f"{mod.UV} run --no-sync python3 -m bts.web")
                loaded = {**props, "ExecStart": structured_exec(start)}
                names = [v for i, v in enumerate(argv) if i and argv[i - 1] == "-p"]
                res_lines = [loaded[n] if "--value" in argv else f"{n}={loaded[n]}" for n in names]
                return 0, ("\n".join(res_lines) + "\n").encode()
            if "rev-parse" in argv:
                return 0, OLD_SHA.encode() + b"\n"
            if "cat-file" in argv:
                return 0, b"commit\n"
            if "show" in argv:
                return 0, installer_bytes
            return 0, b""

        mod.call = call
        try:
            mod.preflight(1000, TARGET_SHA, fetch=False)
            out[cname] = "ACCEPTED"
        except mod.Refused as e:
            out[cname] = f"REFUSED:{e.step}"
    return out


def check_f2_owner_and_abandoned(tree: Path, work: Path, tag: str) -> dict:
    gdir = setup_guard_dir(work / f"f2_g_{tag}")
    pdir = work / f"f2_p_{tag}"
    (pdir / "data").mkdir(parents=True)
    now_ts = int(time.time())
    rec = {
        "schema": "bts_deploy_hold_v1",
        "deploy_id": RUN_ID,
        "owner_unit": f"bts-deploy-{RUN_ID}.service",
        "state": "activating",
        "step": "activation",
        "target": TARGET_SHA,
        "previous": OLD_SHA,
        "armed_at": now_ts - 20,
        "updated_at": now_ts - 10,
    }
    info = (gdir / "transaction.lock").stat()

    liveness = {}
    for label, state, main_pid, holder_pid in [
        ("active_matching_holder", "active", 456, 456),
        ("active_foreign_holder", "active", 456, 789),
        ("deactivating_same_holder", "deactivating", 456, 456),
        ("active_zero_pid", "active", 0, 456),
    ]:
        mod = load_guard_mod(tree, gdir, pdir, f"f2_live_{tag}_{label}")
        plocks = work / f"plocks_{tag}_{label}"
        plocks.write_text(
            f"1: FLOCK ADVISORY WRITE {holder_pid} "
            f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino} 0 EOF\n"
        )
        if hasattr(mod, "PROC_LOCKS"):
            mod.PROC_LOCKS = plocks

        def fake_run(argv, **kw):
            if "--value" in argv:
                return SimpleNamespace(returncode=0, stdout=(state + "\n").encode())
            return SimpleNamespace(
                returncode=0,
                stdout=f"ActiveState={state}\nMainPID={main_pid}\n".encode(),
            )

        mod.subprocess = SimpleNamespace(
            run=fake_run, Popen=subprocess.Popen, DEVNULL=subprocess.DEVNULL,
            PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired,
        )
        with mod.directory() as rfd, mod.lock(rfd, "transaction.lock", fcntl.LOCK_EX):
            mod.publish_json(rfd, "hold", rec)
            liveness[label] = bool(mod.clear(rfd))

    # Abandoned activating + stale/bad target in next owned_deploy
    gdir2 = setup_guard_dir(work / f"f2_ab_{tag}")
    mod2 = load_guard_mod(tree, gdir2, pdir, f"f2_ab_{tag}")
    events = []
    mod2.bts_uid = lambda: 1000
    mod2.verified_stop = lambda uid: (events.append("stop-old"), True)[1]
    mod2.page = lambda r, s: (events.append("page"), 204)[1]
    mod2.verified_target = lambda *a, **kw: (_ for _ in ()).throw(mod2.Refused("guard-stale"))
    with mod2.directory() as rfd:
        mod2.publish_json(rfd, "hold", rec)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = mod2.owned_deploy(rfd, "1795197599-next", "c" * 40, "")
        after_hold = mod2.read_hold(rfd)

    # Hook reconcile under contended transaction.lock / uid failure
    gdir3 = setup_guard_dir(work / f"f2_hk_{tag}")
    mod3 = load_guard_mod(tree, gdir3, pdir, f"f2_hk_{tag}")
    hk_events = []
    orig_lock = mod3.lock
    mod3.lock = lambda r, n, op, timeout=0: (
        (_ for _ in ()).throw(BlockingIOError("busy"))
        if n == "transaction.lock" else orig_lock(r, n, op, timeout)
    )
    mod3.bts_uid = lambda: 1000
    mod3.verified_stop = lambda uid: (hk_events.append("stop-old"), True)[1]
    mod3.page = lambda r, s: (hk_events.append("page"), 204)[1]
    with mod3.directory() as rfd:
        mod3.publish_json(rfd, "hold", rec)
        try:
            with redirect_stdout(io.StringIO()):
                hk_rc = mod3.reconcile(rfd, RUN_ID)
        except BlockingIOError:
            hk_rc = "RAISED_BLOCKING_IO"
        hk_res = mod3.read_json(rfd, f"run-{RUN_ID}.result")

    return {
        "clear_under_lock": liveness,
        "abandoned_before_target_check": {
            "rc": rc,
            "hold_state_after": after_hold["state"],
            "hold_step_after": after_hold["step"],
            "events": events,
        },
        "hook_under_contended_lock": {
            "rc": hk_rc,
            "events": hk_events,
            "result_written": hk_res is not None,
        },
    }


def check_f3_units(tree: Path, work: Path, tag: str) -> dict:
    gdir = setup_guard_dir(work / f"f3_g_{tag}")
    pdir = work / f"f3_p_{tag}"
    pdir.mkdir(parents=True)
    installer_bytes = (tree / "scripts" / "install-deploy-guard-hetzner.sh").read_bytes()
    (gdir / "guard-install.sha256").write_text(hashlib.sha256(installer_bytes).hexdigest() + "\n")

    drift_cases = {
        "cond_spoof_path_true": ("bts-scheduler.service", {"ExecCondition": structured_exec(COND_CMD, path="/usr/bin/true")}),
        "start_spoof_path_true": ("bts-scheduler.service", {"ExecStart": structured_exec(SCHED_CMD, path="/usr/bin/true")}),
        "success_exit_1": ("bts-scheduler.service", {"SuccessExitStatus": "143 1"}),
        "success_exit_143_ok": ("bts-scheduler.service", {"SuccessExitStatus": "143"}),
        "workdir_drift": ("bts-scheduler.service", {"WorkingDirectory": "/srv/other"}),
        "env_path_drift": ("bts-scheduler.service", {"Environment": "UV_CACHE_DIR=/tmp/uv-cache PATH=/tmp/evil:/usr/bin"}),
        "envfile_ignore_yes": ("bts-scheduler.service", {"EnvironmentFiles": f"{pdir}/.env (ignore_errors=yes)"}),
        "dashboard_workdir_drift": ("bts-dashboard.service", {"WorkingDirectory": "/srv/other"}),
    }
    results = {}
    for name, (target_svc, overrides) in drift_cases.items():
        mod = load_guard_mod(tree, gdir, pdir, f"f3_{tag}_{name}")
        os_ns = SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if not k.startswith("__")})
        path_ns = SimpleNamespace(**{k: getattr(os.path, k) for k in dir(os.path) if not k.startswith("__")})
        path_ns.lexists = lambda p: False
        os_ns.path = path_ns
        mod.os = os_ns

        base_props = {
            "LoadState": "loaded",
            "NeedDaemonReload": "no",
            "ExecCondition": structured_exec(COND_CMD),
            "SuccessExitStatus": "143",
            "WorkingDirectory": str(pdir),
            "Environment": "UV_CACHE_DIR=/tmp/uv-cache PATH=/home/bts/.local/bin:/usr/local/bin:/usr/bin:/bin",
            "EnvironmentFiles": f"{pdir}/.env (ignore_errors=no)",
        }

        def call(argv, *, limit=10, t_svc=target_svc, ov=overrides):
            if "/usr/bin/crontab" in argv:
                return 0, ("\n".join(canonical_crontab()) + "\n").encode()
            if "/usr/bin/systemctl" in argv:
                if "--user" not in argv:
                    return 0, b"not-found\n"
                svc = argv[argv.index("show") + 1]
                start = SCHED_CMD if svc == "bts-scheduler.service" else DASH_CMD
                loaded = {**base_props, "ExecStart": structured_exec(start), **(ov if svc == t_svc else {})}
                names = [v for i, v in enumerate(argv) if i and argv[i - 1] == "-p"]
                lines = [loaded[n] if "--value" in argv else f"{n}={loaded[n]}" for n in names]
                return 0, ("\n".join(lines) + "\n").encode()
            if "rev-parse" in argv:
                return 0, OLD_SHA.encode() + b"\n"
            if "cat-file" in argv:
                return 0, b"commit\n"
            if "show" in argv:
                return 0, installer_bytes
            return 0, b""

        mod.call = call
        try:
            mod.preflight(1000, TARGET_SHA, fetch=False)
            results[name] = "ACCEPTED"
        except mod.Refused as e:
            results[name] = f"REFUSED:{e.step}"
    return results


def check_f4_f6_f8_f10(tree: Path, work: Path, tag: str) -> dict:
    # F4 + F6 + F8 in owned_deploy
    gdir = setup_guard_dir(work / f"f468_g_{tag}")
    pdir = work / f"f468_p_{tag}"
    (pdir / "data").mkdir(parents=True)

    def make_configured_owner(suffix: str):
        mod = load_guard_mod(tree, gdir, pdir, f"own_{tag}_{suffix}")
        events = []
        mod.bts_uid = lambda: 1000
        mod.preflight = lambda uid, target, fetch=True: (OLD_SHA, ["git"])
        mod.window = lambda *a, **kw: None
        if hasattr(mod, "verified_stop_dashboard"):
            def stop_dash(uid):
                events.append("stop-dashboard")
                return True
            mod.verified_stop_dashboard = stop_dash
        mod.verified_stop = lambda uid: (events.append("stop-old"), True)[1]
        mod.require = lambda argv, step, expected=None, limit=10: (events.append(step), expected or b"")[1]
        mod.dashboard = lambda uid: events.append("canary-dashboard")
        mod.page = lambda r, s: (events.append("page"), 204)[1]
        mod.write_stamp = lambda: True
        return mod, events

    # 1. F4 + F8 normal pass: check dashboard stop before reset, and released_at vs first heartbeat
    mod_ok, ev_ok = make_configured_owner("ok")
    t_auth = 1_800_000_000
    t_hb = 1_800_000_005
    t_rel = 1_800_000_030
    cur_t = [t_auth]
    mod_ok.time = SimpleNamespace(time=lambda: cur_t[0], monotonic=time.monotonic, sleep=lambda s: None)
    mod_ok.canary = lambda: "2027-07-15"

    def act_ok(uid, auth_at):
        ev_ok.append("activation")
        cur_t[0] = t_rel

    mod_ok.activate_scheduler = act_ok
    with mod_ok.directory() as rfd:
        with redirect_stdout(io.StringIO()):
            rc_ok = mod_ok.owned_deploy(rfd, RUN_ID, TARGET_SHA, "")
        last_rel = mod_ok.read_json(rfd, "last-release")

    # Monitor check right after deploy with scheduler's first heartbeat at t_hb (T+5 < T+30!)
    hb_path = work / f"hb_post_deploy_{tag}.json"
    hb_path.write_text(json.dumps({
        "state": "sleeping",
        "timestamp": datetime.fromtimestamp(t_hb, timezone.utc).isoformat(),
        "sleeping_until": datetime.fromtimestamp(t_rel + 3600, timezone.utc).isoformat(),
    }))
    mon_mod, mon_pings = load_monitor_mod(tree, datetime.fromtimestamp(t_rel + 5, timezone.utc), f"mon_f8_{tag}")
    mon_rc, mon_out = run_monitor_main(
        mon_mod,
        ["check_heartbeat.py", "--heartbeat-path", str(hb_path), "--ping-url", "https://hc-ping.com/t", "--no-churn"],
        str(gdir),
    )

    # 2. F6 midnight rollover during activation (certified on 2027-07-15, post-activation is 2027-07-16)
    gdir_f6 = setup_guard_dir(work / f"f6_g_{tag}")
    mod_f6 = load_guard_mod(tree, gdir_f6, pdir, f"own_f6_{tag}")
    ev_f6 = []
    mod_f6.bts_uid = lambda: 1000
    mod_f6.preflight = lambda uid, target, fetch=True: (OLD_SHA, ["git"])
    mod_f6.window = lambda *a, **kw: None
    if hasattr(mod_f6, "verified_stop_dashboard"):
        mod_f6.verified_stop_dashboard = lambda uid: True
    mod_f6.verified_stop = lambda uid: (ev_f6.append("stop"), True)[1]
    mod_f6.require = lambda argv, step, expected=None, limit=10: expected or b""
    mod_f6.dashboard = lambda uid: None
    mod_f6.page = lambda r, s: (ev_f6.append("page"), 204)[1]
    mod_f6.write_stamp = lambda: True
    activated_flag = [False]

    def f6_canary():
        if activated_flag[0]:
            raise mod_f6.Refused("canary-contest-state")
        return "2027-07-15"

    mod_f6.canary = f6_canary
    mod_f6.activate_scheduler = lambda uid, at: activated_flag.__setitem__(0, True)
    with mod_f6.directory() as rfd:
        with redirect_stdout(io.StringIO()):
            rc_f6 = mod_f6.owned_deploy(rfd, RUN_ID, TARGET_SHA, "")
        hold_f6 = mod_f6.read_hold(rfd)

    # 3. F10 monitor without guard directory vs damaged guard directory
    absent_guard = work / f"absent_guard_{tag}" / "hold"
    mon_abs, pings_abs = load_monitor_mod(tree, datetime.fromtimestamp(t_rel + 5, timezone.utc), f"mon_abs_{tag}")
    rc_abs, out_abs = run_monitor_main(
        mon_abs,
        ["check_heartbeat.py", "--deploy-hold", str(absent_guard), "--heartbeat-path", str(hb_path),
         "--ping-url", "https://hc-ping.com/t", "--no-churn"],
        None,
    )
    damaged_dir = work / f"damaged_guard_{tag}"
    damaged_dir.mkdir(mode=0o755)
    mon_dmg, pings_dmg = load_monitor_mod(tree, datetime.fromtimestamp(t_rel + 5, timezone.utc), f"mon_dmg_{tag}")
    rc_dmg, out_dmg = run_monitor_main(
        mon_dmg,
        ["check_heartbeat.py", "--deploy-hold", str(damaged_dir / "hold"), "--heartbeat-path", str(hb_path),
         "--ping-url", "https://hc-ping.com/t", "--no-churn"],
        None,
    )

    return {
        "f4_stop_dashboard_before_reset": ("stop-dashboard" in ev_ok and ev_ok.index("stop-dashboard") < ev_ok.index("reset")),
        "f8_post_deploy_first_heartbeat": {
            "deploy_rc": rc_ok,
            "released_at": last_rel["released_at"],
            "monitor_rc": mon_rc,
            "monitor_pings": mon_pings,
            "monitor_out": mon_out,
        },
        "f6_midnight_rollover_after_activation": {
            "deploy_rc": rc_f6,
            "hold_kept": hold_f6 is not None,
            "hold_state": hold_f6["state"] if hold_f6 else None,
            "events": ev_f6,
        },
        "f10_absent_vs_damaged_guard": {
            "absent_rc": rc_abs,
            "absent_pings": pings_abs,
            "damaged_rc": rc_dmg,
            "damaged_out": out_dmg,
        },
    }


def check_f5_subprocess(tree: Path, work: Path, tag: str) -> dict:
    script = (
        "import json, sys\n"
        "from datetime import datetime\n"
        "from importlib import import_module\n"
        "from pathlib import Path\n"
        "from zoneinfo import ZoneInfo\n"
        "from click.testing import CliRunner\n"
        "from bts.cli import cli\n"
        "from tests.test_contest_dated_cs4a import observation, write\n"
        "w = Path(sys.argv[1])\n"
        "picks = w / 'picks'\n"
        "write(picks, observation('2027-07-15', recorded_at='2027-07-15T11:00:00-04:00'))\n"
        "import_module('bts.cli')._now_et = lambda: datetime.fromisoformat('2027-07-15T15:00:00-04:00')\n"
        "out = {}\n"
        "for name, body in [\n"
        "  ('bad_cascade', f'[[tiers]]\\nname=\"t\"\\ntype=\"local\"\\n[orchestrator]\\npicks_dir={json.dumps(str(picks))}\\n[scheduler]\\nentry_intent=\"enter\"\\npick_delivery=\"dm\"\\ncontest_state_required=true\\ncascade_budget_min=\"forty\"\\n[bluesky]\\ndm_recipient=\"x\"\\n'),\n"
        "  ('missing_tiers', f'[orchestrator]\\npicks_dir={json.dumps(str(picks))}\\n[scheduler]\\nentry_intent=\"research\"\\npick_delivery=\"private\"\\n'),\n"
        "  ('missing_dm_recipient', f'[[tiers]]\\nname=\"t\"\\ntype=\"local\"\\n[orchestrator]\\npicks_dir={json.dumps(str(picks))}\\n[scheduler]\\nentry_intent=\"enter\"\\npick_delivery=\"dm\"\\ncontest_state_required=true\\n'),\n"
        "]:\n"
        "  cfg = w / f'{name}.toml'\n"
        "  cfg.write_text(body)\n"
        "  r = CliRunner().invoke(cli, ['deploy-canary', '--config', str(cfg)])\n"
        "  out[name] = {'rc': r.exit_code, 'out': r.output.strip()}\n"
        "print(json.dumps(out))\n"
    )
    sub = work / f"f5_{tag}"
    sub.mkdir(parents=True)
    res = subprocess.run(
        [sys.executable, "-c", script, str(sub)],
        env={**os.environ, "PYTHONPATH": f"{tree / 'src'}:{REPO_HEAD}"},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def check_f7_docs(tree: Path) -> dict:
    inc = (tree / "INCIDENT.md").read_text()
    claude = (tree / "CLAUDE.md").read_text()
    season = (tree / "docs" / "ops" / "2027-season-start.md").read_text()
    return {
        "incident_uv_no_sync_picks_dir": "uv run --no-sync bts season-contract status --picks-dir" in inc,
        "incident_revert_guard_protocol": "same guard protocol" in inc,
        "incident_backfill_outside_windows": "outside the grader and reconcile windows" in inc,
        "incident_dashboard_gated_by_s5": "paths in §5 restart `bts-dashboard`" in inc,
        "claude_git_show_staging": "git show <target>:scripts/install-systemd-hetzner.sh" in claude,
        "claude_stop_dashboard_first": "systemctl --user stop bts-dashboard" in claude,
        "a7_canary_on_box_config": "uv run --no-sync bts deploy-canary --config ~/.bts-orchestrator.toml" in season,
    }


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="r2_r1check_", dir=R2_DIR))
    try:
        summary = {}
        for label, tree in [("46ce572d", TREE_R1), ("HEAD_0645be4b", TREE_HEAD)]:
            summary[label] = {
                "F1_crontab": check_f1_crontab(tree, work, label),
                "F2_owner_and_abandoned": check_f2_owner_and_abandoned(tree, work, label),
                "F3_units": check_f3_units(tree, work, label),
                "F4_F6_F8_F10": check_f4_f6_f8_f10(tree, work, label),
                "F5_canary_configs": check_f5_subprocess(tree, work, label),
                "F7_F16_docs": check_f7_docs(tree),
            }
        out_file = R2_DIR / "r2_probe_round1_recheck.json"
        out_file.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
