#!/usr/bin/env python3
"""Item 1 fast comprehensive probe (1c, <10s): every phase boundary under fail, TERM, KILL, SSH loss, RuntimeMaxSec,
checking check-clear --scheduler, all 4 cron jobs, reconcile, next-preflight, and result/evidence/page agreement."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "deploy.yml"
INSTALLER = REPO / "scripts" / "install-deploy-guard-hetzner.sh"
MONITOR = REPO / "scripts" / "check_heartbeat.py"
PYTHON = sys.executable

OLD_SHA = "a" * 40
TARGET_SHA = "b" * 40
JOBS = ("check-results", "reconcile", "fetch-contest-streak", "nightly-preview")


def extract_guard_source() -> str:
    lines = INSTALLER.read_text().splitlines(keepends=True)
    start, = (i for i, ln in enumerate(lines) if "<<'BTS_DEPLOY_HOLD_PYTHON'" in ln)
    end, = (i for i, ln in enumerate(lines) if ln.rstrip("\n") == "BTS_DEPLOY_HOLD_PYTHON")
    return "".join(lines[start + 1 : end])


def load_guard_module(guard_root: Path, project_dir: Path, unit_active_fn=lambda: True, *, sleep_fn=lambda s: None, time_fn=time.time) -> ModuleType:
    mod = ModuleType("guard_1c")
    exec(compile(extract_guard_source(), str(INSTALLER), "exec"), mod.__dict__)
    mod.ROOT = guard_root
    mod.UID = os.geteuid()
    mod.PROJECT = str(project_dir)
    mod.time = SimpleNamespace(time=time_fn, monotonic=time.monotonic, sleep=sleep_fn)
    orig_sub_run = subprocess.run

    def shimmed_run(argv, *a, **kw):
        if argv and argv[0] == "/usr/bin/systemctl":
            return SimpleNamespace(returncode=0, stdout=b"active\n" if unit_active_fn() else b"inactive\n")
        return orig_sub_run(argv, *a, **kw)

    mod.subprocess = SimpleNamespace(
        run=shimmed_run, Popen=subprocess.Popen, DEVNULL=subprocess.DEVNULL,
        PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired,
    )
    return mod


def check_gates_inprocess(mod: ModuleType) -> dict:
    """Evaluate exact check-clear --scheduler and writer() admission logic against current guard_root."""
    try:
        with mod.directory() as rfd:
            sched_rc = 0 if mod.clear(rfd) else 1
    except Exception:
        sched_rc = 1

    jobs = {}
    for job in JOBS:
        try:
            with mod.directory() as rfd:
                with mod.lock(rfd, "writers.lock", fcntl.LOCK_SH):
                    admitted = bool(mod.clear(rfd))
                    rc = 0 if admitted else 75
        except Exception:
            admitted = False
            rc = 75
        jobs[job] = {"rc": rc, "admitted": admitted}
    return {"scheduler_clear_rc": sched_rc, "jobs": jobs}


def init_case(base: Path) -> tuple[Path, Path]:
    guard_root = base / "guard"
    project_dir = base / "project"
    (project_dir / "data").mkdir(parents=True, exist_ok=True)
    (project_dir / ".env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.com/testtoken123\n")
    guard_root.mkdir(parents=True, exist_ok=True)
    os.chmod(guard_root, 0o755)
    for name in ("writers.lock", "transaction.lock", "state.lock"):
        p = guard_root / name
        p.touch()
        os.chmod(p, 0o644)
    return guard_root, project_dir


def configure_mocks(mod: ModuleType, sched_file: Path, pages: list, hit_cb=lambda n: None, fail_at: str | None = None):
    mod.bts_uid = lambda: 1000
    (mod.ROOT / "guard-install.sha256").write_bytes(hashlib.sha256(b"pinned").hexdigest().encode() + b"\n")

    def verified_target(target, *, fetch):
        hit_cb("verified_target")
        if fail_at == "guard-stale":
            raise mod.Refused("guard-stale", OLD_SHA)
        return OLD_SHA, ["fake-git"]

    def preflight(uid, target, *, fetch=True):
        prev, git = verified_target(target, fetch=fetch)
        hit_cb("preflight")
        if fail_at == "preflight":
            raise mod.Refused("crontab", prev)
        return prev, git

    def window_fn(recovery, target, deadline, *, minimum=1800):
        stage = "window" if minimum == 1800 else "window-recheck"
        hit_cb(stage)
        if fail_at == stage:
            raise mod.Refused("window")

    def verified_stop(uid):
        hit_cb("stop-old")
        if fail_at == "stop-old":
            return False
        sched_file.write_text("stopped")
        return True

    def require_fn(argv, step, expected=None, *, limit=10):
        hit_cb(step)
        if fail_at == step:
            raise mod.Refused(step)
        return expected or b""

    def canary_fn():
        hit_cb("canary-contest-state")
        if fail_at == "canary-contest-state":
            raise mod.Refused("canary-contest-state")

    def dashboard_fn(uid):
        hit_cb("canary-dashboard")
        if fail_at == "canary-dashboard":
            raise mod.Refused("canary-dashboard")
        hit_cb("canary-http")
        if fail_at == "canary-http":
            raise mod.Refused("canary-http")

    def activate_fn(uid, authorized_at):
        sched_file.write_text("running")
        hit_cb("activating-started")
        hit_cb("activation")
        if fail_at == "activation":
            raise mod.Refused("activation")
        hb = Path(mod.PROJECT) / "data" / ".heartbeat"
        now_dt = datetime.fromtimestamp(mod.time.time() + 1, timezone.utc)
        hb.write_text(json.dumps({
            "state": "sleeping",
            "timestamp": now_dt.isoformat(),
            "sleeping_until": (now_dt + timedelta(hours=2)).isoformat(),
        }))

    def page_fn(record, stopped):
        pages.append({"deploy_id": record["deploy_id"], "step": record["step"], "stopped": stopped})
        hit_cb("page")
        return True

    def stamp_fn():
        hit_cb("stamp")
        if fail_at == "stamp":
            return False
        (Path(mod.PROJECT) / "data" / ".last_deploy_iso").write_text("2027-07-06T12:00:31Z\n")
        return True

    mod.verified_target = verified_target
    mod.preflight = preflight
    mod.window = window_fn
    mod.verified_stop = verified_stop
    mod.require = require_fn
    mod.canary = canary_fn
    mod.dashboard = dashboard_fn
    mod.activate_scheduler = activate_fn
    mod.page = page_fn
    mod.write_stamp = stamp_fn


def kill_child_entry(case_str: str, phase: str, run_id: str, digest: str) -> int:
    case_dir = Path(case_str)
    guard_root = case_dir / "guard"
    project_dir = case_dir / "project"
    sched_file = case_dir / "sched"
    unit_state_file = case_dir / "unit_active"
    unit_state_file.write_text("1")

    mod = load_guard_module(guard_root, project_dir, unit_active_fn=lambda: unit_state_file.read_text() == "1")
    pages = []
    fired = False

    def hit(name: str):
        nonlocal fired
        if name != phase or fired:
            return
        fired = True
        (case_dir / "gate_during.json").write_text(json.dumps(check_gates_inprocess(mod)))
        unit_state_file.write_text("0")
        os.kill(os.getpid(), signal.SIGKILL)

    configure_mocks(mod, sched_file, pages, hit_cb=hit)
    orig_pub = mod.publish_json
    orig_lock = mod.lock

    def pub(root, name, content):
        orig_pub(root, name, content)
        if name == "hold" and content.get("state") == "pending":
            hit("armed")
        elif name == "hold" and content.get("state") == "activating":
            hit("activating-record")
        elif name.endswith(".evidence") and content.get("activation") == "verified":
            hit("activated")
        elif name.endswith(".result") and content.get("outcome") == "committed":
            hit("reported")

    def lk(root, name, op, timeout=0):
        if name == "writers.lock" and timeout == 120:
            hit("drain")
        return orig_lock(root, name, op, timeout)

    mod.publish_json = pub
    mod.lock = lk
    with mod.directory() as rfd:
        return mod.run_owner(rfd, run_id, TARGET_SHA, digest)


def run_all_boundaries(work_dir: Path) -> dict:
    phases = [
        "preflight", "window", "armed", "stop-old", "drain", "reset", "head",
        "sync", "restart", "canary-contest-state", "canary-dashboard", "canary-http",
        "activating-record", "activating-started", "activation", "activated", "stamp", "reported",
    ]
    modes = ["fail", "TERM", "RuntimeMaxSec", "KILL"]
    checked = 0
    anomalies = []

    # Also verify real subprocess CLI parity on check-clear and run once up front!
    sub_base = work_dir / "cli_parity"
    g0, _ = init_case(sub_base)
    tool = sub_base / "bts-deploy-hold"
    tool.write_text(extract_guard_source())
    tool.chmod(0o755)
    env0 = {"BTS_TEST_DEPLOY_GUARD_DIR": str(g0), "PATH": "/usr/bin:/bin"}
    assert subprocess.run([PYTHON, "-I", str(tool), "check-clear", "--scheduler"], env=env0, timeout=5).returncode == 0
    (g0 / "hold").write_text("{}")
    assert subprocess.run([PYTHON, "-I", str(tool), "check-clear", "--scheduler"], env=env0, timeout=5).returncode == 1
    for job in JOBS:
        assert subprocess.run([PYTHON, "-I", str(tool), "run", job, "--", "/bin/true"], env=env0, capture_output=True, timeout=5).returncode == 75

    for phase in phases:
        for mode in modes:
            if mode == "fail" and phase in ("armed", "activating-record", "activating-started", "activated", "reported"):
                continue
            for rec_kind in ("reconcile", "next-preflight"):
                checked += 1
                cdir = work_dir / f"m_{checked}_{phase}_{mode}_{rec_kind}"
                guard_root, project_dir = init_case(cdir)
                sched_file = cdir / "sched"
                sched_file.write_text("running-old")
                run_id = "1795197580-run1"
                phase_bytes = b"#!/bin/bash\nexec true\n"
                digest = hashlib.sha256(phase_bytes).hexdigest()

                unit_alive = [True]
                mod = load_guard_module(guard_root, project_dir, unit_active_fn=lambda: unit_alive[0])
                with mod.directory() as rfd:
                    mod.publish(rfd, f"run-{run_id}.sh", phase_bytes, 0o700)
                    mod.publish_json(rfd, f"run-{run_id}.metadata", {
                        "deploy_id": run_id, "target": TARGET_SHA, "event": "push",
                        "recovery": "", "phase_sha256": digest,
                    })

                pages_owner = []
                if mode == "KILL":
                    proc = subprocess.run(
                        [PYTHON, __file__, "--kill-child", str(cdir), phase, run_id, digest],
                        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
                    )
                    assert proc.returncode == -signal.SIGKILL
                    unit_alive[0] = False
                    gate_during = json.loads((cdir / "gate_during.json").read_text())
                else:
                    gate_holder = []
                    fired = [False]

                    def hit(name: str):
                        if name != phase or fired[0]:
                            return
                        fired[0] = True
                        gate_holder.append(check_gates_inprocess(mod))
                        if mode in ("TERM", "RuntimeMaxSec"):
                            raise InterruptedError()

                    configure_mocks(mod, sched_file, pages_owner, hit_cb=hit, fail_at=phase if mode == "fail" else None)
                    orig_pub = mod.publish_json
                    orig_lock = mod.lock

                    def pub(root, name, content):
                        orig_pub(root, name, content)
                        if name == "hold" and content.get("state") == "pending":
                            hit("armed")
                        elif name == "hold" and content.get("state") == "activating":
                            hit("activating-record")
                        elif name.endswith(".evidence") and content.get("activation") == "verified":
                            hit("activated")
                        elif name.endswith(".result") and content.get("outcome") == "committed":
                            hit("reported")

                    def lk(root, name, op, timeout=0):
                        if name == "writers.lock" and timeout == 120:
                            hit("drain")
                            if mode == "fail" and phase == "drain":
                                raise BlockingIOError("drain timeout")
                        return orig_lock(root, name, op, timeout)

                    mod.publish_json = pub
                    mod.lock = lk
                    with mod.directory() as rfd:
                        mod.owned_deploy(rfd, run_id, TARGET_SHA, "")
                    unit_alive[0] = False
                    gate_during = gate_holder[0]

                gate_after_death = check_gates_inprocess(mod)

                # Verify hold gates at boundary!
                if phase in ("stop-old", "drain", "reset", "head", "sync", "restart", "canary-contest-state", "canary-dashboard", "canary-http"):
                    assert gate_during["scheduler_clear_rc"] == 1, (phase, mode, gate_during)
                    assert all(j["rc"] == 75 and not j["admitted"] for j in gate_during["jobs"].values())
                if phase in ("activating-record", "activating-started", "activation"):
                    assert gate_during["scheduler_clear_rc"] == 0, (phase, mode, gate_during)
                    assert gate_after_death["scheduler_clear_rc"] == 1, (phase, mode, gate_after_death)
                    assert all(j["rc"] == 75 and not j["admitted"] for j in gate_after_death["jobs"].values())

                # Run recovery (reconcile hook OR next deploy's preflight)
                pages_rec = []
                mod_rec = load_guard_module(guard_root, project_dir, unit_active_fn=lambda: False)
                configure_mocks(mod_rec, sched_file, pages_rec)
                with mod_rec.directory() as rfd:
                    if rec_kind == "reconcile":
                        mod_rec.reconcile(rfd, run_id)
                        res_rec = mod_rec.read_json(rfd, f"run-{run_id}.result")
                        ev_rec = mod_rec.read_json(rfd, f"run-{run_id}.evidence")
                        hold_rec = mod_rec.read_hold(rfd)
                    else:
                        mod_rec.owned_deploy(rfd, "1795197581-next", TARGET_SHA, "")
                        res_rec = mod_rec.read_json(rfd, f"run-{run_id}.result")
                        ev_rec = mod_rec.read_json(rfd, f"run-{run_id}.evidence")
                        hold_rec = mod_rec.read_hold(rfd)

                total_pages = len(pages_owner) + len(pages_rec)
                pre_arm = phase in ("preflight", "window")
                post_release = phase in ("stamp", "reported")
                if not pre_arm and not post_release:
                    if sched_file.read_text() != "stopped" or total_pages != 1 or ev_rec.get("stop") != "verified" or ev_rec.get("page") != "attempted":
                        anomalies.append({
                            "phase": phase, "mode": mode, "rec": rec_kind,
                            "sched": sched_file.read_text(), "pages": total_pages, "ev": ev_rec, "res": res_rec, "hold": hold_rec,
                        })
    return {"total_cases": checked, "anomalies": anomalies}


def run_adversarial_probes(work_dir: Path) -> dict:
    from bts.heartbeat import HeartbeatState, write_heartbeat

    # 1. Abandoned activating + reconcile LOCK_NB race + Deploy 2 guard-stale -> scheduler LEFT RUNNING, 0 pages!
    c1 = work_dir / "adv1"
    g1, p1 = init_case(c1)
    sched1 = c1 / "sched"
    sched1.write_text("stopped")
    run_id = "1795197580-killact"
    phase_bytes = b"#!/bin/bash\nexec true\n"
    digest = hashlib.sha256(phase_bytes).hexdigest()
    mod1 = load_guard_module(g1, p1, unit_active_fn=lambda: False)
    with mod1.directory() as rfd:
        mod1.publish(rfd, f"run-{run_id}.sh", phase_bytes, 0o700)
        mod1.publish_json(rfd, f"run-{run_id}.metadata", {
            "deploy_id": run_id, "target": TARGET_SHA, "event": "push", "recovery": "", "phase_sha256": digest,
        })
    proc = subprocess.run(
        [PYTHON, __file__, "--kill-child", str(c1), "activating-started", run_id, digest],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == -signal.SIGKILL and sched1.read_text() == "running"

    with (g1 / "transaction.lock").open("rb") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with mod1.directory() as rfd:
                mod1.reconcile(rfd, run_id)
            reconcile_blocked = False
        except BlockingIOError:
            reconcile_blocked = True

    pages_d2 = []
    configure_mocks(mod1, sched1, pages_d2, fail_at="guard-stale")
    with mod1.directory() as rfd:
        d2_rc = mod1.owned_deploy(rfd, "1795197585-d2", "c" * 40, "")
        hold_after = mod1.read_hold(rfd)

    # 2. Normal passing sleep-window deploy -> last-release released_at (T0+30) > scheduler SLEEPING heartbeat (T0+2.5) -> monitor pages /fail!
    c2 = work_dir / "adv2"
    g2, p2 = init_case(c2)
    hb_path = p2 / "data" / ".heartbeat"
    clock = [1_800_000_000.4]
    mod2 = load_guard_module(g2, p2, sleep_fn=lambda s: clock.__setitem__(0, clock[0] + s), time_fn=lambda: clock[0])
    pages2 = []
    configure_mocks(mod2, c2 / "sched", pages2)
    pre_dt = datetime.fromtimestamp(clock[0] - 60, timezone.utc)
    write_heartbeat(hb_path, state=HeartbeatState.SLEEPING, now_utc=pre_dt, sleeping_until=pre_dt + timedelta(hours=2))

    def real_activate(uid, authorized_at):
        clock[0] += 1.0
        write_heartbeat(hb_path, state=HeartbeatState.RUNNING, now_utc=datetime.fromtimestamp(clock[0], timezone.utc))
        clock[0] += 1.5
        write_heartbeat(hb_path, state=HeartbeatState.SLEEPING,
                        now_utc=datetime.fromtimestamp(clock[0], timezone.utc),
                        sleeping_until=datetime.fromtimestamp(clock[0], timezone.utc) + timedelta(hours=2))
        mod2.time.sleep(30)
        assert mod2.heartbeat()[1] > authorized_at

    mod2.activate_scheduler = real_activate
    with mod2.directory() as rfd:
        deploy_rc = mod2.owned_deploy(rfd, "1795197600-good", TARGET_SHA, "")
        last_rel = mod2.read_json(rfd, "last-release")

    mon = ModuleType("mon_adv2")
    exec(compile(MONITOR.read_text(), str(MONITOR), "exec"), mon.__dict__)
    tick_now = datetime.fromtimestamp(clock[0] + 300, timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return tick_now.astimezone(tz) if tz else tick_now.replace(tzinfo=None)

    mon.datetime = Clock
    mon_pings = []
    mon.ping = lambda url, suffix="": mon_pings.append(("GET", suffix))
    mon.ping_deploy_failure = lambda url, reason: mon_pings.append(("POST", "/fail", reason))
    old_argv, old_env = sys.argv, os.environ.get("BTS_TEST_DEPLOY_GUARD_DIR")
    os.environ["BTS_TEST_DEPLOY_GUARD_DIR"] = str(g2)
    sys.argv = ["check_heartbeat.py", "--heartbeat-path", str(hb_path), "--ping-url", "https://hc-ping.com/testtoken123", "--no-churn"]
    try:
        try:
            mon.main()
            mon_rc = 0
        except SystemExit as e:
            mon_rc = e.code
    finally:
        sys.argv = old_argv
        if old_env is None:
            os.environ.pop("BTS_TEST_DEPLOY_GUARD_DIR", None)
        else:
            os.environ["BTS_TEST_DEPLOY_GUARD_DIR"] = old_env

    return {
        "abandoned_activating_race": {
            "reconcile_lock_nb_blocked": reconcile_blocked,
            "deploy2_guard_stale_rc": d2_rc,
            "hold_state_left": hold_after["state"],
            "scheduler_left_state": sched1.read_text(),
            "pages_sent": len(pages_d2),
        },
        "sleep_window_post_release_false_fail": {
            "deploy_rc": deploy_rc,
            "heartbeat_ts": json.loads(hb_path.read_text())["timestamp"],
            "last_release_iso": datetime.fromtimestamp(last_rel["released_at"], timezone.utc).isoformat(),
            "monitor_rc_at_plus_5min": mon_rc,
            "monitor_pings": mon_pings,
        },
    }


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--kill-child":
        _, _, case_str, phase, run_id, digest = sys.argv
        return kill_child_entry(case_str, phase, run_id, digest)

    work_dir = Path(tempfile.mkdtemp(prefix="p1c_", dir=Path(__file__).resolve().parent))
    try:
        matrix = run_all_boundaries(work_dir)
        adv = run_adversarial_probes(work_dir)
        report = {"matrix": matrix, "adversarial": adv}
        out_path = Path(__file__).resolve().parent / "probe_1c_lifecycle.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
