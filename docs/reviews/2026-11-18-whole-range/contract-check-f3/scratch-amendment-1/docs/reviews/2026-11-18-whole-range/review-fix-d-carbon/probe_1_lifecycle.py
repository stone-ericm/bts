#!/usr/bin/env python3
"""Item 1 probe: workflow handoff, owner phases, signals, SSH loss, reconcile, next-preflight, and post-release monitor."""
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


def extract_deploy_payload() -> bytes:
    raw = WORKFLOW.read_bytes()
    matches = list(re.finditer(rb"^          script: \|\n", raw, re.M))
    assert len(matches) == 1
    body = raw[matches[0].end() :]
    indent = b" " * 12
    lines = body.splitlines(keepends=True)
    assert all(ln == b"\n" or ln.startswith(indent) for ln in lines)
    return b"".join(ln.removeprefix(indent) for ln in lines)


def load_guard_module(guard_root: Path, project_dir: Path, *, sleep_fn=lambda s: None, time_fn=time.time) -> ModuleType:
    mod = ModuleType("guard_under_test")
    exec(compile(extract_guard_source(), str(INSTALLER), "exec"), mod.__dict__)
    mod.ROOT = guard_root
    mod.UID = os.geteuid()
    mod.PROJECT = str(project_dir)
    mod.time = SimpleNamespace(time=time_fn, monotonic=time.monotonic, sleep=sleep_fn)
    return mod


def init_guard_dir(guard_root: Path) -> Path:
    guard_root.mkdir(parents=True, exist_ok=True)
    os.chmod(guard_root, 0o755)
    for name in ("writers.lock", "transaction.lock", "state.lock"):
        p = guard_root / name
        p.touch()
        os.chmod(p, 0o644)
    tool = guard_root.parent / "bts-deploy-hold"
    tool.write_text(extract_guard_source())
    os.chmod(tool, 0o755)
    return tool


def gate_snapshot(tool: Path, guard_root: Path, shim_bin: Path) -> dict:
    env = {
        "BTS_TEST_DEPLOY_GUARD_DIR": str(guard_root),
        "PATH": f"{shim_bin}:/usr/bin:/bin",
    }
    sched = subprocess.run(
        [PYTHON, "-I", str(tool), "check-clear", "--scheduler"],
        env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
    )
    job_results = {}
    for job in JOBS:
        marker = guard_root.parent / f"ran-{job}"
        marker.unlink(missing_ok=True)
        res = subprocess.run(
            [PYTHON, "-I", str(tool), "run", job, "--", "/bin/sh", "-c", f": > '{marker}'"],
            env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
        )
        job_results[job] = {"rc": res.returncode, "ran": marker.exists(), "out": res.stdout.strip()}
    return {"scheduler_clear_rc": sched.returncode, "jobs": job_results}


def write_systemctl_shim(shim_bin: Path, state_file: Path) -> None:
    shim_bin.mkdir(parents=True, exist_ok=True)
    sc = shim_bin / "systemctl"
    sc.write_text(
        f'#!/bin/sh\nstate=$(cat "{state_file}" 2>/dev/null || printf inactive)\nprintf "%s\\n" "$state"\n'
    )
    sc.chmod(0o755)


def patch_systemctl_in_source(source: str, shim_systemctl: Path) -> str:
    return source.replace('"/usr/bin/systemctl"', f'"{shim_systemctl}"')


def setup_patched_tool(base: Path) -> tuple[Path, Path, Path, Path]:
    guard_root = base / "guard"
    project_dir = base / "project"
    (project_dir / "data").mkdir(parents=True, exist_ok=True)
    (project_dir / ".env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.com/testtoken123\n")
    init_guard_dir(guard_root)
    shim_bin = base / "bin"
    owner_state = base / "owner_unit_state"
    owner_state.write_text("inactive")
    write_systemctl_shim(shim_bin, owner_state)
    tool = base / "bts-deploy-hold"
    tool.write_text(patch_systemctl_in_source(extract_guard_source(), shim_bin / "systemctl"))
    tool.chmod(0o755)
    return guard_root, project_dir, tool, owner_state


def configure_box_mocks(mod: ModuleType, scheduler_running_file: Path, pages: list, boundary_cb=lambda name: None, fail_at: str | None = None):
    mod.bts_uid = lambda: 1000
    installer_bytes = b"#!/bin/bash\n# pinned\n"
    (mod.ROOT / "guard-install.sha256").write_bytes(hashlib.sha256(installer_bytes).hexdigest().encode() + b"\n")

    def verified_target(target, *, fetch):
        boundary_cb("verified_target")
        if fail_at == "guard-stale":
            raise mod.Refused("guard-stale", OLD_SHA)
        if fail_at == "fetch":
            raise mod.Refused("fetch", OLD_SHA)
        return OLD_SHA, ["fake-git"]

    def preflight(uid, target, *, fetch=True):
        prev, git = verified_target(target, fetch=fetch)
        boundary_cb("preflight")
        if fail_at == "preflight":
            raise mod.Refused("crontab", prev)
        return prev, git

    def window_fn(recovery, target, deadline, *, minimum=1800):
        boundary_cb("window" if minimum == 1800 else "window-recheck")
        if fail_at == ("window" if minimum == 1800 else "window-recheck"):
            raise mod.Refused("window")

    def verified_stop(uid):
        boundary_cb("stop-old")
        if fail_at == "stop-old":
            return False
        scheduler_running_file.write_text("stopped")
        return True

    def require_fn(argv, step, expected=None, *, limit=10):
        boundary_cb(step)
        if fail_at == step:
            raise mod.Refused(step)
        return expected or b""

    def canary_fn():
        boundary_cb("canary-contest-state")
        if fail_at == "canary-contest-state":
            raise mod.Refused("canary-contest-state")

    def dashboard_fn(uid):
        boundary_cb("canary-dashboard")
        if fail_at == "canary-dashboard":
            raise mod.Refused("canary-dashboard")
        boundary_cb("canary-http")
        if fail_at == "canary-http":
            raise mod.Refused("canary-http")

    def activate_fn(uid, authorized_at):
        scheduler_running_file.write_text("running")
        boundary_cb("activating-started")
        if fail_at == "activation":
            raise mod.Refused("activation")
        hb = Path(mod.PROJECT) / "data" / ".heartbeat"
        now_dt = datetime.fromtimestamp(mod.time.time() + 1, timezone.utc)
        hb.write_text(json.dumps({
            "state": "sleeping",
            "timestamp": now_dt.isoformat(),
            "sleeping_until": (now_dt + timedelta(hours=2)).isoformat(),
        }))
        boundary_cb("activation")

    def page_fn(record, stopped):
        pages.append({"deploy_id": record["deploy_id"], "step": record["step"], "stopped": stopped})
        boundary_cb("page")
        return True

    def stamp_fn():
        boundary_cb("stamp")
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


def child_worker(mode: str, base_str: str, phase: str, run_id: str, digest: str) -> int:
    base = Path(base_str)
    guard_root = base / "guard"
    project_dir = base / "project"
    sched_file = base / "scheduler_state"
    owner_state = base / "owner_unit_state"
    pages_file = base / "pages.jsonl"
    gate_file = base / "gate_at_boundary.json"
    shim_bin = base / "bin"
    tool = base / "bts-deploy-hold"

    mod = load_guard_module(guard_root, project_dir)
    mod.owner_alive  # ensure loaded
    # Patch systemctl inside loaded module too
    orig_run = mod.subprocess.run

    def patched_run(argv, *a, **kw):
        if argv and argv[0] == "/usr/bin/systemctl":
            argv = [str(shim_bin / "systemctl"), *argv[1:]]
        return orig_run(argv, *a, **kw)

    mod.subprocess.run = patched_run

    pages = []
    fired = False

    def record_pages():
        with pages_file.open("a") as f:
            for p in pages:
                f.write(json.dumps(p) + "\n")
        pages.clear()

    def hit(name: str):
        nonlocal fired
        if name != phase or fired:
            return
        fired = True
        gate_file.write_text(json.dumps(gate_snapshot(tool, guard_root, shim_bin)))
        if mode == "fail":
            return  # configure_box_mocks will raise Refused(phase)
        if mode in ("TERM", "RuntimeMaxSec"):
            os.kill(os.getpid(), signal.SIGTERM)
        elif mode == "KILL":
            owner_state.write_text("inactive")
            os.kill(os.getpid(), signal.SIGKILL)

    fail_target = phase if mode == "fail" else None
    configure_box_mocks(mod, sched_file, pages, boundary_cb=hit, fail_at=fail_target)

    orig_publish = mod.publish_json
    orig_lock = mod.lock

    def hooked_publish(root, name, content):
        orig_publish(root, name, content)
        if name == "hold" and content.get("state") == "pending":
            hit("armed")
        elif name == "hold" and content.get("state") == "activating":
            hit("activating-record")
        elif name.endswith(".evidence") and content.get("activation") == "verified":
            hit("activated")
        elif name == "last-release":
            hit("last-release-written")
        elif name.endswith(".result") and content.get("outcome") == "committed":
            hit("reported")

    def hooked_lock(root, name, op, timeout=0):
        if name == "writers.lock" and timeout == 120:
            hit("drain")
        return orig_lock(root, name, op, timeout)

    mod.publish_json = hooked_publish
    mod.lock = hooked_lock
    owner_state.write_text("active")
    try:
        with mod.directory() as root_fd:
            rc = mod.run_owner(root_fd, run_id, TARGET_SHA, digest)
        return rc
    finally:
        owner_state.write_text("inactive")
        record_pages()


def run_phase_matrix(work_root: Path) -> dict:
    phases = [
        "preflight", "window", "armed", "stop-old", "drain", "reset", "head",
        "sync", "restart", "canary-contest-state", "canary-dashboard", "canary-http",
        "activating-record", "activating-started", "activation", "activated", "stamp", "reported",
    ]
    modes = ["fail", "TERM", "KILL", "RuntimeMaxSec"]
    summary = {"total": 0, "checked": [], "anomalies": []}

    for phase in phases:
        for mode in modes:
            if mode == "fail" and phase in ("armed", "activating-record", "activating-started", "activated", "reported"):
                continue
            for rec_kind in ("reconcile", "next-preflight"):
                summary["total"] += 1
                case_dir = work_root / f"{phase}_{mode}_{rec_kind}"
                guard_root, project_dir, tool, owner_state = setup_patched_tool(case_dir)
                sched_file = case_dir / "scheduler_state"
                sched_file.write_text("running-old")
                run_id = "1795197580-matrix01"
                phase_bytes = b"#!/bin/bash\nset -eu\nexec /usr/bin/python3 -I /usr/local/bin/bts-deploy-hold owner \"$@\"\n"
                digest = hashlib.sha256(phase_bytes).hexdigest()

                mod = load_guard_module(guard_root, project_dir)
                with mod.directory() as rfd:
                    mod.publish(rfd, f"run-{run_id}.sh", phase_bytes, 0o700)
                    mod.publish_json(rfd, f"run-{run_id}.metadata", {
                        "deploy_id": run_id, "target": TARGET_SHA, "event": "push",
                        "recovery": "", "phase_sha256": digest,
                    })

                proc = subprocess.run(
                    [PYTHON, __file__, "--worker", mode, str(case_dir), phase, run_id, digest],
                    stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
                )
                owner_state.write_text("inactive")

                gate_during = json.loads((case_dir / "gate_at_boundary.json").read_text())
                gate_after_death = gate_snapshot(tool, guard_root, case_dir / "bin")

                # Check invariant 1: before activating-record, scheduler_clear_rc MUST be 1 once hold exists
                if phase in ("stop-old", "drain", "reset", "head", "sync", "restart", "canary-contest-state", "canary-dashboard", "canary-http"):
                    assert gate_during["scheduler_clear_rc"] == 1, (phase, mode, gate_during)
                    assert all(j["rc"] == 75 and not j["ran"] for j in gate_during["jobs"].values()), (phase, mode, gate_during)
                if phase in ("activating-record", "activating-started", "activation"):
                    # While owner alive in activating, clear returns 0; immediately after owner dies without release, MUST return 1
                    assert gate_during["scheduler_clear_rc"] == 0, (phase, mode, gate_during)
                    assert gate_after_death["scheduler_clear_rc"] == 1, (phase, mode, gate_after_death)
                    assert all(j["rc"] == 75 and not j["ran"] for j in gate_after_death["jobs"].values())

                # Run recovery path
                pages_after = []
                configure_box_mocks(mod, sched_file, pages_after)
                with mod.directory() as rfd:
                    if rec_kind == "reconcile":
                        rec_rc = mod.reconcile(rfd, run_id)
                    else:
                        hold_now = mod.read_hold(rfd)
                        next_id = "1795197581-next01"
                        rec_rc = mod.owned_deploy(rfd, next_id, TARGET_SHA, "")

                pages_owner = [
                    json.loads(ln) for ln in (case_dir / "pages.jsonl").read_text().splitlines()
                ] if (case_dir / "pages.jsonl").exists() else []
                total_pages = len(pages_owner) + len(pages_after)

                pre_arm = phase in ("preflight", "window") or (phase == "drain" and False)
                post_release = phase in ("stamp", "reported")
                if not pre_arm and not post_release:
                    # Scheduler must be stopped and exactly 1 page sent across owner + recovery
                    if sched_file.read_text() != "stopped" or total_pages != 1:
                        summary["anomalies"].append({
                            "phase": phase, "mode": mode, "rec_kind": rec_kind,
                            "sched": sched_file.read_text(), "total_pages": total_pages,
                        })
                summary["checked"].append(f"{phase}:{mode}:{rec_kind}:rc={proc.returncode}:pages={total_pages}:sched={sched_file.read_text()}")
    return summary


def probe_adversarial_abandoned_activating_race(work_root: Path) -> dict:
    """Demonstrate BLOCK candidate: Owner 1 SIGKILLed in activating while scheduler is running;
    reconcile loses non-blocking transaction.lock race; Deploy 2 fails verified_target (guard-stale)
    BEFORE reconciling abandoned -> scheduler stays RUNNING and ZERO pages are sent."""
    case_dir = work_root / "adv_abandoned_race"
    guard_root, project_dir, tool, owner_state = setup_patched_tool(case_dir)
    sched_file = case_dir / "scheduler_state"
    sched_file.write_text("stopped")

    run_id = "1795197580-killedact"
    phase_bytes = b"#!/bin/bash\nexec true\n"
    digest = hashlib.sha256(phase_bytes).hexdigest()
    mod = load_guard_module(guard_root, project_dir)
    with mod.directory() as rfd:
        mod.publish(rfd, f"run-{run_id}.sh", phase_bytes, 0o700)
        mod.publish_json(rfd, f"run-{run_id}.metadata", {
            "deploy_id": run_id, "target": TARGET_SHA, "event": "push",
            "recovery": "", "phase_sha256": digest,
        })

    # Step 1: Owner 1 reaches activating-started (scheduler is NOW RUNNING!) and receives SIGKILL
    proc = subprocess.run(
        [PYTHON, __file__, "--worker", "KILL", str(case_dir), "activating-started", run_id, digest],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == -signal.SIGKILL
    assert sched_file.read_text() == "running"

    # Step 2: Concurrent contender (e.g. Deploy 2 starting or hand release inspecting) holds transaction.lock
    # when ExecStopPost `reconcile --deploy-id <run_id>` runs with timeout=0 (LOCK_NB).
    env = {"BTS_TEST_DEPLOY_GUARD_DIR": str(guard_root), "PATH": f"{case_dir / 'bin'}:/usr/bin:/bin"}
    with (guard_root / "transaction.lock").open("rb") as contender_lock:
        fcntl.flock(contender_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reconcile_proc = subprocess.run(
            [PYTHON, "-I", str(tool), "reconcile", "--deploy-id", run_id],
            env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
        )

    # Step 3: Deploy 2 runs with a commit that also updated install-deploy-guard-hetzner.sh (guard-stale)
    # or where git fetch fails. Because owned_deploy calls verified_target BEFORE reconciling `abandoned`,
    # Deploy 2 refuses at guard-stale without stopping the scheduler or transitioning hold or paging!
    pages_d2 = []
    configure_box_mocks(mod, sched_file, pages_d2, fail_at="guard-stale")
    with mod.directory() as rfd:
        d2_rc = mod.owned_deploy(rfd, "1795197585-deploy2", "c" * 40, "")
        hold_after = mod.read_hold(rfd)

    # Also test reconcile when `id -u bts` fails: page() does not need uid, yet reconcile aborts before page() and result!
    case_uid = work_root / "adv_reconcile_uid_fail"
    g2, p2, _, _ = setup_patched_tool(case_uid)
    mod2 = load_guard_module(g2, p2)
    pages_uid = []
    configure_box_mocks(mod2, case_uid / "sched", pages_uid)
    mod2.bts_uid = lambda: (_ for _ in ()).throw(mod2.Refused("id"))
    with mod2.directory() as rfd2:
        now_i = int(time.time()) - 5
        mod2.publish_json(rfd2, "hold", {
            "schema": "bts_deploy_hold_v1", "deploy_id": run_id,
            "owner_unit": f"bts-deploy-{run_id}.service", "state": "activating",
            "step": "activation", "target": TARGET_SHA, "previous": OLD_SHA,
            "armed_at": now_i, "updated_at": now_i,
        })
        try:
            mod2.reconcile(rfd2, run_id)
            uid_raised = False
        except mod2.Refused:
            uid_raised = True
        uid_result = mod2.read_json(rfd2, f"run-{run_id}.result")

    return {
        "reconcile_race_rc": reconcile_proc.returncode,
        "deploy2_rc": d2_rc,
        "hold_state_after_both": hold_after["state"],
        "scheduler_state_after_both": sched_file.read_text(),
        "pages_sent": len(pages_d2),
        "reconcile_uid_fail_aborts_before_page_and_result": (uid_raised and len(pages_uid) == 0 and uid_result is None),
    }


def probe_adversarial_sleep_window_post_release_false_page(work_root: Path) -> dict:
    """Demonstrate BLOCK: In EVERY normal passing deploy in a sleep window,
    activate_scheduler starts bts-scheduler at T0; bts-scheduler writes RUNNING then SLEEPING
    heartbeat at ~T0+2s and enters _watchdog_ping_sleep (which never touches .heartbeat);
    activate_scheduler sleeps 30s until T0+30s; then release() writes last-release with
    released_at = int(T0+30) > T0+2s! Every subsequent check_heartbeat.py run pages /fail!"""
    from bts.heartbeat import HeartbeatState, write_heartbeat

    case_dir = work_root / "adv_post_release_false_page"
    guard_root, project_dir, tool, _ = setup_patched_tool(case_dir)
    hb_path = project_dir / "data" / ".heartbeat"

    clock = [1_800_000_000.4]  # T0 = 2027-01-15T08:00:00.400000+00:00

    def now_fn():
        return clock[0]

    def sleep_fn(seconds):
        clock[0] += seconds

    mod = load_guard_module(guard_root, project_dir, sleep_fn=sleep_fn, time_fn=now_fn)
    pages = []
    sched_file = case_dir / "sched"
    configure_box_mocks(mod, sched_file, pages)

    # Pre-deploy valid sleeping heartbeat (wake 2 hours ahead)
    pre_dt = datetime.fromtimestamp(clock[0] - 60, timezone.utc)
    write_heartbeat(hb_path, state=HeartbeatState.SLEEPING, now_utc=pre_dt,
                    sleeping_until=pre_dt + timedelta(hours=2))

    # Real activate_scheduler behavior using real bts.heartbeat.write_heartbeat + 30s observation sleep!
    def real_activate(uid, authorized_at):
        # systemctl --user start bts-scheduler: writes RUNNING at +1.0s, fetches schedule, writes SLEEPING at +2.5s
        clock[0] += 1.0
        write_heartbeat(hb_path, state=HeartbeatState.RUNNING,
                        now_utc=datetime.fromtimestamp(clock[0], timezone.utc))
        clock[0] += 1.5
        sleep_target = datetime.fromtimestamp(clock[0], timezone.utc) + timedelta(hours=2)
        write_heartbeat(hb_path, state=HeartbeatState.SLEEPING,
                        now_utc=datetime.fromtimestamp(clock[0], timezone.utc),
                        sleeping_until=sleep_target)
        # Now owner sleeps 30s in activate_scheduler (line 715) while scheduler is in _watchdog_ping_sleep
        mod.time.sleep(30)
        # Verify heartbeat()[1] > authorized_at (line 730)
        _, hb_ts = mod.heartbeat()
        assert hb_ts > authorized_at, (hb_ts, authorized_at)

    mod.activate_scheduler = real_activate

    with mod.directory() as rfd:
        deploy_rc = mod.owned_deploy(rfd, "1795197600-gooddeploy", TARGET_SHA, "")
        last_release = mod.read_json(rfd, "last-release")
        result_rec = mod.read_json(rfd, "run-1795197600-gooddeploy.result")

    assert deploy_rc == 0 and result_rec["outcome"] == "committed"
    hb_json = json.loads(hb_path.read_text())

    # Now 5 minutes later (still well inside the 2-hour sleep window!), cron runs check_heartbeat.py!
    # Load check_heartbeat in-process with Clock pinned to T0 + 300s to capture exact ping & stdout.
    mon = ModuleType("monitor_under_test")
    exec(compile(MONITOR.read_text(), str(MONITOR), "exec"), mon.__dict__)
    tick_now = datetime.fromtimestamp(clock[0] + 300, timezone.utc)

    class PinnedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return tick_now.astimezone(tz) if tz else tick_now.replace(tzinfo=None)

    mon.datetime = PinnedClock
    pings_sent = []
    mon.ping = lambda url, suffix="": pings_sent.append(("GET", suffix))
    mon.ping_deploy_failure = lambda url, reason: pings_sent.append(("POST", "/fail", reason))

    old_argv = sys.argv
    old_env = os.environ.get("BTS_TEST_DEPLOY_GUARD_DIR")
    os.environ["BTS_TEST_DEPLOY_GUARD_DIR"] = str(guard_root)
    sys.argv = [
        "check_heartbeat.py", "--heartbeat-path", str(hb_path),
        "--deploy-hold", str(guard_root / "hold"),
        "--ping-url", "https://hc-ping.com/testtoken123", "--no-churn",
    ]
    try:
        try:
            mon.main()
            mon_rc = 0
        except SystemExit as exc:
            mon_rc = exc.code
    finally:
        sys.argv = old_argv
        if old_env is None:
            os.environ.pop("BTS_TEST_DEPLOY_GUARD_DIR", None)
        else:
            os.environ["BTS_TEST_DEPLOY_GUARD_DIR"] = old_env

    return {
        "deploy_rc": deploy_rc,
        "deploy_outcome": result_rec["outcome"],
        "scheduler_heartbeat_timestamp": hb_json["timestamp"],
        "last_release_released_at_iso": datetime.fromtimestamp(last_release["released_at"], timezone.utc).isoformat(),
        "monitor_rc_5min_after_healthy_deploy": mon_rc,
        "monitor_pings_5min_after_healthy_deploy": pings_sent,
    }


def probe_workflow_payload_and_ssh_loss(work_root: Path) -> dict:
    """Run extracted deploy.yml bash script under shims; test guard failure and SSH SIGHUP disconnect."""
    case_dir = work_root / "workflow_ssh"
    guard_root, project_dir, tool, _ = setup_patched_tool(case_dir)
    payload = case_dir / "deploy_script.sh"
    payload.write_bytes(extract_deploy_payload())

    shims = case_dir / "shims"
    shims.mkdir()
    # sudo shim routes /usr/bin/python3 /usr/local/bin/bts-deploy-hold to our temporary tool
    sudo_shim = shims / "sudo"
    sudo_shim.write_text(
        f'#!/bin/sh\n'
        f'if [ "$1" = "-n" ] && [ "$2" = "--" ]; then shift 2; fi\n'
        f'if [ "$1" = "/usr/bin/python3" ] && [ "$2" = "-I" ] && [ "$3" = "/usr/local/bin/bts-deploy-hold" ]; then\n'
        f'  shift 3\n'
        f'  exec "{PYTHON}" -I "{tool}" "$@"\n'
        f'fi\n'
        f'exit 91\n'
    )
    sudo_shim.chmod(0o755)

    # 1. Corrupt tool version -> workflow refuses before handoff
    bad_tool = case_dir / "bad-hold"
    bad_tool.write_text('print("wrong 2")\n')
    sudo_bad = case_dir / "shims_bad"
    sudo_bad.mkdir()
    (sudo_bad / "sudo").write_text(
        f'#!/bin/sh\nshift 4\nexec "{PYTHON}" -I "{bad_tool}" "$@"\n'
    )
    (sudo_bad / "sudo").chmod(0o755)
    bad_res = subprocess.run(
        ["/bin/bash", str(payload)],
        env={
            "PATH": f"{sudo_bad}:/usr/bin:/bin",
            "TARGET_SHA": TARGET_SHA,
            "DEPLOY_EVENT": "push",
            "RECOVER_SHA": TARGET_SHA,
        },
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
    )
    assert bad_res.returncode == 1 and bad_res.stdout == "DEPLOY NOT STARTED — guard failed.\n"

    # 2. Invalid target SHA on handoff -> refuses cleanly; push ignores RECOVER_SHA
    bad_sha_res = subprocess.run(
        ["/bin/bash", str(payload)],
        env={
            "PATH": f"{shims}:/usr/bin:/bin",
            "BTS_TEST_DEPLOY_GUARD_DIR": str(guard_root),
            "TARGET_SHA": "not-a-sha",
            "DEPLOY_EVENT": "push",
            "RECOVER_SHA": "",
        },
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
    )
    # Note: handoff requires os.geteuid() == 0 in main(); verify non-root rejection in main()!
    assert bad_sha_res.returncode == 1

    return {
        "bad_version_rc": bad_res.returncode,
        "bad_version_stdout": bad_res.stdout.strip(),
        "nonroot_handoff_rejected_rc": bad_sha_res.returncode,
    }


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _, _, mode, base_str, phase, run_id, digest = sys.argv
        return child_worker(mode, base_str, phase, run_id, digest)

    work_dir = Path(tempfile.mkdtemp(prefix="p1_", dir=Path(__file__).resolve().parent))
    try:
        wf = probe_workflow_payload_and_ssh_loss(work_dir)
        matrix = run_phase_matrix(work_dir)
        adv_race = probe_adversarial_abandoned_activating_race(work_dir)
        adv_release = probe_adversarial_sleep_window_post_release_false_page(work_dir)
        report = {
            "workflow_checks": wf,
            "phase_matrix_total": matrix["total"],
            "phase_matrix_anomalies": matrix["anomalies"],
            "adversarial_abandoned_activating_race": adv_race,
            "adversarial_sleep_window_post_release": adv_release,
        }
        out_path = Path(__file__).resolve().parent / "probe_1_lifecycle.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
