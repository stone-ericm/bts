#!/usr/bin/env python3
"""Probes for Review Items 2 and 3: Every `verified` line, attempt record, mutex, and cancellation mark."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github/workflows/deploy.yml"
OLD = "a" * 40
TARGET = "b" * 40

COMMANDS = {
    "bash", "bts", "cat", "curl", "cut", "date", "env", "flock", "git", "grep", "head",
    "id", "install", "mv", "pgrep", "pkill", "python3", "rm", "sleep", "stat", "sudo",
    "systemctl", "tailscale", "timeout", "uv",
}

SHIM_CODE = r'''
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

command = Path(sys.argv[0]).name
args = sys.argv[1:]
root = Path(os.environ["BOX"])
scenario = os.environ.get("SCENARIO", "active")
with (root / "calls").open("a") as log:
    print(json.dumps([command, *args]), file=log)

def dispatch(words):
    if words[0].startswith("/") and words[0] != "/bin/bash" and not Path(words[0]).is_relative_to(root):
        raise SystemExit(99)
    prog = Path(words[0]).name
    shim = root / "shims" / prog
    if not shim.is_file():
        raise SystemExit(98)
    os.execv(shim, [str(shim), *words[1:]])

if command == "sudo":
    args = args[args.index("bts") + 1:]
    while args and "=" in args[0] and not args[0].startswith("/"):
        k, v = args.pop(0).split("=", 1)
        os.environ[k] = v
    dispatch(args)
elif command == "env":
    while args and "=" in args[0] and not args[0].startswith("/"):
        k, v = args.pop(0).split("=", 1)
        os.environ[k] = v
    dispatch(args)
elif command == "timeout":
    dispatch(args[3:])
elif command == "install":
    Path(args[-1]).mkdir(mode=0o700, parents=True, exist_ok=True)
elif command == "stat":
    p = Path(args[-1])
    uid = 1000 if scenario == "dir-wrong-owner" else 0
    print(f"{uid} {p.stat().st_mode & 0o777:o}")
elif command == "flock":
    try:
        fcntl.flock(int(args[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(1)
elif command == "python3":
    op = args[2] if len(args) > 2 else None
    code = args[1]
    sys.argv = ["-c", *args[2:]]
    orig_fstat = os.fstat
    fake_uid = 1000 if scenario == "file-wrong-owner" else 0
    def patched_fstat(fd):
        st = orig_fstat(fd)
        return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, fake_uid, 0, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))
    if op in {"lockfile", "mark"}:
        os.fstat = patched_fstat
    if op == "marker" and scenario == "marker-swap-inode":
        orig_stat = os.stat
        def swap_stat(p, *a, **kw):
            res = orig_stat(p, *a, **kw)
            if str(p).endswith(".heartbeat"):
                other = root / "swapped-hb"
                other.write_bytes(b"other")
                return orig_stat(other)
            return res
        os.stat = swap_stat
    exec(compile(code, "<fs_code>", "exec"), {"__name__": "__main__"})
elif command == "mv":
    os.replace(args[-2], args[-1])
elif command == "git":
    branch = root / "head"
    if args[0] == "rev-parse":
        print(branch.read_text()[:7] if "--short" in args else branch.read_text())
    elif args[0] == "reset":
        if scenario == "hold-reset":
            (root / "in-reset").touch()
            deadline = time.monotonic() + 8
            while not (root / "release-reset").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
        branch.write_text(args[-1])
    elif args[0] == "cat-file":
        print("commit")
elif command == "id":
    if scenario == "uid-fail-in-contain" and (root / "sync-failed").exists():
        raise SystemExit(1)
    print(1000)
elif command == "systemctl":
    user = args[0] == "--user"
    if user:
        args.pop(0)
    if user and args[0] in {"disable", "enable"}:
        if scenario == "user-disable-fail-in-contain" and (root / "sync-failed").exists():
            raise SystemExit(1)
        (root / "enabled").write_text("disabled" if args[0] == "disable" else "enabled")
    if not user and args[0] == "disable":
        if scenario == "legacy-no-fragment":
            # Debian systemctl disable on a loaded unit whose fragment was deleted exits 1
            raise SystemExit(1)
        (root / "legacy-enabled").write_text("disabled")
    if user and args[0] == "restart":
        (root / "stopped").unlink(missing_ok=True)
    if args[0] == "is-enabled":
        if user:
            if scenario == "user-isenabled-exit0-disabled" and (root / "sync-failed").exists():
                print("disabled")
                raise SystemExit(0)
            st = (root / "enabled").read_text() if (root / "enabled").exists() else "enabled"
        else:
            if scenario == "legacy-no-fragment":
                raise SystemExit(1)
            st = (root / "legacy-enabled").read_text() if (root / "legacy-enabled").exists() else "disabled"
        print(st)
        raise SystemExit(1 if st == "disabled" else 0)
    if args[0] == "show":
        active = "active" if user and not (root / "stopped").exists() else "inactive"
        loaded = "loaded" if user or scenario in {"legacy-no-fragment", "legacy-job", "legacy-cgroup"} else "not-found"
        states = dict(Id=args[1], LoadState=loaded, UnitFileState="enabled",
                      ActiveState=active, SubState="running", Result="success",
                      ExecMainStatus="0", MainPID="0", ControlGroup="")
        if "ControlGroup" in args:
            if user and scenario == "root-cgroup":
                states["ControlGroup"] = "/"
            elif user and scenario == "dotdot-cgroup":
                states["ControlGroup"] = "/sched/../other"
            elif user and scenario == "deactivating":
                states["ActiveState"] = "deactivating"
            elif not user and scenario == "legacy-cgroup":
                states["ControlGroup"] = "/legacy"
        for key in (a for a in args if a in states):
            print(f"{key}={states[key]}")
    if user and args[0] == "stop":
        (root / "stopped").touch()
    if args[0] == "list-jobs":
        if user and scenario == "malformed-jobs" and (root / "sync-failed").exists():
            print("No jobs running.")
        if not user and scenario == "legacy-job" and (root / "sync-failed").exists():
            print("42 bts-scheduler.service start waiting")
elif command == "tailscale":
    print("100.64.0.1")
elif command == "curl":
    if "--data-raw" in args:
        code = os.environ.get("POST_CODE", "200")
        rc = int(os.environ.get("POST_RC", "0"))
        print(code, end="")
        raise SystemExit(rc)
    print("200", end="")
elif command == "bash" and args[0] == "-c" and args[1].startswith("printf "):
    print(*args[3:], sep="\n")
elif command == "bash" and args[0] == "-c" and args[1].startswith("exec 9>&-"):
    os.close(9)
    dispatch(args[3:])
elif command == "bts":
    print("Deploy canary: not a live configuration (delivery private); contest state not checked.")
elif command == "pgrep":
    if scenario == "pgrep-error":
        raise SystemExit(2)
    raise SystemExit(1)
elif command == "pkill":
    raise SystemExit(1)
elif command == "rm":
    for operand in (a for a in args if not a.startswith("-")):
        Path(operand).unlink(missing_ok=True)
elif command == "uv":
    if os.environ.get("FAIL_SYNC") == "1":
        (root / "sync-failed").touch()
        raise SystemExit(17)
elif command == "sleep":
    pass
else:
    raise SystemExit(97)
'''


def extract_scripts() -> tuple[str, str]:
    source = WORKFLOW.read_text()
    steps = [s for s in yaml.safe_load(source)["jobs"]["deploy"]["steps"] if "script" in s.get("with", {})]
    return steps[0]["with"]["script"], steps[1]["with"]["script"]


def make_box(tmp: Path) -> Path:
    shim_dir = tmp / "shims"
    shim_dir.mkdir()
    for name in COMMANDS:
        p = shim_dir / name
        p.write_text(f"#!{sys.executable}\n" + SHIM_CODE)
        p.chmod(0o700)
    (tmp / "home/projects/bts/data").mkdir(parents=True)
    (tmp / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.invalid/uuid\n")
    (tmp / "sys/fs/cgroup").mkdir(parents=True)
    (tmp / "sys/fs/cgroup/cgroup.controllers").touch()
    (tmp / "head").write_text(OLD)
    return tmp


def run_script(box: Path, script: str, *, scenario: str = "active", fail_sync: bool = False,
               run_id: str = "123", attempt: str = "1", target: str = TARGET,
               post_code: str = "200", post_rc: str = "0") -> tuple[subprocess.CompletedProcess, list[list[str]]]:
    for absolute, rel in {
        "/home/bts/.local/bin/uv": "shims/uv",
        "/home/bts/projects/bts/.venv/bin/bts": "shims/bts",
        "/home/bts": "home",
        "/run/bts-deploy": "run/bts-deploy",
        "/run/user": "run/user",
        "/sys/fs/cgroup": "sys/fs/cgroup",
        "/etc/systemd/system": "etc/systemd/system",
        "/usr/bin/python3": "shims/python3",
        "/tmp/uv-cache": "uv-cache",
        "/dev/null": "null",
    }.items():
        script = script.replace(absolute, str(box / rel))
    argv = [
        "/usr/bin/env", "-i", f"PATH={box / 'shims'}", f"BOX={box}",
        f"SCENARIO={scenario}", f"FAIL_SYNC={'1' if fail_sync else '0'}",
        f"POST_CODE={post_code}", f"POST_RC={post_rc}",
        f"TARGET_SHA={target}", f"RUN_ID={run_id}", f"RUN_ATTEMPT={attempt}",
        "LC_ALL=C", "/bin/bash", "-c", script,
    ]
    proc = subprocess.Popen(argv, cwd=box, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with proc:
        try:
            out, err = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=2)
            raise
    calls = [json.loads(l) for l in (box / "calls").read_text().splitlines()] if (box / "calls").exists() else []
    return subprocess.CompletedProcess(argv[:2], proc.returncode, out, err), calls


def main() -> None:
    deploy, recovery = extract_scripts()

    # --- Item 2a: Scheduler stopped: verified edge cases (all MUST be unverified!) ---
    stop_unverified_cases = [
        "uid-fail-in-contain",
        "user-disable-fail-in-contain",
        "user-isenabled-exit0-disabled",
        "root-cgroup",
        "dotdot-cgroup",
        "deactivating",
        "malformed-jobs",
        "legacy-no-fragment",
        "legacy-job",
        "legacy-cgroup",
        "pgrep-error",
        "missing-cgroup-v2",
    ]
    for scen in stop_unverified_cases:
        with tempfile.TemporaryDirectory() as td:
            box = make_box(Path(td))
            if scen == "legacy-cgroup":
                (box / "sys/fs/cgroup/legacy").mkdir(parents=True)
                (box / "sys/fs/cgroup/legacy/cgroup.events").write_text("populated 1\nfrozen 0\n")
            if scen == "missing-cgroup-v2":
                (box / "sys/fs/cgroup/cgroup.controllers").unlink()
            res, calls = run_script(box, deploy, scenario=scen, fail_sync=True)
            assert "Scheduler stopped: unverified." in res.stdout, (scen, res.stdout)
            if scen == "legacy-no-fragment":
                # Even though disable failed on fragmentless loaded unit, stop MUST still have run!
                assert ["systemctl", "stop", "bts-scheduler.service"] in calls
    print(f"Item 2a: all {len(stop_unverified_cases)} hostile stop-verification cases -> unverified (0 false greens)")

    # --- Item 2b: Marker verification edge cases ---
    for kind in ("swap-inode", "dangling-symlink", "fifo-with-reader"):
        with tempfile.TemporaryDirectory() as td:
            box = make_box(Path(td))
            hb = box / "home/projects/bts/data/.heartbeat"
            if kind == "dangling-symlink":
                hb.symlink_to(box / "nonexistent-target")
                res, _ = run_script(box, deploy, fail_sync=True)
                assert "Failure heartbeat written: verified." in res.stdout
                assert not hb.is_symlink() and not (box / "nonexistent-target").exists()
            elif kind == "fifo-with-reader":
                os.mkfifo(hb)
                r_fd = os.open(hb, os.O_RDONLY | os.O_NONBLOCK)
                try:
                    res, _ = run_script(box, deploy, fail_sync=True)
                finally:
                    os.close(r_fd)
                assert "Failure heartbeat written: verified." in res.stdout
                assert hb.is_file() and not hb.is_fifo()
            elif kind == "swap-inode":
                res, _ = run_script(box, deploy, scenario="marker-swap-inode", fail_sync=True)
                assert "Failure heartbeat written: unverified." in res.stdout
    print("Item 2b: marker verification (dangling symlink, live FIFO, swapped inode) -> OK")

    # --- Item 2c: Endpoint & POST verification edge cases ---
    bad_envs = [
        "BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.invalid/uuid\r\n",
        "BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.invalid/uuid \n",
        'BTS_SCHEDULER_HEARTBEAT_PING_URL="https://hc-ping.invalid/uuid\'\n',
        'BTS_SCHEDULER_HEARTBEAT_PING_URL=""https://hc-ping.invalid/uuid""\n',
        "BTS_SCHEDULER_HEARTBEAT_PING_URL=https://@hc-ping.invalid/uuid\n",
        "BTS_SCHEDULER_HEARTBEAT_PING_URL=https://:pw@hc-ping.invalid/uuid\n",
        "BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.invalid:65536/uuid\n",
        "BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.invalid:/uuid\n",
        "BTS_SCHEDULER_HEARTBEAT_PING_URL=https:/hc-ping.invalid/uuid\n",
        "BTS_SCHEDULER_HEARTBEAT_PING_URL=https:hc-ping.invalid/uuid\n",
    ]
    for env_txt in bad_envs:
        with tempfile.TemporaryDirectory() as td:
            box = make_box(Path(td))
            (box / "home/projects/bts/.env").write_text(env_txt)
            res, calls = run_script(box, deploy, fail_sync=True)
            posts = [c for c in calls if c[0] == "curl" and "--data-raw" in c]
            assert not posts and "Failure ping received HTTP 2xx: unverified." in res.stdout, env_txt
    # Also test symlink .env (must be rejected by O_NOFOLLOW!)
    with tempfile.TemporaryDirectory() as td:
        box = make_box(Path(td))
        real_env = box / "real.env"
        real_env.write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.invalid/uuid\n")
        env_p = box / "home/projects/bts/.env"
        env_p.unlink()
        env_p.symlink_to(real_env)
        res, calls = run_script(box, deploy, fail_sync=True)
        assert not [c for c in calls if c[0] == "curl" and "--data-raw" in c]
        assert "Failure ping received HTTP 2xx: unverified." in res.stdout
    # Also test curl codes: 302 redirect, 200200 (glob multi-response), 000, exit 28 with 200
    for code, rc in [("302", "0"), ("200200", "0"), ("000", "0"), ("200", "28")]:
        with tempfile.TemporaryDirectory() as td:
            box = make_box(Path(td))
            res, _ = run_script(box, deploy, fail_sync=True, post_code=code, post_rc=rc)
            assert "Failure ping received HTTP 2xx: unverified." in res.stdout
    # Check curl globbing note: does endpoint accept https://hc-ping.invalid/[1-1]/uuid and pass it without -g?
    with tempfile.TemporaryDirectory() as td:
        box = make_box(Path(td))
        (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.invalid/[1-1]/uuid\n")
        res, calls = run_script(box, deploy, fail_sync=True)
        posts = [c for c in calls if c[0] == "curl" and "--data-raw" in c]
        print(f"Item 2c: bad envs/symlink/codes all unverified; curl glob URL passed? {bool(posts)} has_-g={'-g' in posts[0] or '--globoff' in posts[0] if posts else None}")

    # --- Item 3: Attempt record, mutex, cancellation mark across orders & hostile files ---
    # 3a. Malformed/special/oversized attempt records in BOTH deploy and recovery
    bad_records = [
        b"x" * 513 + b"\n",
        b"bts_deploy_attempt_v1 123 1 " + TARGET.encode() + b" " + OLD.encode() + b" begun - - - -",  # no \n
        b"bts_deploy_attempt_v1 123 1 " + TARGET.encode() + b" " + OLD.encode() + b" begun - - - -\nextra\n",
        b"bts_deploy_attempt_v1 0 1 " + TARGET.encode() + b" " + OLD.encode() + b" begun - - - -\n",
        b"bts_deploy_attempt_v1 123 1 " + TARGET.encode() + b" unknown begun - - - -\n",
        b"bts_deploy_attempt_v1 123 1 " + TARGET.encode() + b" " + OLD.encode() + b" begun sync - - -\n",
        b"bts_deploy_attempt_v1 123 1 " + TARGET.encode() + b" " + OLD.encode() + b" contained - verified verified verified\n",
        b"bts_deploy_attempt_v1 123 1 " + TARGET.encode() + b" " + OLD.encode() + b" contained bogus verified verified verified\n",
        b"bts_deploy_attempt_v1 123 1 " + TARGET.encode() + b" " + OLD.encode() + b" contained sync - verified verified\n",
        b"\xff\n",
    ]
    for rec in bad_records:
        for idx, sc in enumerate((deploy, recovery)):
            with tempfile.TemporaryDirectory() as td:
                box = make_box(Path(td))
                ld = box / "run/bts-deploy"
                ld.mkdir(mode=0o700, parents=True)
                (ld / "attempt").write_bytes(rec)
                res, calls = run_script(box, sc)
                assert res.returncode == 1
                assert not any(c[0] == "systemctl" for c in calls)
                if idx == 0:
                    assert "DEPLOY NOT STARTED — attempt-record failed." in res.stdout
                else:
                    assert "the attempt record is unreadable" in res.stdout

    # Special files at attempt (symlink, dangling symlink, FIFO, dir) and at cancelled.<id>.<attempt>
    for kind in ("symlink", "dangling", "fifo", "dir"):
        with tempfile.TemporaryDirectory() as td:
            box = make_box(Path(td))
            ld = box / "run/bts-deploy"
            ld.mkdir(mode=0o700, parents=True)
            att = ld / "attempt"
            if kind == "symlink":
                real = box / "real-att"
                real.write_text(f"bts_deploy_attempt_v1 999 1 {TARGET} {OLD} committed - - - -\n")
                att.symlink_to(real)
            elif kind == "dangling":
                att.symlink_to(box / "missing")
            elif kind == "fifo":
                os.mkfifo(att)
            else:
                att.mkdir()
            r_d, c_d = run_script(box, deploy)
            r_r, c_r = run_script(box, recovery)
            assert r_d.returncode == 1 and "attempt-record failed" in r_d.stdout
            assert r_r.returncode == 1 and "the attempt record is unreadable" in r_r.stdout
            assert not any(c[0] == "systemctl" for c in c_d + c_r)

    # Malformed cancellation marks in recovery (non-empty, mode 0644, symlink, dangling, FIFO, dir, wrong owner)
    for kind in ("nonempty", "mode", "symlink", "dangling", "fifo", "dir", "wrong-owner"):
        with tempfile.TemporaryDirectory() as td:
            box = make_box(Path(td))
            ld = box / "run/bts-deploy"
            ld.mkdir(mode=0o700, parents=True)
            cm = ld / "cancelled.123.1"
            scen = "active"
            if kind == "nonempty":
                cm.write_bytes(b"x")
                cm.chmod(0o600)
            elif kind == "mode":
                cm.touch(mode=0o644)
            elif kind == "symlink":
                other = box / "other"
                other.touch(mode=0o600)
                cm.symlink_to(other)
            elif kind == "dangling":
                cm.symlink_to(box / "missing")
            elif kind == "fifo":
                os.mkfifo(cm)
            elif kind == "dir":
                cm.mkdir()
            elif kind == "wrong-owner":
                scen = "file-wrong-owner"
            r_r, c_r = run_script(box, recovery, scenario=scen)
            assert r_r.returncode == 1
            if kind == "wrong-owner":
                assert "lock unavailable" in r_r.stdout  # lockfile check catches wrong owner first!
            else:
                assert "cancellation mark could not be verified" in r_r.stdout, (kind, r_r.stdout)
            assert not any(c[0] == "systemctl" for c in c_r)

    # Second run (different run_id=456, same attempt=1) vs re-run (same run_id=123, attempt=2)
    with tempfile.TemporaryDirectory() as td:
        box = make_box(Path(td))
        r1, _ = run_script(box, deploy, fail_sync=True, run_id="123", attempt="1")
        assert r1.returncode == 1
        # Duplicate (123, 1) refused
        r_dup, _ = run_script(box, deploy, run_id="123", attempt="1")
        assert r_dup.returncode == 1 and "attempt-record failed" in r_dup.stdout
        # Re-run (123, 2) succeeds
        r_rerun, _ = run_script(box, deploy, run_id="123", attempt="2")
        assert r_rerun.returncode == 0 and "CANARY PASSED" in r_rerun.stdout
        # Second run (456, 1) succeeds
        (box / "head").write_text(OLD)
        r_run2, _ = run_script(box, deploy, run_id="456", attempt="1")
        assert r_run2.returncode == 0 and "CANARY PASSED" in r_run2.stdout
        # Recovery for earlier run (123, 2) refuses with "different attempt owns the box record"
        r_old_rec, c_old_rec = run_script(box, recovery, run_id="123", attempt="2")
        assert r_old_rec.returncode == 1 and "a different attempt owns the box record" in r_old_rec.stdout
        assert (box / "run/bts-deploy/cancelled.123.2").is_file()
    print("Item 3: all malformed/special/oversized paths and attempt/run orderings -> OK")


if __name__ == "__main__":
    main()
