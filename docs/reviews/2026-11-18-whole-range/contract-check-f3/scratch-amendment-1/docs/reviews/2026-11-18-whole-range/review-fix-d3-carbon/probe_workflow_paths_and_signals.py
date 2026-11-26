#!/usr/bin/env python3
"""Probes for Review Items 1 and 4: Phase-2/F3/F4 failure & timeout paths, signals, EPIPE, and success path."""
from __future__ import annotations

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
BASE_WORKFLOW = REPO / ".codex-review/wr-d3/base/.github/workflows/deploy.yml"
OLD = "1" * 40
TARGET = "2" * 40

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
import re
import signal
import stat as statmod
import sys
import time

command = Path(sys.argv[0]).name
args = sys.argv[1:]
root = Path(os.environ["BOX"])
scenario = os.environ.get("SCENARIO", "active")
with (root / "calls").open("a") as log:
    print(json.dumps([command, *args]), file=log)

def phase():
    if command == "systemctl" and args:
        user = args[0] == "--user"
        op, *rest = args[1:] if user else args
        if user and op in {"disable", "enable", "restart"}:
            return op
        if op == "is-enabled" and user:
            st = (root / "enabled").read_text() if (root / "enabled").exists() else "enabled"
            return "check-enable" if st == "enabled" else "check-disable"
        if op == "show" and user and "ExecMainStatus" in rest:
            return "scheduler-state"
        if op == "show" and user and "LoadState" in rest:
            return "user-units"
        if op == "show" and not user and "LoadState" in rest:
            return "legacy-units"
        if op == "show" and user and "bts-dashboard" in rest and "ActiveState" in rest:
            return "dashboard-state"
        if not user and op in {"disable", "stop", "daemon-reload"}:
            if op in {"disable", "stop"} and rest:
                unit_tag = "sched" if "scheduler" in rest[0] else "dash"
                return f"legacy-{op}-{unit_tag}"
            return "legacy-" + op
    if command == "rm" and args:
        operands = [a for a in args if not a.startswith("-")]
        if operands and "etc/systemd/system" in operands[0]:
            unit_tag = "sched" if "scheduler" in operands[0] else "dash"
            return f"legacy-rm-{unit_tag}"
    if command == "git" and args:
        if args[0] == "rev-parse" and (root / "head").read_text().startswith("2"):
            return "head"
        return args[0]
    if command == "id":
        return "uid"
    if command in {"uv", "pkill", "sleep", "bts", "curl"}:
        return {"uv": "sync", "pkill": "legacy-pkill", "sleep": "wait",
                "bts": "contest-state",
                "curl": "post" if "--data-raw" in args else "dashboard-http"}[command]
    if command == "python3" and len(args) > 2:
        op = args[2]
        if op == "stage":
            return "stamp-stage"
        if op == "stamp":
            return "stamp"
        if op == "stage-record" and len(args) > 4:
            wanted = args[4].split()[5]
            return f"stage-record-{wanted}"
        if op == "read":
            return "read-record"
    if command == "mv" and len(args) >= 2 and Path(args[-2]).name.startswith(".attempt."):
        wanted = Path(args[-2]).read_text().split()[5]
        return f"mv-record-{wanted}"
    return None

at = phase()
fault = os.environ.get("FAULT", "")
if fault and at:
    target_at, *fault_rest = fault.split(":")
    if target_at == at and not (root / "fault-fired").exists():
        # Optional occurrence filter for read-record (e.g. only after mutated)
        allow = True
        if len(fault_rest) >= 2 and fault_rest[1] == "after-mutated":
            allow = (root / "head").read_text() == os.environ["TARGET_SHA"]
        if allow:
            (root / "fault-fired").touch()
            action = fault_rest[0]
            if action in {"2", "17", "124", "137"}:
                raise SystemExit(int(action))
            if action == "mismatch":
                print("3" * 40)
                raise SystemExit(0)
            if action == "advance-and-sig":
                sig_name = fault_rest[1]
                (root / "clock-offset-ms").write_text("795000")
                os.kill(int((root / "owner.pid").read_text()), getattr(signal, "SIG" + sig_name))
            elif action.startswith("SIG"):
                os.kill(int((root / "owner.pid").read_text()), getattr(signal, action))

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
    print(f"0 {p.stat().st_mode & 0o777:o}")
elif command == "flock":
    try:
        fcntl.flock(int(args[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(1)
elif command == "python3":
    op = args[2] if len(args) > 2 else None
    code = args[1]
    sys.argv = ["-c", *args[2:]]
    if op == "clock":
        offset = int((root / "clock-offset-ms").read_text()) if (root / "clock-offset-ms").exists() else 0
        print(time.monotonic_ns() // 1_000_000 + offset)
        raise SystemExit(0)
    # Run the REAL fs_code (including lockfile and mark!) with st_uid mocked to 0 for root-owned paths
    orig_fstat = os.fstat
    def root_fstat(fd):
        st = orig_fstat(fd)
        return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, 0, 0, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))
    if op in {"lockfile", "mark"}:
        os.fstat = root_fstat
    if op == "stamp":
        orig_fsync = os.fsync
        def logged_fsync(fd):
            with (root / "fsyncs").open("a") as fl:
                st = orig_fstat(fd)
                print("dir" if statmod.S_ISDIR(st.st_mode) else "reg", file=fl)
            return orig_fsync(fd)
        os.fsync = logged_fsync
    exec(compile(code, "<fs_code>", "exec"), {"__name__": "__main__"})
elif command == "mv":
    os.replace(args[-2], args[-1])
elif command == "git":
    branch = root / "head"
    if args[0] == "fetch" and scenario == "fetch-fail":
        raise SystemExit(17)
    if args[0] == "rev-parse":
        print(branch.read_text()[:7] if "--short" in args else branch.read_text())
    elif args[0] == "reset":
        branch.write_text(args[-1])
    elif args[0] == "cat-file":
        print("commit")
elif command == "id":
    print(1000)
elif command == "systemctl":
    user = args[0] == "--user"
    if user:
        args.pop(0)
    if user and args[0] in {"disable", "enable"}:
        (root / "enabled").write_text("disabled" if args[0] == "disable" else "enabled")
    if user and args[0] == "restart":
        (root / "stopped").unlink(missing_ok=True)
        (root / "home/projects/bts/data/.heartbeat").write_text('{"state":"running"}')
    if args[0] == "is-enabled":
        en = root / "enabled"
        state = en.read_text() if en.exists() else "enabled"
        print(state)
        raise SystemExit(1 if state == "disabled" else 0)
    if args[0] == "is-active" and "--quiet" not in args:
        print("active")
    if args[0] == "show":
        active = "active" if user and not (root / "stopped").exists() else "inactive"
        loaded = "loaded" if user or scenario == "legacy-present" else "not-found"
        states = dict(Id=args[1], LoadState=loaded, UnitFileState="enabled",
                      ActiveState=active, SubState="running", Result="success",
                      ExecMainStatus="0", MainPID="0", ControlGroup="")
        for key in (a for a in args if a in states):
            print(states[key] if "--value" in args else f"{key}={states[key]}")
    if user and args[0] == "stop":
        if scenario == "second-signal":
            owner = int((root / "owner.pid").read_text())
            os.kill(owner, signal.SIGTERM)
            os.kill(owner, signal.SIGHUP)
        (root / "stopped").touch()
elif command == "tailscale":
    print("100.64.0.1")
elif command == "curl":
    print("200", end="")
elif command == "head":
    print(sys.stdin.readline(), end="")
elif command == "grep":
    raise SystemExit(0 if re.search(args[-1], sys.stdin.read()) else 1)
elif command == "cut":
    print(sys.stdin.read()[:7])
elif command == "bash" and args[0] == "-c" and args[1].startswith("date -u "):
    (root / "home/projects/bts/data/.last_deploy_iso").write_text("2026-11-21T13:00:00Z\n")
elif command == "bash" and args[0] == "-c" and args[1].startswith("printf "):
    print(*args[3:], sep="\n")
elif command == "bash" and args[0] == "-c" and args[1].startswith("exec 9>&-"):
    os.close(9)
    dispatch(args[3:])
elif command == "bts":
    print("Deploy canary: not a live configuration (delivery private); contest state not checked.")
elif command in {"pgrep", "pkill"}:
    raise SystemExit(1)
elif command == "rm":
    for operand in (a for a in args if not a.startswith("-")):
        Path(operand).unlink(missing_ok=True)
elif command in {"sleep", "uv"}:
    pass
else:
    raise SystemExit(97)
'''


def extract_scripts(path: Path) -> list[str]:
    source = path.read_text()
    steps = [s for s in yaml.safe_load(source)["jobs"]["deploy"]["steps"] if "script" in s.get("with", {})]
    literals = re.findall(r"^          script: \|\n((?:^ {12}[^\n]*\n|^\n)*)", source, re.MULTILINE)
    scripts = [re.sub(r"^ {12}", "", lit, flags=re.MULTILINE).rstrip("\n") + "\n" for lit in literals]
    assert scripts == [s["with"]["script"] for s in steps]
    return scripts


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


def run_script(box: Path, script: str, *, scenario: str = "active", fault: str = "",
               run_id: str = "101", attempt: str = "1", target: str = TARGET,
               closed_output: bool = False) -> tuple[subprocess.CompletedProcess, list[list[str]]]:
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
        "/usr/bin/env", "-i", f"PATH={box / 'shims'}", f"BOX={box}", f"FAULT={fault}",
        f"SCENARIO={scenario}", f"TARGET_SHA={target}", f"RUN_ID={run_id}",
        f"RUN_ATTEMPT={attempt}", "LC_ALL=C", "/bin/bash", "-c", script,
    ]
    launcher = (
        "import os, signal, sys\n"
        "for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGPIPE):\n"
        "    signal.signal(sig, signal.SIG_DFL)\n"
        "os.execv(sys.argv[1], sys.argv[1:])\n"
    )
    writer = subprocess.PIPE
    if closed_output:
        r_fd, w_fd = os.pipe()
        os.close(r_fd)
        writer = w_fd
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", launcher, *argv],
            cwd=box, stdout=writer, stderr=subprocess.PIPE, text=True,
        )
    finally:
        if closed_output:
            os.close(writer)
    with proc:
        (box / "owner.pid").write_text(str(proc.pid))
        try:
            out, err = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=2)
            raise
    calls = [json.loads(l) for l in (box / "calls").read_text().splitlines()] if (box / "calls").exists() else []
    return subprocess.CompletedProcess(argv[:2], proc.returncode, out or "", err), calls


def main() -> None:
    deploy, recovery = extract_scripts(WORKFLOW)
    base_deploy, = extract_scripts(BASE_WORKFLOW)

    # RD8 check
    prefix_d = deploy.split("\nnot_started() {", 1)[0]
    prefix_r = recovery.split("\nrecovery_exit() {", 1)[0]
    assert prefix_d == prefix_r, "RD8 containment/helper prefix differs"
    print("RD8 byte-identity of frame + helpers + contain(): OK")

    # 1. Every phase-2, F3, and F4 command failure & timeout
    matrix = [
        ("disable", "disable", "active"),
        ("check-disable", "disable", "active"),
        ("reset", "reset", "active"),
        ("sync", "sync", "active"),
        ("legacy-disable-sched", "legacy-units", "legacy-present"),
        ("legacy-stop-sched", "legacy-units", "legacy-present"),
        ("legacy-rm-sched", "legacy-units", "legacy-present"),
        ("legacy-disable-dash", "legacy-units", "legacy-present"),
        ("legacy-stop-dash", "legacy-units", "legacy-present"),
        ("legacy-rm-dash", "legacy-units", "legacy-present"),
        ("legacy-daemon-reload", "legacy-units", "legacy-present"),
        ("legacy-pkill", "legacy-units", "legacy-present"),
        ("restart", "restart", "active"),
        ("head", "head", "active"),
        ("wait", "wait", "active"),
        ("scheduler-state", "scheduler-state", "active"),
        ("dashboard-state", "dashboard-state", "active"),
        ("dashboard-http", "dashboard-http", "active"),
        ("contest-state", "contest-state", "active"),
        ("stamp-stage", "stamp-stage", "active"),
        ("enable", "enable", "active"),
        ("check-enable", "enable", "active"),
        ("stamp", "stamp", "active"),
        ("stage-record-committed", "commit-record", "active"),
        ("mv-record-committed", "commit-record", "active"),
    ]
    mismatches = []
    for cmd_tag, expected_step, scen in matrix:
        for code in ("17", "124", "137"):
            if cmd_tag == "legacy-pkill" and code == "17":
                code_used = "2"
            else:
                code_used = code
            with tempfile.TemporaryDirectory() as td:
                box = make_box(Path(td))
                if scen == "legacy-present":
                    (box / "etc/systemd/system").mkdir(parents=True)
                    (box / "etc/systemd/system/bts-scheduler.service").touch()
                    (box / "etc/systemd/system/bts-dashboard.service").touch()
                res, calls = run_script(box, deploy, scenario=scen, fault=f"{cmd_tag}:{code_used}")
                assert res.returncode == 1
                assert ["git", "reset", "--hard", OLD] not in calls, f"rollback occurred on {cmd_tag}"
                assert f"DEPLOY FAILED at {expected_step} —" in res.stdout, (cmd_tag, code_used, res.stdout)
                # Ensure boot link is left disabled after containment (even if failure happened after enable!)
                en_state = (box / "enabled").read_text() if (box / "enabled").exists() else "enabled"
                assert en_state == "disabled", f"left enabled for boot after {cmd_tag}:{code_used}"
                if code_used in ("124", "137"):
                    exp_status = "timeout"
                elif cmd_tag in ("check-disable", "check-enable", "stage-record-committed", "mv-record-committed"):
                    exp_status = "1"
                elif cmd_tag in ("scheduler-state", "dashboard-state"):
                    exp_status = "query"
                else:
                    exp_status = code_used
                expected_line = f"Check {expected_step} failed ({exp_status})."
                if expected_line not in res.stdout:
                    actual_check = [l for l in res.stdout.splitlines() if l.startswith("Check ")]
                    mismatches.append((cmd_tag, code_used, expected_line, actual_check))
    print(f"Phase-2/F3/F4 failure & timeout matrix ({len(matrix)*3} runs) mismatches: {mismatches}")

    # 2. Counterexample B1: Signal (SIGHUP / SIGTERM) near end of 800s Phase-2 budget (>=795s) after mutations!
    for sig_name, fault_hook in [
        ("HUP", "wait:advance-and-sig:HUP"),
        ("TERM", "wait:advance-and-sig:TERM"),
        ("TERM", "mv-record-committed:advance-and-sig:TERM"),
    ]:
        with tempfile.TemporaryDirectory() as td:
            box = make_box(Path(td))
            res, calls = run_script(box, deploy, fault=fault_hook)
            head_now = (box / "head").read_text()
            stopped = (box / "stopped").exists()
            hb = json.loads((box / "home/projects/bts/data/.heartbeat").read_text())
            en_state = (box / "enabled").read_text()
            stamp_exists = (box / "home/projects/bts/data/.last_deploy_iso").exists()
            posts = [c for c in calls if c[0] == "curl" and "--data-raw" in c]
            print(
                f"COUNTEREXAMPLE on_signal ({fault_hook}): rc={res.returncode} "
                f"head_mutated={head_now == TARGET} scheduler_stopped={stopped} "
                f"boot_link={en_state} stamp_written={stamp_exists} heartbeat={hb} "
                f"posts={len(posts)} stdout={res.stdout.strip()!r}"
            )

    # 3. Counterexample B2: Signal at t=0s (ample budget) when read-record encounters an error after mutation
    with tempfile.TemporaryDirectory() as td:
        box = make_box(Path(td))
        # Trigger SIGTERM during wait, and make read-record return 1 once after head is mutated
        # We can test this by corrupting attempt right when wait runs? No, under normal flow attempt is begun,
        # test read-record failing with 124 or 1 after mutation:
        res, calls = run_script(box, deploy, fault="read-record:1:after-mutated")
        # Note: read-record after-mutated runs at publish committed! Let's see what publish committed does:
        print(f"read-record error at publish committed: rc={res.returncode} stdout={res.stdout.splitlines()}")

    # 4. Closed stdout (EPIPE) on both passing deploy and failing deploy (containment must still complete!)
    with tempfile.TemporaryDirectory() as td:
        box = make_box(Path(td))
        res_pass, _ = run_script(box, deploy, closed_output=True)
        rec_pass = (box / "run/bts-deploy/attempt").read_text().strip()
    with tempfile.TemporaryDirectory() as td:
        box = make_box(Path(td))
        res_fail, calls_fail = run_script(box, deploy, fault="sync:17", closed_output=True)
        rec_fail = (box / "run/bts-deploy/attempt").read_text().strip()
        hb_fail = json.loads((box / "home/projects/bts/data/.heartbeat").read_text())
        posts_fail = [c for c in calls_fail if c[0] == "curl" and "--data-raw" in c]
    print(f"Closed stdout pass: rc={res_pass.returncode} record={rec_pass}")
    print(f"Closed stdout fail: rc={res_fail.returncode} record={rec_fail} hb_state={hb_fail['state']} posts={len(posts_fail)}")
    assert res_pass.returncode == 0 and " committed " in rec_pass
    assert res_fail.returncode == 1 and " contained sync verified verified verified" in rec_fail and len(posts_fail) == 1

    # 5. Second signal inside contain + preflight failure followed by recovery
    with tempfile.TemporaryDirectory() as td:
        box = make_box(Path(td))
        res2, calls2 = run_script(box, deploy, scenario="second-signal", fault="sync:17")
        stops = [c for c in calls2 if c[:3] == ["systemctl", "--user", "stop"]]
        assert res2.returncode == 1 and len(stops) == 1
        print("Second signal inside contain: single containment verified")

    with tempfile.TemporaryDirectory() as td:
        box = make_box(Path(td))
        r_pre, c_pre = run_script(box, deploy, scenario="fetch-fail")
        (box / "calls").unlink(missing_ok=True)
        r_rec, c_rec = run_script(box, recovery)
        assert r_pre.returncode == 1 and "DEPLOY NOT STARTED — fetch failed." in r_pre.stdout
        assert r_rec.returncode == 1 and "no phase-2 record on this boot" in r_rec.stdout
        assert not any(c[0] == "systemctl" for c in c_pre + c_rec)
        print("Preflight failure + recovery leaves services untouched: OK")

    # 6. Item 4: Success path order, atomic stamp + fsyncs, commit before PASS
    for leg in (False, True):
        with tempfile.TemporaryDirectory() as td:
            box = make_box(Path(td))
            scen = "legacy-present" if leg else "active"
            if leg:
                (box / "etc/systemd/system").mkdir(parents=True)
                (box / "etc/systemd/system/bts-scheduler.service").touch()
                (box / "etc/systemd/system/bts-dashboard.service").touch()
            r_base, calls_base = run_script(box, base_deploy, scenario=scen)
        with tempfile.TemporaryDirectory() as td:
            box = make_box(Path(td))
            if leg:
                (box / "etc/systemd/system").mkdir(parents=True)
                (box / "etc/systemd/system/bts-scheduler.service").touch()
                (box / "etc/systemd/system/bts-dashboard.service").touch()
            r_cand, calls_cand = run_script(box, deploy, scenario=scen)
            fsyncs = (box / "fsyncs").read_text().splitlines()
            # Verify stage -> enable -> stamp -> mv committed -> CANARY PASSED
            idx_stage = next(i for i, c in enumerate(calls_cand) if c[0] == "python3" and len(c) > 2 and c[2] == "stage")
            idx_enable = calls_cand.index(["systemctl", "--user", "enable", "bts-scheduler"])
            idx_stamp = next(i for i, c in enumerate(calls_cand) if c[0] == "python3" and len(c) > 2 and c[2] == "stamp")
            idx_commit = next(i for i, c in enumerate(calls_cand) if c[0] == "mv" and str(box / "run/bts-deploy/attempt") in c)
            idx_pass = next(i for i, c in enumerate(calls_cand) if c[0] == "bash" and any("CANARY PASSED" in a for a in c))
            assert idx_stage < idx_enable < idx_stamp < idx_commit < idx_pass
            assert fsyncs == ["reg", "dir"], f"unexpected fsync sequence: {fsyncs}"
            print(f"Success path (legacy={leg}): base={len(calls_base)} calls, cand={len(calls_cand)} calls, order idx={(idx_stage, idx_enable, idx_stamp, idx_commit, idx_pass)}, fsyncs={fsyncs}")


if __name__ == "__main__":
    main()
