#!/usr/bin/env python3
"""Targeted demonstration of concrete workflow counterexamples (6 runs total, <30s)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github/workflows/deploy.yml"
OLD = "1" * 40
TARGET = "2" * 40

COMMANDS = {
    "bash", "bts", "cat", "curl", "cut", "date", "env", "flock", "git", "grep", "head",
    "id", "install", "mv", "pgrep", "pkill", "python3", "rm", "sleep", "stat", "sudo",
    "systemctl", "tailscale", "timeout", "uv",
}

SHIM_CODE = r'''
import fcntl, json, os, signal, sys, time
from pathlib import Path

command = Path(sys.argv[0]).name
args = sys.argv[1:]
root = Path(os.environ["BOX"])
case = os.environ["CASE"]
with (root / "calls").open("a") as log:
    print(json.dumps([command, *args]), file=log)

def dispatch(words):
    prog = Path(words[0]).name
    shim = root / "shims" / prog
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
    fcntl.flock(int(args[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
elif command == "python3":
    op = args[2] if len(args) > 2 else None
    code = args[1]
    sys.argv = ["-c", *args[2:]]
    if op == "clock":
        off = int((root / "offset").read_text()) if (root / "offset").exists() else 0
        print(time.monotonic_ns() // 1_000_000 + off)
        raise SystemExit(0)
    if op == "stage-record" and len(args) > 4 and " committed " in args[4]:
        if case == "commit_stage_timeout":
            raise SystemExit(124)
        if case == "commit_shared_budget_expired":
            (root / "offset").write_text("801000")
            raise SystemExit(124)
    if op == "read" and (root / "fail-next-read").exists():
        (root / "fail-next-read").unlink()
        raise SystemExit(1)
    orig_fstat = os.fstat
    def root_fstat(fd):
        st = orig_fstat(fd)
        return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, 0, 0, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))
    if op in {"lockfile", "mark"}:
        os.fstat = root_fstat
    exec(compile(code, "<fs_code>", "exec"), {"__name__": "__main__"})
elif command == "mv":
    if Path(args[-2]).name.startswith(".attempt.") and " committed " in Path(args[-2]).read_text():
        if case == "commit_mv_timeout_unpublished":
            raise SystemExit(124)
        if case == "sigterm_after_enable_and_stamp_at_795s":
            (root / "offset").write_text("795000")
            os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)
            raise SystemExit(124)
    os.replace(args[-2], args[-1])
elif command == "git":
    branch = root / "head"
    if args[0] == "rev-parse":
        print(branch.read_text())
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
    elif user and args[0] == "restart":
        (root / "stopped").unlink(missing_ok=True)
        (root / "home/projects/bts/data/.heartbeat").write_text('{"state":"running"}')
    elif args[0] == "is-enabled":
        en = (root / "enabled").read_text() if (root / "enabled").exists() else "enabled"
        print(en)
        raise SystemExit(1 if en == "disabled" else 0)
    elif args[0] == "show":
        active = "active" if user and not (root / "stopped").exists() else "inactive"
        loaded = "loaded" if user else "not-found"
        states = dict(Id=args[1], LoadState=loaded, UnitFileState="enabled",
                      ActiveState=active, SubState="running", Result="success",
                      ExecMainStatus="0", MainPID="0", ControlGroup="")
        for k in (a for a in args if a in states):
            print(f"{k}={states[k]}")
    elif user and args[0] == "stop":
        (root / "stopped").touch()
elif command == "sleep":
    if case == "sighup_at_795s_during_wait":
        (root / "offset").write_text("795000")
        os.kill(int((root / "owner.pid").read_text()), signal.SIGHUP)
    elif case == "sigterm_with_read_error_after_restart":
        (root / "fail-next-read").touch()
        os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)
elif command == "tailscale":
    print("100.64.0.1")
elif command == "curl":
    print("200", end="")
elif command == "bash" and args[0] == "-c" and args[1].startswith("printf "):
    print(*args[3:], sep="\n")
elif command == "bts":
    print("Deploy canary: not a live configuration (delivery private); contest state not checked.")
elif command in {"pgrep", "pkill"}:
    raise SystemExit(1)
elif command == "rm":
    for op in (a for a in args if not a.startswith("-")):
        Path(op).unlink(missing_ok=True)
elif command == "uv":
    pass
'''


def run_case(script: str, case: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        box = Path(td)
        shim_dir = box / "shims"
        shim_dir.mkdir()
        for name in COMMANDS:
            p = shim_dir / name
            p.write_text(f"#!{sys.executable}\n" + SHIM_CODE)
            p.chmod(0o700)
        (box / "home/projects/bts/data").mkdir(parents=True)
        (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc.invalid/u\n")
        (box / "sys/fs/cgroup").mkdir(parents=True)
        (box / "sys/fs/cgroup/cgroup.controllers").touch()
        (box / "head").write_text(OLD)

        s = script
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
            s = s.replace(absolute, str(box / rel))

        launcher = (
            "import os, signal, sys\n"
            "for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGPIPE):\n"
            "    signal.signal(sig, signal.SIG_DFL)\n"
            "os.execv(sys.argv[1], sys.argv[1:])\n"
        )
        argv = [
            "/usr/bin/env", "-i", f"PATH={shim_dir}", f"BOX={box}", f"CASE={case}",
            f"TARGET_SHA={TARGET}", "RUN_ID=555", "RUN_ATTEMPT=1", "LC_ALL=C",
            "/bin/bash", "-c", s,
        ]
        proc = subprocess.Popen([sys.executable, "-c", launcher, *argv],
                                cwd=box, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with proc:
            (box / "owner.pid").write_text(str(proc.pid))
            try:
                out, err = proc.communicate(timeout=45)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate(timeout=2)
                raise
        calls = [json.loads(l) for l in (box / "calls").read_text().splitlines()]
        hb_path = box / "home/projects/bts/data/.heartbeat"
        return {
            "case": case,
            "rc": proc.returncode,
            "head_is_target": (box / "head").read_text() == TARGET,
            "scheduler_stopped": (box / "stopped").exists(),
            "boot_link": (box / "enabled").read_text() if (box / "enabled").exists() else "enabled",
            "stamp_written": (box / "home/projects/bts/data/.last_deploy_iso").exists(),
            "heartbeat": json.loads(hb_path.read_text()) if hb_path.exists() else None,
            "attempt_record": (box / "run/bts-deploy/attempt").read_text().strip(),
            "posts": len([c for c in calls if c[0] == "curl" and "--data-raw" in c]),
            "stdout_lines": out.strip().splitlines(),
        }


def main() -> None:
    wf = yaml.safe_load(WORKFLOW.read_text())
    deploy_script = wf["jobs"]["deploy"]["steps"][3]["with"]["script"]
    for case in (
        "sighup_at_795s_during_wait",
        "sigterm_after_enable_and_stamp_at_795s",
        "sigterm_with_read_error_after_restart",
        "commit_stage_timeout",
        "commit_mv_timeout_unpublished",
        "commit_shared_budget_expired",
    ):
        res = run_case(deploy_script, case)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
