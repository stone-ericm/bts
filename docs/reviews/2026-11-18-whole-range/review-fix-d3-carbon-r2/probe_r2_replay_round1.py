#!/usr/bin/env python3
"""Round-2 Item 1 probe: replay all Round-1 blocking findings (Codex 1-6 + R7, Carbon B1-B3 + N1/N2) on d57f4013."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import yaml

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github/workflows/deploy.yml"
PY311 = Path("/Users/eric/.local/share/uv/python/cpython-3.11.18-macos-aarch64-none/bin/python3.11")
OLD = "1" * 40
TARGET = "2" * 40

COMMANDS = {
    "bash", "bts", "cat", "curl", "cut", "date", "env", "flock", "git", "grep", "head",
    "id", "install", "mv", "pgrep", "pkill", "python3", "rm", "sleep", "stat", "sudo",
    "systemctl", "tailscale", "timeout", "uv",
}

SHIM_CODE = r'''
import fcntl, json, os, pathlib, signal, sys, time
from pathlib import Path

command = Path(sys.argv[0]).name
args = sys.argv[1:]
root = Path(os.environ["BOX"])
case = os.environ["CASE"]
with (root / "calls").open("a") as log:
    print(json.dumps([command, *args]), file=log)

def dispatch(words):
    if words[0].startswith("/") and words[0] != "/bin/bash" and not Path(words[0]).is_relative_to(root):
        raise SystemExit(99)
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
        if (root / "fail-clock-once").exists():
            (root / "fail-clock-once").unlink()
            (root / "clock-failed").touch()
            raise SystemExit(124)
        if (root / "fail-clock-always").exists():
            (root / "clock-failed").touch()
            raise SystemExit(124)
        off = int((root / "offset").read_text()) if (root / "offset").exists() else 0
        print(time.monotonic_ns() // 1_000_000 + off)
        raise SystemExit(0)
    if op == "read":
        if case == "recovery_clock_once" and not (root / "armed").exists():
            (root / "armed").touch()
            (root / "fail-clock-once").touch()
        if case == "recovery_clock_always" and not (root / "armed").exists():
            (root / "armed").touch()
            (root / "fail-clock-always").touch()
        if case in {"codex_read_126", "carbon_sigterm_read_err"} and (root / "fail-next-read").exists():
            (root / "fail-next-read").unlink()
            raise SystemExit(126)
        if case in {"commit_read_124", "commit_read_137"}:
            rec = root / "run/bts-deploy/attempt"
            if rec.exists() and " committed " in rec.read_text():
                raise SystemExit(int(case.rsplit("_", 1)[1]))
    if op == "stage-record" and len(args) > 4 and " committed " in args[4]:
        if case in {"commit_stage_17", "commit_stage_124", "commit_stage_137"}:
            raise SystemExit(int(case.rsplit("_", 1)[1]))
        if case == "commit_shared_budget_expired":
            (root / "offset").write_text("801000")
            raise SystemExit(124)
    orig_fstat = os.fstat
    def root_fstat(fd):
        st = orig_fstat(fd)
        return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, 0, 0, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))
    if op in {"lockfile", "mark"}:
        os.fstat = root_fstat
    exec(compile(code, "<fs_code>", "exec"), {"__name__": "__main__"})
    if op == "stamp" and case == "carbon_sigterm_after_stamp_795s":
        (root / "offset").write_text("795000")
        os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)
elif command == "mv":
    if Path(args[-2]).name.startswith(".attempt.") and " committed " in Path(args[-2]).read_text():
        if case in {"commit_mv_17", "commit_mv_124", "commit_mv_137"}:
            raise SystemExit(int(case.rsplit("_", 1)[1]))
    os.replace(args[-2], args[-1])
elif command == "git":
    branch = root / "head"
    if args[0] == "rev-parse":
        if branch.read_text() == os.environ["TARGET_SHA"] and case in {"deploy_clock_once", "deploy_clock_always", "cgroup_blank_after_stop", "cgroup_always_blank", "glob_url"}:
            if case == "deploy_clock_once":
                (root / "fail-clock-once").touch()
            elif case == "deploy_clock_always":
                (root / "fail-clock-always").touch()
            raise SystemExit(17)
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
        if user and "ExecMainStatus" in args and case in {"codex_late_signal", "codex_read_126"}:
            if case == "codex_late_signal":
                (root / "offset").write_text("805000")
            else:
                (root / "fail-next-read").touch()
            os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)
        active = "active" if user and not (root / "stopped").exists() else "inactive"
        loaded = "loaded" if user else "not-found"
        cg = "/user-sched" if user else ""
        if user and case == "cgroup_blank_after_stop":
            cg = "" if (root / "stopped").exists() else "/populated-sched"
        elif user and case == "cgroup_always_blank":
            cg = ""
        states = dict(Id=args[1], LoadState=loaded, UnitFileState="enabled",
                      ActiveState=active, SubState="running", Result="success",
                      ExecMainStatus="0", MainPID="0", ControlGroup=cg)
        for k in (a for a in args if a in states):
            print(f"{k}={states[k]}")
    elif user and args[0] == "stop":
        (root / "stopped").touch()
elif command == "sleep":
    if case == "codex_late_signal":
        (root / "offset").write_text("790000")
    elif case == "carbon_sighup_795s_wait":
        (root / "offset").write_text("795000")
        os.kill(int((root / "owner.pid").read_text()), signal.SIGHUP)
    elif case == "carbon_sigterm_read_err":
        (root / "fail-next-read").touch()
        os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)
elif command == "tailscale":
    print("100.64.0.1")
elif command == "curl":
    print("200", end="")
elif command == "bash" and args[0] == "-c" and args[1].startswith("printf "):
    print(*args[3:], sep="\n")
elif command == "bts":
    if case == "codex_nul_in_prefix":
        sys.stdout.buffer.write(b"Deploy canary: not a li\x00ve configuration (delivery private); contest state not checked.\n")
        raise SystemExit(0)
    print("Deploy canary: not a live configuration (delivery private); contest state not checked.")
elif command in {"pgrep", "pkill"}:
    raise SystemExit(1)
elif command == "rm":
    for op in (a for a in args if not a.startswith("-")):
        Path(op).unlink(missing_ok=True)
elif command == "uv":
    pass
'''


def run_box(script: str, case: str, *, env_url: str = "https://checks.invalid/a",
            is_recovery: bool = False) -> dict:
    with tempfile.TemporaryDirectory() as td:
        box = Path(td)
        shim_dir = box / "shims"
        shim_dir.mkdir()
        for name in COMMANDS:
            p = shim_dir / name
            p.write_text(f"#!{sys.executable}\n" + SHIM_CODE)
            p.chmod(0o700)
        (box / "home/projects/bts/data").mkdir(parents=True)
        (box / "home/projects/bts/.env").write_text(f"BTS_SCHEDULER_HEARTBEAT_PING_URL={env_url}\n")
        (box / "sys/fs/cgroup/populated-sched").mkdir(parents=True)
        (box / "sys/fs/cgroup/populated-sched/cgroup.events").write_text("populated 1\nfrozen 0\n")
        (box / "sys/fs/cgroup/cgroup.controllers").touch()
        (box / "head").write_text(OLD)
        if is_recovery:
            ld = box / "run/bts-deploy"
            ld.mkdir(mode=0o700, parents=True)
            (ld / "attempt").write_text(f"bts_deploy_attempt_v1 555 1 {TARGET} {OLD} begun - - - -\n")

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
        hb = box / "home/projects/bts/data/.heartbeat"
        return {
            "case": case,
            "rc": proc.returncode,
            "stopped": (box / "stopped").exists(),
            "enabled": (box / "enabled").read_text() if (box / "enabled").exists() else "enabled",
            "heartbeat": json.loads(hb.read_text())["state"] if hb.exists() else None,
            "attempt": (box / "run/bts-deploy/attempt").read_text().strip(),
            "curls": [c for c in calls if c[0] == "curl"],
            "clock_failed": (box / "clock-failed").exists(),
            "stdout": out.strip().splitlines(),
        }


def main() -> None:
    wf = yaml.safe_load(WORKFLOW.read_text())
    deploy_s = wf["jobs"]["deploy"]["steps"][3]["with"]["script"]
    recov_s = wf["jobs"]["deploy"]["steps"][4]["with"]["script"]

    # Check RD8 byte-identity on d57f4013
    assert deploy_s.split("\nnot_started() {", 1)[0] == recov_s.split("\nrecovery_exit() {", 1)[0]

    # Check Python 3.11 under LC_ALL=C with updated fs_code (contains UTF-8 em-dash in capture-contest)
    m = re.search(r"fs_code='\n(.*?)\n'", deploy_s, re.DOTALL)
    assert m
    r311 = subprocess.run(
        ["/usr/bin/env", "-i", "LC_ALL=C", str(PY311), "-c", m.group(1),
         "capture-contest", "4096", str(PY311), "-c", 'import sys; print("Deploy canary: REFUSED \u2014 bad"); sys.exit(1)'],
        capture_output=True, text=True, timeout=5,
    )
    assert r311.returncode == 1 and r311.stdout == "Deploy canary: REFUSED \u2014 bad\n", r311
    print("Python 3.11 LC_ALL=C capture-contest with UTF-8 em-dash: OK")

    # R1: Codex 1 (clock failure once and always, in deploy and recovery)
    for c, rec in [("deploy_clock_once", False), ("deploy_clock_always", False),
                   ("recovery_clock_once", True), ("recovery_clock_always", True)]:
        res = run_box(recov_s if rec else deploy_s, c, is_recovery=rec)
        assert res["rc"] == 1 and res["clock_failed"] and res["stopped"]
        assert res["heartbeat"] == "deploy_failed" and " contained " in res["attempt"]
        assert any("--data-raw" in k and "--globoff" in k for k in res["curls"])
    print("R1 (Codex 1 — containment clock failure once & always in deploy & recovery): FIXED")

    # R2: Codex 2 + Carbon B1 (late signal & post-begun read error)
    for c in ("codex_late_signal", "codex_read_126", "carbon_sighup_795s_wait",
              "carbon_sigterm_after_stamp_795s", "carbon_sigterm_read_err"):
        res = run_box(deploy_s, c)
        assert res["rc"] == 1 and res["stopped"] and res["enabled"] == "disabled", res
        assert res["heartbeat"] == "deploy_failed" and " contained interrupted " in res["attempt"]
        assert not any("DEPLOY NOT STARTED" in l for l in res["stdout"])
        assert any("DEPLOY FAILED at interrupted" in l for l in res["stdout"])
    print("R2 (Codex 2 / Carbon B1 — original late-signal & post-begun read-failure inputs): FIXED")

    # R3: Codex 3 (blank ControlGroup after stop with populated subtree, and always-blank ControlGroup)
    for c in ("cgroup_blank_after_stop", "cgroup_always_blank"):
        res = run_box(deploy_s, c)
        assert "Scheduler stopped: unverified." in res["stdout"], res
    print("R3 (Codex 3 — blank/unknown ControlGroup with populated subtree): FIXED")

    # R4: Codex 4 + Carbon B3 (curl globs & --globoff)
    for bad_url in ("https://checks.invalid/{other}", "https://checks.invalid/{one,two}", "https://hc-ping.invalid/[1-1]/uuid"):
        res = run_box(deploy_s, "glob_url", env_url=bad_url)
        assert not res["curls"] and "Failure ping received HTTP 2xx: unverified." in res["stdout"], (bad_url, res)
    print("R4 (Codex 4 / Carbon B3 — URL glob rejection & --globoff): FIXED")

    # R5: Codex 5 + Carbon B2 (commit-record status preservation for 17, 124, 137, read timeout, shared budget)
    for c, exp in [
        ("commit_stage_17", "17"), ("commit_mv_17", "17"),
        ("commit_stage_124", "timeout"), ("commit_stage_137", "timeout"),
        ("commit_mv_124", "timeout"), ("commit_mv_137", "timeout"),
        ("commit_read_124", "timeout"), ("commit_read_137", "timeout"),
        ("commit_shared_budget_expired", "timeout"),
    ]:
        res = run_box(deploy_s, c)
        assert f"Check commit-record failed ({exp})." in res["stdout"], (c, res["stdout"])
        assert " contained commit-record " in res["attempt"]
    print("R5 (Codex 5 / Carbon B2 — commit-record numeric & timeout statuses): FIXED")

    # R7: Codex note (embedded NUL inside valid canary prefix)
    res_nul = run_box(deploy_s, "codex_nul_in_prefix")
    assert res_nul["rc"] == 1 and "Check contest-state failed (query)." in res_nul["stdout"]
    print("R7 (Codex NUL byte inside canary prefix): FIXED")


if __name__ == "__main__":
    main()
