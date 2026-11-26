#!/usr/bin/env python3
"""Round-2 Item 2 probe: test new on_signal publication states, fallback clock slices, and systemd inactive cgroup behavior."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import yaml

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github/workflows/deploy.yml"
OLD = "1" * 40
TARGET = "2" * 40
PRIOR_TARGET = "9" * 40
PRIOR_OLD = "8" * 40

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
        if case == "slow_failing_contain_clock" and (root / "in-contain").exists():
            time.sleep(1.1)
            raise SystemExit(124)
        print(time.monotonic_ns() // 1_000_000)
        raise SystemExit(0)
    if op == "stage-record" and len(args) > 4 and " begun " in args[4]:
        if case in {"sig_during_beginning_with_prior_committed", "sig_during_beginning_rerun_after_contained", "sig_during_beginning_fresh_boot"}:
            os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)
            raise SystemExit(143)
    orig_fstat = os.fstat
    def root_fstat(fd):
        st = orig_fstat(fd)
        return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, 0, 0, st.st_size, st.st_atime, st.st_mtime, st.st_ctime))
    if op in {"lockfile", "mark"}:
        os.fstat = root_fstat
    exec(compile(code, "<fs_code>", "exec"), {"__name__": "__main__"})
elif command == "mv":
    if Path(args[-2]).name.startswith(".attempt."):
        txt = Path(args[-2]).read_text()
        if " begun " in txt and case == "sig_during_beginning_after_rename":
            os.replace(args[-2], args[-1])
            os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)
            raise SystemExit(143)
        if " committed " in txt and case == "sig_during_committing_unpublished":
            os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)
            raise SystemExit(143)
        if " committed " in txt and case == "sig_during_committing_published":
            os.replace(args[-2], args[-1])
            os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)
            raise SystemExit(143)
    os.replace(args[-2], args[-1])
elif command == "git":
    branch = root / "head"
    if args[0] == "rev-parse":
        if branch.read_text() == os.environ["TARGET_SHA"] and case in {"slow_failing_contain_clock", "debian_pruned_cgroup"}:
            (root / "in-contain").touch()
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
        active = "active" if user and not (root / "stopped").exists() else "inactive"
        loaded = "loaded" if user else "not-found"
        if user:
            if case == "debian_pruned_cgroup":
                # Real Debian systemd: active unit has ControlGroup=/user-sched; inactive unit has ControlGroup=""
                cg = "/user-sched" if active == "active" else ""
            else:
                cg = "/user-sched"
        else:
            cg = ""
        states = dict(Id=args[1], LoadState=loaded, UnitFileState="enabled",
                      ActiveState=active, SubState="running", Result="success",
                      ExecMainStatus="0", MainPID="0", ControlGroup=cg)
        for k in (a for a in args if a in states):
            print(f"{k}={states[k]}")
    elif user and args[0] == "stop":
        (root / "stopped").touch()
elif command == "sleep":
    pass
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


def setup_box(tmp: Path, *, prior_record: str | None = None, head_sha: str = OLD) -> Path:
    shim_dir = tmp / "shims"
    shim_dir.mkdir()
    for name in COMMANDS:
        p = shim_dir / name
        p.write_text(f"#!{sys.executable}\n" + SHIM_CODE)
        p.chmod(0o700)
    (tmp / "home/projects/bts/data").mkdir(parents=True)
    (tmp / "home/projects/bts/data/.heartbeat").write_text('{"state":"running"}')
    (tmp / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/u\n")
    (tmp / "sys/fs/cgroup").mkdir(parents=True)
    (tmp / "sys/fs/cgroup/cgroup.controllers").touch()
    (tmp / "head").write_text(head_sha)
    (tmp / "enabled").write_text("enabled")
    if prior_record is not None:
        ld = tmp / "run/bts-deploy"
        ld.mkdir(mode=0o700, parents=True)
        (ld / "attempt").write_text(prior_record)
    return tmp


def invoke(box: Path, script: str, case: str, *, run_id: str = "200", attempt: str = "1") -> dict:
    (box / "calls").unlink(missing_ok=True)
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
        "/usr/bin/env", "-i", f"PATH={box / 'shims'}", f"BOX={box}", f"CASE={case}",
        f"TARGET_SHA={TARGET}", f"RUN_ID={run_id}", f"RUN_ATTEMPT={attempt}", "LC_ALL=C",
        "/bin/bash", "-c", s,
    ]
    t0 = time.monotonic()
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
    elapsed = round(time.monotonic() - t0, 2)
    calls = [json.loads(l) for l in (box / "calls").read_text().splitlines()] if (box / "calls").exists() else []
    att = box / "run/bts-deploy/attempt"
    return {
        "case": case,
        "rc": proc.returncode,
        "elapsed_s": elapsed,
        "head": (box / "head").read_text(),
        "stopped": (box / "stopped").exists(),
        "enabled": (box / "enabled").read_text(),
        "heartbeat": json.loads((box / "home/projects/bts/data/.heartbeat").read_text()),
        "attempt": att.read_text().strip() if att.exists() else None,
        "posts": len([c for c in calls if c[0] == "curl" and "--data-raw" in c]),
        "stdout": out.strip().splitlines(),
    }


def main() -> None:
    wf = yaml.safe_load(WORKFLOW.read_text())
    deploy_s = wf["jobs"]["deploy"]["steps"][3]["with"]["script"]
    recov_s = wf["jobs"]["deploy"]["steps"][4]["with"]["script"]

    # 1. COUNTEREXAMPLE: Signal during publication=beginning when prior deploy record exists in /run/bts-deploy/attempt!
    with tempfile.TemporaryDirectory() as td:
        prior = f"bts_deploy_attempt_v1 100 1 {PRIOR_TARGET} {PRIOR_OLD} committed - - - -\n"
        box = setup_box(Path(td), prior_record=prior, head_sha=PRIOR_TARGET)
        res_prior = invoke(box, deploy_s, "sig_during_beginning_with_prior_committed", run_id="200", attempt="1")
        print("COUNTEREXAMPLE (prior committed record + signal before begun rename):")
        print(json.dumps(res_prior, indent=2))

    # 2. Re-run (run_id=100, attempt=2) interrupted during publication=beginning after attempt 1 was contained:
    with tempfile.TemporaryDirectory() as td:
        prior = f"bts_deploy_attempt_v1 100 1 {TARGET} {OLD} contained sync verified verified verified\n"
        box = setup_box(Path(td), prior_record=prior, head_sha=TARGET)
        (box / "stopped").touch()
        (box / "enabled").write_text("disabled")
        res_rerun = invoke(box, deploy_s, "sig_during_beginning_rerun_after_contained", run_id="100", attempt="2")
        print("COUNTEREXAMPLE (re-run attempt=2 + signal before begun rename):")
        print(json.dumps(res_rerun, indent=2))

    # 3. Fresh-boot control (attempt absent, rc==3):
    with tempfile.TemporaryDirectory() as td:
        box = setup_box(Path(td), prior_record=None, head_sha=PRIOR_TARGET)
        res_fresh = invoke(box, deploy_s, "sig_during_beginning_fresh_boot", run_id="200", attempt="1")
        assert res_fresh["rc"] == 1 and not res_fresh["stopped"] and res_fresh["enabled"] == "enabled"
        assert any("DEPLOY NOT STARTED — attempt-record failed." in l for l in res_fresh["stdout"])
        print("Control (fresh boot, no prior record): DEPLOY NOT STARTED without touching services -> OK")

    # 4. Other publication boundaries (beginning after rename, committing before rename, committing after rename)
    for case, exp_rc, exp_stopped in [
        ("sig_during_beginning_after_rename", 1, True),
        ("sig_during_committing_unpublished", 1, True),
        ("sig_during_committing_published", 0, False),
    ]:
        with tempfile.TemporaryDirectory() as td:
            box = setup_box(Path(td))
            r = invoke(box, deploy_s, case)
            assert r["rc"] == exp_rc and r["stopped"] == exp_stopped, (case, r)
    print("Other publication boundaries (renamed begun, unpublished commit, published commit): OK")

    # 5. Slow failing clock in contain (1.1s sleep + exit 124) -> SECONDS fallback finishes cleanly with all sub-steps
    with tempfile.TemporaryDirectory() as td:
        box = setup_box(Path(td))
        r_clk = invoke(box, deploy_s, "slow_failing_contain_clock")
        assert r_clk["rc"] == 1 and r_clk["stopped"] and r_clk["posts"] == 1
        assert r_clk["heartbeat"]["state"] == "deploy_failed" and " contained head " in r_clk["attempt"]
        print(f"Slow failing clock in contain -> SECONDS fallback completed all sub-steps in {r_clk['elapsed_s']}s: OK")

    # 6. Real Debian systemd behavior (inactive unit has ControlGroup="") across deploy -> recovery
    with tempfile.TemporaryDirectory() as td:
        box = setup_box(Path(td))
        r_dep = invoke(box, deploy_s, "debian_pruned_cgroup", run_id="200", attempt="1")
        r_rec = invoke(box, recov_s, "debian_pruned_cgroup", run_id="200", attempt="1")
        print("Debian pruned ControlGroup='' on already-stopped unit:")
        print("  deploy contain:", [l for l in r_dep["stdout"] if "Scheduler stopped:" in l], "record:", r_dep["attempt"])
        print("  recovery contain:", [l for l in r_rec["stdout"] if "Scheduler stopped:" in l or "DEPLOY RECOVERY" in l], "record:", r_rec["attempt"])


if __name__ == "__main__":
    main()
