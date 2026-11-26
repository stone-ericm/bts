"""Draft 4 Bash owner model, with actual signals and local flock. Never calls systemd."""

import fcntl
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from review_probe import base_replay


HERE = Path(__file__).resolve().parent
TOP = HERE.parents[1]
HANDOVER = """set -eu
systemd-run --unit=bts-deploy-fixture --collect --property=RuntimeMaxSec=1380 \\
  --property=TimeoutStopSec=120 --property=KillMode=mixed /bin/bash "$CASE/phase"
exec follow
"""
PHASE = """set -eu
trap 'printf failed > "$CASE/hold"; printf TERM > "$CASE/trap"; result failed; exit 1' TERM
result() {
    printf '%s\\n' "$1" > "$CASE/result.tmp"
    /bin/mv "$CASE/result.tmp" "$CASE/result"
}
point() {
    : > "$CASE/reached-$1"
    if [ "$GATE" = "$1" ]; then gate; fi
}
point preflight
printf pending > "$CASE/hold"
point armed
point reset
point candidate
if [ "$MODE" = reject ]; then
    printf failed > "$CASE/hold"
    result failed
    exit 1
fi
/bin/rm "$CASE/hold"
point released
: > "$CASE/started"
point activation
result committed
"""


def wait_until(predicate, limit=3):
    end = time.monotonic() + limit
    while not predicate():
        if time.monotonic() >= end:
            raise TimeoutError("bounded fixture wait expired")
        time.sleep(0.01)


def sysrun(args):
    assert args[:5] == ["--unit=bts-deploy-fixture", "--collect", "--property=RuntimeMaxSec=1380",
                        "--property=TimeoutStopSec=120", "--property=KillMode=mixed"], args
    assert args[5] == "/bin/bash"
    subprocess.Popen([sys.executable, __file__, "--supervise", *args[5:]],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    return 0


def is_busy(path):
    with path.open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def supervise(root, mode, command):
    with (root / "transaction.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root / "owner.log").open("wb") as log:
            owner = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True, pass_fds=(lock.fileno(),))
    events = []
    try:
        if mode in {"term-short", "term-stuck", "kill-main", "kill-post-release"}:
            wait_until((root / "gate-ready").exists)
            if mode in {"term-short", "term-stuck"}:
                owner.send_signal(signal.SIGTERM)
                events.append("TERM main")
                grace = 0.35
                if mode == "term-short":
                    time.sleep(0.08)
                    (root / "continue").touch()
                try:
                    owner.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    os.killpg(owner.pid, signal.SIGKILL)
                    events.append("KILL group")
            elif mode == "kill-main":
                owner.kill()
                owner.wait(timeout=1)
                events.append(f"after KILL main, inherited lock busy={is_busy(root / 'transaction.lock')}")
                (root / "continue").touch()
                wait_until(lambda: not is_busy(root / "transaction.lock"))
            else:
                os.killpg(owner.pid, signal.SIGKILL)
                events.append("KILL group after release")
        rc = owner.wait(timeout=3)
        wait_until(lambda: not is_busy(root / "transaction.lock"))
    except BaseException:
        try:
            os.killpg(owner.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        owner.wait(timeout=2)
        raise
    (root / "supervisor-result").write_text(json.dumps({"events": events, "owner_rc": rc}))
    (root / "supervisor-done").touch()
    return 0


def gate(root):
    (root / "gate-ready").touch()
    wait_until((root / "continue").exists, limit=2)
    return 0


def follow(root):
    (root / "following").touch()
    wait_until((root / "supervisor-done").exists, limit=4)
    path = root / "result"
    return 0 if path.exists() and path.read_text().strip() == "committed" else 1


def run_case(root, mode, point, disconnect=False):
    root.mkdir()
    (root / "transaction.lock").touch()
    (root / "phase").write_text(PHASE)
    (root / "handover").write_text(HANDOVER)
    shim_dir = root / "bin"
    shim_dir.mkdir()
    for name in ("systemd-run", "follow", "gate"):
        shim = shim_dir / name
        shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{__file__}" --{name} "$@"\n')
        shim.chmod(0o700)
    env = {"PATH": str(shim_dir), "CASE": str(root), "MODE": mode, "GATE": point,
           "PYTHONDONTWRITEBYTECODE": "1"}
    ssh = subprocess.Popen(["/bin/bash", str(root / "handover")], env=env,
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if disconnect:
            wait_until(lambda: (root / "following").exists() and (root / "gate-ready").exists())
            ssh.send_signal(signal.SIGHUP)
            assert ssh.wait(timeout=1) == -signal.SIGHUP
            assert is_busy(root / "transaction.lock")
            (root / "continue").touch()
        ssh_rc = ssh.wait(timeout=5)
        wait_until((root / "supervisor-done").exists)
    finally:
        if ssh.poll() is None:
            ssh.kill()
            ssh.wait(timeout=1)
    outcome = json.loads((root / "supervisor-result").read_text())
    outcome.update({"ssh_rc": ssh_rc, "lock_busy_after": is_busy(root / "transaction.lock"),
                    "trap_ran": (root / "trap").exists(), "started": (root / "started").exists()})
    for field in ("hold", "result"):
        p = root / field
        outcome[field] = p.read_text().strip() if p.exists() else None
    return outcome


def current_tree_checks(root):
    from importlib.machinery import PathFinder

    assert PathFinder.find_spec("bts.predict", [str(TOP / "src/bts")]) is None
    assert PathFinder.find_spec("bts.model.predict", [str(TOP / "src/bts/model")]) is not None
    spec = importlib.util.spec_from_file_location("heartbeat_monitor", TOP / "scripts/check_heartbeat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    heartbeat = root / ".heartbeat"
    heartbeat.write_text(json.dumps({"timestamp": "2027-07-06T12:00:00+00:00", "state": "sleeping",
                                     "sleeping_until": "2027-07-06T20:00:00+00:00"}))
    verdict = module.is_stale(heartbeat, datetime(2027, 7, 6, 18, tzinfo=timezone.utc))
    assert verdict == (False, "fresh sleeping")
    # G5 cannot distinguish a hand-release/preflight or merely another FD holder from an owner by flock alone.
    lock_path = root / "foreign.lock"
    lock_path.touch()
    with lock_path.open("rb") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert is_busy(lock_path)
    return {"draft_import_found": False, "actual_module_found": True, "after_release_old_heartbeat": verdict,
            "anonymous_exclusive_holder_passes_flock_only_liveness_test": True}


def main():
    if len(sys.argv) > 1:
        root = Path(os.environ["CASE"])
        command = sys.argv[1]
        if command == "--systemd-run":
            return sysrun(sys.argv[2:])
        if command == "--supervise":
            return supervise(root, os.environ["MODE"], sys.argv[2:])
        if command == "--gate":
            return gate(root)
        if command == "--follow":
            return follow(root)
        raise AssertionError(command)
    with TemporaryDirectory(prefix="r4-owner-", dir=HERE) as tmp:
        root = Path(tmp)
        base_root = root / "base"
        base_root.mkdir()
        base = base_replay(base_root)
        cases = {point: run_case(root / point, "normal", point, disconnect=True)
                 for point in ("preflight", "armed", "reset", "candidate", "released", "activation")}
        assert all(case["result"] == "committed" and case["started"] for case in cases.values()), cases
        cases["rejected"] = run_case(root / "rejected", "reject", "candidate", disconnect=True)
        assert cases["rejected"]["hold"] == "failed" and not cases["rejected"]["started"], cases
        for mode in ("term-short", "term-stuck", "kill-main", "kill-post-release"):
            cases[mode] = run_case(root / mode, mode, "released" if mode == "kill-post-release" else "reset")
        assert cases["term-short"]["trap_ran"] and cases["term-short"]["result"] == "failed", cases
        assert not cases["term-stuck"]["trap_ran"] and cases["term-stuck"]["hold"] == "pending", cases
        assert cases["term-stuck"]["events"] == ["TERM main", "KILL group"], cases
        assert cases["kill-main"]["events"] == ["after KILL main, inherited lock busy=True"], cases
        post = cases["kill-post-release"]
        assert post["hold"] is None and post["result"] is None and not post["started"], post
        assert all(not case["lock_busy_after"] for case in cases.values()), cases
        result = {"exact_base_replay": {name: case["rc"] for name, case in base.items()}, "owner_model": cases,
                  "current_tree": current_tree_checks(root)}
    (HERE / "owner-probe.log").write_text(json.dumps(result, indent=2) + "\n")
    print("Passed 6 base shell cases, 11 draft-owner cases and current-tree checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
