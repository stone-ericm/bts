"""Offline contract probes. Only the extracted base is executable workflow code."""

import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = "619abda7" + "0" * 32
TARGET = "41952802" + "0" * 32


def workflow():
    raw = (REPO / ".github/workflows/deploy.yml").read_bytes()
    starts = list(re.finditer(rb"^          script: \|\n", raw, re.M))
    assert len(starts) == 1
    body = raw[starts[0].end():]
    lines = body.splitlines(keepends=True)
    assert all(line == b"\n" or line.startswith(b" " * 12) for line in lines)
    payload = b"".join(line[12:] if line.startswith(b" " * 12) else line for line in lines)
    assert b"".join(b" " * 12 + line if line != b"\n" else line
                    for line in payload.splitlines(keepends=True)) == body
    return payload


def shim():
    name, *args = sys.argv[2:]
    name = Path(name).name
    root = Path(os.environ["WRD3_CASE"])
    scenario = os.environ["WRD3_SCENARIO"]
    head = root / "head"
    with (root / "calls").open("a") as log:
        log.write(json.dumps([name, *args]) + "\n")
    if name == "id":
        print("1000")
    elif name == "sleep":
        pass
    elif name == "curl":
        print("503" if scenario in ("rollback", "broken-rollback") else "200")
        return 28 if scenario == "curl-28" else 0
    elif name == "sudo":
        if "git" in args and "rev-parse" in args:
            print(head.read_text()[:7] if "--short" in args else head.read_text())
        elif "git" in args and "reset" in args:
            if scenario == "broken-rollback" and args[-1] == OLD:
                return 1
            head.write_text(args[-1])
        elif "git" in args or "tailscale" in args:
            pass
        elif "sync" in args:
            return int(scenario == "sync-error")
        elif "systemctl" in args and "restart" in args:
            return int(scenario == "restart-error")
        elif "systemctl" in args and "is-active" in args:
            if "--quiet" not in args:
                print("active")
        elif "bash" in args and ".last_deploy_iso" in args[-1]:
            (root / "stamped").touch()
        else:
            raise AssertionError(args)
    else:
        raise AssertionError(name)
    return 0


def base_replay(root):
    script = root / "base.sh"
    script.write_bytes(workflow())
    env_script = root / "bash_env"
    env_script.write_text('cd() { builtin cd "$WRD3_CASE"; }\n'
                          '[() {\n'
                          '  if [[ "$2" == /etc/systemd/system/bts-scheduler.service ]]; then\n'
                          '    return 1\n'
                          '  fi\n'
                          '  builtin [ "$@"\n}\n')
    shims = root / "bin"
    shims.mkdir()
    for name in ("id", "curl", "sleep", "sudo"):
        path = shims / name
        path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{__file__}" --shim "$0" "$@"\n')
        path.chmod(0o700)
    for name in ("cut", "head", "grep"):
        (shims / name).symlink_to(Path("/usr/bin") / name)
    results = {}
    for name in ("pass", "sync-error", "restart-error", "rollback", "broken-rollback", "curl-28"):
        case = root / name
        case.mkdir()
        (case / "head").write_text(OLD)
        env = dict(PATH=str(shims), BASH_ENV=str(env_script), TARGET_SHA=TARGET,
                   WRD3_CASE=str(case), WRD3_SCENARIO=name)
        run = subprocess.run(["/bin/bash", str(script)], env=env, cwd=case,
                             capture_output=True, text=True, timeout=5)
        calls = [json.loads(line) for line in (case / "calls").read_text().splitlines()]
        results[name] = {"rc": run.returncode, "head": (case / "head").read_text()[:8],
                         "stamp": (case / "stamped").exists(),
                         "output": run.stdout.splitlines(), "calls": calls,
                         **({"stderr": run.stderr} if run.stderr else {})}
    assert results["pass"]["rc"] == results["curl-28"]["rc"] == 0, results
    assert results["rollback"]["head"] == OLD[:8], results
    assert "Rolled back cleanly" in "\n".join(results["broken-rollback"]["output"]), results
    assert all(results[n]["head"] == TARGET[:8] and not results[n]["stamp"]
               for n in ("sync-error", "restart-error")), results
    return results


def wait_for(path, seconds=3):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(str(path))
        time.sleep(0.01)


def writer(root, name, mode):
    fd = os.open(root / "writers.lock", os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return 75
    if os.path.lexists(root / "hold"):
        return 75
    os.set_inheritable(fd, True)
    os.execv(sys.executable, [sys.executable, __file__, "--job", str(root), name, mode, str(fd)])


def job(root, name, mode, fd):
    os.fstat(fd)
    if mode == "daemon":
        if os.fork():
            os._exit(0)
        os.setsid()
        if os.fork():
            os._exit(0)
    (root / f"{name}.admitted").touch()
    wait_for(root / f"{name}.finish")
    os.close(fd)
    (root / f"{name}.done").touch()


def writer_proc(root, name, mode):
    return subprocess.Popen([sys.executable, __file__, "--writer", str(root), name, mode],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def try_drain(root, seconds=0.15):
    deadline = time.monotonic() + seconds
    with (root / "writers.lock").open("rb") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.01)


def lock_probe(root):
    (root / "writers.lock").touch()
    first = writer_proc(root, "before", "normal")
    wait_for(root / "before.admitted")
    (root / "hold").touch()
    between = writer_proc(root, "between", "normal")
    between_status = between.wait(timeout=3)
    assert between_status == 75 and not (root / "between.admitted").exists()
    assert not try_drain(root)
    (root / "before.finish").touch()
    assert first.wait(timeout=3) == 0 and try_drain(root)
    (root / "hold").unlink()
    daemon = writer_proc(root, "detached", "daemon")
    assert daemon.wait(timeout=3) == 0
    wait_for(root / "detached.admitted")
    (root / "hold").touch()
    assert not try_drain(root)
    (root / "detached.finish").touch()
    wait_for(root / "detached.done")
    assert try_drain(root)
    return {"forked_after_arm_before_drain": between_status, "prearm_writer_blocks": True,
            "detached_grandchild_blocks_after_launcher_exits": True, "drains_after_fd_close": True}


def model_and_shell(root):
    stamp = root / "data" / ".last_deploy_iso"
    stamp.mkdir(parents=True)
    tmp = stamp.parent / "temp"
    tmp.write_text("stamp\n")
    run = subprocess.run(["/bin/bash", "-c", '/bin/mv "$1" "$2"', "stamp", str(tmp), str(stamp)],
                         capture_output=True, timeout=3)
    assert run.returncode == 0 and (stamp / "temp").exists()
    shadow = root / "python3"
    shadow.write_text("#!/bin/sh\nexit 79\n")
    shadow.chmod(0o700)
    run = subprocess.run(["/bin/bash", "-c", "python3"], env={"PATH": str(root)}, timeout=3)
    assert run.returncode == 79
    # An existing, harmless absolute executable is independent of PATH on this host.
    run = subprocess.run(["/bin/bash", "-c", "/bin/sh -c 'exit 0'"],
                         env={"PATH": str(root)}, timeout=3)
    assert run.returncode == 0
    # The following traces are legal orderings in the prose, not observed systemd behavior.
    lease_traces = {
        "old_process": ["arm", "controller lost before drain/stop", "old process still runnable"],
        "timer_gap": ["canary record persisted", "controller lost before systemd-run",
                      "explicit start passes condition", "record expires, started process still runnable"],
        "early_send": ["lease", "restart", "scheduler READY", "scheduler can send or write",
                       "contest or dashboard canary fails", "stop cannot undo action"],
        "release_expire": ["release reads canary", "expiry writes failed and stops", "release unlinks failed"],
        "late_release": ["lease deadline passes", "timer delayed", "release finds canary and unlinks"],
        "after_release_signal": ["release", "TERM before trap disarm", "E4 recreates failed hold and stops"],
    }
    record = None
    arms = 0
    while len(json.dumps(record)) <= 8192:
        record = {"schema": "bts_deploy_hold_v1", "deploy_id": f"1800000000-{arms}", "state": "failed",
                  "step": "sync", "target": TARGET, "previous": OLD, "armed_at": 1800000000,
                  "updated_at": 1800000001, "boot_id": "6a6597ba-a27d-4f81-a86c-74e13b37aaf4",
                  "lease_until": None, "prior": record}
        arms += 1
    return {"mv_to_directory_rc": 0, "absolute_path_ignores_PATH": True,
            "model_lease_traces": lease_traces, "nested_priors_exceed_limit_at": arms}


def main():
    if sys.argv[1:2] == ["--shim"]:
        return shim()
    if sys.argv[1:2] == ["--writer"]:
        return writer(Path(sys.argv[2]), *sys.argv[3:])
    if sys.argv[1:2] == ["--job"]:
        job(Path(sys.argv[2]), *sys.argv[3:5], int(sys.argv[5]))
        return 0
    with TemporaryDirectory(prefix="r3-probe-", dir=HERE) as tmp:
        root = Path(tmp)
        for kind in ("base", "locks", "models"):
            (root / kind).mkdir()
        result = {"base_bash": subprocess.run(["/bin/bash", "--version"], capture_output=True,
                                               text=True, timeout=3).stdout.splitlines()[0],
                  "base": base_replay(root / "base"), "lock": lock_probe(root / "locks"),
                  "models": model_and_shell(root / "models")}
    (HERE / "review-probe.log").write_text(json.dumps(result, indent=2) + "\n")
    print("Passed 6 exact-base shell cases, flock probe and shell/model checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
