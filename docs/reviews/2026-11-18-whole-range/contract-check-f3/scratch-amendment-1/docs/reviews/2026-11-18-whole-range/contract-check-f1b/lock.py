"""Exercise the proposed shell line in a throwaway home and track all spawned PIDs."""
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
PYTHON = sys.executable


def record(event, **fields):
    with open(os.environ["WR_TRACE"], "a") as out:
        out.write(json.dumps(dict(event=event, pid=os.getpid(), pgid=os.getpgrp(), **fields)) + "\n")


def keep_running():
    time.sleep(6)


def flock_shim():
    _, _, option, path, *command = sys.argv
    assert option == "-n"
    record("flock")
    try:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o666)
        except IsADirectoryError:
            fd = os.open(path, os.O_RDONLY)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return 1
        print(f"flock shim: {exc}", file=sys.stderr)
        return 73
    os.environ["WR_LOCK_FD"] = str(fd)
    child = subprocess.Popen(command, pass_fds=(fd,))
    record("flock-child", child=child.pid)
    result = child.wait(timeout=6)
    return result if result >= 0 else 128 - result


def timeout_shim():
    _, _, option, grace, duration, *command = sys.argv
    assert option == "-k"
    record("timeout")
    child = subprocess.Popen(command, start_new_session=True, pass_fds=(int(os.environ["WR_LOCK_FD"]),))
    record("timeout-child", child=child.pid)
    try:
        return child.wait(timeout=float(duration))
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=float(grace))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=1)
            return 137
        return 124


def transport_child():
    mode = os.environ["WR_MODE"]
    if mode == "escape":
        os.setsid()
    fd = int(os.environ["WR_LOCK_FD"])
    try:
        inherits_lock = os.fstat(fd).st_ino == Path(os.environ["WR_LOCK_PATH"]).stat().st_ino
    except OSError:
        inherits_lock = False
    def on_term(*_):
        record("python-TERM")
        if mode not in ("ignore-child", "ignore-all"):
            raise SystemExit(143)
    signal.signal(signal.SIGTERM, on_term)
    record("python", inherits_lock=inherits_lock)
    if mode != "quick":
        keep_running()


def uv_shim():
    record("uv")
    if os.environ["WR_MODE"] == "ignore-all":
        signal.signal(signal.SIGTERM, lambda *_: record("uv-TERM"))
    fd = int(os.environ["WR_LOCK_FD"])
    child = subprocess.Popen([PYTHON, str(SELF), "child"], pass_fds=(fd,))
    record("uv-child", child=child.pid)
    return child.wait(timeout=6.5)


def run_checks():
    raw = (HERE / "generated-cron.txt").read_text().strip()
    minute, hours, _, _, _, shell_line = raw.split(maxsplit=5)
    limit = re.search(r"/usr/bin/timeout -k (\d+) (\d+) ", shell_line)
    assert limit and sum(map(int, limit.groups())) < int(minute.split("/")[1]) * 60
    print(f"generated schedule {minute} {hours}; grace + limit = {sum(map(int, limit.groups()))} seconds", flush=True)
    lock = HERE / "probe.lock"
    uv_word = shlex.split(shell_line)[shell_line.split().index("840") + 1]
    log = Path(re.search(r">> (\S+) 2>&1", shell_line)[1])
    log.parent.mkdir(exist_ok=True)
    with log.open("a"):
        pass
    processes = []
    traces = []

    def launch(group, mode, *, real_timeout=False, real_uv=False, path=lock):
        def command(*parts):
            return shlex.join(parts)

        trace = HERE / f"pids-{group}-{mode}.jsonl"
        traces.append(trace)
        env = dict(os.environ, WR_TRACE=str(trace), WR_MODE=mode, WR_LOCK_PATH=str(path),
                   UV_OFFLINE="1", UV_NO_SYNC="1", UV_CACHE_DIR=str(HERE / "uv-cache"))
        line = shell_line.replace("/usr/bin/flock", command(PYTHON, str(SELF), "flock"), 1)
        line = line.replace("/home/bts/.check-pick-entered.lock", shlex.quote(str(path)), 1)
        time_command = "/usr/bin/timeout" if real_timeout else command(PYTHON, str(SELF), "timeout")
        line = line.replace("/usr/bin/timeout -k 30 840", f"{time_command} -k 0.2 1.2", 1)
        uv_command = (command("/opt/homebrew/bin/uv", "run", "--offline", "--no-project", "--no-sync",
                              PYTHON, str(SELF), "child") if real_uv else command(PYTHON, str(SELF), "uv"))
        line = line.replace(f"{uv_word} run bts check-pick-entered", uv_command, 1)
        start = time.monotonic()
        process = subprocess.Popen(["/bin/sh", "-c", line], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(process)
        return process, trace, start

    def events(trace):
        return [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []

    def ready(run):
        end = time.monotonic() + 0.9
        while time.monotonic() < end:
            if rows := [row for row in events(run[1]) if row["event"] == "python"]:
                return rows[0]
            time.sleep(0.01)
        raise AssertionError(f"child did not start: {events(run[1])}")

    def ended(run):
        process, trace, start = run
        out, err = process.communicate(timeout=2)
        return dict(code=process.returncode, elapsed=round(time.monotonic() - start, 3), out=out.decode(),
                    err=err.decode(), events=events(trace), shell_pid=process.pid)

    def check(group, mode, *, real_timeout=False, real_uv=False, retained=False, kill=False):
        before = log.read_bytes()
        holder = launch(group, mode, real_timeout=real_timeout, real_uv=real_uv)
        python = ready(holder)
        contender = ended(launch(group, "blocked", real_timeout=real_timeout))
        assert contender["code"] == 1 and [row["event"] for row in contender["events"]] == ["flock"]
        assert contender["elapsed"] < 1 and not contender["out"] and not contender["err"]
        result = ended(holder)
        assert result["code"] == (137 if kill else 124), result
        if retained:
            still_locked = ended(launch(group, "retained", real_timeout=real_timeout))
            assert still_locked["code"] == 1, still_locked
            os.kill(python["pid"], signal.SIGKILL)
            end = time.monotonic() + 0.5
            while time.monotonic() < end:
                try:
                    with lock.open() as fd:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.01)
        followup = ended(launch(group, "quick", real_timeout=real_timeout))
        assert followup["code"] == 0 and any(row["event"] == "python" for row in followup["events"]), followup
        print(group, "holder", result, "contender", contender, "followup", followup,
              "retained", retained, "log delta", log.read_bytes()[len(before):], flush=True)

    try:
        check("shim-term", "term")
        check("shim-kill", "ignore-all", kill=True)
        check("shim-early-exit", "ignore-child", retained=True)
        check("shim-escape", "escape", retained=True)
        check("bsd-uv", "term", real_timeout=True, real_uv=True)
        check("bsd-kill", "ignore-all", real_timeout=True, kill=True)
        missing = HERE / "absent-parent" / "lock"
        before = log.read_bytes()
        result = ended(launch("path", "missing-parent", path=missing))
        assert result["code"] == 73 and b"flock shim:" in log.read_bytes()[len(before):], result
        print("unopenable path", result, log.read_bytes()[len(before):], flush=True)
        for name, path in (("directory", HERE), ("read-only", lock)):
            if name == "read-only":
                path.chmod(0o444)
            result = ended(launch("path", name, path=path))
            assert result["code"] == 0, result
            print(name, result, flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=1)
        for trace in traces:
            for row in events(trace):
                if row["event"] in ("uv", "python", "timeout", "flock"):
                    try:
                        os.kill(row["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    roles = {"flock": flock_shim, "timeout": timeout_shim, "uv": uv_shim, "child": transport_child}
    raise SystemExit(roles[sys.argv[1]]() if len(sys.argv) > 1 else run_checks())
