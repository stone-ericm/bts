"""Run targeted Bash models of the v2 deploy and recovery rules, offline."""

from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TARGET = "b" * 40
OLD = "a" * 40

MODEL = r'''
note() { printf '%s\n' "$*" >> events; }
record() {
    printf 'bts_deploy_attempt_v1 %s %s %s %s %s %s %s %s %s\n' \
        "$RUN_ID" "$RUN_ATTEMPT" "$TARGET" "$OLD" "$@" > staged
    mv -f staged attempt
}
flock_shim() { "$PY" "$PROBE" --lock; }
stop() {
    printf 'stopped\n' > scheduler
    note "stop $RUN_ID.$RUN_ATTEMPT"
}
contain() {
    trap '' HUP INT TERM PIPE
    stop
    printf 'deploy_failed\n' > heartbeat
    note "page $RUN_ID.$RUN_ATTEMPT"
    record contained "$1" verified verified verified
    exit 1
}
on_signal() {
    read -r magic id number sha old stage step s m p < attempt
    if [[ $stage == committed ]]; then exit 0; fi
    contain interrupted
}
exec 9>>lock
if ! flock_shim; then note lock-unavailable; exit 1; fi
mark=cancelled.$RUN_ID.$RUN_ATTEMPT
if [[ $MODE == deploy ]]; then
    if [[ -e $mark ]]; then note 'not-started cancelled'; exit 1; fi
    if [[ $CASE == no-write ]]; then note 'not-started attempt-record'; exit 1; fi
    record begun - - - -
    if [[ $CASE == late-status ]]; then note 'not-started attempt-record'; exit 1; fi
    if [[ $CASE == early-signal ]]; then kill -TERM "$$"; fi
    trap 'on_signal' HUP INT TERM PIPE
    printf 'disabled-running\n' > scheduler
    note "disable $RUN_ID.$RUN_ATTEMPT"
    if [[ $CASE == lost ]]; then kill -KILL "$$"; fi
    if [[ $CASE == stamp ]]; then
        printf 'new-stamp\n' > stamp-next
        mv -f stamp-next stamp
        contain commit-record
    fi
    printf 'enabled-running\n' > scheduler
    if [[ $CASE == after-enable ]]; then kill -KILL "$$"; fi
    record committed - - - -
    if [[ $CASE == after-commit ]]; then kill -TERM "$$"; fi
    exit 0
fi
if [[ -f attempt ]]; then
    read -r magic id number sha old stage step s m p < attempt
fi
if [[ $id != "$RUN_ID" || $number != "$RUN_ATTEMPT" ]]; then
    : > "$mark"
    note "fenced $RUN_ID.$RUN_ATTEMPT"
    exit 1
fi
case $stage in
    committed) note 'recovery committed'; exit 1 ;;
    begun) contain deploy-step-incomplete ;;
    contained)
        stop
        if [[ $m == unverified ]]; then printf 'deploy_failed\n' > heartbeat; fi
        if [[ $p == unverified ]]; then note "page $RUN_ID.$RUN_ATTEMPT"; fi
        note "marker $m"
        exit 1
        ;;
esac
'''


def run(box, mode="deploy", ident="101", attempt="1", scenario="", body=MODEL):
    env = dict(os.environ, MODE=mode, RUN_ID=ident, RUN_ATTEMPT=attempt,
               TARGET=TARGET, OLD=OLD, CASE=scenario, PY=sys.executable, PROBE=str(__file__))
    return subprocess.run(["/bin/bash", "-c", body], cwd=box, env=env,
                          capture_output=True, timeout=3)


def start(box, name):
    path = box / name
    path.mkdir()
    (path / "scheduler").write_text("enabled-running\n")
    (path / "heartbeat").write_text("running\n")
    (path / "stamp").write_text("old-stamp\n")
    return path


def order(box):
    root = start(box, "first-recovery")
    assert run(root, "recovery").returncode == 1
    assert run(root).returncode == 1
    assert run(root, attempt="2").returncode == 0
    assert run(root, "recovery").returncode == 1
    assert (root / "scheduler").read_text() == "enabled-running\n"
    print("recovery first blocks late attempt 1; same SHA attempt 2 runs; old recovery leaves it alone")

    root = start(box, "owner-first")
    assert run(root, scenario="lost").returncode == -signal.SIGKILL
    assert run(root, "recovery").returncode == 1
    assert (root / "scheduler").read_text() == "stopped\n"
    assert (root / "heartbeat").read_text() == "deploy_failed\n"
    print("owner dies after disable; matching recovery stops and marks")

    root = start(box, "later-owner")
    assert run(root, scenario="lost").returncode == -signal.SIGKILL
    assert run(root, ident="102").returncode == 0
    assert run(root, "recovery").returncode == 1
    assert (root / "scheduler").read_text() == "enabled-running\n"
    print("later different run commits before earlier recovery; old recovery leaves it alone")

    root = start(box, "reboot")
    assert run(root, scenario="after-enable").returncode == -signal.SIGKILL
    (root / "attempt").unlink()
    (root / "lock").unlink()
    (root / "scheduler").write_text("boot-running-from-link\n")
    assert run(root, "recovery").returncode == 1
    assert (root / "scheduler").read_text() == "boot-running-from-link\n"
    assert (root / "cancelled.101.1").exists()
    print("after simulated /run loss, recovery has no owner evidence and marks cancellation")

    root = start(box, "mutex-busy")
    with (root / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert run(root, "recovery").returncode == 1
        assert (root / "events").read_text() == "lock-unavailable\n"
        assert not (root / "cancelled.101.1").exists()
        assert (root / "scheduler").read_text() == "enabled-running\n"
    print("a genuinely held lock prevents recovery mutation")

    root = start(box, "after-commit")
    assert run(root, scenario="after-commit").returncode == 0
    assert run(root, "recovery").returncode == 1
    assert "recovery committed" in (root / "events").read_text()
    assert (root / "scheduler").read_text() == "enabled-running\n"
    print("TERM at committed record is treated as committed; recovery leaves services alone")


def counterexamples(box):
    root = start(box, "record-status")
    assert run(root, scenario="late-status").returncode == 1
    assert (root / "scheduler").read_text() == "enabled-running\n"
    assert run(root, "recovery").returncode == 1
    events = (root / "events").read_text().splitlines()
    assert events == ["not-started attempt-record", "stop 101.1", "page 101.1"]
    print("rename happened but writer failed: NOT STARTED followed by stop/page of untouched services")

    root = start(box, "early-signal")
    assert run(root, scenario="early-signal").returncode == -signal.SIGTERM
    assert (root / "scheduler").read_text() == "enabled-running\n"
    assert run(root, "recovery").returncode == 1
    assert (root / "scheduler").read_text() == "stopped\n"
    print("TERM between begun publish and trap skipped the local handler; recovery stops")

    root = start(box, "lost-handover")
    assert run(root, scenario="lost").returncode == -signal.SIGKILL
    assert run(root, ident="102", scenario="early-signal").returncode == -signal.SIGTERM
    assert run(root, "recovery").returncode == 1
    assert (root / "scheduler").read_text() == "disabled-running\n"
    assert (root / "heartbeat").read_text() == "running\n"
    assert run(root, "recovery", ident="102").returncode == 1
    assert (root / "scheduler").read_text() == "stopped\n"
    print("lost new owner overwrites old evidence before mutation; only new recovery can stop")

    root = start(box, "late-marker")
    (root / "attempt").write_text(
        f"bts_deploy_attempt_v1 101 1 {TARGET} {OLD} contained sync unverified verified verified\n"
    )
    assert run(root, "recovery").returncode == 1
    assert (root / "scheduler").read_text() == "stopped\n"
    assert (root / "heartbeat").read_text() == "running\n"
    assert "marker verified" in (root / "events").read_text()
    print("recovery stops previous survivor but repeats a stale marker verification; reader sees running")

    root = start(box, "commit-failure")
    assert run(root, scenario="stamp").returncode == 1
    assert (root / "stamp").read_text() == "new-stamp\n"
    print("commit-record error after stamp rename retains new complete stamp")

    root = start(box, "dangling-lock")
    (root / "lock").symlink_to("unexpected")
    assert not (root / "lock").exists()
    result = run(root, body="if [[ ! -e lock ]]; then : >> lock; fi")
    assert result.returncode == 0 and (root / "unexpected").is_file()
    print("a dangling entry looks absent under -e; append-open follows it")

    result = run(root, body=r'''
is_enabled() { printf 'disabled\n'; return 1; }
out=$(is_enabled)
printf '%s %s\n' "$?" "$out"
''')
    assert result.stdout == b"1 disabled\n"
    print("disabled shim presents the documented nonzero exit with the required disabled value")


def heartbeat(box):
    path = box / ".heartbeat"
    spec = importlib.util.spec_from_file_location(
        "check_heartbeat", ROOT / "scripts/check_heartbeat.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    now = datetime(2027, 7, 15, 16, tzinfo=timezone.utc)
    path.write_text(json.dumps({"state": "running", "timestamp": now.isoformat(), "extra": "x" * 2**20}))
    assert module.is_stale(path, now=now) == (False, "fresh running")
    path.unlink()
    path.mkdir()
    result = module.is_stale(path, now=now)
    assert result[0] and result[1].startswith("heartbeat unreadable: ")
    print("base handles a 1 MiB valid heartbeat and a directory without a hang or exception")


if __name__ == "__main__":
    if sys.argv[1:] == ["--lock"]:
        try:
            fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(1)
        raise SystemExit(0)
    with TemporaryDirectory(dir=HERE) as temp:
        directory = Path(temp)
        order(directory)
        counterexamples(directory)
        heartbeat(directory)
