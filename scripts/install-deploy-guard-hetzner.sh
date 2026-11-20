#!/bin/bash
# Standalone provisioning entry point. The deploy never invokes it.
set -euo pipefail

root="${DESTDIR:-}"
if [[ -z "$root" && "$EUID" != 0 ]]; then
    printf '%s\n' 'FAILED: run the installer as root' >&2
    exit 1
fi
if [[ -n "$root" && ( "$root" != /* || "$root" == / || ! -d "$root" || -L "$root" ) ]]; then
    printf '%s\n' 'FAILED: invalid DESTDIR' >&2
    exit 1
fi
trap 'printf "%s\n" "FAILED: deploy guard installation or verification" >&2' ERR

directory="$root/var/lib/bts-deploy"
tool="$root/usr/local/bin/bts-deploy-hold"
drop="$root/etc/systemd/user/bts-scheduler.service.d/50-deploy-hold.conf"
condition='/usr/bin/python3 -I /usr/local/bin/bts-deploy-hold check-clear --scheduler'
umask 022
ensure_directory() {
    local path="$1" mode
    if [[ "$path" != "$root" && "$path" != / && -n "$path" ]]; then
        ensure_directory "${path%/*}"
        [[ ! -L "$path" ]]
        if [[ ! -d "$path" ]]; then
            mkdir "$path"
        fi
        if [[ -z "$root" ]]; then
            [[ "$(stat -c '%u' "$path")" == 0 ]]
            mode="$(stat -c '%a' "$path")"
            (( (8#$mode & 8#022) == 0 ))
        fi
    fi
}
for path in "$directory" "${tool%/*}" "${drop%/*}"; do
    ensure_directory "$path"
done
chown root:root "$directory"
chmod 0755 "$directory"

install_fingerprint() {
    /usr/bin/python3 -I -c '
import hashlib
import os
import stat
import sys
import tempfile
from pathlib import Path

directory, source, action = sys.argv[1:4]
record = Path(directory) / "guard-install.sha256"
flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
with os.fdopen(os.open(source, flags), "rb") as stream:
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        raise SystemExit("FAILED: installer source must be a regular file")
    data = stream.read(65537)
if not data or len(data) > 65536:
    raise SystemExit("FAILED: installer source size")
digest = hashlib.sha256(data).hexdigest()
fd = os.open(directory, flags | os.O_DIRECTORY)
try:
    if action == "prepare":
        record.unlink(missing_ok=True)
        os.fsync(fd)
        print(digest)
    elif digest == sys.argv[4]:
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix=".guard-install-", delete=False) as stream:
            path = Path(stream.name)
            try:
                os.fchmod(stream.fileno(), 0o644)
                stream.write(digest + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(path, record)
                os.fsync(fd)
            finally:
                path.unlink(missing_ok=True)
    else:
        raise SystemExit("FAILED: installer source changed during installation")
finally:
    os.close(fd)
' "$directory" "$0" "$@"
}

# A partial install must invalidate any proof from a previous run.
installer_sha="$(install_fingerprint prepare)"
for name in writers.lock transaction.lock state.lock; do
    path="$directory/$name"
    if [[ -f "$path" && ! -L "$path" ]]; then
        printf '%s already current\n' "$name"
    else
        (set -o noclobber; : > "$path")
        printf '%s installed\n' "$name"
    fi
    chown root:root "$path"
    chmod 0644 "$path"
done

install_content() {
    local temporary target="$1" mode="$2"
    temporary="$(mktemp "$target.XXXXXXXX")"
    cat > "$temporary"
    chown root:root "$temporary"
    chmod "$mode" "$temporary"
    if [[ -f "$target" && ! -L "$target" ]] && cmp -s "$temporary" "$target"; then
        chown root:root "$target"
        chmod "$mode" "$target"
        rm -f "$temporary"
        printf '%s already current\n' "${target##*/}"
    else
        [[ ! -d "$target" ]]
        mv -f "$temporary" "$target"
        printf '%s installed\n' "${target##*/}"
    fi
}

install_content "$tool" 0755 <<'BTS_DEPLOY_HOLD_PYTHON'
#!/usr/bin/python3 -I
"""Standalone system-Python guard for deploy holds."""
import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

TEST_ROOT = os.environ.get("BTS_TEST_DEPLOY_GUARD_DIR") if os.geteuid() else None
ROOT = Path(TEST_ROOT or "/var/lib/bts-deploy")
UID = os.geteuid() if TEST_ROOT else 0
FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
JOBS = ("check-results", "reconcile", "fetch-contest-streak", "nightly-preview")
RUN_ID = re.compile(r"[0-9]{1,20}-[A-Za-z0-9]{1,64}")
SHA = re.compile(r"[0-9a-f]{40}")
PROC_LOCKS = Path("/proc/locks")
STEPS = frozenset((
    "arm", "stop-dashboard", "stop-old", "drain", "reset", "head", "sync", "restart", "wait",
    "canary-contest-state", "canary-dashboard", "canary-http", "release",
    "activation", "interrupted", "abandoned", "owner-ended", "window",
))


@contextmanager
def directory():
    fd = os.open(ROOT, FLAGS | os.O_DIRECTORY)
    try:
        info = os.fstat(fd)
        if info.st_uid != UID or stat.S_IMODE(info.st_mode) != 0o755:
            raise PermissionError("guard directory")
        yield fd
    finally:
        os.close(fd)


@contextmanager
def lock(root, name, operation, timeout=0):
    fd = os.open(name, FLAGS, dir_fd=root)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != UID or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o644):
            raise PermissionError("guard lock")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, operation | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        yield fd
    finally:
        os.close(fd)


def unique_fields(items):
    items = tuple(items)
    result = dict(items)
    if len(result) != len(items):
        raise ValueError("duplicate field")
    return result


def read_json(root, name):
    try:
        fd = os.open(name, FLAGS, dir_fd=root)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != UID or info.st_mode & 0o022:
            raise ValueError("record type")
        raw = os.read(fd, 8193)
        if len(raw) > 8192:
            raise ValueError("record size")
    finally:
        os.close(fd)
    record = json.loads(raw, object_pairs_hook=unique_fields)
    if not isinstance(record, dict):
        raise ValueError("record shape")
    return record


class LookupFailed(OSError):
    pass


def read_hold(root):
    try:
        os.stat("hold", dir_fd=root, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError:
        raise LookupFailed("hold lookup") from None
    record = read_json(root, "hold")
    if not record or set(record) != {
        "schema", "deploy_id", "owner_unit", "state", "step", "target",
        "previous", "armed_at", "updated_at",
    }:
        raise ValueError("record fields")
    run_id = record["deploy_id"]
    if not (
        record["schema"] == "bts_deploy_hold_v1"
        and isinstance(run_id, str) and RUN_ID.fullmatch(run_id)
        and record["owner_unit"] == f"bts-deploy-{run_id}.service"
        and record["state"] in ("pending", "activating", "failed")
        and record["step"] in STEPS
        and isinstance(record["target"], str) and SHA.fullmatch(record["target"])
        and isinstance(record["previous"], str)
        and (record["previous"] == "unknown" or SHA.fullmatch(record["previous"]))
        and type(record["armed_at"]) is int and type(record["updated_at"]) is int
        and 0 <= record["armed_at"] <= record["updated_at"] <= int(time.time())
    ):
        raise ValueError("record values")
    return record


def owner_alive(root, record):
    try:
        info = os.stat("transaction.lock", dir_fd=root, follow_symlinks=False)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != UID
                or stat.S_IMODE(info.st_mode) != 0o644 or info.st_nlink != 1):
            return False
        result = subprocess.run(
            ["/usr/bin/systemctl", "show", record["owner_unit"], "-p", "ActiveState", "-p", "MainPID"],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=3,
        )
        if result.returncode != 0:
            return False
        fields = unique_fields(line.split("=", 1) for line in result.stdout.decode().splitlines())
        pid = fields.get("MainPID", "")
        if fields.get("ActiveState") != "active" or not re.fullmatch(r"[1-9][0-9]*", pid):
            return False
        with PROC_LOCKS.open() as stream:
            locks = stream.read(1_048_577)
        if len(locks) > 1_048_576:
            return False
        for line in locks.splitlines():
            parts = line.split()
            if parts[1:5] != ["FLOCK", "ADVISORY", "WRITE", pid] or len(parts) != 8:
                continue
            major, minor, inode = parts[5].split(":")
            if (int(major, 16), int(minor, 16), int(inode)) == (
                os.major(info.st_dev), os.minor(info.st_dev), info.st_ino,
            ):
                return True
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False
    return False


def clear(root):
    record = read_hold(root)
    return record is None or (record["state"] == "activating" and owner_alive(root, record))


def status(root, machine):
    try:
        record = read_hold(root)
    except LookupFailed:
        print('{"state":"lookup-failed"}' if machine else "deploy hold lookup failed")
        return 2
    except (ValueError, OSError):
        print('{"state":"unreadable"}' if machine else "unreadable")
        return 1
    if record is None:
        print('{"state":"clear"}' if machine else "no deploy hold")
        return 0
    if machine:
        print(json.dumps(record, sort_keys=True))
    else:
        armed = datetime.fromtimestamp(record["armed_at"], timezone.utc).isoformat()
        print(f'{record["state"]} at {record["step"]} (deploy {record["deploy_id"]}, '
              f'target {record["target"][:7]}, previous {record["previous"][:7]}, armed {armed})')
    return 1


def writer(name, argv):
    reason = "lock unavailable"
    try:
        with directory() as root:
            with lock(root, "writers.lock", fcntl.LOCK_SH) as fd:
                reason = "unreadable"
                if clear(root):
                    reason = "command unavailable"
                    os.set_inheritable(fd, True)
                    os.execv(argv[0], argv)
                reason = "held"
    except BlockingIOError:
        reason = "lock busy"
    except (OSError, ValueError):
        pass
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] bts-deploy-hold: {name} skipped: {reason}")
    return 75


def publish(root, name, content, mode=0o644):
    temporary = ".deploy-" + secrets.token_hex(12)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode, dir_fd=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, name, src_dir_fd=root, dst_dir_fd=root)
        os.fsync(root)
    finally:
        try:
            os.unlink(temporary, dir_fd=root)
        except FileNotFoundError:
            pass


def publish_json(root, name, record):
    publish(root, name, (json.dumps(record, sort_keys=True) + "\n").encode())


def transition(root, record, state, step):
    with lock(root, "state.lock", fcntl.LOCK_EX, timeout=30):
        updated = dict(record, state=state, step=step, updated_at=int(time.time()))
        publish_json(root, "hold", updated)
        return updated


def release(root, record, kind, *, authorized_at=None, certified_date=None):
    with lock(root, "state.lock", fcntl.LOCK_EX, timeout=30):
        if read_hold(root) != record:
            raise ValueError("release ownership changed")
        publish_json(root, "last-release", {
            "released_at": int(time.time()) if authorized_at is None else authorized_at,
            "kind": kind, "deploy_id": record["deploy_id"],
        })
        if certified_date is not None:
            today = datetime.fromtimestamp(time.time(), ZoneInfo("America/New_York")).date().isoformat()
            if today != certified_date:
                raise Refused("activation")
        os.unlink("hold", dir_fd=root)
        os.fsync(root)


PROJECT = "/home/bts/projects/bts"
CONFIG = "/home/bts/.bts-orchestrator.toml"
UV = "/home/bts/.local/bin/uv"


class Refused(Exception):
    def __init__(self, step, previous=None):
        self.step = step
        self.previous = previous


def call(argv, *, limit=10):
    with subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        start_new_session=True, cwd="/",
    ) as process:
        try:
            deadline = time.monotonic() + limit
            output = bytearray()
            fd = process.stdout.fileno()
            os.set_blocking(fd, False)
            with selectors.DefaultSelector() as selector:
                selector.register(fd, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise subprocess.TimeoutExpired(argv[0], limit)
                    try:
                        chunk = os.read(fd, 8192)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        break
                    output.extend(chunk)
                    if len(output) > 65536:
                        raise OSError("command output limit")
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
            raise
        return process.returncode, bytes(output)


def require(argv, step, expected=None, *, limit=10):
    try:
        code, output = call(argv, limit=limit)
    except InterruptedError:
        raise
    except (OSError, subprocess.TimeoutExpired):
        raise Refused(step) from None
    if code != 0 or expected is not None and output != expected:
        raise Refused(step)
    return output


def as_bts(*argv):
    return ["/usr/bin/sudo", "-n", "-Hu", "bts", "--", *argv]


def user_manager(uid, *argv):
    return as_bts("/usr/bin/env", f"XDG_RUNTIME_DIR=/run/user/{uid}", "/usr/bin/systemctl", "--user", *argv)


def stop_and_verify(uid, unit, allowed):
    try:
        code, _ = call(user_manager(uid, "stop", unit), limit=30)
        query, value = call(user_manager(uid, "show", unit, "-p", "ActiveState", "--value"))
        return code == query == 0 and value in allowed
    except InterruptedError:
        raise
    except (OSError, subprocess.TimeoutExpired):
        return False


def verified_stop(uid):
    return stop_and_verify(uid, "bts-scheduler.service", (b"inactive\n", b"failed\n"))


def verified_stop_dashboard(uid):
    return stop_and_verify(uid, "bts-dashboard.service", (b"inactive\n",))


def page(record, stopped):
    import urllib.error
    import urllib.request

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    def deadline(signum, frame):
        raise TimeoutError()

    try:
        fd = os.open(PROJECT + "/.env", FLAGS)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return None
            raw = stream.read(65537)
        if len(raw) > 65536:
            return None
        assignments = [
            line for line in raw.splitlines()
            if re.match(rb"\s*(?:export\s+)?BTS_SCHEDULER_HEARTBEAT_PING_URL\s*=", line)
        ]
        if len(assignments) != 1:
            return None
        match = re.fullmatch(
            rb"BTS_SCHEDULER_HEARTBEAT_PING_URL=(https://hc-ping\.com/[A-Za-z0-9_-]+)", assignments[0],
        )
        if not match:
            return None
        body = (
            f'Deploy failed at {record["step"]}, requested target {record["target"][:7]}, '
            f'scheduler stop {"verified" if stopped else "unverified"}. See INCIDENT.md §4.'
        ).encode()
        request = urllib.request.Request(match[1].decode() + "/fail", data=body, method="POST")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        previous = signal.signal(signal.SIGALRM, deadline)
        signal.alarm(10)
        try:
            with opener.open(request, timeout=10) as response:
                return response.status if 200 <= response.status <= 299 else None
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    except InterruptedError:
        raise
    except (OSError, ValueError, urllib.error.URLError):
        return None


def containment(root, record, uid):
    name = f'run-{record["deploy_id"]}.evidence'
    try:
        evidence = read_json(root, name) or {}
    except (OSError, ValueError):
        evidence = {}
    stopped = evidence.get("stop") == "verified"
    if not stopped and uid is not None:
        try:
            stopped = verified_stop(uid)
        except InterruptedError:
            raise
        except (OSError, Refused, subprocess.TimeoutExpired):
            stopped = False
    status = "verified" if stopped else "unverified"
    print(f"Scheduler stopped: {status}.", flush=True)
    send = False
    try:
        with lock(root, "state.lock", fcntl.LOCK_EX, timeout=2):
            evidence = read_json(root, name) or {}
            if evidence.get("stop") != "verified":
                evidence["stop"] = status
            claim = evidence.get("page", "not-attempted")
            attempts = evidence.get("page_attempts", 1 if claim == "claimed" else 0)
            if claim in ("not-attempted", "claimed") and type(attempts) is int and 0 <= attempts < 2:
                evidence.update(page="claimed", page_attempts=attempts + 1)
                send = True
            publish_json(root, name, evidence)
    except (OSError, ValueError):
        # A failed evidence write cannot establish a durable claim.
        send = False
    if send:
        code = page(record, stopped)
        outcome = f"sent {code}" if type(code) is int and 200 <= code <= 299 else "failed"
        try:
            with lock(root, "state.lock", fcntl.LOCK_EX, timeout=2):
                latest = read_json(root, name) or {}
                if latest.get("page") == "claimed" and latest.get("page_attempts") == attempts + 1:
                    latest["page"] = outcome
                    publish_json(root, name, latest)
        except (OSError, ValueError):
            pass
        print("Failure ping received HTTP 2xx: "
              + ("verified." if outcome != "failed" else "unverified."), flush=True)
    return stopped


def bts_uid():
    output = require(["/usr/bin/id", "-u", "bts"], "id")
    if not re.fullmatch(rb"[0-9]+\n", output):
        raise Refused("id")
    return int(output)


def finish_incident(root, record):
    try:
        uid = bts_uid()
    except (Refused, OSError):
        uid = None
    contained = containment(root, record, uid)
    run_id = record["deploy_id"]
    if read_json(root, run_name(run_id, "result")) is None:
        record_result(root, run_id, "failed", record["step"])
    return contained


def reconcile(root, run_id):
    with ExitStack() as transaction:
        try:
            transaction.enter_context(lock(root, "transaction.lock", fcntl.LOCK_EX, timeout=60))
        except BlockingIOError:
            record = read_hold(root)
            if record is not None and record["deploy_id"] == run_id:
                finish_incident(root, dict(record, step="owner-ended"))
            return 1
        record = read_hold(root)
        if record is None or record["deploy_id"] != run_id:
            return 0
        if record["state"] != "failed":
            record = transition(root, record, "failed", "owner-ended")
        return 0 if finish_incident(root, record) else 1


def heartbeat():
    fd = os.open(PROJECT + "/data/.heartbeat", FLAGS)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("heartbeat type")
        payload = stream.read(8193)
    if len(payload) > 8192:
        raise ValueError("heartbeat size")
    data = json.loads(payload, object_pairs_hook=unique_fields)
    timestamp = datetime.fromisoformat(data["timestamp"])
    if timestamp.utcoffset() is None or timestamp.timestamp() > time.time():
        raise ValueError("heartbeat timestamp")
    return data, timestamp.timestamp()


def window(recovery, target, deadline, *, minimum=1800):
    try:
        data, timestamp = heartbeat()
        state = data["state"]
        age = time.time() - timestamp
        if state in ("running", "waiting_for_games"):
            if age <= (300 if state == "running" else 600):
                raise Refused("window")
        elif state == "sleeping":
            expires = datetime.fromisoformat(data["sleeping_until"])
            if expires.utcoffset() is None:
                raise ValueError("sleep timestamp")
            if expires.timestamp() >= time.time() - 600:
                if expires.timestamp() < max(deadline, time.time() + minimum):
                    raise Refused("window")
                return
    except (OSError, KeyError, TypeError, ValueError, OverflowError):
        pass
    if recovery != target:
        raise Refused("window")


def target_installer(git, target, *, fetch):
    object_name = f"{target}:scripts/install-deploy-guard-hetzner.sh"
    try:
        require([*git, "cat-file", "-t", target], "target", b"commit\n")
    except Refused:
        if not fetch:
            raise
    else:
        return require([*git, "show", object_name], "guard-stale")

    # Inspect a remote-only target in an isolated object store. Until the
    # fingerprint matches, even the checkout's Git config and refs stay put.
    remote = require([*git, "remote", "get-url", "origin"], "fetch").decode().removesuffix("\n")
    temporary = require(as_bts("/usr/bin/mktemp", "-d", "/tmp/bts-deploy-target.XXXXXXXX"), "fetch")
    path = temporary.decode().removesuffix("\n")
    if not re.fullmatch(r"/tmp/bts-deploy-target\.[A-Za-z0-9]{8}", path):
        raise Refused("fetch")
    isolated_git = as_bts("/usr/bin/git", "-C", path)
    try:
        require([*isolated_git, "init", "--bare", "--quiet"], "fetch")
        require([*isolated_git, "fetch", "--depth=1", "--no-tags", "--no-write-fetch-head",
                 "--", remote, target], "fetch", limit=90)
        require([*isolated_git, "cat-file", "-t", target], "target", b"commit\n")
        return require([*isolated_git, "show", object_name], "guard-stale")
    finally:
        try:
            call(as_bts("/usr/bin/rm", "-rf", "--", path))
        except (OSError, subprocess.TimeoutExpired):
            pass


def verified_target(target, *, fetch):
    git = as_bts("/usr/bin/git", "-C", PROJECT)
    previous = require([*git, "rev-parse", "HEAD"], "old-sha").decode().strip()
    if not SHA.fullmatch(previous):
        raise Refused("old-sha")
    try:
        with directory() as root:
            fd = os.open("guard-install.sha256", FLAGS, dir_fd=root)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != UID
                        or stat.S_IMODE(info.st_mode) != 0o644 or info.st_nlink != 1):
                    raise ValueError("invalid fingerprint file")
                fingerprint = stream.read(66)
        if not re.fullmatch(rb"[0-9a-f]{64}\n", fingerprint):
            raise ValueError("invalid fingerprint")
    except (OSError, ValueError):
        raise Refused("guard-stale", previous) from None

    try:
        script = target_installer(git, target, fetch=fetch)
    except (Refused, InterruptedError) as error:
        error.previous = previous
        raise
    if hashlib.sha256(script).hexdigest().encode() + b"\n" != fingerprint:
        raise Refused("guard-stale", previous)
    return previous, git


CONDITION_COMMAND = "/usr/bin/python3 -I /usr/local/bin/bts-deploy-hold check-clear --scheduler"
RUNTIME_ENVIRONMENT = {"UV_CACHE_DIR=/tmp/uv-cache", "PATH=/home/bts/.local/bin:/usr/local/bin:/usr/bin:/bin"}
UNIT_PROPERTIES = (
    "ExecStart", "WorkingDirectory", "Environment", "EnvironmentFiles",
)
SCHEDULER_PROPERTIES = (*UNIT_PROPERTIES, "ExecCondition", "SuccessExitStatus")


def execution_matches(value, command):
    match = re.fullmatch(r"\{\s*([^{}]+?)\s*\}", value)
    if not match:
        return False
    fields = unique_fields(piece.strip().split("=", 1) for piece in match[1].split(";") if piece.strip())
    return (
        fields.get("path") == command.split()[0]
        and fields.get("argv[]") == command
        and fields.get("ignore_errors") == "no"
    )


def verify_unit_fields(unit, fields):
    scheduler = unit == "bts-scheduler.service"
    command = (f"{UV} run --no-sync bts schedule --config {CONFIG}" if scheduler
               else f"{UV} run --no-sync python3 -m bts.web")
    try:
        environment = shlex.split(fields["Environment"])
        if (
            not execution_matches(fields["ExecStart"], command)
            or fields["WorkingDirectory"] != PROJECT
            or len(environment) != 2
            or set(environment) != RUNTIME_ENVIRONMENT
            or fields["EnvironmentFiles"] != f"{PROJECT}/.env (ignore_errors=no)"
        ):
            raise Refused("guard")
        if scheduler and (
            not execution_matches(fields["ExecCondition"], CONDITION_COMMAND)
            or "1" in fields["SuccessExitStatus"].split()
        ):
            raise Refused("guard")
    except (KeyError, ValueError):
        raise Refused("guard") from None


CRON_WRITERS = {job: ("preview" if job == "nightly-preview" else job) for job in JOBS}
WRITER_INVOKES = re.compile(
    r"(?<![\w.-])(?:bts|bts\.cli)\s+(" + "|".join(CRON_WRITERS.values()) + r")(?![\w-])"
)
CRON_GATE = re.compile(
    r"\s*(?:[\d*,/\-]+\s+){5}/usr/local/bin/bts-deploy-hold run (\S+) -- /bin/sh -c "
    r"'([^']+)' >> [^\s;&|<>]+ 2>&1 # BTS-HETZNER"
)


def runs_heartbeat(command):
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    separators = {";", "&&", "||", "|", "&", "(", ")"}
    for i, token in enumerate(tokens):
        if (token != "/usr/bin/python3" or tokens[i + 1:i + 2] != ["scripts/check_heartbeat.py"]
                or i and tokens[i - 1] not in separators):
            continue
        end = next((j for j in range(i + 2, len(tokens)) if tokens[j] in separators), len(tokens))
        flags = tokens[i + 2:end]
        if flags.count("--deploy-hold") == 1:
            offset = flags.index("--deploy-hold")
            if flags[offset + 1:offset + 2] == ["/var/lib/bts-deploy/hold"]:
                return True
    return False


def verify_crontab(raw):
    counts = {job: 0 for job in JOBS}
    heartbeat = False
    for line in raw.decode().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and runs_heartbeat(parts[5]):
            heartbeat = True
        invoked = set(WRITER_INVOKES.findall(line.translate({ord("'"): None, ord('"'): None})))
        if not invoked:
            continue
        gated = CRON_GATE.fullmatch(line)
        if not gated or gated[1] not in CRON_WRITERS or invoked != {CRON_WRITERS[gated[1]]}:
            raise Refused("crontab")
        if set(WRITER_INVOKES.findall(gated[2])) != invoked:
            raise Refused("crontab")
        counts[gated[1]] += 1
    if not heartbeat or tuple(counts.values()) != (1, 2, 4, 1):
        raise Refused("crontab")


def preflight(uid, target, *, fetch=True):
    previous, git = verified_target(target, fetch=fetch)
    try:
        if fetch:
            require(as_bts("/usr/bin/git", "config", "--global", "--add", "safe.directory", PROJECT), "git-config")
            require([*git, "fetch", "origin"], "fetch", limit=90)
        for unit in ("bts-scheduler.service", "bts-dashboard.service"):
            require(user_manager(uid, "show", unit, "-p", "LoadState", "--value"), "user-manager", b"loaded\n")
            require(user_manager(uid, "show", unit, "-p", "NeedDaemonReload", "--value"), "guard", b"no\n")
            if os.path.lexists(f"/etc/systemd/system/{unit}"):
                raise Refused("legacy-unit")
            require(["/usr/bin/systemctl", "show", unit, "-p", "LoadState", "--value"], "legacy-unit", b"not-found\n")
            properties = SCHEDULER_PROPERTIES if unit == "bts-scheduler.service" else UNIT_PROPERTIES
            options = [argument for prop in properties for argument in ("-p", prop)]
            output = require(user_manager(uid, "show", unit, "--all", *options), "guard").decode()
            try:
                fields = unique_fields(line.split("=", 1) for line in output.splitlines())
            except ValueError:
                raise Refused("guard") from None
            verify_unit_fields(unit, fields)
        crontab = require(["/usr/bin/crontab", "-u", "bts", "-l"], "crontab")
        verify_crontab(crontab)
    except (Refused, InterruptedError) as error:
        error.previous = previous
        raise
    except ValueError:
        raise Refused("guard", previous) from None
    return previous, git


def canary():
    argv = as_bts(
        "/usr/bin/env", "UV_CACHE_DIR=/tmp/uv-cache", UV, "run", "--no-sync",
        "--directory", PROJECT, "bts", "deploy-canary", "--config", CONFIG,
    )
    try:
        code, output = call(argv, limit=60)
        line = output.decode().removesuffix("\n")
        match = re.fullmatch(r"deploy canary ([^\s]+): ([^\r\n]+)", line)
        if match is None:
            raise ValueError("unclassified canary")
        sample = datetime.fromisoformat(match[1])
        if sample.utcoffset() is None:
            raise ValueError("canary clock")
    except InterruptedError:
        raise
    except (OSError, ValueError, subprocess.TimeoutExpired):
        raise Refused("canary-contest-state") from None
    print(line, flush=True)
    if code != 0:
        raise Refused("canary-contest-state")
    return sample.date().isoformat()


def dashboard(uid):
    require(user_manager(uid, "show", "bts-dashboard.service", "-p", "ActiveState", "--value"),
            "canary-dashboard", b"active\n")
    cgroup = require(user_manager(uid, "show", "bts-dashboard.service", "-p", "ControlGroup", "--value"),
                     "canary-dashboard").decode().strip()
    if not cgroup.startswith("/") or not cgroup.endswith("/bts-dashboard.service"):
        raise Refused("canary-dashboard")
    try:
        address = require(as_bts("/usr/bin/tailscale", "ip", "-4"), "canary-http").decode().strip()
    except Refused:
        address = ""
    if address:
        from ipaddress import IPv4Address
        try:
            address = str(IPv4Address(address))
        except ValueError:
            raise Refused("canary-http") from None
    else:
        address = "127.0.0.1"
    listener = require(["/usr/bin/ss", "-H", "-ltnp", "sport", "=", ":3003"], "canary-dashboard")
    socket_lines = [line for line in listener.decode().splitlines()
                    if len(line.split()) >= 5 and line.split()[3] in
                    (f"{address}:3003", "0.0.0.0:3003", "*:3003")]
    if len(socket_lines) != 1:
        raise Refused("canary-dashboard")
    pids = set(re.findall(r"\bpid=([0-9]+)\b", socket_lines[0]))
    if not pids or len(pids) > 16:
        raise Refused("canary-dashboard")
    for pid in pids:
        try:
            payload = Path(f"/proc/{pid}/cgroup").read_text()
            working_dir = os.readlink(f"/proc/{pid}/cwd")
            if "0::" + cgroup not in payload.splitlines() or working_dir != PROJECT:
                raise Refused("canary-dashboard")
        except OSError:
            raise Refused("canary-dashboard") from None
    code = require([
        "/usr/bin/curl", "--noproxy", "*", "-sS", "--max-time", "5",
        "-o", "/dev/null", "-w", "%{http_code}", f"http://{address}:3003/",
    ], "canary-http")
    if not re.fullmatch(rb"2[0-9]{2}", code):
        raise Refused("canary-http")


def activate_scheduler(uid, authorized_at):
    require(user_manager(uid, "start", "bts-scheduler.service"), "activation", limit=90)
    time.sleep(30)
    output = require(user_manager(
        uid, "show", "bts-scheduler.service",
        "-p", "ActiveState", "-p", "SubState", "-p", "Result", "-p", "ExecMainStatus",
    ), "activation")
    try:
        fields = unique_fields(line.split("=", 1) for line in output.decode().splitlines())
        names = {"ActiveState", "SubState", "Result", "ExecMainStatus"}
        if set(fields) != names:
            raise Refused("activation")
        if fields["ActiveState"] != "active" and fields != {
            "ActiveState": "activating", "SubState": "auto-restart",
            "Result": "success", "ExecMainStatus": "0",
        }:
            raise Refused("activation")
        if heartbeat()[1] <= authorized_at:
            raise Refused("activation")
    except (OSError, KeyError, TypeError, ValueError):
        raise Refused("activation") from None


def run_name(run_id, suffix):
    return f"run-{run_id}.{suffix}"


def record_result(root, run_id, outcome, step=None, **extra):
    publish_json(root, run_name(run_id, "result"), {
        "deploy_id": run_id, "outcome": outcome, "step": step, **extra,
    })


def activation_was_released(root, run_id):
    try:
        if read_hold(root) is not None:
            return False
        marker = read_json(root, "last-release")
        evidence = read_json(root, run_name(run_id, "evidence"))
        return bool(marker and evidence and marker.get("kind") == "deploy"
                    and marker.get("deploy_id") == run_id and evidence.get("activation") == "verified")
    except (OSError, ValueError):
        return False


def owned_deploy(root, run_id, target, recovery):
    record = None
    previous = "unknown"
    step = "guard"
    committed = False
    uid = None
    try:
        with lock(root, "transaction.lock", fcntl.LOCK_EX):
            publish_json(root, run_name(run_id, "accepted"), {"deploy_id": run_id, "target": target})
            try:
                try:
                    abandoned = read_hold(root)
                except (OSError, ValueError):
                    if recovery != target:
                        raise Refused("guard") from None
                    abandoned = None
                if abandoned is not None:
                    if abandoned["state"] != "failed":
                        abandoned = transition(root, abandoned, "failed", "abandoned")
                        finish_incident(root, abandoned)
                        raise Refused("abandoned")
                    if not finish_incident(root, abandoned):
                        raise Refused("abandoned")
                uid = bts_uid()
                if not Path(PROJECT).is_dir():
                    raise Refused("cd")
                previous, git = preflight(uid, target)
                deadline = time.time() + 1200
                window(recovery, target, deadline)
                step = "writers-busy"
                with lock(root, "writers.lock", fcntl.LOCK_EX):
                    armed = int(time.time())
                    step = "arm"
                    record = {
                        "schema": "bts_deploy_hold_v1", "deploy_id": run_id,
                        "owner_unit": f"bts-deploy-{run_id}.service", "state": "pending",
                        "step": "arm", "target": target, "previous": previous,
                        "armed_at": armed, "updated_at": armed,
                    }
                    record = transition(root, record, "pending", "arm")
                    step = "stop-dashboard"
                    if not verified_stop_dashboard(uid):
                        raise Refused(step)
                step = "stop-old"
                if not verified_stop(uid):
                    raise Refused(step)
                step = "drain"
                with lock(root, "writers.lock", fcntl.LOCK_EX, timeout=120):
                    pass
                require([*git, "reset", "--hard", target], "reset", limit=90)
                require([*git, "rev-parse", "HEAD"], "head", target.encode() + b"\n")
                require(as_bts("/usr/bin/env", "UV_CACHE_DIR=/tmp/uv-cache", UV, "sync",
                               "--directory", PROJECT, "--locked", "--extra", "model"), "sync", limit=480)
                require(user_manager(uid, "restart", "bts-dashboard.service"), "restart", limit=90)
                time.sleep(30)
                canary()
                dashboard(uid)
                window(recovery, target, deadline, minimum=0)
                require([*git, "rev-parse", "HEAD"], "head", target.encode() + b"\n")
                certified_date = canary()
                step = "release"
                authorized_at = time.time()
                record = transition(root, record, "activating", "activation")
                step = "activation"
                activate_scheduler(uid, authorized_at)
                try:
                    release_date = canary()
                except Refused:
                    raise Refused("activation") from None
                if release_date != certified_date:
                    raise Refused("activation")
                publish_json(root, run_name(run_id, "evidence"), {
                    "activation": "verified", "stop": "not-attempted", "page": "not-attempted",
                })
                step = "release"
                release(root, record, "deploy", authorized_at=authorized_at, certified_date=certified_date)
                committed = True
                stamp = write_stamp()
                if not stamp:
                    print("Scheduler activation passed. Calibration timestamp update failed (stamp write failed). "
                          "Record the warning and follow §4 for calibration diagnostics.")
                print(f"Deployed {target[:7]}\nCANARY PASSED — {target[:7]} is live and healthy", flush=True)
                record_result(root, run_id, "committed", stamp="verified" if stamp else "unverified")
                return 0
            except (Refused, OSError, ValueError, InterruptedError) as error:
                if committed or activation_was_released(root, run_id):
                    return 1
                step = error.step if isinstance(error, Refused) else step
                if isinstance(error, InterruptedError):
                    step = "interrupted"
                if record is None:
                    if getattr(error, "previous", None) is not None:
                        previous = error.previous
                    if step == "abandoned":
                        text = "DEPLOY NOT STARTED — abandoned deployment reconciled."
                    elif step == "guard-stale":
                        text = (f"DEPLOY NOT STARTED — guard-stale failed. Re-run scripts/install-deploy-guard-hetzner.sh "
                                f"as root from {target[:7]} before deploying it. This run did not change the checkout or services.")
                    else:
                        text = f"DEPLOY NOT STARTED — {step} failed. This run did not change the checkout or services."
                    print(f"{text} Previous SHA: {previous[:7]}.", flush=True)
                    record_result(root, run_id, "not-started", step)
                    return 1
                print(f"DEPLOY FAILED at {step} — requested target {target[:7]}, previous SHA {previous[:7]}. "
                      "No automatic rollback was attempted. Recovery has not been verified.", flush=True)
                try:
                    record = transition(root, record, "failed", step)
                except (OSError, ValueError):
                    print("Deployment hold: unverified.", flush=True)
                else:
                    print("Deployment hold: verified.", flush=True)
                containment(root, record, uid)
                if step == "activation":
                    print("Activation failed after certification; the hold was kept. Verify the incident hold "
                          "and any delivery/entry during activation before retrying.", flush=True)
                print("Recovery procedure: INCIDENT.md §4.", flush=True)
                record_result(root, run_id, "failed", step)
                return 1
    except BlockingIOError:
        print("DEPLOY NOT STARTED — hold-busy failed.", flush=True)
        record_result(root, run_id, "not-started", "hold-busy")
        return 1


def release_audit(root, record, event, reason):
    fd = os.open("releases.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                 0o600, dir_fd=root)
    with os.fdopen(fd, "ab") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != UID or info.st_nlink != 1 or info.st_mode & 0o022:
            raise OSError("release log")
        line = json.dumps({
            "event": event, "date": datetime.now(timezone.utc).isoformat(),
            "deploy_id": record["deploy_id"], "state": record["state"], "step": record["step"],
            "target": record["target"], "reason": reason,
        }, ensure_ascii=True)
        stream.write((line + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.fsync(root)


def release_by_hand(root, reason):
    if not reason or not 1 <= len(reason) <= 200 or not reason.strip() or not reason.isprintable():
        print("hand release refused: invalid reason")
        return 1
    try:
        with lock(root, "transaction.lock", fcntl.LOCK_EX):
            record = read_hold(root)
            if record is None or record["state"] != "failed":
                raise Refused("failed hold required")
            target = record["target"]
            head, git = preflight(bts_uid(), target, fetch=False)
            if head != target:
                raise Refused("target mismatch; use a deploy")
            require([*git, "status", "--porcelain", "--untracked-files=no"], "checkout differs; use a deploy", b"")
            try:
                require(as_bts(
                    "/usr/bin/env", "UV_CACHE_DIR=/tmp/uv-cache", "UV_PYTHON_DOWNLOADS=never",
                    UV, "sync", "--directory", PROJECT, "--locked", "--check", "--offline", "--extra", "model",
                ), "sync", limit=120)
            except Refused:
                print(f"hand release refused: the environment could not be verified as synced for {target[:7]}; use a deploy")
                return 1
            canary()
            window(target, target, time.time() + 30, minimum=30)
            release_audit(root, record, "intent", reason)
            release(root, record, "hand")
            try:
                release_audit(root, record, "completion", reason)
            except OSError:
                print("hand release outcome unverified: inspect hold and release audit")
                return 1
            stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            print(f'Deploy hold released by hand at {stamp} (deploy {record["deploy_id"]}, '
                  f'failed at {record["step"]}, target {target[:7]}). Reason: {reason}.')
            print("Cron and boot activation are now permitted. This command did not request a scheduler start.")
            return 0
    except Refused as error:
        print(f"hand release refused: {error.step}")
    except (OSError, ValueError):
        print("hand release refused: guard or transaction unavailable")
    return 1


def write_stamp():
    script = (
        "from datetime import datetime,timezone\n"
        "import os,stat,tempfile\n"
        f"directory={PROJECT!r}+'/data'\n"
        "destination=directory+'/.last_deploy_iso'\n"
        "try:\n"
        "    destination_mode=os.lstat(destination).st_mode\n"
        "except FileNotFoundError:\n"
        "    pass\n"
        "else:\n"
        "    if not stat.S_ISREG(destination_mode):\n"
        "        raise OSError('non-regular stamp destination')\n"
        "fd,path=tempfile.mkstemp(prefix='.deploy-',dir=directory)\n"
        "try:\n"
        "    with os.fdopen(fd,'w') as stream:\n"
        "        stream.write(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ\\n'))\n"
        "        stream.flush()\n"
        "        os.fsync(stream.fileno())\n"
        "    os.replace(path,destination)\n"
        "finally:\n"
        "    if os.path.exists(path): os.unlink(path)\n"
    )
    try:
        return call(as_bts("/usr/bin/python3", "-I", "-c", script))[0] == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def handoff(root, target, event, recovery):
    if not SHA.fullmatch(target) or event not in ("push", "workflow_dispatch"):
        print("DEPLOY NOT STARTED — target failed.")
        return 1
    if event != "workflow_dispatch":
        recovery = ""
    def read_timeout(signum, frame):
        raise TimeoutError("phase input")

    old_alarm_handler = signal.signal(signal.SIGALRM, read_timeout)
    signal.alarm(10)
    try:
        phase = sys.stdin.buffer.read(65537)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm_handler)
    if not phase or len(phase) > 65536 or not phase.endswith(b"\n"):
        print("DEPLOY NOT STARTED — phase payload invalid.")
        return 1
    run_id = f"{int(time.time())}-{secrets.token_hex(12)}"
    name = run_name(run_id, "sh")
    digest = hashlib.sha256(phase).hexdigest()
    publish(root, name, phase, 0o700)
    publish_json(root, run_name(run_id, "metadata"), {
        "deploy_id": run_id, "target": target, "event": event,
        "recovery": recovery, "phase_sha256": digest,
    })
    log = ROOT / run_name(run_id, "log")
    publish(root, log.name, b"", 0o600)
    print(f"Deploy run {run_id}.", flush=True)
    hook = f"/usr/bin/python3 -I /usr/local/bin/bts-deploy-hold reconcile --deploy-id {run_id}"
    argv = [
        "/usr/bin/systemd-run", f"--unit=bts-deploy-{run_id}", "--collect",
        "--property=Type=exec", "--property=RuntimeMaxSec=1200",
        "--property=TimeoutStopSec=120", "--property=KillMode=mixed",
        f"--property=ExecStopPost={hook}", "--property=StandardInput=null",
        f"--property=StandardOutput=append:{log}", f"--property=StandardError=append:{log}",
        "--property=WorkingDirectory=/", "--setenv=PATH=/usr/sbin:/usr/bin:/sbin:/bin",
        "--setenv=LC_ALL=C", "--setenv=BASH_ENV=/dev/null",
        "/bin/bash", str(ROOT / name), run_id, target, digest,
    ]
    try:
        call(argv, limit=15)
    except (OSError, subprocess.TimeoutExpired):
        pass
    deadline = time.monotonic() + 1410
    while time.monotonic() < deadline:
        result = read_json(root, run_name(run_id, "result"))
        if result and result.get("deploy_id") == run_id:
            fd = os.open(log.name, FLAGS, dir_fd=root)
            try:
                if stat.S_ISREG(os.fstat(fd).st_mode):
                    print(os.read(fd, 65536).decode(errors="replace"), end="")
            finally:
                os.close(fd)
            return int(result.get("outcome") != "committed")
        time.sleep(1)
    print(f"Deploy outcome unknown (run {run_id}). Inspect the on-box result and hold read-only. "
          "Contact Eric before any retry.")
    return 1


def run_owner(root, run_id, target, digest):
    if not RUN_ID.fullmatch(run_id) or not SHA.fullmatch(target):
        return 1
    metadata = read_json(root, run_name(run_id, "metadata"))
    if not metadata or (metadata.get("deploy_id"), metadata.get("target"), metadata.get("phase_sha256")) != (
        run_id, target, digest,
    ):
        return 1
    fd = os.open(run_name(run_id, "sh"), FLAGS, dir_fd=root)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != UID or stat.S_IMODE(info.st_mode) != 0o700
                or not 0 < info.st_size <= 65536 or hashlib.sha256(os.read(fd, 65537)).hexdigest() != digest):
            return 1
    finally:
        os.close(fd)

    def interrupted(signum, frame):
        raise InterruptedError()

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    return owned_deploy(root, run_id, target, metadata.get("recovery", ""))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version")
    commands.add_parser("status").add_argument("--json", action="store_true")
    commands.add_parser("verify-unit").add_argument(
        "unit", choices=("bts-scheduler.service", "bts-dashboard.service"),
    )
    commands.add_parser("check-clear").add_argument("--scheduler", action="store_true", required=True)
    run = commands.add_parser("run")
    run.add_argument("name", choices=JOBS)
    run.add_argument("argv", nargs=argparse.REMAINDER)
    commands.add_parser("reconcile").add_argument("--deploy-id", required=True)
    launch = commands.add_parser("handoff")
    launch.add_argument("target")
    launch.add_argument("event")
    launch.add_argument("recovery")
    own = commands.add_parser("owner")
    own.add_argument("run_id")
    own.add_argument("target")
    own.add_argument("digest")
    hand = commands.add_parser("release")
    hand.add_argument("--by-hand", action="store_true", required=True)
    hand.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.command == "version":
        print("bts-deploy-hold 1")
        return 0
    if args.command == "verify-unit":
        try:
            raw = sys.stdin.buffer.read(8193)
            if len(raw) > 8192:
                raise ValueError("unit property size")
            fields = unique_fields(line.split("=", 1) for line in raw.decode().splitlines())
            verify_unit_fields(args.unit, fields)
        except (Refused, ValueError):
            print("FAILED: effective user unit differs from the template", file=sys.stderr)
            return 1
        return 0
    if args.command == "run":
        argv = args.argv
        if argv[:1] == ["--"]:
            argv = argv[1:]
        if not argv or not argv[0].startswith("/"):
            parser.error("expected an absolute command")
        return writer(args.name, argv)
    try:
        with directory() as root:
            if args.command == "handoff":
                return handoff(root, args.target, args.event, args.recovery) if os.geteuid() == 0 else 1
            if args.command == "owner":
                return run_owner(root, args.run_id, args.target, args.digest) if os.geteuid() == 0 else 1
            if args.command == "release":
                return release_by_hand(root, args.reason) if os.geteuid() == 0 else 1
            if args.command == "reconcile":
                if os.geteuid() != 0 or not RUN_ID.fullmatch(args.deploy_id):
                    return 1
                return reconcile(root, args.deploy_id)
            if args.command == "status":
                return status(root, args.json)
            return 0 if clear(root) else 1
    except (ValueError, OSError, Refused):
        if args.command == "status":
            print("deploy hold lookup failed")
            return 2
        return 1


if __name__ == "__main__":
    sys.exit(main())
BTS_DEPLOY_HOLD_PYTHON

printf '[Service]\nExecCondition=%s\n' "$condition" | install_content "$drop" 0644
uid="$(id -u bts)"
[[ "$uid" =~ ^[0-9]+$ ]]
manager() {
    runuser -u bts -- env "XDG_RUNTIME_DIR=/run/user/$uid" systemctl --user "$@"
}
manager daemon-reload
for unit in bts-scheduler.service bts-dashboard.service; do
    if ! reload_pending="$(manager show "$unit" -p NeedDaemonReload --value)" || [[ "$reload_pending" != no ]]; then
        printf '%s\n' 'FAILED: user manager needs reload or could not be queried' >&2
        exit 1
    fi
    properties=(-p ExecStart -p WorkingDirectory -p Environment -p EnvironmentFiles)
    if [[ "$unit" == bts-scheduler.service ]]; then
        properties+=(-p ExecCondition -p SuccessExitStatus)
    fi
    manager show "$unit" --all "${properties[@]}" | /usr/bin/python3 -I "$tool" verify-unit "$unit"
done
install_fingerprint publish "$installer_sha"
