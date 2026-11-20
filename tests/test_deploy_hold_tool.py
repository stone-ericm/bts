"""Replay the exact standalone guard source with local files and commands."""

import ast
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
import urllib.error
import urllib.request

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install-deploy-guard-hetzner.sh"
GUARD_MARKER = "BTS_DEPLOY_HOLD_PYTHON"


def guard_source():
    lines = INSTALLER.read_text().splitlines(keepends=True)
    start, = (i for i, line in enumerate(lines) if f"<<'{GUARD_MARKER}'" in line)
    end, = (i for i, line in enumerate(lines) if line.rstrip("\n") == GUARD_MARKER)
    return "".join(lines[start + 1:end])


def test_guard_and_monitor_parse_as_311_with_standard_library_imports():
    for source in (guard_source(), (ROOT / "scripts" / "check_heartbeat.py").read_text()):
        tree = ast.parse(source, feature_version=(3, 11))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] in sys.stdlib_module_names for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0 and node.module is not None
                assert node.module.split(".")[0] in sys.stdlib_module_names


@pytest.fixture
def guard(tmp_path):
    tool = tmp_path / "bts-deploy-hold"
    tool.write_text(guard_source())
    root = tmp_path / "guard"
    root.mkdir(mode=0o755)
    for name in ("writers.lock", "transaction.lock", "state.lock"):
        (root / name).touch(mode=0o644)

    def run(*args):
        return subprocess.run(
            [sys.executable, "-I", str(tool), *args],
            env={"BTS_TEST_DEPLOY_GUARD_DIR": str(root), "PATH": "/nonexistent"},
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=3,
        )

    return root, run


def test_clear_status_and_version(guard):
    _, run = guard
    result = run("version")
    assert (result.returncode, result.stdout) == (0, "bts-deploy-hold 1\n")
    result = run("status")
    assert (result.returncode, result.stdout) == (0, "no deploy hold\n")
    assert run("check-clear", "--scheduler").returncode == 0


@pytest.mark.parametrize("entry", ["directory", "fifo", "symlink", "malformed"])
def test_nonregular_and_malformed_records_fail_closed(guard, tmp_path, entry):
    root, run = guard
    target = tmp_path / "target"
    target.write_text("preserve")
    hold = root / "hold"
    if entry == "directory":
        hold.mkdir()
    elif entry == "fifo":
        os.mkfifo(hold)
    elif entry == "symlink":
        hold.symlink_to(target)
    else:
        hold.write_text('{"state":"failed","state":"pending"}')
    assert run("check-clear", "--scheduler").returncode == 1
    assert run("status").returncode == 1
    assert run("run", "reconcile", "--", "/nonexistent").returncode == 75
    assert target.read_text() == "preserve"


def test_writer_gate_inherits_lock_and_preserves_job_exit_status(guard):
    root, run = guard
    lock = root / "writers.lock"
    with lock.open("rb") as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blocked = run("run", "reconcile", "--", "/nonexistent")
        assert blocked.returncode == 75
        assert "lock busy" in blocked.stdout
    program = (
        "import fcntl, sys\n"
        "with open(sys.argv[1], 'rb') as file:\n"
        "    try:\n"
        "        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "    except BlockingIOError:\n"
        "        raise SystemExit(37)\n"
        "raise SystemExit(99)\n"
    )
    assert run("run", "reconcile", "--", sys.executable, "-c", program, str(lock)).returncode == 37


def _python311():
    candidate = os.environ.get("BTS_TEST_PYTHON311")
    if not candidate and (uv := shutil.which("uv")):
        try:
            found = subprocess.run(
                [uv, "python", "find", "3.11"], env={**os.environ, "UV_PYTHON_DOWNLOADS": "never"},
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
            )
            if found.returncode == 0:
                candidate = found.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        if candidate and Path(candidate).is_absolute():
            checked = subprocess.run(
                [candidate, "-I", "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
            )
            if (checked.returncode, checked.stdout) == (0, "3.11\n"):
                return candidate
    except (OSError, subprocess.TimeoutExpired):
        pass
    message = "no Python 3.11 interpreter on this machine (set BTS_TEST_PYTHON311)"
    if os.environ.get("BTS_REQUIRE_TEST_PYTHON311") == "1":
        pytest.fail(message)
    pytest.skip(message)


def test_guard_and_monitor_run_on_python311(tmp_path):
    python = _python311()
    tool = tmp_path / "bts-deploy-hold"
    tool.write_text(guard_source())
    for name in ("writers.lock", "transaction.lock", "state.lock"):
        (tmp_path / name).touch(mode=0o644)
    env = {"BTS_TEST_DEPLOY_GUARD_DIR": str(tmp_path), "PATH": "/nonexistent"}

    def run(*args):
        return subprocess.run(
            [python, "-I", *map(str, args)], env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=5,
        )

    version = run(tool, "version")
    assert (version.returncode, version.stdout) == (0, "bts-deploy-hold 1\n")
    hb = tmp_path / ".heartbeat"
    now = datetime.now(timezone.utc)
    hb.write_text(json.dumps({
        "state": "sleeping", "timestamp": now.isoformat(),
        "sleeping_until": (now + timedelta(hours=1)).isoformat(),
    }))
    args = (ROOT / "scripts" / "check_heartbeat.py", "--heartbeat-path", hb, "--no-churn")
    clear = run(*args)
    assert clear.returncode == 0 and "fresh sleeping" in clear.stdout
    (tmp_path / "hold").write_text("bad schema")
    assert run(tool, "check-clear", "--scheduler").returncode == 1
    held = run(*args)
    assert held.returncode == 1 and "Deploy hold present and unreadable" in held.stdout


def load_guard(root):
    implementation = ModuleType("fixture_deploy_guard")
    exec(compile(guard_source(), str(INSTALLER), "exec"), implementation.__dict__)
    implementation.ROOT = root
    implementation.UID = os.geteuid()
    implementation.PROJECT = str(root.parent)
    implementation.time = SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=lambda seconds: None)
    implementation.verified_stop_dashboard = lambda uid: True
    return implementation


@pytest.fixture
def owner(tmp_path, monkeypatch):
    root = tmp_path / "guard"
    root.mkdir(mode=0o755)
    monkeypatch.setenv("BTS_TEST_DEPLOY_GUARD_DIR", str(root))
    for name in ("writers.lock", "transaction.lock", "state.lock"):
        (root / name).touch(mode=0o644)
    return load_guard(root)


@pytest.mark.parametrize("failure", [None, "canary-contest-state", "activation"])
def test_owner_transaction_uses_real_hold_and_terminal_evidence(owner, capsys, failure):
    events = []
    old = "a" * 40
    new = "b" * 40
    run_id = "1795197580-testowner"
    owner.bts_uid = lambda: 1000
    owner.preflight = lambda uid, target: (old, ["fake-git"])
    owner.window = lambda *args, **kwargs: None

    def required(argv, step, expected=None, **kwargs):
        if step == "reset":
            assert argv[-1] == new
        events.append(step)
        return expected or b""

    def check_state(state, stage):
        record = json.loads((owner.ROOT / "hold").read_text())
        assert record["state"] == state
        assert record["target"] == new
        events.append(stage)
        if stage == failure:
            raise owner.Refused(stage)

    owner.require = required
    owner.verified_stop = lambda uid: (events.append("stop"), True)[1]
    owner.canary = lambda: check_state(
        "activating" if "activation" in events else "pending", "canary-contest-state",
    )
    owner.dashboard = lambda uid: check_state("pending", "canary-dashboard")
    owner.activate_scheduler = lambda uid, at: check_state("activating", "activation")
    owner.page = lambda hold, stopped: (events.append("page"), stopped)[1]
    owner.write_stamp = lambda: (events.append("stamp"), True)[1]

    with owner.directory() as root:
        result = owner.owned_deploy(root, run_id, new, "")
    terminal = json.loads((owner.ROOT / f"run-{run_id}.result").read_text())
    assert events[:5] == ["stop", "reset", "head", "sync", "restart"]
    if failure is None:
        assert result == 0 and terminal["outcome"] == "committed"
        assert events[-3:] == ["activation", "canary-contest-state", "stamp"]
        assert "CANARY PASSED" in capsys.readouterr().out
        assert not (owner.ROOT / "hold").exists()
        assert json.loads((owner.ROOT / "last-release").read_text())["kind"] == "deploy"
    else:
        assert result == 1 and (terminal["outcome"], terminal["step"]) == ("failed", failure)
        assert events[-2:] == ["stop", "page"]
        assert "stamp" not in events and "CANARY PASSED" not in capsys.readouterr().out
        assert json.loads((owner.ROOT / "hold").read_text())["state"] == "failed"
        assert not (owner.ROOT / "last-release").exists()
        if failure == "canary-contest-state":
            assert "activation" not in events


def test_reconcile_finishes_failed_evidence_once_and_ignores_other_run(owner):
    run_id = "1795197580-fixture"
    armed_at = int(time.time()) - 30
    record = {
        "schema": "bts_deploy_hold_v1", "deploy_id": run_id,
        "owner_unit": f"bts-deploy-{run_id}.service", "state": "failed", "step": "sync",
        "previous": "a" * 40, "target": "b" * 40, "armed_at": armed_at, "updated_at": armed_at,
    }
    events = []
    owner.bts_uid = lambda: 1000
    owner.verified_stop = lambda uid: (events.append("stop"), True)[1]
    owner.page = lambda current, stopped: (events.append("page"), stopped)[1]
    with owner.directory() as root:
        owner.publish_json(root, "hold", record)
        owner.publish_json(root, f"run-{run_id}.evidence", {
            "stop": "unverified", "page": "not-attempted",
        })
        assert owner.reconcile(root, "1795197580-other") == 0
        assert events == []
        assert owner.reconcile(root, run_id) == 0
        assert owner.reconcile(root, run_id) == 0
        assert events == ["stop", "page"]


@pytest.mark.parametrize("state", ["matching", "mismatch", "missing", "unreadable", "fifo", "symlink"])
def test_preflight_checks_installed_guard_against_exact_target_blob(owner, monkeypatch, state):
    target, previous = "b" * 40, "a" * 40
    source = b"#!/bin/bash\n# exact target bytes\n"
    digest = hashlib.sha256(source).hexdigest().encode() + b"\n"
    record = owner.ROOT / "guard-install.sha256"
    if state in ("matching", "unreadable"):
        record.write_bytes(digest)
    elif state == "mismatch":
        record.write_bytes(b"0" * 64 + b"\n")
    elif state == "fifo":
        os.mkfifo(record)
    elif state == "symlink":
        link_target = owner.ROOT / "digest"
        link_target.write_bytes(digest)
        record.symlink_to(link_target)
    if state == "unreadable":
        real_open = owner.os.open

        def deny_record(path, *args, **kwargs):
            if path == record.name:
                raise PermissionError("injected read refusal")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(owner.os, "open", deny_record)

    calls = []

    def require(argv, step, expected=None, *, limit=10):
        calls.append((step, argv))
        if "rev-parse" in argv:
            return previous.encode() + b"\n"
        if "git-config" == step and state == "matching":
            return b""
        if argv[-2:] == ["fetch", "origin"] and state == "matching":
            return b""
        if "cat-file" in argv:
            return b"commit\n"
        if argv[-2:] == ["show", f"{target}:scripts/install-deploy-guard-hetzner.sh"]:
            assert 0 < limit <= 10
            return source
        if step == "user-manager":
            raise owner.Refused("fixture-reached-unit-check")
        pytest.fail(f"unexpected command: {argv!r}")

    owner.require = require
    with pytest.raises(owner.Refused) as refused:
        owner.preflight(1000, target)
    if state == "matching":
        assert refused.value.step == "fixture-reached-unit-check"
    else:
        assert refused.value.step == "guard-stale"
        assert refused.value.previous == previous
        assert not any(step == "user-manager" for step, _ in calls)
    assert not (owner.ROOT / "hold").exists()


def test_stale_guard_refusal_identifies_target_and_previous_sha(owner, capsys):
    target, previous, run_id = "b" * 40, "a" * 40, "1795197580-stale"
    owner.bts_uid = lambda: 1000

    def stale(*args):
        raise owner.Refused("guard-stale", previous)

    owner.preflight = stale
    with owner.directory() as root:
        assert owner.owned_deploy(root, run_id, target, "") == 1
    assert capsys.readouterr().out == (
        "DEPLOY NOT STARTED — guard-stale failed. Re-run scripts/install-deploy-guard-hetzner.sh "
        "as root from bbbbbbb before deploying it. This run did not change the checkout or services. "
        "Previous SHA: aaaaaaa.\n"
    )
    assert not (owner.ROOT / "hold").exists()


def test_remote_only_target_with_stale_guard_never_fetches_into_checkout(owner):
    target, previous = "b" * 40, "a" * 40
    temporary = "/tmp/bts-deploy-target.abcdefgh"
    (owner.ROOT / "guard-install.sha256").write_bytes(b"0" * 64 + b"\n")
    calls = []

    def require(argv, step, expected=None, *, limit=10):
        calls.append(argv)
        if argv[-2:] == ["rev-parse", "HEAD"]:
            return previous.encode() + b"\n"
        if argv[-3:] == ["cat-file", "-t", target]:
            if temporary not in argv:
                raise owner.Refused("target")
            return b"commit\n"
        if argv[-3:] == ["remote", "get-url", "origin"]:
            return b"fixture-origin\n"
        if "/usr/bin/mktemp" in argv:
            return temporary.encode() + b"\n"
        if temporary in argv and ("init" in argv or "fetch" in argv):
            return b""
        if temporary in argv and argv[-2:] == ["show", f"{target}:scripts/install-deploy-guard-hetzner.sh"]:
            return b"new installation bytes\n"
        pytest.fail(f"unexpected command: {argv!r}")

    owner.require = require
    owner.call = lambda argv: calls.append(argv)
    with pytest.raises(owner.Refused) as refused:
        owner.preflight(1000, target)
    assert (refused.value.step, refused.value.previous) == ("guard-stale", previous)
    assert all(temporary in argv for argv in calls if "fetch" in argv)
    assert calls[-1][-4:] == ["/usr/bin/rm", "-rf", "--", temporary]
    assert not (owner.ROOT / "hold").exists()


TARGET = "b" * 40
PREVIOUS = "a" * 40
RUN_ID = "1795197580-coverage"


def hold_record(state="failed"):
    armed = int(time.time()) - 10
    return {
        "schema": "bts_deploy_hold_v1", "deploy_id": RUN_ID,
        "owner_unit": f"bts-deploy-{RUN_ID}.service", "state": state, "step": "sync",
        "previous": PREVIOUS, "target": TARGET, "armed_at": armed, "updated_at": armed,
    }


@pytest.mark.parametrize("case", [
    "no-hold", "pending", "activating", "target", "dirty", "uv-unsynced", "uv-unsupported",
    "uv-missing", "reason-empty", "reason-blank", "reason-newline", "reason-long", "busy", "audit", "success",
])
def test_hand_release_checks_and_orders_durable_audit(owner, capsys, case):
    state = case if case in {"pending", "activating"} else "failed"
    record = hold_record(state)
    events = []
    reason = {
        "reason-empty": "", "reason-blank": "  ", "reason-newline": "foo\nbar", "reason-long": "a" * 201,
    }.get(case, "verified delivery, entry and window")
    owner.bts_uid = lambda: 1000
    owner.preflight = lambda uid, target, fetch: (PREVIOUS if case == "target" else TARGET, ["fixture-git"])
    owner.canary = lambda: events.append("canary")
    owner.window = lambda *args, **kwargs: events.append("window")

    def call(argv, *, limit):
        if "status" in argv:
            return 0, b" M file\n" if case == "dirty" else b""
        assert [arg for arg in argv if arg in ("--check", "--locked", "--offline")] == [
            "--locked", "--check", "--offline",
        ]
        events.append("uv")
        if case == "uv-missing":
            raise FileNotFoundError("fixture uv")
        return {"uv-unsynced": 1, "uv-unsupported": 2}.get(case, 0), b""

    owner.call = call
    original = owner.release_audit

    def audit(root, current, event, why):
        assert (owner.ROOT / "hold").exists() == (event == "intent")
        events.append(event)
        if case == "audit":
            raise OSError("fixture audit write failure")
        original(root, current, event, why)

    owner.release_audit = audit
    with owner.directory() as root:
        if case != "no-hold":
            owner.publish_json(root, "hold", record)
        busy = owner.lock(root, "transaction.lock", fcntl.LOCK_EX) if case == "busy" else nullcontext()
        with busy:
            rc = owner.release_by_hand(root, reason)
        if case == "success":
            assert rc == 0
            assert events == ["uv", "canary", "window", "intent", "completion"]
            assert owner.read_hold(root) is None
            assert owner.read_json(root, "last-release")["kind"] == "hand"
            audit_lines = (owner.ROOT / "releases.log").read_text().splitlines()
            assert [json.loads(line)["event"] for line in audit_lines] == ["intent", "completion"]
        else:
            assert rc == 1
            assert owner.read_hold(root) == (None if case == "no-hold" else record)
            assert owner.read_json(root, "last-release") is None
            assert "completion" not in events
            if case.startswith("uv-"):
                assert "environment could not be verified as synced" in capsys.readouterr().out
            elif case != "audit":
                assert "intent" not in events


@pytest.mark.parametrize("state,age,wake,recover,allowed", [
    ("running", 300, None, False, False), ("running", 5, None, True, False),
    ("waiting_for_games", 600, None, False, False), ("waiting_for_games", 5, None, True, False),
    ("sleeping", 2, 1799, False, False), ("sleeping", 2, 1799, True, False),
    ("sleeping", 2, 1800, False, True), ("running", 301, None, False, False),
    ("running", 301, None, True, True), (None, None, None, False, False),
    (None, None, None, True, True),
])
def test_window_uses_pinned_heartbeat_and_exact_recovery(owner, state, age, wake, recover, allowed):
    now = 1_800_000_000
    owner.time.time = lambda: now
    if state is not None:
        timestamp = datetime.fromtimestamp(now - age, timezone.utc)
        record = {"state": state, "timestamp": timestamp.isoformat()}
        if wake is not None:
            record["sleeping_until"] = datetime.fromtimestamp(now + wake, timezone.utc).isoformat()
        data = Path(owner.PROJECT) / "data"
        data.mkdir()
        (data / ".heartbeat").write_text(json.dumps(record))
    if allowed:
        owner.window(TARGET if recover else "", TARGET, now + 1200)
    else:
        with pytest.raises(owner.Refused, match="window"):
            owner.window(TARGET if recover else PREVIOUS, TARGET, now + 1200)


@pytest.mark.parametrize("response,code,lookup,listener,step", [
    (b"204", 0, True, True, None), (b"503", 0, True, True, "canary-http"),
    (b"2A0", 0, True, True, "canary-http"), (b"999", 0, True, True, "canary-http"),
    (b"200", 28, True, True, "canary-http"), (b"200", 0, False, True, None),
    (b"200", 0, False, False, "canary-dashboard"),
    (b"200", 0, True, False, "canary-dashboard"),
])
def test_dashboard_checks_listener_and_exact_success_code(owner, monkeypatch, response, code, lookup, listener, step):
    cgroup = "/fixture/bts-dashboard.service"
    address = "100.72.1.2" if lookup else "127.0.0.1"
    commands = {
        "/usr/bin/tailscale": (0, b"100.72.1.2\n") if lookup else (1, b""),
        "/usr/bin/ss": (0, f'LISTEN 0 5 {address}:3003 0.0.0.0:* users:(("web",pid=123,fd=6))\n'.encode()),
        "/usr/bin/curl": (code, response),
    }
    calls = []

    def call(argv, *, limit):
        calls.append(argv)
        if "ActiveState" in argv:
            return 0, b"active\n"
        if "ControlGroup" in argv:
            return 0, cgroup.encode() + b"\n"
        return next(result for command, result in commands.items() if command in argv)

    owner.call = call
    owner.Path = lambda path: SimpleNamespace(
        read_text=lambda: "0::" + (cgroup if listener else "/fixture/other") + "\n",
    )
    monkeypatch.setattr(owner.os, "readlink", lambda path: owner.PROJECT)
    if step is None:
        owner.dashboard(1000)
    else:
        with pytest.raises(owner.Refused, match=step):
            owner.dashboard(1000)
    assert any(f"http://{address}:3003/" in argv for argv in calls) == listener


@pytest.mark.parametrize("kind", ["directory", "fifo", "regular"])
def test_stamp_rejects_nonregular_destination_without_opening_it(owner, kind):
    data = Path(owner.PROJECT) / "data"
    data.mkdir()
    stamp = data / ".last_deploy_iso"
    if kind == "directory":
        stamp.mkdir()
    elif kind == "fifo":
        os.mkfifo(stamp)
    else:
        stamp.write_text("old\n")
    owner.as_bts = lambda python, *args: [sys.executable, *args]
    assert owner.write_stamp() == (kind == "regular")
    assert not list(data.glob(".deploy-*"))
    if kind == "directory":
        assert stamp.is_dir()
    elif kind == "fifo":
        assert stamp.is_fifo()
    else:
        assert stamp.read_text().strip().endswith("Z")


_PING = "BTS_SCHEDULER_HEARTBEAT_PING_URL="


@pytest.mark.parametrize("value,send", [
    ("https://hc-ping.com/token_9-a", True), ("http://hc-ping.com/token", False),
    ("https://hc-ping.com.attacker.invalid/token", False), ("https://user@hc-ping.com/token", False),
    ("https://hc-ping.com:443/token", False), ("https://hc-ping.com/token?q=1", False),
    ("https://hc-ping.com/token#f", False), ("https://hc-ping.com/foo/token", False),
    ('"https://hc-ping.com/token"', False),
    ("https://hc-ping.com/token\n" + _PING + "https://hc-ping.com/other", False),
])
def test_ping_allowlist_and_failed_post(owner, monkeypatch, value, send):
    (Path(owner.PROJECT) / ".env").write_text(_PING + value + "\n")
    requests = []

    def open_request(request, *, timeout):
        assert timeout == 10
        requests.append(request)
        raise urllib.error.URLError("fixture refused POST")

    def opener(proxy, redirect):
        assert proxy.proxies == {}
        assert redirect.redirect_request(None, None, 302, "", {}, "https://elsewhere.invalid") is None
        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(urllib.request, "build_opener", opener)
    assert owner.page(hold_record(), True) is None
    assert len(requests) == int(send)
    if send:
        assert requests[0].get_method() == "POST"
        assert requests[0].full_url == value + "/fail"
        assert b"requested target bbbbbbb" in requests[0].data


def configure_transaction(owner, boundary=lambda phase: None):
    owner.bts_uid = lambda: 1000

    def preflight(uid, target):
        boundary("preflight")
        return PREVIOUS, ["fixture-git"]

    def require(argv, step, expected=None, **kwargs):
        boundary(step)
        return expected or b""

    owner.preflight = preflight
    owner.verified_target = lambda *args, **kwargs: (PREVIOUS, ["fixture-git"])
    owner.require = require
    owner.window = lambda *args, **kwargs: boundary("window")
    owner.verified_stop = lambda uid: (boundary("stop-old"), True)[1]
    owner.canary = lambda: boundary("candidate")
    owner.dashboard = lambda uid: boundary("dashboard")
    owner.activate_scheduler = lambda uid, moment: boundary("activation")
    owner.write_stamp = lambda: (boundary("released"), True)[1]
    owner.page = lambda record, stopped: (boundary("page"), True)[1]


def child_command(owner, operation, *args):
    return [sys.executable, __file__, operation, str(owner.ROOT), *args]


def prepare_owner(owner, root):
    phase = b"exec fixture\n"
    digest = hashlib.sha256(phase).hexdigest()
    owner.publish(root, owner.run_name(RUN_ID, "sh"), phase, 0o700)
    owner.publish_json(root, owner.run_name(RUN_ID, "metadata"), {
        "deploy_id": RUN_ID, "target": TARGET, "phase_sha256": digest,
    })
    return digest


def interrupt_owner(owner, phase, termination, digest):
    delivered = False

    def boundary(name):
        nonlocal delivered
        if name == phase and not delivered:
            delivered = True
            # The shim models systemd's RuntimeMaxSec expiry as TERM to MainPID.
            os.kill(os.getpid(), signal.SIGKILL if termination == "KILL" else signal.SIGTERM)

    configure_transaction(owner, boundary)
    publish_json = owner.publish_json
    lock = owner.lock

    def publish(root, name, content):
        publish_json(root, name, content)
        if name == "hold" and content["state"] == "pending":
            boundary("armed")
        elif name.endswith(".evidence") and content.get("activation") == "verified":
            boundary("activated")
        elif name.endswith(".result") and content["outcome"] == "committed":
            boundary("reported")

    def lock_at_boundary(root, name, operation, timeout=0):
        if name == "writers.lock" and timeout:
            boundary("drain")
        return lock(root, name, operation, timeout)

    owner.publish_json = publish
    owner.lock = lock_at_boundary
    with owner.directory() as root:
        return owner.run_owner(root, RUN_ID, TARGET, digest)


@pytest.mark.parametrize("phase", [
    "preflight", "window", "armed", "stop-old", "drain", "reset", "sync", "restart", "candidate",
    "dashboard", "activation", "activated", "released", "reported",
])
@pytest.mark.parametrize("termination", ["TERM", "KILL", "RuntimeMaxSec"])
@pytest.mark.parametrize("recovery", ["hook", "next-preflight"])
def test_owner_interruption_at_phase_boundary(owner, phase, termination, recovery):
    with owner.directory() as root:
        digest = prepare_owner(owner, root)
        child = subprocess.run(
            child_command(owner, "interrupt", phase, termination, digest),
            stdin=subprocess.DEVNULL, capture_output=True, timeout=5, text=True,
        )
        assert child.returncode == (-signal.SIGKILL if termination == "KILL" else 1), child.stderr
        early = phase in {"preflight", "window"}
        released = phase in {"released", "reported"}
        assert owner.read_json(root, owner.run_name(RUN_ID, "accepted"))["target"] == TARGET
        record = owner.read_hold(root)
        terminal = owner.read_json(root, owner.run_name(RUN_ID, "result"))
        if released:
            assert record is None
            assert owner.read_json(root, "last-release")["deploy_id"] == RUN_ID
        if phase == "reported":
            assert terminal["outcome"] == "committed"
        elif termination == "KILL" or released:
            assert terminal is None
        else:
            assert (terminal["outcome"], terminal["step"]) == (
                "not-started" if early else "failed", "interrupted",
            )
        events = []
        configure_transaction(owner, events.append)
        if recovery == "hook":
            assert owner.reconcile(root, RUN_ID) == 0
            if early or released:
                assert events == []
                assert owner.read_hold(root) is None
            else:
                assert owner.read_hold(root)["state"] == "failed"
                assert events == (["stop-old", "page"] if termination == "KILL" else [])
        elif record is not None and record["state"] != "failed":
            follower = "1795197581-next"
            assert owner.owned_deploy(root, follower, TARGET, "") == 1
            assert owner.read_hold(root)["step"] == "abandoned"
            assert events == ["stop-old", "page"]
            result = owner.read_json(root, owner.run_name(follower, "result"))
            assert result["step"] == "abandoned"
        else:
            assert owner.owned_deploy(root, "1795197581-next", TARGET, "") == 0
            assert events.count("page") == 0
            assert owner.read_hold(root) is None
        if released:
            assert owner.read_json(root, "last-release")["kind"] == "deploy"


def await_file(path):
    deadline = time.monotonic() + 4
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(path.name)
        time.sleep(0.01)


def disconnected_follower(owner):
    def launch(argv, *, limit):
        assert limit == 15
        assert "--property=RuntimeMaxSec=1200" in argv
        assert "--property=TimeoutStopSec=120" in argv
        assert "--property=KillMode=mixed" in argv
        run_id, target, digest = argv[-3:]
        assert any(arg.endswith(f"reconcile --deploy-id {run_id}") for arg in argv)
        temporary = owner.ROOT / "launch-request.tmp"
        temporary.write_text(json.dumps([run_id, target, digest]))
        temporary.replace(owner.ROOT / "launch-request")
        await_file(owner.ROOT / "launch-reply")
        return 0, b""

    def poll(seconds):
        (owner.ROOT / "following").touch()
        time.sleep(seconds)

    signal.signal(signal.SIGHUP, signal.SIG_DFL)
    owner.sys.stdin = SimpleNamespace(buffer=BytesIO(b"#!/bin/bash\nexec fixture\n"))
    owner.call = launch
    owner.time.sleep = poll
    with owner.directory() as root:
        return owner.handoff(root, TARGET, "push", TARGET)


def detached_service(owner, run_id, target, digest):
    def boundary(phase):
        if phase == "preflight":
            await_file(owner.ROOT / "continue-owner")

    configure_transaction(owner, boundary)
    with owner.directory() as root:
        return owner.run_owner(root, run_id, target, digest)


def test_handoff_survives_loss_of_follower_and_owner_reaches_result(owner):
    with owner.directory() as root:
        follower = subprocess.Popen(
            child_command(owner, "follower"), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        )
        service = None
        try:
            launch_request = owner.ROOT / "launch-request"
            await_file(launch_request)
            run_id, target, digest = json.loads(launch_request.read_text())
            service = subprocess.Popen(
                child_command(owner, "service", run_id, target, digest),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            )
            await_file(owner.ROOT / owner.run_name(run_id, "accepted"))
            (owner.ROOT / "launch-reply").touch()
            await_file(owner.ROOT / "following")
            follower.send_signal(signal.SIGHUP)
            assert follower.wait(timeout=5) == -signal.SIGHUP
            (owner.ROOT / "continue-owner").touch()
            assert service.wait(timeout=5) == 0
        finally:
            for process in (follower, service):
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
        assert owner.read_json(root, owner.run_name(run_id, "result"))["outcome"] == "committed"
        assert owner.read_json(root, owner.run_name(run_id, "metadata"))["recovery"] == ""


@pytest.mark.parametrize("conflict", ["owner", "writer-before-arm", "drain-timeout"])
def test_transaction_refuses_lock_contention_at_correct_boundary(owner, conflict):
    events = []
    configure_transaction(owner, events.append)
    original_lock = owner.lock

    if conflict == "drain-timeout":
        def lock(root, name, operation, timeout=0):
            if name == "writers.lock" and timeout:
                assert timeout == 120
                raise BlockingIOError("fixture drain deadline elapsed")
            return original_lock(root, name, operation, timeout)
        owner.lock = lock

    with owner.directory() as root:
        if conflict == "owner":
            with owner.lock(root, "transaction.lock", fcntl.LOCK_EX):
                assert owner.owned_deploy(root, RUN_ID, TARGET, "") == 1
            assert not (owner.ROOT / owner.run_name(RUN_ID, "accepted")).exists()
        elif conflict == "writer-before-arm":
            open_writer = []

            def acquire_writer(*args, **kwargs):
                context = owner.lock(root, "writers.lock", fcntl.LOCK_SH)
                context.__enter__()
                open_writer.append(context)

            owner.window = acquire_writer
            try:
                assert owner.owned_deploy(root, RUN_ID, TARGET, "") == 1
            finally:
                for context in open_writer:
                    context.__exit__(None, None, None)
        else:
            assert owner.owned_deploy(root, RUN_ID, TARGET, "") == 1
        result = owner.read_json(root, owner.run_name(RUN_ID, "result"))
        assert result["step"] == {"owner": "hold-busy", "writer-before-arm": "writers-busy",
                                  "drain-timeout": "drain"}[conflict]
        assert "reset" not in events
        if conflict == "drain-timeout":
            assert owner.read_hold(root)["state"] == "failed"
            assert events.count("page") == 1
        else:
            assert owner.read_hold(root) is None
            assert "stop-old" not in events


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_owner_records_stamp_failure_as_committed(owner, capsys, kind):
    stamp_writer = owner.write_stamp
    configure_transaction(owner)
    owner.write_stamp = stamp_writer
    owner.as_bts = lambda python, *args: [sys.executable, *args]
    data = Path(owner.PROJECT) / "data"
    data.mkdir()
    stamp = data / ".last_deploy_iso"
    if kind == "directory":
        stamp.mkdir()
    else:
        os.mkfifo(stamp)
    with owner.directory() as root:
        assert owner.owned_deploy(root, RUN_ID, TARGET, "") == 0
        result = owner.read_json(root, owner.run_name(RUN_ID, "result"))
        assert (result["outcome"], result["stamp"]) == ("committed", "unverified")
        assert owner.read_hold(root) is None
    assert "Calibration timestamp update failed" in capsys.readouterr().out
    assert stamp.is_dir() if kind == "directory" else stamp.is_fifo()


@pytest.mark.parametrize("locked", [False, True])
def test_f12_valid_pending_fences_scheduler_and_every_writer(guard, locked):
    root, run = guard
    (root / "hold").write_text(json.dumps(hold_record("pending")))
    with (root / "transaction.lock").open("rb") as channel:
        if locked:
            fcntl.flock(channel, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run("check-clear", "--scheduler").returncode == 1
        for job in ("check-results", "reconcile", "fetch-contest-streak", "nightly-preview"):
            result = run("run", job, "--", "/usr/bin/true")
            assert result.returncode == 75
            assert "skipped: held" in result.stdout


if __name__ == "__main__":
    operation, root, *args = sys.argv[1:]
    command = {"interrupt": interrupt_owner, "follower": disconnected_follower, "service": detached_service}[operation]
    raise SystemExit(command(load_guard(Path(root)), *args))
