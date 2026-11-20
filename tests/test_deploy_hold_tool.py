"""Replay the exact standalone guard source with local files and commands."""

import ast
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

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


@pytest.fixture
def owner(tmp_path, monkeypatch):
    implementation = ModuleType("fixture_deploy_guard")
    monkeypatch.setenv("BTS_TEST_DEPLOY_GUARD_DIR", str(tmp_path))
    exec(compile(guard_source(), str(INSTALLER), "exec"), implementation.__dict__)
    implementation.ROOT = tmp_path / "guard"
    implementation.ROOT.mkdir(mode=0o755)
    implementation.UID = os.geteuid()
    implementation.PROJECT = str(tmp_path)
    for name in ("writers.lock", "transaction.lock", "state.lock"):
        (implementation.ROOT / name).touch(mode=0o644)
    implementation.time = SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=lambda seconds: None)
    return implementation


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
    owner.canary = lambda: check_state("pending", "canary-contest-state")
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
        assert events[-2:] == ["activation", "stamp"]
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
