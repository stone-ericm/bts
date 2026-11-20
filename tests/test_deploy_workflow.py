"""Replay the exact deploy shell payload with a restricted local command PATH."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/deploy.yml"
OLD = "619abda7" + "0" * 32
TARGET = "b128854d" + "0" * 32
PHASE_MARKER = b"<<'BTS_DEPLOY_PHASE'\n"


def deploy_payload():
    workflow = WORKFLOW.read_bytes()
    scripts = list(re.finditer(rb"^          script: \|\n", workflow, re.M))
    assert len(scripts) == 1
    body = workflow[scripts[0].end():]
    lines = body.splitlines(keepends=True)
    indent = b" " * 12
    assert all(line == b"\n" or line.startswith(indent) for line in lines)
    executable = b"".join(line.removeprefix(indent) for line in lines)
    assert b"".join(
        indent + line if line != b"\n" else line
        for line in executable.splitlines(keepends=True)
    ) == body
    assert executable.endswith(b"\n") and not executable.endswith(b"\n\n")
    return executable


def shim(name, args):
    """The base can reach reset/sync/restart, but no command can reach a real host."""
    root = Path(os.environ["WRD_TEST_ROOT"])
    with (root / "calls").open("a") as log:
        log.write(json.dumps([name, *args]) + "\n")
    command = set(args)
    head = root / "head"
    if name == "id":
        print("1000")
    elif name == "sleep":
        return 0
    elif name == "curl":
        print("503")
    elif name == "sudo":
        if os.environ["WRD_TEST_CASE"] == "accepted" and args[-1:] == ["version"]:
            print("bts-deploy-hold 1")
        elif os.environ["WRD_TEST_CASE"] == "accepted" and "handoff" in args:
            path = root / "received-phase"
            path.write_bytes(sys.stdin.buffer.read(65537))
            print("accepted phase fixture")
        elif {"git", "rev-parse"} <= command:
            print(head.read_text()[:7] if "--short" in args else head.read_text())
        elif {"git", "reset"} <= command:
            head.write_text(args[-1])
        elif "git" in args or "tailscale" in args:
            return 0
        elif "sync" in args:
            return int(os.environ["WRD_TEST_CASE"] == "sync-error")
        elif {"systemctl", "restart"} <= command:
            return 0
        elif {"systemctl", "is-active"} <= command:
            if "--quiet" not in args:
                print("active")
        elif "bash" in args and ".last_deploy_iso" in args[-1]:
            (root / "stamped").touch()
        else:
            return 94
    else:
        return 95
    return 0


@pytest.fixture
def replay(tmp_path):
    payload = tmp_path / "payload"
    payload.write_bytes(deploy_payload())
    shims = tmp_path / "bin"
    shims.mkdir()
    for command in ("sudo", "systemctl", "systemd-run", "id", "sleep", "curl"):
        executable = shims / command
        executable.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{__file__}" --shim "{command}" "$@"\n')
        executable.chmod(0o700)
    for command in ("cut", "head", "grep"):
        (shims / command).symlink_to(Path("/usr/bin") / command)
    bash_env = tmp_path / "bash_env"
    bash_env.write_text(
        'cd() { builtin cd "$WRD_TEST_ROOT"; }\n'
        '[() {\n'
        '  if [[ "$2" == /etc/systemd/system/bts-scheduler.service ]]; then return 1; fi\n'
        '  builtin [ "$@"\n'
        '}\n'
    )
    (tmp_path / "head").write_text(OLD)

    def run(case):
        env = {
            "PATH": str(shims), "BASH_ENV": str(bash_env), "TARGET_SHA": TARGET,
            "WRD_TEST_ROOT": str(tmp_path), "WRD_TEST_CASE": case,
            "DEPLOY_EVENT": "workflow_dispatch", "RECOVER_SHA": TARGET,
        }
        result = subprocess.run(
            ["/bin/bash", str(payload)], env=env, cwd=tmp_path, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=10,
        )
        calls = tmp_path / "calls"
        return result, [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []

    return run


def test_guard_unavailable_refuses_before_reset_or_service_changes(replay, tmp_path):
    result, calls = replay("guard-unavailable")
    assert (tmp_path / "head").read_text() == OLD
    assert not any({"git", "reset"} <= set(call) for call in calls)
    assert not any("restart" in call or "start" in call or "stop" in call for call in calls)
    assert result.returncode == 1
    assert "DEPLOY NOT STARTED" in result.stdout


def test_phase_handoff_preserves_exact_payload_bytes_and_dispatch_sha(replay, tmp_path):
    result, calls = replay("accepted")
    received = tmp_path / "received-phase"
    assert received.is_file(), (result.stdout, calls)
    assert result.returncode == 0
    literal = deploy_payload()
    assert literal.count(PHASE_MARKER) == 1
    expected, trailer = literal.split(PHASE_MARKER)[1].split(b"BTS_DEPLOY_PHASE\n")
    assert trailer == b""
    assert hashlib.sha256(received.read_bytes()).digest() == hashlib.sha256(expected).digest()
    assert calls[-1][-4:] == ["handoff", TARGET, "workflow_dispatch", TARGET]


if __name__ == "__main__" and sys.argv[1:2] == ["--shim"]:
    raise SystemExit(shim(sys.argv[2], sys.argv[3:]))
