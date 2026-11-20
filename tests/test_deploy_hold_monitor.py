"""Hold-aware monitor decisions against local files and a recording ping shim."""

import json
import os
import fcntl
import sys
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import pytest

from scripts import check_heartbeat


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    now = datetime(2026, 11, 20, 18, tzinfo=timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    monkeypatch.setattr(check_heartbeat, "datetime", Clock)
    guard = tmp_path / "guard"
    guard.mkdir()
    for name in ("writers.lock", "transaction.lock", "state.lock"):
        (guard / name).touch(mode=0o644)
    monkeypatch.setenv("BTS_TEST_DEPLOY_GUARD_DIR", str(guard))
    heartbeat = tmp_path / ".heartbeat"
    pings = []
    monkeypatch.setattr(check_heartbeat, "ping", lambda url, suffix="": pings.append(suffix))
    if hasattr(check_heartbeat, "ping_deploy_failure"):
        monkeypatch.setattr(check_heartbeat, "ping_deploy_failure", lambda url, reason: pings.append("/fail"))
    monkeypatch.setattr(sys, "argv", [
        "check_heartbeat.py", "--heartbeat-path", str(heartbeat), "--no-churn",
        "--ping-url", "https://hc-ping.com/test-token",
    ])

    def run(timestamp, *, state="sleeping"):
        heartbeat.write_text(json.dumps({
            "timestamp": timestamp.isoformat(),
            "state": state,
            "sleeping_until": (now + timedelta(hours=2)).isoformat(),
        }))
        with pytest.raises(SystemExit) as exited:
            check_heartbeat.main()
        return exited.value.code

    return guard, now, pings, run


def test_held_monitor_overrides_fresh_sleeping_heartbeat(monitor, capsys):
    guard, now, pings, run = monitor
    os.mkfifo(guard / "hold")
    assert run(now - timedelta(seconds=20)) == 1
    assert pings == ["/fail"]
    assert "Deploy hold present and unreadable" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["hand", "deploy"])
def test_release_marker_requires_strictly_newer_heartbeat(monitor, capsys, kind):
    guard, now, pings, run = monitor
    released = now - timedelta(seconds=20)
    (guard / "last-release").write_text(json.dumps({
        "released_at": int(released.timestamp()), "kind": kind, "deploy_id": "1795197580-412",
    }))
    assert run(released) == 1
    assert pings == ["/fail"]
    assert f"({kind}); no scheduler heartbeat since." in capsys.readouterr().out
    assert run(released + timedelta(seconds=1)) == 0
    assert pings == ["/fail", ""]


def test_monitor_without_hold_or_release_keeps_legacy_output(monitor, capsys):
    _, now, pings, run = monitor
    assert run(now - timedelta(seconds=20)) == 0
    assert pings == [""]
    assert capsys.readouterr().out == "[2026-11-20T18:00:00+00:00] stale=False  reason=fresh sleeping\n"


def test_release_marker_does_not_make_a_stale_heartbeat_fresh(monitor):
    guard, now, pings, run = monitor
    marker = guard / "last-release"
    marker.write_text(json.dumps({
        "released_at": int((now - timedelta(hours=1)).timestamp()),
        "kind": "deploy", "deploy_id": "1795194000-412",
    }))
    assert run(now - timedelta(minutes=6), state="running") == 1
    assert pings == ["/fail"]


@pytest.mark.parametrize("state", ["pending", "activating"])
def test_monitor_suppresses_only_for_live_matching_owner(monitor, monkeypatch, capsys, state):
    guard, now, pings, run = monitor
    run_id = "1795197580-412"
    (guard / "hold").write_text("{}")
    record = {
        "state": state, "step": "activation", "target": "a" * 40,
        "deploy_id": run_id, "armed_at": int(now.timestamp()) - 30,
    }
    seen = []

    def command(argv, **kwargs):
        seen.append(argv)
        assert kwargs["timeout"] <= 3
        if argv[0] == "/usr/bin/python3":
            return SimpleNamespace(returncode=1, stdout=json.dumps(record).encode())
        assert argv == [
            "/usr/bin/systemctl", "show", f"bts-deploy-{run_id}.service", "-p", "ActiveState", "--value",
        ]
        return SimpleNamespace(returncode=0, stdout=b"active\n")

    monkeypatch.setattr(check_heartbeat.subprocess, "run", command)
    with (guard / "transaction.lock").open("rb") as channel:
        fcntl.flock(channel, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run(now - timedelta(seconds=1)) == 0
    assert pings == []
    assert f"run {run_id}" in capsys.readouterr().out
    assert len(seen) == 2
    assert run(now - timedelta(seconds=1)) == 1
    assert pings == ["/fail"]
