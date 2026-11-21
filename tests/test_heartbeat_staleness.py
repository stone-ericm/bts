"""Tests for scripts/check_heartbeat.py staleness decision logic."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import check_heartbeat
from scripts.check_heartbeat import is_stale


@pytest.mark.parametrize("timestamp", [{}, {"timestamp": "bad"}, {"timestamp": "2026-11-21T12:00:00Z"}])
@pytest.mark.parametrize(
    ("step", "target", "expected_step", "expected_target"),
    [
        ("contest-state", "012abcd", "contest-state", "012abcd"),
        (["reset"], "012abcd", "unknown", "012abcd"),
        ("reset", {"sha": "012abcd"}, "reset", "unknown"),
        ("RESET", "012ABCD", "unknown", "unknown"),
        ("deploy-step-incomplete", "012abcd", "deploy-step-incomplete", "012abcd"),
    ],
)
def test_deploy_failure_precedes_timestamp(tmp_path, timestamp, step, target, expected_step, expected_target):
    path = tmp_path / ".heartbeat"
    path.write_text(json.dumps(dict(timestamp, state="deploy_failed", step=step, target_sha=target)))
    assert is_stale(path) == (
        True, f"Deploy failed at {expected_step} (requested target {expected_target}). See INCIDENT.md §4."
    )


@pytest.mark.parametrize(
    "payload",
    [b"[]", b"\xff", b" " * 65537, b'{"timestamp":"2026-11-21T12:00:00"}'],
    ids=["non-object", "invalid-encoding", "oversized", "naive-timestamp"],
)
def test_unsupported_heartbeat_payload(tmp_path, payload):
    path = tmp_path / ".heartbeat"
    path.write_bytes(payload)
    assert is_stale(path) == (True, "heartbeat unreadable or malformed")


def test_fifo_heartbeat_returns_without_a_writer(tmp_path):
    path = tmp_path / ".heartbeat"
    os.mkfifo(path)
    command = [sys.executable, check_heartbeat.__file__, "--heartbeat-path", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=3)
    assert result.returncode == 1
    assert "heartbeat unreadable or malformed" in result.stdout


def test_deploy_failed_marker_pings_fail_and_skips_churn(tmp_path, monkeypatch, capsys):
    marker = tmp_path / ".heartbeat"
    marker.write_text('{"state":"deploy_failed","step":"sync","target_sha":"012abcd"}')
    pings = []
    churn = []
    monkeypatch.setattr(check_heartbeat, "ping", lambda *args: pings.append(args))
    monkeypatch.setattr(check_heartbeat, "read_nrestarts", lambda *args: churn.append(args))
    monkeypatch.setattr(sys, "argv", ["check_heartbeat.py", "--heartbeat-path", str(marker),
                                     "--ping-url", "https://example.invalid/check"])
    with pytest.raises(SystemExit) as exc:
        check_heartbeat.main()
    assert exc.value.code == 1
    assert pings == [("https://example.invalid/check", "/fail")]
    assert not churn
    assert "Deploy failed at sync (requested target 012abcd). See INCIDENT.md §4." in capsys.readouterr().out


@pytest.fixture(scope="module")
def base_heartbeat_source():
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "show", "24019a9a:scripts/check_heartbeat.py"], cwd=root,
        capture_output=True, text=True, check=True, timeout=5,
    ).stdout


@pytest.mark.parametrize(("state", "fields"), [
    ("running", {}), ("waiting_for_games", {}),
    ("sleeping", {"sleeping_until": "2026-11-21T13:00:00Z"}),
    ("sleeping", {"sleeping_until": "2026-11-21T11:00:00Z"}),
    ("sleeping", {}), ("idle_end_of_day", {}),
    ("stalled", {"stage": "predict", "stalled_for_s": 30}),
    ("something-else", {}),
])
@pytest.mark.parametrize("timestamp", ["2026-11-21T12:00:00Z", "2026-11-20T12:00:00Z"])
def test_ordinary_outputs_match_base(tmp_path, monkeypatch, capsys, base_heartbeat_source, state, fields, timestamp):
    now = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)

    class PinnedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz)

    marker = tmp_path / ".heartbeat"
    marker.write_text(json.dumps(dict(state=state, timestamp=timestamp, **fields)))
    symlink = tmp_path / "heartbeat-link"
    symlink.symlink_to(marker)
    monkeypatch.setattr(sys, "argv", [
        "check_heartbeat.py", "--heartbeat-path", str(symlink), "--no-churn",
        "--ping-url", "https://example.invalid/check",
    ])
    baseline = {"__name__": "heartbeat_baseline"}
    exec(compile(base_heartbeat_source, "<pinned base heartbeat>", "exec"), baseline)
    baseline["datetime"] = PinnedDatetime
    monkeypatch.setattr(check_heartbeat, "datetime", PinnedDatetime)

    def run(main, set_ping):
        pings = []
        set_ping(lambda *args: pings.append(args))
        with pytest.raises(SystemExit) as exc:
            main()
        output = capsys.readouterr()
        return exc.value.code, output.out.encode(), output.err.encode(), pings

    old = run(baseline["main"], lambda ping: baseline.update(ping=ping))
    current = run(check_heartbeat.main, lambda ping: monkeypatch.setattr(check_heartbeat, "ping", ping))
    assert current == old


def _write_hb(path: Path, **kv) -> None:
    path.write_text(json.dumps(kv))


def test_fresh_running_state(tmp_path):
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    _write_hb(hb_path, state="running", timestamp=now.isoformat())
    stale, _ = is_stale(hb_path, now=now)
    assert not stale


def test_stale_running_state(tmp_path):
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    old = now - timedelta(minutes=8)
    _write_hb(hb_path, state="running", timestamp=old.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
    assert "running" in reason.lower()


def test_sleeping_with_future_wakeup_is_fresh(tmp_path):
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    wake = now + timedelta(hours=2)
    _write_hb(hb_path, state="sleeping",
              timestamp=(now - timedelta(hours=1)).isoformat(),
              sleeping_until=wake.isoformat())
    stale, _ = is_stale(hb_path, now=now)
    assert not stale


def test_sleeping_past_wakeup_is_stale(tmp_path):
    """If sleeping_until is in the past by >10 min, daemon should have woken."""
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    wake = now - timedelta(minutes=15)  # past due
    _write_hb(hb_path, state="sleeping",
              timestamp=(now - timedelta(hours=2)).isoformat(),
              sleeping_until=wake.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
    assert "sleeping" in reason.lower()


def test_missing_file_is_stale(tmp_path):
    hb_path = tmp_path / "missing.json"
    stale, reason = is_stale(hb_path)
    assert stale
    assert "not found" in reason.lower()


def test_corrupt_file_is_stale(tmp_path):
    hb_path = tmp_path / "hb.json"
    hb_path.write_text("{not valid json")
    stale, reason = is_stale(hb_path)
    assert stale


def test_idle_end_of_day_stale_after_max_age(tmp_path):
    """idle_end_of_day should be stale after 90 min — it's a brief transitional state."""
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    old = now - timedelta(minutes=95)
    _write_hb(hb_path, state="idle_end_of_day", timestamp=old.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
    assert "idle_end_of_day" in reason.lower() or "stuck" in reason.lower()


def test_idle_end_of_day_fresh_within_max_age(tmp_path):
    """idle_end_of_day with recent timestamp is still fresh."""
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    old = now - timedelta(minutes=30)
    _write_hb(hb_path, state="idle_end_of_day", timestamp=old.isoformat())
    stale, _ = is_stale(hb_path, now=now)
    assert not stale


def test_unknown_state_is_stale(tmp_path):
    """Unknown states (new writer state not yet supported by monitor) trigger stale."""
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    _write_hb(hb_path, state="teleporting", timestamp=now.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
    assert "unknown" in reason.lower()


def test_sleeping_without_wake_target_is_stale(tmp_path):
    """Malformed sleeping heartbeat (no sleeping_until field) is stale."""
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    _write_hb(hb_path, state="sleeping", timestamp=now.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
    assert "sleeping_until" in reason.lower()


def test_fresh_waiting_for_games(tmp_path):
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    _write_hb(hb_path, state="waiting_for_games", timestamp=now.isoformat())
    stale, _ = is_stale(hb_path, now=now)
    assert not stale


def test_stale_waiting_for_games(tmp_path):
    hb_path = tmp_path / "hb.json"
    now = datetime.now(timezone.utc)
    old = now - timedelta(minutes=15)
    _write_hb(hb_path, state="waiting_for_games", timestamp=old.isoformat())
    stale, reason = is_stale(hb_path, now=now)
    assert stale
    assert "waiting" in reason.lower()


def test_stalled_state_is_stale_with_stage_in_reason(tmp_path):
    path = tmp_path / ".heartbeat"
    _write_hb(
        path,
        state="stalled",
        timestamp=datetime.now(timezone.utc).isoformat(),
        stage="computing_features",
        stalled_for_s=1234,
    )

    stale, reason = is_stale(path)

    assert stale is True
    assert "computing_features" in reason
    assert "1234" in reason
