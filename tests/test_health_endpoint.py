"""Tests for the /health endpoint logic."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bts.heartbeat import write_heartbeat, HeartbeatState


@pytest.fixture
def heartbeat_path(tmp_path):
    return tmp_path / ".heartbeat"


def test_health_200_when_heartbeat_fresh(heartbeat_path):
    """Fresh RUNNING heartbeat should return 200 with status=ok."""
    write_heartbeat(heartbeat_path, state=HeartbeatState.RUNNING)

    from bts.web import health_check
    status_code, data = health_check(heartbeat_path)

    assert status_code == 200
    assert data["status"] == "ok"
    assert data["scheduler_state"] == HeartbeatState.RUNNING


def test_health_503_when_heartbeat_missing(heartbeat_path):
    """Missing heartbeat file should return 503 with status=stale."""
    from bts.web import health_check
    status_code, data = health_check(heartbeat_path)

    assert status_code == 503
    assert data["status"] == "stale"


def test_health_200_when_sleeping(heartbeat_path):
    """Sleeping heartbeat with future wake time should return 200."""
    wake = datetime.now(timezone.utc) + timedelta(hours=3)
    write_heartbeat(
        heartbeat_path,
        state=HeartbeatState.SLEEPING,
        sleeping_until=wake,
    )

    from bts.web import health_check
    status_code, data = health_check(heartbeat_path)

    assert status_code == 200
    assert data["scheduler_state"] == HeartbeatState.SLEEPING
