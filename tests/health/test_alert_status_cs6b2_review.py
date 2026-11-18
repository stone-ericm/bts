"""Round-1 review of change set 6b-2 (Codex BLOCK 1 and 2).

The status path can be replaced between the dispatcher's type check and its open.
A FIFO swapped in at that moment must not block the read (before any gate) or the
write (after the DM has gone out): the open itself is non-blocking and the type is
re-checked on the opened descriptor.
"""

import json
import logging
import os
import signal
from pathlib import Path
from unittest.mock import Mock

import pytest

from bts.health import alert
from tests.health.test_alert_status_cs6b2 import CRITICAL, DAY, dispatch


@pytest.fixture
def transport(monkeypatch):
    send = Mock()
    monkeypatch.setattr(alert, "send_dm", send)
    return send


@pytest.fixture
def bounded():
    """Fail, not hang, when a dispatcher call blocks."""
    def timeout(*_):
        pytest.fail("dispatcher blocked on a replaced status path")

    prior_handler = signal.signal(signal.SIGALRM, timeout)
    prior_timer = signal.setitimer(signal.ITIMER_REAL, 3)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior_handler)
        signal.setitimer(signal.ITIMER_REAL, *prior_timer)


def swap_to_fifo_after_stat(monkeypatch, path, nth, *, with_reader=False):
    """After the nth ``Path.stat`` of ``path`` has answered, put a FIFO there."""
    real_stat = Path.stat
    seen = {"count": 0, "reader": None}
    fifo = path.with_name(path.name + ".fifo")
    os.mkfifo(fifo)

    def stat_then_swap(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        if self == path:
            seen["count"] += 1
            if seen["count"] == nth:
                if with_reader:
                    seen["reader"] = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
                os.replace(fifo, path)
        return result

    monkeypatch.setattr(Path, "stat", stat_then_swap)
    return seen


@pytest.mark.parametrize("mode", ["critical", "empty", "no_recipient"])
def test_fifo_swapped_in_before_the_read_cannot_block(tmp_path, monkeypatch, transport, bounded, mode):
    path = tmp_path / "health_dm_delivery_status.json"
    path.write_text("{}")
    seen = swap_to_fifo_after_stat(monkeypatch, path, 1)
    alerts = [] if mode == "empty" else [CRITICAL]
    recipient = None if mode == "no_recipient" else "offline.test"
    result = dispatch(path, alerts, recipient=recipient)
    assert seen["count"] >= 1
    # Exactly what a missing status file gives.
    assert result is (mode == "critical")
    assert transport.call_count == (1 if mode == "critical" else 0)


@pytest.mark.parametrize("with_reader", [False, True])
@pytest.mark.parametrize("mode", ["critical", "no_recipient"])
def test_fifo_swapped_in_before_the_write_cannot_block(
        tmp_path, monkeypatch, transport, bounded, caplog, mode, with_reader):
    path = tmp_path / "health_dm_delivery_status.json"
    path.write_text("{}")
    seen = swap_to_fifo_after_stat(monkeypatch, path, 2, with_reader=with_reader)
    recipient = None if mode == "no_recipient" else "offline.test"
    try:
        with caplog.at_level(logging.WARNING):
            result = dispatch(path, [CRITICAL], recipient=recipient)
    finally:
        if seen["reader"] is not None:
            os.close(seen["reader"])
    assert seen["count"] >= 2
    assert result is (mode == "critical")
    assert transport.call_count == (1 if mode == "critical" else 0)
    assert "could not write health DM delivery status" in caplog.text


def test_regular_and_dangling_writes_are_as_before(tmp_path, transport):
    """Controls: the non-blocking open must not change an ordinary write."""
    plain = tmp_path / "plain.json"
    assert dispatch(plain, [CRITICAL]) is True
    first = plain.read_bytes()
    assert json.loads(first)["status"] == "sent" and first.endswith(b"\n")
    reference = tmp_path / "reference.txt"
    reference.write_text("x")
    assert (plain.stat().st_mode & 0o777) == (reference.stat().st_mode & 0o777)
    # A longer prior body is fully replaced (truncation), not overwritten in place.
    plain.write_text(json.dumps({"status": "failed", "critical_count": 0, "pad": "p" * 5000}))
    assert dispatch(plain, [CRITICAL], day=DAY) is True
    assert json.loads(plain.read_bytes())["status"] == "sent"
    target = tmp_path / "nested" / "target.json"
    target.parent.mkdir()
    link = tmp_path / "link.json"
    link.symlink_to(target)
    assert dispatch(link, [CRITICAL]) is True
    assert link.is_symlink() and json.loads(target.read_bytes())["status"] == "sent"
    missing_parent = tmp_path / "a" / "b" / "status.json"
    assert dispatch(missing_parent, [CRITICAL]) is True
    assert json.loads(missing_parent.read_bytes())["status"] == "sent"
