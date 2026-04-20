"""Test the sd_notify helper handles missing NOTIFY_SOCKET gracefully
and sends correct messages when present."""
import socket

from bts.sd_notify import notify_watchdog, notify_ready, notify_raw


def test_no_socket_is_noop(monkeypatch):
    """If NOTIFY_SOCKET env is missing, notify_watchdog() silently no-ops."""
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    notify_watchdog()  # must not raise
    notify_ready()  # must not raise


def test_sends_watchdog_message_when_socket_set(monkeypatch, tmp_path):
    """When NOTIFY_SOCKET is a valid unix socket, WATCHDOG=1 is sent."""
    sock_path = str(tmp_path / "notify.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(sock_path)
    srv.settimeout(2)
    try:
        monkeypatch.setenv("NOTIFY_SOCKET", sock_path)
        notify_watchdog()
        data, _ = srv.recvfrom(64)
        assert data == b"WATCHDOG=1"
    finally:
        srv.close()


def test_sends_ready_message_when_socket_set(monkeypatch, tmp_path):
    """notify_ready() sends READY=1."""
    sock_path = str(tmp_path / "notify.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(sock_path)
    srv.settimeout(2)
    try:
        monkeypatch.setenv("NOTIFY_SOCKET", sock_path)
        notify_ready()
        data, _ = srv.recvfrom(64)
        assert data == b"READY=1"
    finally:
        srv.close()


def test_bad_socket_path_does_not_raise(monkeypatch):
    """If NOTIFY_SOCKET points to a nonexistent path, notify_raw silently
    swallows the OSError — this is a fire-and-forget primitive."""
    monkeypatch.setenv("NOTIFY_SOCKET", "/nonexistent/socket/path.sock")
    notify_raw("WATCHDOG=1")  # must not raise
