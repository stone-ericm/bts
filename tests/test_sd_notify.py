"""Test the sd_notify helper handles missing NOTIFY_SOCKET gracefully
and sends correct messages when present.

Note on path length: AF_UNIX paths are capped at ~104 chars on macOS
(108 on Linux). pytest's tmp_path on macOS deep-TMPDIR setups can exceed
this, so we use a short dedicated temp dir under /tmp (explicitly short)
for socket binds.
"""
import os
import socket
import tempfile

import pytest

from bts.sd_notify import notify_watchdog, notify_ready, notify_raw


@pytest.fixture
def short_sock_path():
    """Yield a short AF_UNIX path (respects $TMPDIR but skips pytest's deep
    tmp_path nesting) so macOS's 104-char sun_path cap isn't exceeded.

    pytest's tmp_path nests as $TMPDIR/pytest-of-USER/pytest-N/test_NAME/...
    which on macOS (TMPDIR = /var/folders/xy/abcdef.../T/) can already exceed
    100 chars before any filename is appended. mkdtemp() with no dir= uses
    $TMPDIR directly, yielding ~65-char paths with margin to spare.
    """
    d = tempfile.mkdtemp(prefix="sd_")
    path = os.path.join(d, "s")
    if len(path) >= 104:
        pytest.skip(f"AF_UNIX path too long for test environment ({len(path)} chars)")
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(d)
        except OSError:
            pass


def test_no_socket_is_noop(monkeypatch):
    """If NOTIFY_SOCKET env is missing, notify_watchdog() silently no-ops."""
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    notify_watchdog()  # must not raise
    notify_ready()  # must not raise


def test_sends_watchdog_message_when_socket_set(monkeypatch, short_sock_path):
    """When NOTIFY_SOCKET is a valid unix socket, WATCHDOG=1 is sent."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(short_sock_path)
    srv.settimeout(2)
    try:
        monkeypatch.setenv("NOTIFY_SOCKET", short_sock_path)
        notify_watchdog()
        data, _ = srv.recvfrom(64)
        assert data == b"WATCHDOG=1"
    finally:
        srv.close()


def test_sends_ready_message_when_socket_set(monkeypatch, short_sock_path):
    """notify_ready() sends READY=1."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(short_sock_path)
    srv.settimeout(2)
    try:
        monkeypatch.setenv("NOTIFY_SOCKET", short_sock_path)
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


def test_non_encodable_message_does_not_raise(monkeypatch, short_sock_path):
    """notify_raw with a non-ASCII-surrogate message must not raise
    UnicodeEncodeError — it's fire-and-forget."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    srv.bind(short_sock_path)
    srv.settimeout(2)
    try:
        monkeypatch.setenv("NOTIFY_SOCKET", short_sock_path)
        # Surrogate pairs are not UTF-8 encodable — .encode() raises
        notify_raw("WATCHDOG=1\ud800")  # must not raise
    finally:
        srv.close()
