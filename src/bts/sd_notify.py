"""Minimal sd_notify client for systemd watchdog integration.

No dependencies beyond stdlib. Silently no-ops when not running under systemd
(NOTIFY_SOCKET env var not set).

This exists because the project doesn't pull in `systemd-python` (a C-extension
dep). sd_notify is a trivial line-oriented protocol over a unix datagram
socket — we can just implement the write directly.
"""
from __future__ import annotations

import os
import socket


def notify_raw(message: str) -> None:
    """Send a raw sd_notify message to $NOTIFY_SOCKET. No-op if unset or unreachable.

    Fire-and-forget: catches any exception (OSError for socket failures,
    UnicodeEncodeError for non-ASCII input, etc.) so the notify path can
    never take down the daemon.
    """
    sock_path = os.environ.get("NOTIFY_SOCKET")
    if not sock_path:
        return
    # '@' prefix indicates abstract namespace on Linux
    if sock_path.startswith("@"):
        sock_path = "\0" + sock_path[1:]
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.sendto(message.encode(), sock_path)
    except Exception:
        # fire-and-forget — heartbeat file + external cron watcher still cover us
        pass
    finally:
        sock.close()


def notify_watchdog() -> None:
    """Reset systemd's watchdog timer. Call periodically (well within WatchdogSec)."""
    notify_raw("WATCHDOG=1")


def notify_ready() -> None:
    """Tell systemd the daemon is ready. Required for Type=notify units at startup."""
    notify_raw("READY=1")
