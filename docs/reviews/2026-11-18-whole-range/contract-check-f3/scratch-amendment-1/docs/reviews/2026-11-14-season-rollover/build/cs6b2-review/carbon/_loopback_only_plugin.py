"""Pytest plugin enforcing loopback-only sockets and forbidding external DNS / connects."""

from __future__ import annotations

import socket

_LOOPBACK_NAMES = {"127.0.0.1", "::1", "localhost", ""}
try:
    _LOOPBACK_NAMES.add(socket.gethostname())
except OSError:
    pass

_real_getaddrinfo = socket.getaddrinfo
_real_gethostbyname = socket.gethostbyname
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _is_allowed_host(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="ignore")
    return host in _LOOPBACK_NAMES or host.startswith("127.")


def _guarded_getaddrinfo(host, port, *args, **kwargs):
    if not _is_allowed_host(host):
        raise AssertionError(f"Forbidden non-loopback DNS lookup: {host!r}")
    return _real_getaddrinfo(host, port, *args, **kwargs)


def _guarded_gethostbyname(hostname):
    if not _is_allowed_host(hostname):
        raise AssertionError(f"Forbidden non-loopback gethostbyname: {hostname!r}")
    return _real_gethostbyname(hostname)


def _check_sockaddr(addr):
    if isinstance(addr, tuple) and len(addr) >= 2:
        if not _is_allowed_host(addr[0]):
            raise AssertionError(f"Forbidden outbound socket connect: {addr!r}")


def _guarded_connect(self, addr):
    _check_sockaddr(addr)
    return _real_connect(self, addr)


def _guarded_connect_ex(self, addr):
    _check_sockaddr(addr)
    return _real_connect_ex(self, addr)


socket.getaddrinfo = _guarded_getaddrinfo
socket.gethostbyname = _guarded_gethostbyname
socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
