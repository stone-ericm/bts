"""Network guard helper for offline CS4c adversarial review probes.

Forbids DNS lookups and outward socket connections while optionally allowing
loopback when explicitly requested.
"""

from __future__ import annotations

import socket
import urllib.request
from contextlib import contextmanager


class NetworkForbiddenError(AssertionError):
    """Raised when code under test attempts DNS or an outward network call."""


def _is_loopback_address(address: object) -> bool:
    if isinstance(address, (str, bytes)):
        # AF_UNIX local path
        return True
    if isinstance(address, tuple) and address:
        host = str(address[0])
        return host in {"127.0.0.1", "::1", "localhost"}
    return False


@contextmanager
def forbid_network(*, allow_loopback: bool = False):
    orig_getaddrinfo = socket.getaddrinfo
    orig_gethostbyname = socket.gethostbyname
    orig_gethostbyname_ex = socket.gethostbyname_ex
    orig_gethostbyaddr = socket.gethostbyaddr
    orig_create_connection = socket.create_connection
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_urlopen = urllib.request.urlopen

    def blocked_dns(*args, **kwargs):
        if allow_loopback and args and str(args[0]) in {"127.0.0.1", "::1", "localhost"}:
            return orig_getaddrinfo(*args, **kwargs)
        raise NetworkForbiddenError(f"DNS lookup forbidden: {args!r}")

    def blocked_host(*args, **kwargs):
        raise NetworkForbiddenError(f"Host lookup forbidden: {args!r}")

    def guarded_create_connection(address, *args, **kwargs):
        if allow_loopback and _is_loopback_address(address):
            return orig_create_connection(address, *args, **kwargs)
        raise NetworkForbiddenError(f"Outward create_connection forbidden: {address!r}")

    def guarded_connect(self, address):
        if allow_loopback and _is_loopback_address(address):
            return orig_connect(self, address)
        raise NetworkForbiddenError(f"Outward socket.connect forbidden: {address!r}")

    def guarded_connect_ex(self, address):
        if allow_loopback and _is_loopback_address(address):
            return orig_connect_ex(self, address)
        raise NetworkForbiddenError(f"Outward socket.connect_ex forbidden: {address!r}")

    def blocked_urlopen(*args, **kwargs):
        raise NetworkForbiddenError(f"urllib.request.urlopen forbidden: {args!r}")

    socket.getaddrinfo = blocked_dns
    socket.gethostbyname = blocked_host
    socket.gethostbyname_ex = blocked_host
    socket.gethostbyaddr = blocked_host
    socket.create_connection = guarded_create_connection
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    urllib.request.urlopen = blocked_urlopen
    try:
        yield
    finally:
        socket.getaddrinfo = orig_getaddrinfo
        socket.gethostbyname = orig_gethostbyname
        socket.gethostbyname_ex = orig_gethostbyname_ex
        socket.gethostbyaddr = orig_gethostbyaddr
        socket.create_connection = orig_create_connection
        socket.socket.connect = orig_connect
        socket.socket.connect_ex = orig_connect_ex
        urllib.request.urlopen = orig_urlopen
