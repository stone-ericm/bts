"""Offline socket guard inherited by subprocesses launched by the review."""

import ipaddress
import os
import socket
import sys


def loopback(host):
    if isinstance(host, bytes):
        host = host.decode("ascii")
    if host in ("localhost", "localhost."):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def install():
    original_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if not loopback(host):
            raise AssertionError(f"review blocked DNS for {host!r}")
        if host in ("localhost", "localhost.", b"localhost", b"localhost."):
            host = "::1" if family == socket.AF_INET6 else "127.0.0.1"
        return original_getaddrinfo(host, port, family, type, proto, flags | socket.AI_NUMERICHOST)

    def gethostbyaddr(host):
        if not loopback(host):
            raise AssertionError(f"review blocked reverse DNS for {host!r}")
        return "localhost", [], [host]

    def audit(event, args):
        if event in {
            "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo",
        }:
            raise AssertionError(f"review blocked resolver call {event}")
        if event == "socket.getaddrinfo" and not loopback(args[0]):
            raise AssertionError(f"review blocked resolver call {event}")
        if event in {"socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"}:
            sock, address = args[:2]
            if address is not None and sock.family != socket.AF_UNIX:
                # A numeric address is required so a connect or bind cannot resolve a name.
                if not isinstance(address, tuple) or not ipaddress.ip_address(address[0]).is_loopback:
                    raise AssertionError(f"review blocked {event} to {address!r}")

    socket.getaddrinfo = getaddrinfo
    socket.gethostbyaddr = gethostbyaddr
    sys.addaudithook(audit)


if os.environ.get("CS6B2_OFFLINE") == "1":
    install()
