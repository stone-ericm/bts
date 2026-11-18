"""Offline process guard for the review harness and its collection subprocess."""

import ipaddress
import os
import socket
import sys


def loopback(host):
    if host is None:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def deny(event, args):
    if event in {"socket.getaddrinfo", "socket.gethostbyname",
                 "socket.gethostbyaddr", "socket.getnameinfo"}:
        host = args[0][0] if isinstance(args[0], tuple) else args[0]
        allowed = loopback(host)
    elif event in {"socket.connect", "socket.sendto", "socket.sendmsg", "socket.bind"}:
        sock, address = args[:2]
        allowed = (
            sock.family in {socket.AF_INET, socket.AF_INET6}
            and isinstance(address, tuple)
            and loopback(address[0])
        )
    else:
        return
    if not allowed:
        print(f"OFFLINE GUARD BLOCKED {event}", file=sys.__stderr__)
        raise AssertionError(f"offline review forbids {event}")


if os.environ.get("CS4C_OFFLINE") == "1":
    sys.addaudithook(deny)
