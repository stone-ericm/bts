"""Keep the review and its Python subprocesses offline."""

import ipaddress
import os
from pathlib import Path
import socket
import sys


def loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def audit(event, args):
    if event in {"socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"}:
        address = args[-1]
        if isinstance(address, tuple) and address and loopback(address[0]):
            return
    elif event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex"}:
        if loopback(args[0]):
            return
    elif event not in {"socket.gethostbyaddr", "socket.getnameinfo"}:
        return
    marker = os.environ.get("CS6B1R2_DENIED")
    if marker:
        with Path(marker).open("a") as log:
            log.write(event + "\n")
    raise AssertionError(f"review denied {event}")


real_gethostbyaddr = socket.gethostbyaddr


def local_hostbyaddr(host):
    # HTTPServer.getfqdn() needs its own name. Supply it without a resolver.
    if loopback(host):
        return "localhost", [], [host]
    return real_gethostbyaddr(host)


sys.addaudithook(audit)
socket.gethostbyaddr = local_hostbyaddr
