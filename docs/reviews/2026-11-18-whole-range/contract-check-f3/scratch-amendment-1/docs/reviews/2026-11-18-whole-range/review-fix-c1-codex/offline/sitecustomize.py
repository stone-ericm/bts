"""Deny Internet traffic while preserving the web tests' synthetic loopback server."""

import ipaddress
import socket
import sys


def local(host):
    if host is None or host in ("localhost", b"localhost"):
        return True
    try:
        return ipaddress.ip_address(host.decode() if isinstance(host, bytes) else host).is_loopback
    except (ValueError, TypeError):
        return False


def guard(event, args):
    if event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex"):
        if not local(args[0]):
            raise OSError("External network is disabled for the C1 review")
    if event in ("socket.connect", "socket.sendto", "socket.bind"):
        sock, address = args
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not local(address[0]):
            raise OSError("External network is disabled for the C1 review")


sys.addaudithook(guard)
