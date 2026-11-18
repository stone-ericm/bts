"""Network guard for the review runners. Test cases can still use loopback."""

import ipaddress
import os
import socket
import sys


def forbid_network(*, loopback=False):
    def local(address):
        if not loopback or not isinstance(address, str):
            return False
        try:
            return ipaddress.ip_address(address).is_loopback
        except ValueError:
            return False

    def audit(event, args):
        if event == "socket.connect":
            sock, address = args
            if sock.family == socket.AF_UNIX:
                return
            if isinstance(address, tuple) and local(address[0]):
                return
            raise AssertionError(f"outward connect forbidden: {address!r}")
        if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
                     "socket.getnameinfo"}:
            address = args[0]
            host = address[0] if isinstance(address, tuple) else address
            if local(host) or host in {None, "localhost"} and loopback:
                return
            raise AssertionError(f"DNS forbidden: {address!r}")
        if event == "socket.sendto":
            sock, address = args
            if sock.family != socket.AF_UNIX and not (isinstance(address, tuple) and local(address[0])):
                raise AssertionError(f"outward send forbidden: {address!r}")

    sys.addaudithook(audit)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)
