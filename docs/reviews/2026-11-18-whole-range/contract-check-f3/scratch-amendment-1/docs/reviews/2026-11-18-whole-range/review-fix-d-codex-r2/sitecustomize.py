"""Keep review interpreters offline, including child interpreters using PYTHONPATH."""

import sys


def offline(event, arguments):
    if event in {
        "socket.connect", "socket.bind", "socket.sendto", "socket.getaddrinfo",
        "socket.gethostbyname", "socket.gethostbyname_ex",
    }:
        raise PermissionError(f"network disabled in deploy review ({event})")


sys.addaudithook(offline)
