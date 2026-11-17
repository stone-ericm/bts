"""Local-only audit guard shared by the review probes and Python subprocesses."""

import os
import sys
from pathlib import Path

opens = []
network = []
watch = []


def audit(event, args):
    if event in {
        "socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
        "socket.gethostbyname_ex", "socket.gethostbyaddr", "socket.getnameinfo",
        "socket.sendto", "socket.sendmsg",
    }:
        network.append((event, repr(args)))
        raise PermissionError(f"review blocked {event}")
    if event == "open" and watch and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).absolute()
        if any(path.is_relative_to(parent) for parent in watch):
            opens.append((str(path), args[1], args[2]))


sys.addaudithook(audit)
