"""Run pytest with only loopback sockets available to Python."""

from datetime import date, datetime
import os
from pathlib import Path
import sys
import time


def local_only(event, args):
    if event in ("socket.connect", "socket.bind", "socket.sendto"):
        address = args[1]
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
            return
        raise AssertionError(f"Offline review blocked {event} {address!r}")
    if event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"):
        if args[0] not in (None, "127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"Offline review blocked {event} {args[0]!r}")


sys.addaudithook(local_only)
sys.path.insert(0, str(Path.cwd()))
import pytest

if year := os.environ.get("WR_C3_WALL_YEAR"):
    print("Python process wall clock:", datetime.now(), date.today(), time.time(), flush=True)
    assert datetime.now().year == date.today().year == int(year)
raise SystemExit(pytest.main(sys.argv[1:]))
