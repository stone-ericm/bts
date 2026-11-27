import socket
import sys
from pathlib import Path


def offline(event, args):
    if event == "socket.getaddrinfo":
        raise RuntimeError("review: DNS disabled")
    if event in {"socket.connect", "socket.sendto"} and args[0].family in (
        socket.AF_INET, socket.AF_INET6
    ):
        raise RuntimeError("review: network disabled")


if __name__ == "__main__":
    sys.addaudithook(offline)
    import bts
    import pytest

    print("bts:", Path(bts.__file__).resolve(), flush=True)
    raise SystemExit(pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", *sys.argv[1:]]))
