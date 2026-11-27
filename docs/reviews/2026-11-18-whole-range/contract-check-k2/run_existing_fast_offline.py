import socket
import sys

import pytest

from bts import scheduler, strategy


def reject_network(event, args):
    if event == "socket.getaddrinfo":
        raise RuntimeError("contract check: external DNS blocked")
    if event in {"socket.connect", "socket.sendto"} and args[0].family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError("contract check: network blocked")


if __name__ == "__main__":
    sys.addaudithook(reject_network)
    print("Strategy:", strategy.__file__, flush=True)
    print("Scheduler:", scheduler.__file__, flush=True)
    sys.exit(pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", *sys.argv[1:]]))
