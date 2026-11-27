import socket
import sys

import pytest

from bts import scheduler, strategy


def reject_network(event, args):
    if event == "socket.getaddrinfo":
        raise OSError("contract check: external DNS blocked")
    if event in {"socket.connect", "socket.sendto"} and args[0].family in (socket.AF_INET, socket.AF_INET6):
        raise OSError("contract check: network blocked")


def main():
    sys.addaudithook(reject_network)
    print("Strategy:", strategy.__file__, flush=True)
    print("Scheduler:", scheduler.__file__, flush=True)
    return pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
