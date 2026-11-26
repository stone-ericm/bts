"""Run the requested test files with external sockets disabled."""
import ipaddress
import os
import socket
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TESTS = [
    "tests/test_wr_e_texts.py",
    "tests/test_scheduler_skip_visibility.py",
    "tests/test_scheduler_tail_objective.py",
    "tests/test_stale_delivery_cs4c.py",
    "tests/test_morning_state_cs6b1.py",
    "tests/test_late_delivery_guard.py",
    "tests/test_postponed_delivery_guard.py",
    "tests/test_wr_b_contract.py",
    "tests/scripts/season_walk/",
    "tests/test_lightgbm_optional.py",
    "tests/test_web_saver.py",
]


def guard_socket(method):
    def checked(sock, *args):
        address = args[-1]
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            if not ipaddress.ip_address(address[0]).is_loopback:
                raise AssertionError(f"external socket forbidden: {method.__name__} {address}")
        return method(sock, *args)
    return checked


def guard_dns(host, *args, **kw):
    if host not in ("127.0.0.1", "::1", None):
        raise AssertionError(f"external lookup forbidden: {host!r}")
    return original_dns(host, *args, **kw)


original_dns = socket.getaddrinfo


def main():
    for method_name in ("bind", "connect", "connect_ex", "sendto"):
        setattr(socket.socket, method_name, guard_socket(getattr(socket.socket, method_name)))
    socket.getaddrinfo = guard_dns
    os.environ.setdefault("TZ", "America/New_York")
    time.tzset()
    sys.path.insert(0, str(ROOT))
    variant = os.environ.get("WR_E_MUTATION_SRC")
    if variant:
        sys.path.insert(0, variant)
    os.chdir(ROOT)
    args = sys.argv[1:] or TESTS
    return pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", *args])


if __name__ == "__main__":
    sys.exit(main())
