"""Scoped socket guard for the review and its pytest subprocesses."""

from contextlib import ExitStack, contextmanager
from ipaddress import ip_address
import socket
from unittest.mock import patch


def loopback(host):
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


@contextmanager
def offline():
    with ExitStack() as stack:
        for name in ("connect", "connect_ex", "sendto"):
            original = getattr(socket.socket, name)

            def checked(sock, *args, _name=name, _original=original, **kwargs):
                address = args[-1]
                if sock.family in (socket.AF_INET, socket.AF_INET6):
                    if not isinstance(address, tuple) or not loopback(address[0]):
                        raise AssertionError(f"unexpected outward socket {_name}: {address!r}")
                return _original(sock, *args, **kwargs)

            stack.enter_context(patch.object(socket.socket, name, checked))

        original_lookup = socket.getaddrinfo

        def lookup(host, port, family=0, type=0, proto=0, flags=0):
            if not loopback(host):
                raise AssertionError(f"unexpected DNS lookup: {host!r}")
            return original_lookup(host, port, family, type, proto, flags | socket.AI_NUMERICHOST)

        def deny(*args, **kwargs):
            raise AssertionError(f"unexpected name lookup: {args!r}")

        stack.enter_context(patch.object(socket, "getaddrinfo", lookup))
        for name in ("gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo"):
            stack.enter_context(patch.object(socket, name, deny))
        yield


def pytest_configure(config):
    import datetime as calendar
    import os
    from zoneinfo import ZoneInfo

    with ExitStack() as setup:
        setup.enter_context(offline())
        year = os.environ.get("REVIEW_WALL_YEAR")
        if year:
            class WallClock(calendar.datetime):
                @classmethod
                def now(cls, tz=None):
                    instant = cls(int(year), 7, 15, 12, tzinfo=ZoneInfo("America/New_York"))
                    return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

                @classmethod
                def utcnow(cls):
                    return cls.now(calendar.UTC).replace(tzinfo=None)

            setup.enter_context(patch.object(calendar, "datetime", WallClock))
            print(f"Review wall clock: {WallClock.now()}", flush=True)
        config.add_cleanup(setup.pop_all().close)
