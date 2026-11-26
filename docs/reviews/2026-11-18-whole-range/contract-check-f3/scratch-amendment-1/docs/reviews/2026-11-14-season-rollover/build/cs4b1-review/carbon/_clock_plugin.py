"""Pytest plugin for faking wall clock (after importing pandas) and forbidding DNS/network connect."""
from __future__ import annotations

import datetime as _dt
import os
import sys

import pandas as _pd  # noqa: F401 - MUST initialize against real datetime types before replacement!


def _forbid_net(event: str, args: tuple) -> None:
    if event in {
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
        "socket.getnameinfo",
        "socket.sendto",
        "urllib.Request",
    }:
        raise RuntimeError(f"FORBIDDEN NETWORK AUDIT EVENT: {event} {args!r}")


sys.addaudithook(_forbid_net)

_TARGET_ISO = os.environ.get("FAKE_CLOCK_ISO")
if _TARGET_ISO:
    _REAL_DT = _dt.datetime
    _REAL_DATE = _dt.date
    _TARGET_DT = _REAL_DT.fromisoformat(_TARGET_ISO)

    class _FakeDate(_REAL_DATE):
        @classmethod
        def today(cls):
            return _TARGET_DT.date()

    class _FakeDateTime(_REAL_DT):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return _TARGET_DT.replace(tzinfo=None)
            return _TARGET_DT.astimezone(tz)

        @classmethod
        def utcnow(cls):
            return _TARGET_DT.astimezone(_dt.timezone.utc).replace(tzinfo=None)

    _dt.date = _FakeDate
    _dt.datetime = _FakeDateTime
