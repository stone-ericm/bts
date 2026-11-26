"""Pytest plugin for pinning Python's wall clock (datetime.datetime, datetime.date, time.time)
after pre-initializing pandas Cython tslibs so C extension type init sees C PyDateTimeAPI.
"""

from __future__ import annotations

import datetime as _dt_mod
import os
import time as _time_mod

import pandas as _pd  # noqa: F401 — pre-initialize Cython PyDateTimeAPI before replacing datetime.datetime

_ORIG_DATE = _dt_mod.date
_ORIG_DATETIME = _dt_mod.datetime


def _install_wall_clock(iso_str: str) -> None:
    pinned = _ORIG_DATETIME.fromisoformat(iso_str)
    assert pinned.tzinfo is not None
    pinned_ts = pinned.timestamp()

    class PinnedDate(_ORIG_DATE):
        @classmethod
        def today(cls):
            d = pinned.date()
            return _ORIG_DATE(d.year, d.month, d.day)

    class PinnedDatetime(_ORIG_DATETIME):
        @classmethod
        def now(cls, tz=None):
            return pinned.astimezone(tz) if tz is not None else pinned.replace(tzinfo=None)

        @classmethod
        def today(cls):
            return pinned.replace(tzinfo=None)

        @classmethod
        def utcnow(cls):
            return pinned.astimezone(_dt_mod.timezone.utc).replace(tzinfo=None)

    _dt_mod.date = PinnedDate  # type: ignore[misc]
    _dt_mod.datetime = PinnedDatetime  # type: ignore[misc]
    _time_mod.time = lambda: pinned_ts


def pytest_configure(config) -> None:
    iso = os.environ.get("FAKE_WALL_CLOCK_ISO")
    if iso:
        _install_wall_clock(iso)
