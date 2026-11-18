"""Pytest plugin for pinning Python's wall clock (datetime.datetime, datetime.date, time.time)
to FAKE_WALL_CLOCK_ISO across an entire test session.
"""

from __future__ import annotations

import datetime as _dt_mod
import os
import time as _time_mod

_ORIG_DATE = _dt_mod.date
_ORIG_DATETIME = _dt_mod.datetime
_ORIG_TIME = _time_mod.time


def _install_wall_clock(iso_str: str) -> None:
    pinned = _ORIG_DATETIME.fromisoformat(iso_str)
    assert pinned.tzinfo is not None
    pinned_ts = pinned.timestamp()

    class PinnedDate(_ORIG_DATE):
        @classmethod
        def today(cls):
            d = pinned.date()
            return cls(d.year, d.month, d.day)

    class PinnedDatetime(_ORIG_DATETIME):
        @classmethod
        def now(cls, tz=None):
            target = pinned.astimezone(tz) if tz is not None else pinned.replace(tzinfo=None)
            return cls(
                target.year, target.month, target.day,
                target.hour, target.minute, target.second, target.microsecond,
                tzinfo=target.tzinfo, fold=target.fold,
            )

        @classmethod
        def today(cls):
            return cls.now(None)

        @classmethod
        def utcnow(cls):
            u = pinned.astimezone(_dt_mod.timezone.utc).replace(tzinfo=None)
            return cls(u.year, u.month, u.day, u.hour, u.minute, u.second, u.microsecond)

    _dt_mod.date = PinnedDate  # type: ignore[misc]
    _dt_mod.datetime = PinnedDatetime  # type: ignore[misc]
    _time_mod.time = lambda: pinned_ts


_iso = os.environ.get("FAKE_WALL_CLOCK_ISO")
if _iso:
    _install_wall_clock(_iso)
