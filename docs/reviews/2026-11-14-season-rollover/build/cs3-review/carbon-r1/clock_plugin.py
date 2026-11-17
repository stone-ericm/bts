"""Pytest plugin to freeze datetime.datetime.now/utcnow and datetime.date.today.

Imports numpy and pandas BEFORE replacing datetime.datetime and datetime.date so Cython
PyDateTimeAPI initialization binds the real C types without confusion.
"""
from __future__ import annotations

import datetime as _dt
import os
import sys

import numpy  # noqa: F401
import pandas  # noqa: F401

_REAL_DATETIME = _dt.datetime
_REAL_DATE = _dt.date

_ISO = os.environ.get("FAKE_CLOCK_ISO", "2027-07-15T16:00:00+00:00")
_ANCHOR_UTC = _REAL_DATETIME.fromisoformat(_ISO).astimezone(_dt.timezone.utc)


class FrozenDate(_REAL_DATE):
    @classmethod
    def today(cls):
        local_dt = _ANCHOR_UTC.astimezone()
        return _REAL_DATE(local_dt.year, local_dt.month, local_dt.day)


class FrozenDatetime(_REAL_DATETIME):
    @classmethod
    def now(cls, tz=None):
        if tz is None:
            local_dt = _ANCHOR_UTC.astimezone()
            return _REAL_DATETIME(
                local_dt.year, local_dt.month, local_dt.day,
                local_dt.hour, local_dt.minute, local_dt.second, local_dt.microsecond,
            )
        converted = _ANCHOR_UTC.astimezone(tz)
        return _REAL_DATETIME(
            converted.year, converted.month, converted.day,
            converted.hour, converted.minute, converted.second, converted.microsecond,
            tzinfo=converted.tzinfo, fold=converted.fold,
        )

    @classmethod
    def utcnow(cls):
        return _REAL_DATETIME(
            _ANCHOR_UTC.year, _ANCHOR_UTC.month, _ANCHOR_UTC.day,
            _ANCHOR_UTC.hour, _ANCHOR_UTC.minute, _ANCHOR_UTC.second, _ANCHOR_UTC.microsecond,
        )


def _install() -> None:
    _dt.date = FrozenDate  # type: ignore[misc]
    _dt.datetime = FrozenDatetime  # type: ignore[misc]
    for mod in list(sys.modules.values()):
        if mod is None or mod is _dt:
            continue
        name = getattr(mod, "__name__", "")
        if name.startswith(("bts", "tests")):
            if getattr(mod, "datetime", None) is _REAL_DATETIME:
                setattr(mod, "datetime", FrozenDatetime)
            if getattr(mod, "date", None) is _REAL_DATE:
                setattr(mod, "date", FrozenDate)


_install()


def pytest_configure(config):
    _install()


def pytest_collection_finish(session):
    _install()
