"""Pytest plugin to pin the default wall clock (datetime.now / date.today) to FAKE_WALL_CLOCK_ISO."""
from __future__ import annotations

import datetime as dt
import os
from zoneinfo import ZoneInfo

import pandas  # noqa: F401 - bind real C datetime types before replacing dt.datetime / dt.date

_REAL_DATETIME = dt.datetime
_REAL_DATE = dt.date
_ET = ZoneInfo("America/New_York")


def pytest_configure(config):
    raw = os.environ.get("FAKE_WALL_CLOCK_ISO")
    if not raw:
        return
    target = _REAL_DATETIME.fromisoformat(raw)
    if target.tzinfo is None:
        target = target.replace(tzinfo=_ET)

    class PinnedDate(_REAL_DATE):
        @classmethod
        def today(cls):
            return target.astimezone(_ET).date()

    class PinnedDatetime(_REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            return target.astimezone(tz) if tz is not None else target.replace(tzinfo=None)

        @classmethod
        def utcnow(cls):
            return target.astimezone(dt.timezone.utc).replace(tzinfo=None)

    dt.date = PinnedDate
    dt.datetime = PinnedDatetime
