"""Pytest plugin to fake Python wall-clock APIs (datetime.now, date.today, time.time).

Activated via `-p fake_clock_plugin` with `BTS_FAKE_NOW_ISO` in the environment
(e.g. `2027-07-15T18:00:00+00:00` or `2028-07-15T18:00:00+00:00`).

How it works & known harness limits:
- Installs subclasses `FakeDate` and `FakeDatetime` onto `datetime.date` and
  `datetime.datetime` in `pytest_configure` BEFORE test/application modules are
  imported, plus patches `time.time` and `time.time_ns`.
- Uses `time.monotonic()` delta from startup so elapsed wall time advances
  normally (avoiding zero-delta hangs in wait loops, logging, or pytest timers),
  while `time.monotonic` and `time.perf_counter` remain untouched so thread locks
  and `Event.wait(timeout=...)` work normally.
- `__new__`, `now`, `utcnow`, `today`, `fromtimestamp`, `fromisoformat`, and
  `strptime` return real `datetime.date` / `datetime.datetime` C instances so
  C extensions (pandas, json, zoneinfo) see genuine C types; metaclass
  `__instancecheck__` / `__subclasscheck__` delegate to the real C types so
  `isinstance(x, datetime.date)` and `issubclass(datetime.datetime, datetime.date)`
  remain True.
- Harness limits: OS-level file `st_mtime` timestamps and C libraries calling
  `gettimeofday()` directly still see the kernel clock; `type(d) is datetime.date`
  (strict identity against the rebound attribute, as opposed to `isinstance`)
  would be False because `datetime.date` is `FakeDate` while instances have
  `type(d) is _real_date`.
"""
from __future__ import annotations

import datetime as _dt
import os
import time as _time
from zoneinfo import ZoneInfo

_real_date = _dt.date
_real_datetime = _dt.datetime
_real_time = _time.time
_real_monotonic = _time.monotonic

_ANCHOR_UTC: _real_datetime | None = None
_BOOT_MONO: float = 0.0


def _current_utc() -> _real_datetime:
    assert _ANCHOR_UTC is not None
    elapsed = _real_monotonic() - _BOOT_MONO
    return _ANCHOR_UTC + _dt.timedelta(seconds=elapsed)


def _local_tz():
    tz_name = os.environ.get("TZ", "America/New_York")
    return ZoneInfo(tz_name)


class _DateMeta(type):
    def __instancecheck__(cls, instance) -> bool:
        return isinstance(instance, _real_date)

    def __subclasscheck__(cls, subclass) -> bool:
        return issubclass(subclass, _real_date)


class _DatetimeMeta(_DateMeta):
    def __instancecheck__(cls, instance) -> bool:
        return isinstance(instance, _real_datetime)

    def __subclasscheck__(cls, subclass) -> bool:
        return issubclass(subclass, _real_datetime)


class FakeDate(_real_date, metaclass=_DateMeta):
    def __new__(cls, year, month=None, day=None):
        target_cls = _real_date if cls is FakeDate else cls
        if month is None and isinstance(year, (bytes, str)):
            return _real_date.__new__(target_cls, year)
        return _real_date.__new__(target_cls, year, month, day)

    @classmethod
    def today(cls):
        now_local = _current_utc().astimezone(_local_tz())
        return _real_date(now_local.year, now_local.month, now_local.day)

    @classmethod
    def fromtimestamp(cls, t):
        dt_local = _real_datetime.fromtimestamp(t, tz=_local_tz())
        return _real_date(dt_local.year, dt_local.month, dt_local.day)

    @classmethod
    def fromisoformat(cls, date_string):
        return _real_date.fromisoformat(date_string)

    @classmethod
    def fromordinal(cls, n):
        return _real_date.fromordinal(n)


class FakeDatetime(_real_datetime, metaclass=_DatetimeMeta):
    def __new__(cls, *args, **kwargs):
        target_cls = _real_datetime if cls is FakeDatetime else cls
        return _real_datetime.__new__(target_cls, *args, **kwargs)

    @classmethod
    def now(cls, tz=None):
        cur = _current_utc()
        if tz is not None:
            return cur.astimezone(tz)
        local = cur.astimezone(_local_tz())
        return local.replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        return _current_utc().replace(tzinfo=None)

    @classmethod
    def today(cls):
        return cls.now()

    @classmethod
    def fromtimestamp(cls, t, tz=None):
        return _real_datetime.fromtimestamp(t, tz=tz)

    @classmethod
    def utcfromtimestamp(cls, t):
        return _real_datetime.fromtimestamp(t, tz=_dt.timezone.utc).replace(tzinfo=None)

    @classmethod
    def fromisoformat(cls, date_string):
        return _real_datetime.fromisoformat(date_string)

    @classmethod
    def strptime(cls, date_string, fmt):
        return _real_datetime.strptime(date_string, fmt)

    @classmethod
    def combine(cls, date, time, tzinfo=True):
        if tzinfo is True:
            return _real_datetime.combine(date, time)
        return _real_datetime.combine(date, time, tzinfo=tzinfo)


def pytest_configure(config):
    global _ANCHOR_UTC, _BOOT_MONO
    raw = os.environ.get("BTS_FAKE_NOW_ISO")
    if not raw:
        return
    # Pre-import pandas before replacing datetime.datetime so Cython's NaTType
    # binds to the C PyDateTime_DateTimeType layout cleanly.
    import pandas  # noqa: F401

    parsed = _real_datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    _ANCHOR_UTC = parsed.astimezone(_dt.timezone.utc)
    _BOOT_MONO = _real_monotonic()

    _dt.date = FakeDate
    _dt.datetime = FakeDatetime
    _time.time = lambda: _current_utc().timestamp()
    _time.time_ns = lambda: int(_current_utc().timestamp() * 1_000_000_000)
