"""Quick direct sandbox check that the 2027/2028 wall-clock shim works cleanly with
datetime subclasses, ZoneInfo, pandas, and bts modules before invoking yield_run.sh.
"""

import datetime as _dt
import time as _time
from zoneinfo import ZoneInfo

_real_time = _time.time
_real_time_ns = _time.time_ns
_real_date = _dt.date
_real_datetime = _dt.datetime
_target_epoch = _real_datetime.fromisoformat("2028-07-15T12:00:00-04:00").timestamp()
_delta = _target_epoch - _real_time()
_delta_ns = int(_delta * 1_000_000_000)

_time.time = lambda: _real_time() + _delta
_time.time_ns = lambda: _real_time_ns() + _delta_ns


class _DateMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, _real_date)

    def __subclasscheck__(cls, subclass):
        return issubclass(subclass, _real_date)


class _DatetimeMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, _real_datetime)

    def __subclasscheck__(cls, subclass):
        return issubclass(subclass, _real_datetime)


class _ShiftedDate(_real_date, metaclass=_DateMeta):
    @classmethod
    def today(cls):
        return _real_date.fromtimestamp(_time.time())


class _ShiftedDatetime(_real_datetime, metaclass=_DatetimeMeta):
    @classmethod
    def now(cls, tz=None):
        return cls.fromtimestamp(_time.time(), tz=tz)

    @classmethod
    def today(cls):
        return cls.fromtimestamp(_time.time())

    @classmethod
    def utcnow(cls):
        return cls.fromtimestamp(_time.time(), tz=_dt.timezone.utc).replace(tzinfo=None)


_dt.date = _ShiftedDate
_dt.datetime = _ShiftedDatetime

import pandas as pd
from bts import contest_state, scheduler, strategy
from bts.health import alert

et = ZoneInfo("America/New_York")
now_et = _dt.datetime.now(et)
assert now_et.year == 2028 and (now_et.month, now_et.day) == (7, 15)
assert _dt.date.today() == _real_date(2028, 7, 15)


class SubClock(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2027, 7, 14, 12, tzinfo=tz)


assert SubClock.now(et).year == 2027
assert isinstance(SubClock.now(et), _dt.datetime)
assert isinstance(SubClock.now(et).date(), _dt.date)
df = pd.DataFrame([{"t": now_et}])
assert len(df) == 1
print("CLOCK_SHIM_VERIFIED_OK", now_et.isoformat())
