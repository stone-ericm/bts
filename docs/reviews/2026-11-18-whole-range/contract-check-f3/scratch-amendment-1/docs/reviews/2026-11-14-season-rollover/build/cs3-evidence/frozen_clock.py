"""Run with -p frozen_clock and CS3_CLOCK=YYYY-MM-DD; load pandas before replacing dates."""
import os
import datetime as dt
import pandas  # pandas extensions must see the genuine datetime types first

_real_datetime = dt.datetime
_real_date = dt.date
_day = _real_date.fromisoformat(os.environ['CS3_CLOCK'])
_fixed = _real_datetime(_day.year, _day.month, _day.day, 16, tzinfo=dt.timezone.utc)

class FrozenDateTime(_real_datetime):
    @classmethod
    def now(cls, tz=None):
        value = _fixed.astimezone(tz) if tz else _fixed.replace(tzinfo=None)
        return cls.fromisoformat(value.isoformat())

    @classmethod
    def today(cls):
        return cls.now()

    @classmethod
    def utcnow(cls):
        return cls.now(dt.timezone.utc).replace(tzinfo=None)

class FrozenDate(_real_date):
    @classmethod
    def today(cls):
        return cls(_day.year, _day.month, _day.day)

dt.datetime = FrozenDateTime
dt.date = FrozenDate
