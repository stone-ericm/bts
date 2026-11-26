"""Offline audit and an independent wall clock for review subprocesses."""
import datetime as dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REAL_DATETIME, REAL_DATE = dt.datetime, dt.date
ACTIVE = None
EVENTS = []


def audit(event, args):
    if event in {"socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
        raise AssertionError(f"network forbidden: {event}")
    if ACTIVE is not None and event in {"open", "os.rename", "os.mkdir", "os.remove", "os.rmdir", "os.symlink"}:
        EVENTS.append((event, tuple(str(x) for x in args)))


sys.addaudithook(audit)


def begin(label):
    global ACTIVE
    EVENTS.clear()
    ACTIVE = label


def end():
    global ACTIVE
    ACTIVE = None
    return list(EVENTS)


def freeze(now):
    # Initialize pandas before changing datetime's public types.
    import pandas  # noqa: F401
    class DateMeta(type):
        def __instancecheck__(cls, value):
            return isinstance(value, REAL_DATE)
    class TimeMeta(type):
        def __instancecheck__(cls, value):
            return isinstance(value, REAL_DATETIME)
    class Date(REAL_DATE, metaclass=DateMeta):
        @classmethod
        def today(cls):
            return now.astimezone().date()
    class Clock(REAL_DATETIME, metaclass=TimeMeta):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.astimezone().replace(tzinfo=None)
        @classmethod
        def today(cls):
            return cls.now()
        @classmethod
        def utcnow(cls):
            return now.astimezone(dt.timezone.utc).replace(tzinfo=None)
    dt.datetime, dt.date = Clock, Date


source = os.environ.get("REVIEW_SRC")
if source:
    sys.path.insert(0, source)
if os.environ.get("REVIEW_YEAR"):
    from zoneinfo import ZoneInfo
    freeze(REAL_DATETIME(int(os.environ["REVIEW_YEAR"]), 7, 15, 12, tzinfo=ZoneInfo("America/New_York")))
