"""Pin Python wall time in BTS module bindings without replacing the C API types."""

from contextlib import ExitStack
import datetime as calendar
import os
import sys
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo


def pytest_collection_finish(session):
    datetime = calendar.datetime
    date = calendar.date
    instant = datetime(int(os.environ["CS6B1_CLOCK_YEAR"]), 7, 15, 12, tzinfo=ZoneInfo("America/New_York"))

    class WallClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

        @classmethod
        def utcnow(cls):
            return instant.astimezone(calendar.UTC).replace(tzinfo=None)

    class WallDate(date):
        @classmethod
        def today(cls):
            return instant.date()

    with ExitStack() as bindings:
        for name, module in tuple(sys.modules.items()):
            if not name.startswith(("bts.", "tests.")) or module is None:
                continue
            for attribute, value in tuple(vars(module).items()):
                if value is datetime:
                    bindings.enter_context(patch.object(module, attribute, WallClock))
                elif value is date:
                    bindings.enter_context(patch.object(module, attribute, WallDate))
        timestamp = instant.timestamp()
        bindings.enter_context(patch.object(time, "time", lambda: timestamp))
        bindings.enter_context(patch.object(time, "time_ns", lambda: int(timestamp * 1e9)))
        session.config.add_cleanup(bindings.pop_all().close)
    print(f"Review Python wall clock {instant.isoformat()}", flush=True)
