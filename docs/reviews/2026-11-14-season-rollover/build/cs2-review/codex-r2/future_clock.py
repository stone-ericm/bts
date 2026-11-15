"""Pytest plugin: freeze default datetime/date reads, not explicit fixture datetimes.

Loaded through -p future_clock. REVIEW_CLOCK_YEAR controls the default year.
After collection replace bound aliases in all bts/tests modules as well as the
stdlib module attributes used by later local imports. Binary dependencies are
already loaded by collection. Explicit now= arguments and monkeypatches win.
"""
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

REAL_DATETIME = dt.datetime
REAL_DATE = dt.date
YEAR = int(os.environ["REVIEW_CLOCK_YEAR"])
FIXED = REAL_DATETIME(YEAR, 1, 1, 18, tzinfo=dt.timezone.utc)
REAL_MONOTONIC = time.monotonic
START = REAL_MONOTONIC()
PATCHED = []


class FrozenDatetime(REAL_DATETIME):
    @classmethod
    def now(cls, tz=None):
        running = FIXED + dt.timedelta(seconds=REAL_MONOTONIC() - START)
        value = running.astimezone(tz) if tz is not None else running.astimezone().replace(tzinfo=None)
        return cls.fromisoformat(value.isoformat())

    @classmethod
    def utcnow(cls):
        return cls.fromisoformat((FIXED + dt.timedelta(seconds=REAL_MONOTONIC() - START)).replace(tzinfo=None).isoformat())

    @classmethod
    def today(cls):
        return cls.now()


class FrozenDate(REAL_DATE):
    @classmethod
    def today(cls):
        value = FIXED.astimezone().date()
        return cls(value.year, value.month, value.day)


def pytest_collection_finish(session):
    # Load binary datetime consumers before replacing the stdlib class, also
    # for an isolated contest-state file that doesn't itself import pandas.
    import pandas
    from bts import cli, scheduler, contest_state
    for name, module in list(sys.modules.items()):
        if name == "datetime" or name.startswith(("bts.", "tests.", "test_")):
            for key, value in list(vars(module).items()):
                replacement = FrozenDatetime if value is REAL_DATETIME else FrozenDate if value is REAL_DATE else None
                if replacement is not None:
                    PATCHED.append((module, key, value))
                    setattr(module, key, replacement)
    # Verify real production defaults, not only a test helper's clock, changed.
    assert cli._now_et().year == YEAR
    assert scheduler._now_et().year == YEAR
    assert contest_state.datetime.now(dt.timezone.utc).year == YEAR
    out = Path(__file__).with_name(f"clock-{YEAR}-aliases.json")
    out.write_text(json.dumps({"fixed": FIXED.isoformat(), "aliases": [f"{m.__name__}.{k}" for m,k,_ in PATCHED]}, indent=2))


def pytest_sessionfinish(session, exitstatus):
    for module, key, value in reversed(PATCHED):
        setattr(module, key, value)
