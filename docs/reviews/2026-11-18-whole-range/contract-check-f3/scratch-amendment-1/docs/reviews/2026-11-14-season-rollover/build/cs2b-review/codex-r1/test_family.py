"""Run only the changed files under future Python datetime clocks, or base src.

Invoked by run_tests.py through yield_run.sh. No tracked files are modified.
"""
import datetime as dt
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "base/src" if sys.argv[1] == "base" else ROOT / "src"))
# Load compiled extensions before replacing Python's datetime classes.
import pandas  # noqa: E402
import pytest  # noqa: E402

real_date, real_datetime = dt.date, dt.datetime
mode = sys.argv[1]
if mode.startswith("clock"):
    fixed = real_datetime(int(mode[-4:]), 11, 16, 18, tzinfo=dt.timezone.utc)

    class FrozenDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            value = fixed.astimezone(tz) if tz else fixed.astimezone().replace(tzinfo=None)
            return cls.fromisoformat(value.isoformat())

        @classmethod
        def utcnow(cls):
            return cls.fromisoformat(fixed.replace(tzinfo=None).isoformat())

        @classmethod
        def today(cls):
            return cls.now()

    class FrozenDate(real_date):
        @classmethod
        def today(cls):
            return cls.fromisoformat(fixed.astimezone().date().isoformat())

    dt.datetime, dt.date = FrozenDatetime, FrozenDate

# Fail a leaked external connection before any network traffic.
connect = socket.socket.connect
def offline(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in ("127.0.0.1", "::1", "localhost"):
        raise RuntimeError(f"review forbids network: {address!r}")
    return connect(self, address)
socket.socket.connect = offline

if mode == "base":
    files = ["tests/test_streak_season_cs2.py", "tests/test_streak_season_cs2_review.py"]
else:
    files = (HERE / "changed-files.txt").read_text().splitlines()
print("SOURCE", __import__("bts.picks", fromlist=["x"]).__file__, "CLOCK", dt.datetime.now(dt.timezone.utc), flush=True)
raise SystemExit(pytest.main(["-p", "no:cacheprovider", *files, "-q", "--tb=short",
                             f"--junitxml={HERE / (mode + '.xml')}"]))
