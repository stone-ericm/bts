"""Run all CS6a touched files under independent 2027 and 2028 Python clocks."""

import local_guard

import datetime as dt
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from pathlib import Path
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def freeze(year):
    instant = dt.datetime(year, 7, 15, 12, tzinfo=ZoneInfo("America/New_York"))
    stamp = instant.timestamp()
    real_datetime, real_date = dt.datetime, dt.date
    localtime, gmtime = time.localtime, time.gmtime

    class Clock(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(stamp, tz)

        @classmethod
        def today(cls):
            return cls.now()

        @classmethod
        def utcnow(cls):
            return cls.now(dt.UTC).replace(tzinfo=None)

    class Calendar(real_date):
        @classmethod
        def today(cls):
            return cls.fromtimestamp(stamp)

    dt.datetime, dt.date = Clock, Calendar
    time.time = lambda: stamp
    time.time_ns = lambda: int(stamp) * 1_000_000_000
    time.localtime = lambda seconds=None: localtime(stamp if seconds is None else seconds)
    time.gmtime = lambda seconds=None: gmtime(stamp if seconds is None else seconds)
    assert dt.datetime.now(ZoneInfo("America/New_York")) == instant
    assert dt.date.today().year == year and time.localtime().tm_year == year
    return instant


def child(year, temp, files):
    import pytest

    # Keep the native numpy/pandas datetime types initialized before patching.
    import numpy
    import pandas

    instant = freeze(year)
    from bts import cli

    assert Path(cli.__file__).is_relative_to(ROOT / "src"), cli.__file__
    assert cli._today_et() == instant.date().isoformat()
    print(f"Python and CLI ET wall clock {instant.isoformat()}", flush=True)
    result = pytest.main(["-q", "-o", "addopts=", "-p", "no:cacheprovider", "--tb=short",
                          "--basetemp", temp, *files])
    assert not local_guard.network, local_guard.network
    return result


def main():
    files = subprocess.check_output(
        ["git", "diff", "--name-only", "09d9b51f", "HEAD", "--", "tests"], cwd=ROOT, text=True
    ).splitlines()
    assert len(files) == 6 and all(file.endswith(".py") for file in files), files
    print("Touched test files:", *files, sep="\n", flush=True)
    with TemporaryDirectory(prefix="clock-runs-", dir=HERE) as folder:
        for year in (2027, 2028):
            args = [sys.executable, __file__, "child", str(year), str(Path(folder) / str(year)), *files]
            subprocess.run(args, cwd=ROOT, check=True, timeout=120)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        raise SystemExit(child(int(sys.argv[2]), sys.argv[3], sys.argv[4:]))
    main()
