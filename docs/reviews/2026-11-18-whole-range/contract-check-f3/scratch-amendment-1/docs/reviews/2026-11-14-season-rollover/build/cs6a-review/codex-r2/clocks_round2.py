"""Run each CS6a test file in isolation and last, ten times at each ET date."""

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
from itertools import product
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = tuple(f"tests/{name}.py" for name in (
    "test_cli_integration", "test_cli_schedule_date", "test_entry_intent",
    "test_live_gate_cs6a", "test_live_gate_cs6a_review", "test_season_contract_cs4b2_edges",
))


def freeze(year):
    import time

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


def child(year, temp, files):
    import local_guard
    import pytest

    # Preload C extensions that cache the standard datetime and date classes.
    import numpy
    import pandas

    freeze(year)
    result = pytest.main(["-q", "-o", "addopts=", "-p", "no:cacheprovider",
                          "--basetemp", temp, *files])
    if local_guard.network:
        print("Forbidden external transport attempts:", local_guard.network)
        return 1
    return result


def run(job, temp):
    year, repeat, target, mode = job
    name = f"{year}/{repeat:02}/{mode}/{Path(target).stem}"
    files = (target,) if mode == "alone" else tuple(f for f in FILES if f != target) + (target,)
    basedir = temp / name.replace("/", "-")
    result = subprocess.run([sys.executable, str(__file__), "child", str(year), str(basedir), *files],
                            cwd=ROOT, capture_output=True, text=True, timeout=150)
    summary = re.search(r"^(\d+) passed in .*$", result.stdout, flags=re.M)
    if result.returncode or not summary:
        return name, 0, result.stdout + result.stderr
    return name, int(summary[1]), summary[0]


def main():
    jobs = list(product((2027, 2028), range(1, 11), FILES, ("alone", "last")))
    cases = errors = 0
    with TemporaryDirectory(prefix="clock-runs-", dir=HERE) as folder:
        from functools import partial

        with ThreadPoolExecutor(max_workers=2) as pool:
            for name, count, output in pool.map(partial(run, temp=Path(folder)), jobs):
                print(f"{name}: {output}", flush=True)
                errors += not count
                cases += count
    print(f"{len(jobs)} runs, {cases} passed instances, {errors} non-green runs", flush=True)
    return bool(errors)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        raise SystemExit(child(int(sys.argv[2]), sys.argv[3], sys.argv[4:]))
    raise SystemExit(main())
