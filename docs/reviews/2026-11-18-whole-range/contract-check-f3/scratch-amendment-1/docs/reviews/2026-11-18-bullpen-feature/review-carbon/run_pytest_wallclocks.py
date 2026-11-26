"""Run 2027 and 2028 wall-clock pytest verification (item 7)."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PLUGIN_DIR = HERE / "clock_pytest_plugin"

PA_DIR = "/Users/eric/projects/bts/data/processed"
LOOKUP_PATH = "/Users/eric/bts-runs/inputs/probable_pitcher_lookup.2017-2025.json"
ADOPT_FILE = ".codex-review/bullpen-review/adopt/test_bullpen_review_probes.py"

PLUGIN_CODE = '''"""Pytest plugin shifting wall clock safely after C datetime capsules initialize."""
import datetime as _dt
import os as _os
import time as _time
import numpy as _np
import pandas as _pd
import pyarrow as _pa

_REAL_TIME = _time.time
_REAL_TIME_NS = _time.time_ns
_REAL_DATE = _dt.date
_REAL_DATETIME = _dt.datetime


def _install():
    target_iso = _os.environ.get("BTS_TEST_WALL_ISO")
    if not target_iso:
        return
    offset = _REAL_DATETIME.fromisoformat(target_iso).timestamp() - _REAL_TIME()

    def shifted_time():
        return _REAL_TIME() + offset

    def shifted_time_ns():
        return _REAL_TIME_NS() + int(offset * 1_000_000_000)

    class _DateMeta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, _REAL_DATE)

        def __subclasscheck__(cls, subclass):
            return issubclass(subclass, _REAL_DATE)

    class _DatetimeMeta(_DateMeta):
        def __instancecheck__(cls, instance):
            return isinstance(instance, _REAL_DATETIME)

        def __subclasscheck__(cls, subclass):
            return issubclass(subclass, _REAL_DATETIME)

    class ShiftedDate(_REAL_DATE, metaclass=_DateMeta):
        def __new__(cls, *args, **kwargs):
            return _REAL_DATE(*args, **kwargs)

        @classmethod
        def today(cls):
            return _REAL_DATETIME.fromtimestamp(shifted_time()).date()

    class ShiftedDatetime(_REAL_DATETIME, metaclass=_DatetimeMeta):
        def __new__(cls, *args, **kwargs):
            return _REAL_DATETIME(*args, **kwargs)

        @classmethod
        def now(cls, tz=None):
            return _REAL_DATETIME.fromtimestamp(shifted_time(), tz=tz)

        @classmethod
        def utcnow(cls):
            return _REAL_DATETIME.fromtimestamp(shifted_time(), tz=_dt.timezone.utc).replace(tzinfo=None)

    _time.time = shifted_time
    _time.time_ns = shifted_time_ns
    _dt.date = ShiftedDate
    _dt.datetime = ShiftedDatetime
    _pd.Timestamp.now = classmethod(lambda cls, tz=None: _pd.Timestamp(ShiftedDatetime.now(tz=tz)))
    _pd.Timestamp.today = classmethod(lambda cls, tz=None: _pd.Timestamp(ShiftedDatetime.now(tz=tz)))


_install()


def pytest_report_header(config):
    return f"wall_clock_override: date.today()={_dt.date.today().isoformat()}, Timestamp.now()={_pd.Timestamp.now().isoformat()}"
'''


def setup_plugin():
    PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
    (PLUGIN_DIR / "wallclock_plugin.py").write_text(PLUGIN_CODE)


def main():
    setup_plugin()
    if "--selftest" in sys.argv:
        env = dict(os.environ, PYTHONPATH=f"{PLUGIN_DIR}:{REPO}", BTS_TEST_WALL_ISO="2028-02-29T12:00:00-05:00")
        chk = subprocess.run(
            [
                sys.executable, "-c",
                "import wallclock_plugin, datetime, pandas as pd, pyarrow.parquet; "
                "d = datetime.date.today(); dt = datetime.datetime.now(); "
                "assert d.isoformat() == '2028-02-29' and isinstance(d, datetime.date); "
                "assert isinstance(dt, datetime.datetime) and isinstance(dt, datetime.date); "
                "print('SELFTEST OK:', d, dt, pd.Timestamp.now())"
            ],
            cwd=REPO, env=env, check=True, text=True, capture_output=True,
        )
        print(chk.stdout, end="")
        return

    targets = [
        "tests/test_bullpen_same_day.py",
        ADOPT_FILE,
        "tests/test_context_features.py",
        "tests/test_park_drag_feature.py",
        "tests/test_swing_features.py",
        "tests/model/test_predict_feature_coverage.py",
        "tests/experiment/test_features.py",
    ]
    for label, iso in (
        ("wall_clock_2027", "2027-06-15T12:00:00-04:00"),
        ("wall_clock_2028_leap", "2028-02-29T12:00:00-05:00"),
    ):
        env = dict(os.environ)
        env["TZ"] = "America/New_York"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = f"{PLUGIN_DIR}:{REPO}"
        env["BTS_TEST_WALL_ISO"] = iso
        env["BULLPEN_TEST_DATA"] = PA_DIR
        env["BULLPEN_TEST_LOOKUP"] = LOOKUP_PATH
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "wallclock_plugin", *targets],
            cwd=REPO, env=env, capture_output=True, text=True,
        )
        print(f"\n================ [{label}] exit={proc.returncode} ================", flush=True)
        print(proc.stdout, end="", flush=True)
        if proc.stderr:
            print(f"--- stderr ---\n{proc.stderr}", flush=True)
        assert proc.returncode == 0, f"{label} failed with exit={proc.returncode}"

    print("\nALL 2027 AND 2028 WALL-CLOCK PYTEST RUNS PASSED.", flush=True)


if __name__ == "__main__":
    main()
