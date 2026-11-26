"""Run the focused 9-file suite under 2027 and 2028 wall clocks sequentially via nice -n 19.

Keeps sys.modules['datetime'] untouched for C extensions (pandas/pyarrow/dateutil) while
intercepting `datetime`/`date` imports and bindings across all `bts.*` modules so every
unpinned wall-clock read in `bts` sees the target 2027 or 2028 timestamp.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess


REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
WORK_DIR = REVIEW_DIR / "wallclock_work"

ALL_FOCUSED = [
    "tests/test_wr_c3v_reconcile.py",
    "tests/test_wr_c3v_corpus.py",
    "tests/test_wr_c3v_delivery.py",
    "tests/test_wr_c3_r2_entered_pick_blockers.py",
    "tests/test_streak_replay.py",
    "tests/test_streak_season_cs2.py",
    "tests/test_reconcile_receipt.py",
    "tests/test_lightgbm_optional.py",
    "tests/test_web_saver.py",
]

WALL_PLUGIN = '''"""Safe bts-scoped wall-clock override plugin (leaves stdlib datetime module intact for C extensions)."""
import builtins as _builtins
import datetime as _dt
import os
import sys
import time as _time
import types as _types

_WALL_ISO = os.environ["BTS_WALL_CLOCK_ISO"]
_BASE_WALL = _dt.datetime.fromisoformat(_WALL_ISO)
_START_MONO = _time.monotonic()
_REAL_DATETIME = _dt.datetime
_REAL_DATE = _dt.date
_REAL_IMPORT = _builtins.__import__


def _now_wall():
    return _BASE_WALL + _dt.timedelta(seconds=_time.monotonic() - _START_MONO)


class _BtsWallDate(_REAL_DATE):
    @classmethod
    def today(cls):
        w = _now_wall().astimezone()
        return _REAL_DATE(w.year, w.month, w.day)


class _BtsWallDatetime(_REAL_DATETIME):
    @classmethod
    def now(cls, tz=None):
        w = _now_wall()
        if tz is None:
            loc = w.astimezone()
            return _REAL_DATETIME(
                loc.year, loc.month, loc.day, loc.hour, loc.minute, loc.second, loc.microsecond
            )
        return w.astimezone(tz)

    @classmethod
    def utcnow(cls):
        u = _now_wall().astimezone(_dt.timezone.utc)
        return _REAL_DATETIME(
            u.year, u.month, u.day, u.hour, u.minute, u.second, u.microsecond
        )


_PROXY_DT_MOD = _types.ModuleType("datetime")
_PROXY_DT_MOD.__dict__.update(_dt.__dict__)
_PROXY_DT_MOD.datetime = _BtsWallDatetime
_PROXY_DT_MOD.date = _BtsWallDate


def _bts_import(name, globals=None, locals=None, fromlist=(), level=0):
    mod = _REAL_IMPORT(name, globals, locals, fromlist, level)
    if name == "datetime" and level == 0 and isinstance(globals, dict):
        caller = globals.get("__name__", "")
        if isinstance(caller, str) and (caller == "bts" or caller.startswith("bts.")):
            return _PROXY_DT_MOD
    return mod


def _patch_loaded_bts():
    for mname, mod in list(sys.modules.items()):
        if mod is not None and (mname == "bts" or mname.startswith("bts.")):
            if getattr(mod, "datetime", None) is _REAL_DATETIME:
                setattr(mod, "datetime", _BtsWallDatetime)
            if getattr(mod, "date", None) is _REAL_DATE:
                setattr(mod, "date", _BtsWallDate)
            if getattr(mod, "date_type", None) is _REAL_DATE:
                setattr(mod, "date_type", _BtsWallDate)


_builtins.__import__ = _bts_import
_patch_loaded_bts()


def pytest_sessionstart(session):
    _patch_loaded_bts()
    import bts.picks as p
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    expected_et_year = _BASE_WALL.astimezone(et).year
    actual_et_year = p.datetime.now(et).year
    assert actual_et_year == expected_et_year, (actual_et_year, expected_et_year)
'''


def _parse_summary(stdout: str) -> dict:
    summary_line = ""
    for line in reversed(stdout.splitlines()):
        if re.search(r"\b(passed|failed|error|errors|skipped)\b", line) and "=" in line:
            summary_line = line.strip()
            break
    counts = {}
    for key in ("passed", "failed", "error", "errors", "skipped", "deselected"):
        m = re.search(rf"(\d+)\s+{key}\b", summary_line)
        if m:
            counts[key] = int(m.group(1))
    failures = [
        line.strip()
        for line in stdout.splitlines()
        if line.startswith("FAILED ") or line.startswith("ERROR ")
    ]
    return {"summary_line": summary_line, "counts": counts, "failures": failures}


def main() -> int:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)
    plugin_dir = WORK_DIR / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "_bts_wallclock_plugin.py").write_text(WALL_PLUGIN)

    results = {}
    clocks = [
        ("2027_midseason", "2027-07-15T14:00:00-04:00"),
        ("2027_dec31_utc_2028", "2027-12-31T23:30:00-05:00"),
        ("2028_jan01", "2028-01-01T00:30:00-05:00"),
        ("2028_midseason", "2028-07-15T14:00:00-04:00"),
    ]
    for label, iso_val in clocks:
        env = dict(os.environ)
        env["TZ"] = "America/New_York"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["BTS_WALL_CLOCK_ISO"] = iso_val
        env["PYTHONPATH"] = os.pathsep.join(
            [str(plugin_dir), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
        )
        cmd = [
            "/usr/bin/nice", "-n", "19",
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "-m", "pytest",
            "-p", "no:cacheprovider",
            "-p", "_bts_wallclock_plugin",
            "--deselect", "tests/test_web_saver.py::test_do_post_saver_transition_socket_wiring",
            "-ra",
            *ALL_FOCUSED,
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=240)
        log_path = REVIEW_DIR / f"wallclock_clean_{label}.log"
        log_path.write_text(proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else ""))
        parsed = _parse_summary(proc.stdout)
        parsed["returncode"] = proc.returncode
        parsed["wall_clock_iso"] = iso_val
        parsed["log"] = str(log_path.relative_to(REPO_ROOT))
        results[label] = parsed

    shutil.rmtree(WORK_DIR)
    out_path = REVIEW_DIR / "wallclock_2027_2028_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
