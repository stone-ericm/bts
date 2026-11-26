"""Verify 2027/2028 wall-clock runs on the focused suite and verify adopt/test_wr_c3v_r2_adopt.py
passes on HEAD and kills X3_drop_blocked_date_guard and X4_regex_ascii_flag.
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
HEAD_SRC = REPO_ROOT / "src"
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
WORK_DIR = REVIEW_DIR / "_wc_adopt_scratch"
ADOPT_FILE = str(REVIEW_DIR / "adopt" / "test_wr_c3v_r2_adopt.py")

ALL_FOCUSED = [
    "tests/test_wr_c3v_reconcile.py",
    "tests/test_wr_c3v_corpus.py",
    "tests/test_wr_c3v_delivery.py",
    "tests/test_wr_c3_r2_entered_pick_blockers.py",
    "tests/test_wr_c3v_review_gaps.py",
    "tests/test_streak_replay.py",
    "tests/test_streak_season_cs2.py",
    "tests/test_reconcile_receipt.py",
    "tests/test_lightgbm_optional.py",
    "tests/test_web_saver.py",
]

SOCKET_DESELECT = [
    "--deselect",
    "tests/test_web_saver.py::test_do_post_saver_transition_socket_wiring",
]

PLUGIN_CODE = '''"""Pydantic- and pandas-compatible bts-scoped wall-clock override + src verification plugin."""
import builtins as _builtins
import datetime as _dt
import os
from pathlib import Path
import sys
import time as _time
import types as _types

_REAL_DATETIME = _dt.datetime
_REAL_DATE = _dt.date
_REAL_IMPORT = _builtins.__import__
_WALL_ISO = os.environ.get("BTS_WALL_CLOCK_ISO")

if _WALL_ISO:
    _BASE_WALL = _REAL_DATETIME.fromisoformat(_WALL_ISO)
    _START_MONO = _time.monotonic()

    def _now_wall():
        return _BASE_WALL + _dt.timedelta(seconds=_time.monotonic() - _START_MONO)

    class _BtsWallDate(_REAL_DATE):
        @classmethod
        def today(cls):
            w = _now_wall().astimezone()
            return _REAL_DATE(w.year, w.month, w.day)

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type, handler):
            return handler(_REAL_DATE)

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
        def __get_pydantic_core_schema__(cls, source_type, handler):
            return handler(_REAL_DATETIME)

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
    if _WALL_ISO:
        _patch_loaded_bts()
        import bts.picks as p
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        assert p.datetime.now(et).year == _BASE_WALL.astimezone(et).year

    expected_src = os.environ.get("BTS_EXPECTED_SRC")
    if expected_src:
        import bts.cli
        import bts.picks
        import bts.reconcile_receipt
        exp = Path(expected_src).resolve()
        for mod in (bts.picks, bts.cli, bts.reconcile_receipt):
            actual = Path(mod.__file__).resolve()
            if not actual.is_relative_to(exp):
                raise RuntimeError(f"Expected {mod.__name__} from {exp}, got {actual}")
'''


def _parse_summary(stdout: str) -> dict:
    summary = ""
    for line in reversed(stdout.splitlines()):
        if re.search(r"\b(passed|failed|error|errors|skipped)\b", line) and "=" in line:
            summary = line.strip()
            break
    counts = {}
    for key in ("passed", "failed", "error", "errors", "skipped", "deselected"):
        m = re.search(rf"(\d+)\s+{key}\b", summary)
        if m:
            counts[key] = int(m.group(1))
    failures = [ln.strip() for ln in stdout.splitlines() if ln.startswith(("FAILED ", "ERROR "))]
    return {"summary_line": summary, "counts": counts, "failures": failures}


def _run_pytest(log_name: str, args: list[str], *, src_override: Path | None = None, wall_iso: str | None = None) -> dict:
    plugin_dir = WORK_DIR / "plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "_wc_adopt_r2_plugin.py").write_text(PLUGIN_CODE)

    expected_src = (src_override or HEAD_SRC).resolve()
    py_paths = [str(plugin_dir.resolve())]
    if src_override is not None:
        py_paths.insert(0, str(expected_src))
    if os.environ.get("PYTHONPATH"):
        py_paths.append(os.environ["PYTHONPATH"])

    env = dict(os.environ)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    env["BTS_EXPECTED_SRC"] = str(expected_src)
    if wall_iso is not None:
        env["BTS_WALL_CLOCK_ISO"] = wall_iso
    else:
        env.pop("BTS_WALL_CLOCK_ISO", None)

    cmd = [
        "/usr/bin/nice", "-n", "19",
        str(PYTHON_BIN), "-m", "pytest",
        "-p", "no:cacheprovider",
        "-p", "_wc_adopt_r2_plugin",
        "-ra",
        *args,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=240)
    log_path = REVIEW_DIR / log_name
    log_path.write_text(proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else ""))
    parsed = _parse_summary(proc.stdout)
    parsed["returncode"] = proc.returncode
    parsed["log"] = str(log_path.relative_to(REPO_ROOT))
    return parsed


def main() -> int:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)
    try:
        wall_clocks = {}
        for label, iso_val in (
            ("2027_midseason", "2027-07-15T14:00:00-04:00"),
            ("2027_dec31_utc_2028", "2027-12-31T23:30:00-05:00"),
            ("2028_jan01", "2028-01-01T00:30:00-05:00"),
            ("2028_midseason", "2028-07-15T14:00:00-04:00"),
        ):
            wall_clocks[label] = _run_pytest(
                f"wallclock_r2_{label}.log",
                [*ALL_FOCUSED, ADOPT_FILE, *SOCKET_DESELECT],
                wall_iso=iso_val,
            )
            assert wall_clocks[label]["returncode"] == 0, wall_clocks[label]

        cli_orig = (HEAD_SRC / "bts" / "cli.py").read_text()
        picks_orig = (HEAD_SRC / "bts" / "picks.py").read_text()

        killed = {}
        for m_name, apply_fn in (
            (
                "X3_drop_blocked_date_guard",
                lambda r: (r / "bts" / "cli.py").write_text(
                    cli_orig.replace(
                        '    blocked = replay["state"] == "unavailable" and "blocked_date" in replay',
                        '    blocked = replay["state"] == "unavailable"',
                        1,
                    )
                ),
            ),
            (
                "X4_regex_ascii_flag",
                lambda r: (r / "bts" / "picks.py").write_text(
                    picks_orig.replace(
                        '_ISO_DATE_RE = re.compile(r"^\\d{4}-\\d{2}-\\d{2}$")',
                        '_ISO_DATE_RE = re.compile(r"^\\d{4}-\\d{2}-\\d{2}$", re.ASCII)',
                        1,
                    )
                ),
            ),
        ):
            m_src = WORK_DIR / m_name / "src"
            shutil.copytree(HEAD_SRC, m_src, symlinks=True)
            apply_fn(m_src)
            res = _run_pytest(f"adopt_kills_{m_name}.log", [ADOPT_FILE], src_override=m_src)
            assert res["returncode"] != 0, res
            killed[m_name] = res
            shutil.rmtree(m_src.parent)

        out = {"wall_clocks": wall_clocks, "adopt_kills": killed}
        out_path = REVIEW_DIR / "wallclock_and_adopt_r2_results.json"
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
    finally:
        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
