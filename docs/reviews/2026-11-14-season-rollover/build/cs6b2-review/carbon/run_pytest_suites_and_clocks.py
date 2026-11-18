"""Pytest Family B (run under ~/bts-runs/yield_run.sh):
1. Ordering & process-wide state isolation: runs new CS6b-2 files (+ adopt file) BEFORE
   and AFTER tests/test_morning_state_cs6b1.py and tests/test_saved_pick_paths_cs5a.py
   in one process, both orders, with _state_audit_plugin verifying zero leaked state.
2. Wall clocks in 2027 (2027-07-15 ET) and 2028 (2028-07-15 ET) across all added/changed
   test files and the review adopt file, leaving monotonic clocks untouched.
3. Combined one-process run with test_lightgbm_optional.py, test_web_saver.py,
   test_morning_state_cs6b1.py, test_saved_pick_paths_cs5a.py, tests/health/test_alert.py,
   and the two new CS6b-2 test files.
4. No-LightGBM collection check including .codex-review/cs6b2-review/adopt/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")

NEW_FILES = [
    "tests/health/test_alert_status_cs6b2.py",
    "tests/test_alert_status_callers_cs6b2.py",
]
ADOPT_FILE = ".codex-review/cs6b2-review/adopt/test_alert_status_cs6b2_review.py"
NEIGHBOR_FILES = [
    "tests/test_morning_state_cs6b1.py",
    "tests/test_saved_pick_paths_cs5a.py",
]

SITECUSTOMIZE_CLOCK = """\
import datetime as _dt
import os as _os
import time as _time

_target_iso = _os.environ.get("CS6B2_WALL_CLOCK_ISO")
if _target_iso:
    _real_time = _time.time
    _real_time_ns = _time.time_ns
    _real_date = _dt.date
    _real_datetime = _dt.datetime
    _target_epoch = _real_datetime.fromisoformat(_target_iso).timestamp()
    _delta = _target_epoch - _real_time()
    _delta_ns = int(_delta * 1_000_000_000)

    _time.time = lambda: _real_time() + _delta
    _time.time_ns = lambda: _real_time_ns() + _delta_ns

    class _DateMeta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, _real_date)

        def __subclasscheck__(cls, subclass):
            return issubclass(subclass, _real_date)

    class _DatetimeMeta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, _real_datetime)

        def __subclasscheck__(cls, subclass):
            return issubclass(subclass, _real_datetime)

    class _ShiftedDate(_real_date, metaclass=_DateMeta):
        @classmethod
        def today(cls):
            return _real_date.fromtimestamp(_time.time())

    class _ShiftedDatetime(_real_datetime, metaclass=_DatetimeMeta):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(_time.time(), tz=tz)

        @classmethod
        def utcnow(cls):
            return cls.fromtimestamp(_time.time(), tz=_dt.timezone.utc).replace(tzinfo=None)

    _dt.date = _ShiftedDate
    _dt.datetime = _ShiftedDatetime
"""


def make_env(extra_pythonpath: str | None = None, extra_vars: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    parts = []
    if extra_pythonpath:
        parts.append(extra_pythonpath)
    parts.append(str(REVIEW_DIR))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if extra_vars:
        env.update(extra_vars)
    return env


def run_pytest(label: str, args: list[str], env: dict[str, str], allow_socket_wiring_fail: bool = False) -> dict:
    t0 = time.monotonic()
    proc = subprocess.run(
        [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = round(time.monotonic() - t0, 2)
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    summary = lines[-1] if lines else ""
    print(f"[{label}] rc={proc.returncode} ({elapsed}s): {summary}", flush=True)
    if not allow_socket_wiring_fail:
        assert proc.returncode == 0, f"{label} FAILED:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    else:
        if proc.returncode != 0:
            failed_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("FAILED ")]
            assert len(failed_lines) == 1 and (
                "tests/test_web_saver.py::test_do_post_saver_transition_socket_wiring" in failed_lines[0]
            ) and "PermissionError" in proc.stdout, (
                f"Unexpected failure in {label}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
    return {
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "summary": summary,
    }


def main() -> None:
    results: dict[str, dict] = {}

    # 1. Ordering & process-wide state audit (both orders in one process)
    results["order_new_before_existing"] = run_pytest(
        "ORDER_NEW_BEFORE_EXISTING",
        ["-p", "_state_audit_plugin", "-q", *NEW_FILES, ADOPT_FILE, *NEIGHBOR_FILES],
        make_env(),
    )
    results["order_new_after_existing"] = run_pytest(
        "ORDER_NEW_AFTER_EXISTING",
        ["-p", "_state_audit_plugin", "-q", *NEIGHBOR_FILES, *NEW_FILES, ADOPT_FILE],
        make_env(),
    )

    # 2. Wall clocks in 2027 and 2028
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as clock_td:
        clock_dir = Path(clock_td)
        (clock_dir / "sitecustomize.py").write_text(SITECUSTOMIZE_CLOCK)
        for year, iso_str in (
            (2027, "2027-07-15T12:00:00-04:00"),
            (2028, "2028-07-15T12:00:00-04:00"),
        ):
            env_clock = make_env(
                extra_pythonpath=str(clock_dir),
                extra_vars={"CS6B2_WALL_CLOCK_ISO": iso_str},
            )
            # Verify sitecustomize actually shifted the Python process clock first
            chk = subprocess.run(
                [
                    PYTHON, "-c",
                    "from datetime import date, datetime; import time; "
                    f"assert datetime.now().year == {year} and date.today().year == {year}; "
                    "t0 = time.monotonic(); t1 = time.monotonic(); assert t1 >= t0; print(datetime.now().isoformat())",
                ],
                cwd=REPO_ROOT, env=env_clock, capture_output=True, text=True, check=True,
            )
            print(f"[CLOCK CHECK {year}] {chk.stdout.strip()}", flush=True)
            results[f"wall_clock_{year}"] = run_pytest(
                f"WALL_CLOCK_{year}",
                [
                    "-p", "_state_audit_plugin", "-q",
                    *NEW_FILES,
                    "tests/test_saved_pick_paths_cs5a.py",
                    ADOPT_FILE,
                ],
                env_clock,
            )

    # 3. Combined one-process run specified in brief.md item 6
    combined_files = [
        "tests/test_lightgbm_optional.py",
        "tests/test_web_saver.py",
        "tests/test_morning_state_cs6b1.py",
        "tests/test_saved_pick_paths_cs5a.py",
        "tests/health/test_alert.py",
        *NEW_FILES,
        ADOPT_FILE,
    ]
    results["combined_one_process"] = run_pytest(
        "COMBINED_ONE_PROCESS",
        ["-p", "_state_audit_plugin", "-rA", "-q", *combined_files],
        make_env(),
        allow_socket_wiring_fail=True,
    )

    # 4. Verify no-lightgbm collection on tests + adopt
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as lgb_td:
        for err_expr in ("ImportError('no lgbm')", "OSError('dlopen libomp failed')"):
            (Path(lgb_td) / "lightgbm.py").write_text(f"raise {err_expr}\n")
            env_nolgb = make_env(extra_pythonpath=lgb_td)
            proc = subprocess.run(
                [PYTHON, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests", ADOPT_FILE],
                cwd=REPO_ROOT, env=env_nolgb, capture_output=True, text=True, timeout=120,
            )
            assert proc.returncode == 0, f"Collection failed under {err_expr}:\n{proc.stdout}\n{proc.stderr}"
        results["no_lightgbm_collection_including_adopt"] = "ok_importerror_and_oserror"

    (REVIEW_DIR / "pytest_suites_and_clocks.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    print("ALL_SUITE_ORDER_AND_CLOCK_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
    main()
