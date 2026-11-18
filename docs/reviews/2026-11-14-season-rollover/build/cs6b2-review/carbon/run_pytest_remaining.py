"""Remaining pytest verification runs (under ~/bts-runs/yield_run.sh):
1. Categorize exact Base RED/GREEN passed & failed IDs + run the 13th mutation (swallow_send).
2. Ordering & process-wide state isolation (both orders in one process, with _audit_plugin_v2).
3. Wall clocks in 2027 (2027-07-15 ET) and 2028 (2028-07-15 ET) across added/changed/adopt files.
4. Combined one-process run with test_lightgbm_optional.py, test_web_saver.py,
   test_morning_state_cs6b1.py, test_saved_pick_paths_cs5a.py, tests/health/test_alert.py,
   the two new CS6b-2 test files, and the review adopt file.
5. No-LightGBM collection check (ImportError and OSError) including .codex-review/cs6b2-review/adopt/.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_SRC = REVIEW_DIR / "scratch_base" / "src"
MUT_DIR = REVIEW_DIR / "scratch_mut_send"
MUT_SRC = MUT_DIR / "src"
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
        def today(cls):
            return cls.fromtimestamp(_time.time())

        @classmethod
        def utcnow(cls):
            return cls.fromtimestamp(_time.time(), tz=_dt.timezone.utc).replace(tzinfo=None)

    _dt.date = _ShiftedDate
    _dt.datetime = _ShiftedDatetime
"""


def make_env(prepend_paths: list[str] | None = None, extra_vars: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    parts = list(prepend_paths or [])
    parts.append(str(REVIEW_DIR))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if extra_vars:
        env.update(extra_vars)
    return env


def run_base_breakdown_and_swallow_send() -> dict:
    # 1a. Fast breakdown on base (deselecting the 12 known FIFO 2s-timeout RED cases already logged)
    targets = [
        "tests/health/test_alert_status_cs6b2.py",
        "tests/test_alert_status_callers_cs6b2.py",
        "tests/test_saved_pick_paths_cs5a.py::test_refusal_survives_non_object_health_status",
    ]
    proc = subprocess.run(
        [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-v", "--tb=no", "-k", "not fifo", *targets],
        cwd=REPO_ROOT,
        env=make_env([str(BASE_SRC), str(REPO_ROOT)]),
        capture_output=True,
        text=True,
        timeout=90,
    )
    passed_ids = [ln.split(" PASSED")[0].strip() for ln in proc.stdout.splitlines() if " PASSED" in ln]
    failed_ids = [ln.split(" FAILED")[0].strip() for ln in proc.stdout.splitlines() if " FAILED" in ln]
    assert len(passed_ids) == 219, f"Expected 219 passed on base, got {len(passed_ids)}"
    assert len(failed_ids) == 129, f"Expected 129 non-fifo failed on base (+12 fifo = 141), got {len(failed_ids)}"

    named_green = [t for t in passed_ids if "green" in t]
    unprefixed = [t for t in passed_ids if "green" not in t]
    buckets: dict[str, list[str]] = {
        "dangling_and_loop_symlink": [],
        "failed_count_1_0_neg1": [],
        "prior_sources_stale_or_non_sent_or_empty_null": [],
        "dedup_persistent_dir": [],
        "classified_unconfirmed_corrupt": [],
        "other": [],
    }
    for tid in unprefixed:
        if "test_status_shapes[" in tid and ("-dangling]" in tid or "-loop]" in tid):
            buckets["dangling_and_loop_symlink"].append(tid)
        elif "test_failed_count[" in tid:
            buckets["failed_count_1_0_neg1"].append(tid)
        elif "test_prior_sources[" in tid:
            buckets["prior_sources_stale_or_non_sent_or_empty_null"].append(tid)
        elif "test_dedup_after_corruption[" in tid:
            buckets["dedup_persistent_dir"].append(tid)
        elif "classified_unconfirmed" in tid:
            buckets["classified_unconfirmed_corrupt"].append(tid)
        else:
            buckets["other"].append(tid)

    # 1b. Run 13th mutation: swallow_send inside dispatch_dm_for_health_alerts
    if MUT_DIR.exists():
        shutil.rmtree(MUT_DIR)
    shutil.copytree(REPO_ROOT / "src", MUT_SRC)
    orig_path = REPO_ROOT / "src" / "bts" / "health" / "alert.py"
    mut_path = MUT_SRC / "bts" / "health" / "alert.py"
    orig_text = orig_path.read_text()
    target = "    try:\n        send_dm(dm_recipient, body)\n        _persist_health_dm_delivery_status("
    replacement = "    try:\n        pass\n        _persist_health_dm_delivery_status("
    assert orig_text.count(target) == 1
    try:
        mut_path.write_text(orig_text.replace(target, replacement, 1))
        t0 = time.monotonic()
        mproc = subprocess.run(
            [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--tb=no", *NEW_FILES],
            cwd=REPO_ROOT,
            env=make_env([str(MUT_SRC), str(REPO_ROOT)]),
            capture_output=True,
            text=True,
            timeout=60,
        )
        elapsed = round(time.monotonic() - t0, 2)
        last = mproc.stdout.strip().splitlines()[-1] if mproc.stdout.strip() else ""
        print(f"[MUTATION swallow_send] rc={mproc.returncode} ({elapsed}s): {last}", flush=True)
        assert mproc.returncode == 1
        m_fail = re.search(r"(\d+)\s+failed", last)
        swallow_res = {
            "killed": True,
            "returncode": mproc.returncode,
            "failed_count": int(m_fail.group(1)) if m_fail else None,
            "elapsed_sec": elapsed,
            "summary_line": last,
        }
    finally:
        shutil.rmtree(MUT_DIR, ignore_errors=True)

    # 1c. Also run our adopt test file against Base src to confirm its newly-recovered cases are RED on base!
    aproc = subprocess.run(
        [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--tb=no", ADOPT_FILE],
        cwd=REPO_ROOT,
        env=make_env([str(BASE_SRC), str(REPO_ROOT)]),
        capture_output=True,
        text=True,
        timeout=60,
    )
    alast = aproc.stdout.strip().splitlines()[-1] if aproc.stdout.strip() else ""
    print(f"[ADOPT ON BASE] rc={aproc.returncode}: {alast}", flush=True)
    assert aproc.returncode == 1

    return {
        "base_total_failed_with_fifo": 141,
        "base_total_passed": 219,
        "named_green_count": len(named_green),
        "unprefixed_passed_count": len(unprefixed),
        "unprefixed_bucket_counts": {k: len(v) for k, v in buckets.items()},
        "unprefixed_other_examples": buckets["other"],
        "mutation_swallow_send": swallow_res,
        "adopt_on_base_summary": alast,
    }


def run_pytest_checked(label: str, args: list[str], env: dict[str, str], allow_socket_fail: bool = False) -> dict:
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
    if not allow_socket_fail:
        assert proc.returncode == 0, f"{label} FAILED:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    elif proc.returncode != 0:
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
    results: dict[str, object] = {}
    results["base_and_swallow_send"] = run_base_breakdown_and_swallow_send()

    # 2. Ordering & process-wide state isolation (both orders in one process)
    results["order_new_before_existing"] = run_pytest_checked(
        "ORDER_NEW_BEFORE_EXISTING",
        ["-p", "_audit_plugin_v2", "-q", *NEW_FILES, ADOPT_FILE, *NEIGHBOR_FILES],
        make_env(),
    )
    results["order_new_after_existing"] = run_pytest_checked(
        "ORDER_NEW_AFTER_EXISTING",
        ["-p", "_audit_plugin_v2", "-q", *NEIGHBOR_FILES, *NEW_FILES, ADOPT_FILE],
        make_env(),
    )

    # 3. Wall clocks in 2027 and 2028
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as clock_td:
        clock_dir = Path(clock_td)
        (clock_dir / "sitecustomize.py").write_text(SITECUSTOMIZE_CLOCK)
        for year, iso_str in (
            (2027, "2027-07-15T12:00:00-04:00"),
            (2028, "2028-07-15T12:00:00-04:00"),
        ):
            env_clock = make_env(
                prepend_paths=[str(clock_dir)],
                extra_vars={"CS6B2_WALL_CLOCK_ISO": iso_str},
            )
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
            results[f"wall_clock_{year}"] = run_pytest_checked(
                f"WALL_CLOCK_{year}",
                [
                    "-p", "_audit_plugin_v2", "-q",
                    *NEW_FILES,
                    "tests/test_saved_pick_paths_cs5a.py",
                    ADOPT_FILE,
                ],
                env_clock,
            )

    # 4. Combined one-process run specified in brief.md item 6
    combined_files = [
        "tests/test_lightgbm_optional.py",
        "tests/test_web_saver.py",
        "tests/test_morning_state_cs6b1.py",
        "tests/test_saved_pick_paths_cs5a.py",
        "tests/health/test_alert.py",
        *NEW_FILES,
        ADOPT_FILE,
    ]
    results["combined_one_process"] = run_pytest_checked(
        "COMBINED_ONE_PROCESS",
        ["-p", "_audit_plugin_v2", "-rA", "-q", *combined_files],
        make_env(),
        allow_socket_fail=True,
    )

    # 5. No-LightGBM collection check (ImportError and OSError) including adopt/
    with tempfile.TemporaryDirectory(dir=REVIEW_DIR) as lgb_td:
        for err_expr in ("ImportError('no lgbm')", "OSError('dlopen libomp failed')"):
            (Path(lgb_td) / "lightgbm.py").write_text(f"raise {err_expr}\n")
            env_nolgb = make_env(prepend_paths=[lgb_td])
            proc = subprocess.run(
                [PYTHON, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests", ADOPT_FILE],
                cwd=REPO_ROOT, env=env_nolgb, capture_output=True, text=True, timeout=120,
            )
            assert proc.returncode == 0, f"Collection failed under {err_expr}:\n{proc.stdout}\n{proc.stderr}"
        results["no_lightgbm_collection_including_adopt"] = "ok_importerror_and_oserror"

    (REVIEW_DIR / "pytest_remaining.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    print("ALL_REMAINING_PYTEST_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
    main()
