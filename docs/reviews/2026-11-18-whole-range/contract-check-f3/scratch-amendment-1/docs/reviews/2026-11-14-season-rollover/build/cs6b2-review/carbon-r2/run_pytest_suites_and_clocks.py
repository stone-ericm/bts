"""Pytest Family B runner (invoked once via `~/bts-runs/yield_run.sh`):
1. Runs all 6b-2 test files + `adopt/` in one pytest process with
   `tests/test_lightgbm_optional.py`, `tests/test_web_saver.py`,
   `tests/test_morning_state_cs6b1.py`, `tests/test_saved_pick_paths_cs5a.py`,
   and `tests/health/test_alert.py`, in BOTH orders, auditing per-test process-wide
   state (`SIGALRM`, `ITIMER_REAL`, `os.open`, `os.fstat`, `os.ftruncate`, `Path.stat`,
   `Path.open`, `builtins.open`, `io.open`, `umask`, `cwd`, `recursionlimit`,
   `int_max_str_digits`, `strategy._mdp_cache`).
2. Runs the suite under shifted native wall clocks in 2027 (`2027-07-15 12:00 EDT`)
   and 2028 (`2028-07-15 12:00 EDT`) via `libclock_interpose.dylib`.
3. Verifies `test_green_writer_parity` succeeds when pytest is invoked from a
   non-root working directory (confirming Round-1 Note 3's `cwd` fix).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
DYLIB_PATH = REVIEW_DIR / "libclock_interpose.dylib"


CS6B2_FILES = [
    "tests/health/test_alert_status_cs6b2.py",
    "tests/health/test_alert_status_cs6b2_review.py",
    "tests/test_alert_status_callers_cs6b2.py",
    "tests/test_alert_status_cs6b2_carbon.py",
    ".codex-review/cs6b2-r2/adopt/test_alert_status_cs6b2_r2_adopt.py",
]

PEER_FILES = [
    "tests/test_lightgbm_optional.py",
    "tests/test_web_saver.py",
    "tests/test_morning_state_cs6b1.py",
    "tests/test_saved_pick_paths_cs5a.py",
    "tests/health/test_alert.py",
]


def run_suite(
    label: str,
    targets: list[str],
    summary_file: Path,
    *,
    clock_offset_sec: int | None = None,
    cwd: Path = REPO_ROOT,
    extra_args: list[str] | None = None,
) -> dict[str, object]:
    env = dict(os.environ)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["BTS_SUMMARY_OUT"] = str(summary_file)
    pp_parts = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REVIEW_DIR)]
    if env.get("PYTHONPATH"):
        pp_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp_parts)

    if clock_offset_sec is not None:
        assert DYLIB_PATH.is_file(), f"missing {DYLIB_PATH}"
        env["DYLD_INSERT_LIBRARIES"] = str(DYLIB_PATH)
        env["BTS_CLOCK_OFFSET_SEC"] = str(clock_offset_sec)

    summary_file.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-p",
        "review_pytest_plugin",
        "-q",
        *(extra_args or []),
        *targets,
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    elapsed = round(time.monotonic() - t0, 2)
    assert summary_file.is_file(), (
        f"[{label}] no summary written (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
    )
    data = json.loads(summary_file.read_text(encoding="utf-8"))
    print(
        f"SUITE {label}: et_now={data['et_now']} counts={data['counts']} "
        f"leaks={len(data['state_leaks'])} ({elapsed}s)",
        flush=True,
    )
    assert data["state_leaks"] == [], f"[{label}] process-wide state leaks: {data['state_leaks']}"

    failed_ids = [it["nodeid"] for it in data["failed_items"]]
    allowed_sandbox_socket = ["tests/test_web_saver.py::test_do_post_saver_transition_socket_wiring"]
    if failed_ids:
        print(f"  [{label}] failed_ids={failed_ids}", flush=True)
    assert failed_ids == [] or (
        failed_ids == allowed_sandbox_socket
        and "Operation not permitted" in data["failed_items"][0]["longrepr"]
    ), f"[{label}] unexpected failures: {data['failed_items']}"
    return data


def main() -> None:
    real_now_utc = int(time.time())
    target_2027_utc = int(datetime(2027, 7, 15, 16, 0, 0, tzinfo=timezone.utc).timestamp())
    target_2028_utc = int(datetime(2028, 7, 15, 16, 0, 0, tzinfo=timezone.utc).timestamp())
    offset_2027 = target_2027_utc - real_now_utc
    offset_2028 = target_2028_utc - real_now_utc

    with tempfile.TemporaryDirectory(dir=REVIEW_DIR, prefix="tmp_suites_") as tmp_s:
        tmp = Path(tmp_s)
        summary_json = tmp / "summary.json"

        # 1. Combined one-process run — Order 1 (6b-2 first, then peers)
        d_order1 = run_suite("combined_order_1", CS6B2_FILES + PEER_FILES, summary_json)

        # 2. Combined one-process run — Order 2 (peers first, then 6b-2 reversed)
        d_order2 = run_suite(
            "combined_order_2", PEER_FILES + list(reversed(CS6B2_FILES)), summary_json
        )
        assert d_order1["counts"] == d_order2["counts"], (
            f"order count mismatch: {d_order1['counts']} vs {d_order2['counts']}"
        )

        # 3. Wall clock 2027-07-15 12:00 EDT
        d_2027 = run_suite(
            "wall_clock_2027",
            CS6B2_FILES + PEER_FILES,
            summary_json,
            clock_offset_sec=offset_2027,
        )
        assert d_2027["et_year"] == 2027, f"expected 2027 clock, got {d_2027['et_now']}"
        assert d_2027["counts"] == d_order1["counts"]

        # 4. Wall clock 2028-07-15 12:00 EDT
        d_2028 = run_suite(
            "wall_clock_2028",
            CS6B2_FILES + PEER_FILES,
            summary_json,
            clock_offset_sec=offset_2028,
        )
        assert d_2028["et_year"] == 2028, f"expected 2028 clock, got {d_2028['et_now']}"
        assert d_2028["counts"] == d_order1["counts"]

        # 5. Subdirectory cwd check for base_alert git archive (Round-1 Note 3 fix)
        sub_cwd = tmp / "non_root_cwd"
        sub_cwd.mkdir()
        d_sub = run_suite(
            "subdir_cwd_parity",
            [str(REPO_ROOT / "tests/health/test_alert_status_cs6b2.py")],
            summary_json,
            cwd=sub_cwd,
            extra_args=["-k", "test_green_writer_parity"],
        )
        assert d_sub["counts"]["passed"] == 36 and d_sub["counts"]["failed"] == 0

    print("\nPYTEST FAMILY B COMPLETE: BOTH ORDERS, 0 STATE LEAKS, AND 2027/2028 CLOCKS PASSED", flush=True)


if __name__ == "__main__":
    main()
