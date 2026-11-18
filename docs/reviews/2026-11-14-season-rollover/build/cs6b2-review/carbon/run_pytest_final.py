"""Final unified pytest verification driver (run under ~/bts-runs/yield_run.sh):
1. Base (59efa825) RED/GREEN exact item-level breakdown (-v --tb=no -rN, regex-anchored),
   swallow_send mutation check, and adopt/ RED verification against Base.
2. Process-wide state & ordering checks (new + adopt BEFORE and AFTER
   test_morning_state_cs6b1.py and test_saved_pick_paths_cs5a.py in one process, both orders).
3. Native 2027-07-15 and 2028-07-15 ET wall clocks via libclock_interpose.dylib
   (monotonic time untouched) on all added, changed, and adopted test files.
4. Combined one-process pytest run with test_lightgbm_optional.py, test_web_saver.py,
   test_morning_state_cs6b1.py, test_saved_pick_paths_cs5a.py, tests/health/test_alert.py,
   the two new CS6b-2 test files, and the adopt test file.
5. No-LightGBM collection check (ImportError and OSError) across tests/ and adopt/.
"""

from __future__ import annotations

from datetime import datetime
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
MUT_DIR = REVIEW_DIR / "scratch_mut_send2"
MUT_SRC = MUT_DIR / "src"
DYLIB = REVIEW_DIR / "libclock_interpose.dylib"
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

STATUS_LINE_RE = re.compile(r"^(\S+::\S+)\s+(PASSED|FAILED)\b")


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


def run_base_and_swallow_send() -> dict:
    # 1a. Fast exact breakdown on Base (deselecting the 12 FIFO 2s-timeout cases already verified RED in Family A)
    targets = [
        "tests/health/test_alert_status_cs6b2.py",
        "tests/test_alert_status_callers_cs6b2.py",
        "tests/test_saved_pick_paths_cs5a.py::test_refusal_survives_non_object_health_status",
    ]
    proc = subprocess.run(
        [
            PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-p", "_loopback_only_plugin",
            "-v", "--tb=no", "-rN", "-k", "not fifo", *targets,
        ],
        cwd=REPO_ROOT,
        env=make_env([str(BASE_SRC), str(REPO_ROOT)]),
        capture_output=True,
        text=True,
        timeout=90,
    )
    passed_ids: list[str] = []
    failed_ids: list[str] = []
    for ln in proc.stdout.splitlines():
        m = STATUS_LINE_RE.match(ln.strip())
        if m:
            tid, status = m.group(1), m.group(2)
            if status == "PASSED":
                passed_ids.append(tid)
            else:
                failed_ids.append(tid)

    print(
        f"[BASE BREAKDOWN (non-fifo)] passed={len(passed_ids)}, failed={len(failed_ids)} (+12 fifo RED = {len(failed_ids) + 12})",
        flush=True,
    )
    assert proc.returncode == 1
    assert len(passed_ids) + len(failed_ids) == 348, (
        f"Expected 348 non-fifo items, got {len(passed_ids)} + {len(failed_ids)}"
    )

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

    # 1b. 13th mutation: swallow_send inside dispatch_dm_for_health_alerts
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
            [
                PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-p", "_loopback_only_plugin",
                "-q", "--tb=no", "-rN", *NEW_FILES,
            ],
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

    # 1c. Verify adopt/ test file is RED against Base src
    aproc = subprocess.run(
        [
            PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-p", "_loopback_only_plugin",
            "-q", "--tb=no", "-rN", ADOPT_FILE,
        ],
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
        "base_total_passed": len(passed_ids),
        "base_total_failed_including_12_fifo": len(failed_ids) + 12,
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
        [
            PYTHON, "-m", "pytest", "-p", "no:cacheprovider",
            "-p", "_loopback_only_plugin", "-p", "_audit_plugin_v2", *args,
        ],
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
    results["base_and_swallow_send"] = run_base_and_swallow_send()

    # 2. Ordering & process-wide state isolation (both orders in one process)
    results["order_new_before_existing"] = run_pytest_checked(
        "ORDER_NEW_BEFORE_EXISTING",
        ["-q", *NEW_FILES, ADOPT_FILE, *NEIGHBOR_FILES],
        make_env(),
    )
    results["order_new_after_existing"] = run_pytest_checked(
        "ORDER_NEW_AFTER_EXISTING",
        ["-q", *NEIGHBOR_FILES, *NEW_FILES, ADOPT_FILE],
        make_env(),
    )

    # 3. Native wall clocks in 2027 and 2028 via libclock_interpose.dylib
    for year, iso_str in (
        (2027, "2027-07-15T12:00:00-04:00"),
        (2028, "2028-07-15T12:00:00-04:00"),
    ):
        epoch = int(datetime.fromisoformat(iso_str).timestamp())
        env_clock = make_env(
            extra_vars={
                "DYLD_INSERT_LIBRARIES": str(DYLIB),
                "CS6B2_TARGET_EPOCH": str(epoch),
            }
        )
        results[f"wall_clock_{year}"] = run_pytest_checked(
            f"WALL_CLOCK_{year}",
            [
                "-q",
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
        ["-rA", "-q", *combined_files],
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

    (REVIEW_DIR / "pytest_final.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    print("ALL_FINAL_PYTEST_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
    main()
