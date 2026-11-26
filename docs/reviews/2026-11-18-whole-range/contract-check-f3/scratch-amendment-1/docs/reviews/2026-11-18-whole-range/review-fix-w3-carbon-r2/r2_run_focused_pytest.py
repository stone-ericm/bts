"""Run the required 11-file one-process pytest suite plus the two 59b72c04 hook-regression files against checked-out c0917aa7."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
HEAD_SRC = REPO / "src"

REQUIRED_11 = [
    "tests/test_wr_w3.py",
    "tests/test_incident_2026_08_30.py",
    "tests/test_saved_pick_paths_cs5a.py",
    "tests/test_fallback_plan.py",
    "tests/test_scheduler.py",
    "tests/test_scheduler_decision_record_integration.py",
    "tests/test_e2_delivery_idempotency.py",
    "tests/test_late_delivery_guard.py",
    "tests/test_alert_status_cs6b2_carbon.py",
    "tests/test_lightgbm_optional.py",
    "tests/test_web_saver.py",
]

REGRESSION_HOOK_TESTS = [
    "tests/test_wr_a_pick.py::test_live_fresh_pick_refuses_observation_changed_after_selection",
    "tests/test_wr_a_review_carbon.py::test_real_midnight_rollover_on_late_slate_refuses_expired_dated_stamp",
]


def main() -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HEAD_SRC.resolve())
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    check = subprocess.run(
        [
            str(REPO / ".venv/bin/python"),
            "-c",
            f"import bts, pathlib; p = pathlib.Path(bts.__file__).resolve(); print('bts loaded from:', p); assert p.is_relative_to(pathlib.Path({str(HEAD_SRC.resolve())!r})), p",
        ],
        cwd=str(REPO),
        env=env,
        timeout=15,
    )
    if check.returncode != 0:
        return check.returncode

    print("=== Running required 11-file suite in one process ===", flush=True)
    r1 = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", *REQUIRED_11],
        cwd=str(REPO),
        env=env,
        timeout=600,
    )
    print(f"=== 11-file returncode: {r1.returncode} ===", flush=True)

    print("=== Running save_pick hook regression tests ===", flush=True)
    r2 = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", *REGRESSION_HOOK_TESTS],
        cwd=str(REPO),
        env=env,
        timeout=180,
    )
    print(f"=== hook-regression returncode: {r2.returncode} ===", flush=True)
    return r1.returncode or r2.returncode


if __name__ == "__main__":
    sys.exit(main())
