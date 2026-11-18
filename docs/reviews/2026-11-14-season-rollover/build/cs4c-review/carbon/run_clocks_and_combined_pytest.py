"""Final Item 8 pytest driver (run inside ~/bts-runs/yield_run.sh):
1. Base 1c6cbc77 (both src and scripts) audit on the two new CS4c test files.
2. 2027-07-15 and 2028-07-15 wall-clock runs across all 4 added/touched test files + adopt tests.
3. Single-process combined pytest run including tests/test_lightgbm_optional.py and tests/test_web_saver.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_ROOT = REVIEW_DIR / "scratch" / "base_1c6cbc77"
ADOPT_TEST = str(REVIEW_DIR / "adopt" / "test_cs4c_adversarial_review.py")

AUTHOR_NEW_TESTS = [
    "tests/test_automatic_reading_cs4c.py",
    "tests/test_stale_delivery_cs4c.py",
]
ALL_TOUCHED_AND_ADOPT = [
    "tests/test_automatic_reading_cs4c.py",
    "tests/test_stale_delivery_cs4c.py",
    "tests/test_pick_chosen_on_cs5a.py",
    "tests/test_saved_pick_paths_cs5a.py",
    ADOPT_TEST,
]
COMBINED_FILES = [
    *ALL_TOUCHED_AND_ADOPT,
    "tests/test_lightgbm_optional.py",
    "tests/test_web_saver.py",
]


def _run_pytest(args: list[str], *, py_paths: list[str], extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = ":".join(py_paths)
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)


def main() -> None:
    # Ensure scripts/ in BASE_ROOT is a package just like REPO_ROOT/scripts/__init__.py
    base_scripts_init = BASE_ROOT / "scripts" / "__init__.py"
    assert base_scripts_init.exists()

    # 1. Base 1c6cbc77 audit with BOTH base src and base scripts preceding REPO_ROOT
    p_base = _run_pytest(
        [*AUTHOR_NEW_TESTS, "-v", "--tb=no"],
        py_paths=[str(BASE_ROOT / "src"), str(BASE_ROOT), str(REPO_ROOT)],
    )
    passed = [
        ln.split(" PASSED")[0].strip()
        for ln in p_base.stdout.splitlines() if " PASSED" in ln and "::" in ln
    ]
    failed = [
        ln.split(" FAILED")[0].strip()
        for ln in p_base.stdout.splitlines() if " FAILED" in ln and "::" in ln
    ]
    (REVIEW_DIR / "base_full_audit.json").write_text(
        json.dumps({"passed": passed, "failed": failed}, indent=2)
    )
    print(f"=== BASE 1c6cbc77 (src + scripts) AUDIT: {len(passed)} passed, {len(failed)} failed ===")
    for item in passed:
        print(f"  GREEN_CONTROL: {item}")

    # 2. 2027 and 2028 wall-clock runs
    print("=== WALL CLOCK RUNS (2027 & 2028) ===")
    for label, iso in [
        ("clock_2027", "2027-07-15T14:00:00-04:00"),
        ("clock_2028", "2028-07-15T14:00:00-04:00"),
    ]:
        proc = _run_pytest(
            ["-p", "clock_plugin_v2", *ALL_TOUCHED_AND_ADOPT, "-q"],
            py_paths=[str(REVIEW_DIR), str(REPO_ROOT)],
            extra_env={"FAKE_WALL_CLOCK_ISO": iso},
        )
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        print(f"  {label} ({iso}): rc={proc.returncode} | {tail}")
        assert proc.returncode == 0, f"{label} failed:\n{proc.stdout}\n{proc.stderr}"

    # 3. Single-process combined run with test_lightgbm_optional.py and test_web_saver.py
    print("=== SINGLE-PROCESS COMBINED PYTEST RUN ===")
    p_comb = _run_pytest(
        [*COMBINED_FILES, "-q"],
        py_paths=[str(REPO_ROOT)],
    )
    tail_comb = p_comb.stdout.strip().splitlines()[-1] if p_comb.stdout.strip() else ""
    print(f"  combined: rc={p_comb.returncode} | {tail_comb}")
    assert p_comb.returncode == 0, f"combined failed:\n{p_comb.stdout}\n{p_comb.stderr}"
    print("CLOCKS_AND_COMBINED_ALL_PASSED")


if __name__ == "__main__":
    main()
