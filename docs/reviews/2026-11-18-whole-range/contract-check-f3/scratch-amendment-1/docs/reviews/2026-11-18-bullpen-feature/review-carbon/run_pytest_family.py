"""Run all pytest verification checks for item 7 in one wrapper invocation."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_TREE = HERE / "base_tree"
CLOCK_SHIM = HERE / "clock_shim"
NO_LGB_SHIM = HERE / "no_lgb_shim"

PA_DIR = "/Users/eric/projects/bts/data/processed"
LOOKUP_PATH = "/Users/eric/bts-runs/inputs/probable_pitcher_lookup.2017-2025.json"

ADOPT_FILE = ".codex-review/bullpen-review/adopt/test_bullpen_review_probes.py"


def run_cmd(label: str, args: list[str], extra_env: dict[str, str]) -> tuple[int, str]:
    env = dict(os.environ)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra_env)
    proc = subprocess.run(args, cwd=REPO, env=env, capture_output=True, text=True)
    print(f"\n================ [{label}] exit={proc.returncode} ================", flush=True)
    print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(f"--- stderr ---\n{proc.stderr}", flush=True)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def main():
    # Verify base_tree/src overrides editable install
    rc_chk, out_chk = run_cmd(
        "verify_base_import_path",
        [sys.executable, "-c", "import bts.features.compute as c; print(c.__file__)"],
        {"PYTHONPATH": f"{BASE_TREE / 'src'}:{REPO}"},
    )
    assert rc_chk == 0 and "base_tree" in out_chk, f"base_tree import override failed: {out_chk}"

    # 1. Base src with real data: expect 4 failed, 2 passed
    rc1, out1 = run_cmd(
        "1_base_src_with_real_data",
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "tests/test_bullpen_same_day.py"],
        {
            "PYTHONPATH": f"{BASE_TREE / 'src'}:{REPO}",
            "BULLPEN_TEST_DATA": PA_DIR,
            "BULLPEN_TEST_LOOKUP": LOOKUP_PATH,
        },
    )
    assert rc1 == 1 and "4 failed, 2 passed" in out1, f"Unexpected base pytest result: rc={rc1}"

    # 2. Branch with real data hidden: clean skip
    rc2, out2 = run_cmd(
        "2_branch_hidden_real_data",
        [
            sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
            "tests/test_bullpen_same_day.py", ADOPT_FILE,
        ],
        {
            "PYTHONPATH": str(REPO),
            "BULLPEN_TEST_DATA": str(HERE / "nonexistent_data_dir"),
            "BULLPEN_TEST_LOOKUP": str(HERE / "nonexistent_lookup.json"),
        },
    )
    assert rc2 == 0 and "11 passed, 2 skipped" in out2, f"Unexpected hidden-data result: rc={rc2}"

    # 3. Without lightgbm (both collect-only and execution on new + adopt tests)
    if NO_LGB_SHIM.exists():
        shutil.rmtree(NO_LGB_SHIM)
    NO_LGB_SHIM.mkdir(parents=True)
    (NO_LGB_SHIM / "lightgbm.py").write_text("raise ImportError('simulated missing lightgbm')\n")
    rc3, out3 = run_cmd(
        "3_without_lightgbm_new_and_adopt",
        [
            sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
            "tests/test_bullpen_same_day.py", ADOPT_FILE,
        ],
        {
            "PYTHONPATH": f"{NO_LGB_SHIM}:{REPO}",
            "BULLPEN_TEST_DATA": str(HERE / "nonexistent_data_dir"),
            "BULLPEN_TEST_LOOKUP": str(HERE / "nonexistent_lookup.json"),
        },
    )
    assert rc3 == 0 and "10 passed, 3 skipped" in out3, f"Unexpected no-lightgbm result: rc={rc3}"

    # 4. Combined single pytest process: new file + adopt + test_lightgbm_optional + existing feature tests + web_saver
    combined_targets = [
        "tests/test_bullpen_same_day.py",
        ADOPT_FILE,
        "tests/test_lightgbm_optional.py",
        "tests/test_context_features.py",
        "tests/test_park_drag_feature.py",
        "tests/test_swing_features.py",
        "tests/model/test_predict_feature_coverage.py",
        "tests/experiment/test_features.py",
        "tests/test_cold_start_integration.py",
        "tests/test_lgb_params.py",
        "tests/test_web_saver.py",
    ]
    rc4, out4 = run_cmd(
        "4_combined_single_process_with_real_data",
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *combined_targets],
        {
            "PYTHONPATH": str(REPO),
            "BULLPEN_TEST_DATA": PA_DIR,
            "BULLPEN_TEST_LOOKUP": LOOKUP_PATH,
        },
    )
    assert rc4 == 0, f"Combined pytest process failed with rc={rc4}"

    # 5 & 6. Wall clocks in 2027 and 2028
    clock_targets = [
        "tests/test_bullpen_same_day.py",
        ADOPT_FILE,
        "tests/test_context_features.py",
        "tests/test_park_drag_feature.py",
        "tests/test_swing_features.py",
        "tests/model/test_predict_feature_coverage.py",
        "tests/experiment/test_features.py",
    ]
    for year_label, iso_str, expected_date in (
        ("5_wall_clock_2027", "2027-06-15T12:00:00-04:00", "2027-06-15"),
        ("6_wall_clock_2028_leap", "2028-02-29T12:00:00-05:00", "2028-02-29"),
    ):
        rc_clk, out_clk = run_cmd(
            f"{year_label}_verify_date",
            [sys.executable, "-c", "import datetime; print('TODAY=', datetime.date.today().isoformat())"],
            {
                "PYTHONPATH": f"{CLOCK_SHIM}:{REPO}",
                "BTS_WALL_CLOCK_ISO": iso_str,
            },
        )
        assert rc_clk == 0 and f"TODAY= {expected_date}" in out_clk, f"Clock shim check failed: {out_clk}"

        rc_w, _ = run_cmd(
            year_label,
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *clock_targets],
            {
                "PYTHONPATH": f"{CLOCK_SHIM}:{REPO}",
                "BTS_WALL_CLOCK_ISO": iso_str,
                "BULLPEN_TEST_DATA": PA_DIR,
                "BULLPEN_TEST_LOOKUP": LOOKUP_PATH,
            },
        )
        assert rc_w == 0, f"{year_label} failed with rc={rc_w}"

    print("\nALL PYTEST FAMILY CHECKS PASSED.", flush=True)


if __name__ == "__main__":
    main()
