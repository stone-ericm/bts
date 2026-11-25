"""Change one C27 behavior per scratch source tree and run its targeted tests."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


WORK = Path(__file__).resolve().parent
ROOT = WORK.parent.parent
TEXTS = "tests/test_wr_e_texts.py"
DELIVERY = "tests/test_stale_delivery_cs4c.py"
MORNING = "tests/test_morning_state_cs6b1.py"

CASES = {
    "delivery-slate": (
        'f"Write for the {warning_day.isoformat()} slate only on that ET date, with prior entries settled or absent. "',
        'f"Write only if the pick\'s slate and today\'s ET date both equal {warning_day.isoformat()}. "',
        [f"{DELIVERY}::test_warning_date_is_emission_date", f"{TEXTS}::test_stale_warning_maximum_detail[delivery]"],
    ),
    "delivery-saver-year": (
        'f"In {warning_day.year:04d} ET away from midnight,',
        'f"On {warning_day.isoformat()} ET away from midnight,',
        [f"{DELIVERY}::test_attempt_snapshot_and_confirmed_warning", f"{TEXTS}::test_stale_warning_maximum_detail[delivery]"],
    ),
    "morning-saver-year": (
        'In {date[:4]} ET away from midnight,',
        'On {date} ET away from midnight,',
        [f"{MORNING}::test_stale_warning_uses_each_snapshot", f"{TEXTS}::test_stale_warning_maximum_detail[morning]"],
    ),
    "legacy-streak-used": (
        'sentence += f" Streak recorded with this check: {streak}."',
        'sentence += f" Streak used at this check: {streak}."',
        [f"{TEXTS}::test_skip_streak_basis", f"{TEXTS}::test_skip_surfaces"],
    ),
    "stored-text-unchanged": (
        '.replace("below the", "under the")',
        '',
        [f"{TEXTS}::test_skip_stored_text_surfaces"],
    ),
}

DRIVER = """
from pathlib import Path
import sys
import bts.scheduler
import pytest
result = Path(bts.scheduler.__file__).resolve()
assert result.is_relative_to(Path(sys.argv[1]).resolve()), result
raise SystemExit(pytest.main(sys.argv[2:]))
"""


def run_case(label):
    old, changed, tests = CASES[label]
    with TemporaryDirectory(dir=WORK) as temp:
        source = Path(temp) / "src"
        shutil.copytree(ROOT / "src", source, ignore=shutil.ignore_patterns("__pycache__"))
        scheduler = source / "bts" / "scheduler.py"
        original = scheduler.read_text()
        assert original.count(old) == 1, label
        scheduler.write_text(original.replace(old, changed))
        result = subprocess.run(
            [sys.executable, "-B", "-c", DRIVER, str(source), "-p", "no:cacheprovider",
             "-q", "--tb=short", *tests],
            env=os.environ | {"PYTHONPATH": f"{source}{os.pathsep}{ROOT}"},
            cwd=ROOT, timeout=12, capture_output=True, text=True,
        )
    (WORK / f"mutation-{label}.log").write_text(result.stdout + result.stderr)
    assert result.returncode in (0, 1), (label, result.returncode)
    failures = [line for line in result.stdout.splitlines() if line.startswith("FAILED ")]
    if result.returncode:
        assert failures and "AssertionError" in result.stdout, label
    print(label, "detected" if failures else "unnoticed", *failures, sep="\n", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or CASES:
        run_case(name)
