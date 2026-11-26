"""Run each pytest family serially under the yield wrapper."""

import os
from pathlib import Path
import subprocess
import sys


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
NEW = "tests/test_morning_state_cs6b1.py"
LEGACY = [
    "tests/test_lightgbm_optional.py", "tests/test_web_saver.py",
    "tests/test_saved_pick_paths_cs5a.py", "tests/test_stale_delivery_cs4c.py",
    "tests/test_stale_delivery_cs4c_carbon.py",
]


def pytest_run(name, paths, *, source=None, year=None, options=()):
    env = {
        **os.environ,
        "TZ": "America/New_York",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(str(path) for path in (source or ROOT / "src", ROOT, REVIEW)),
        "PYTEST_PLUGINS": "offline_review",
    }
    if year:
        env["REVIEW_WALL_YEAR"] = str(year)
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *paths, "-q", *options]
    with (REVIEW / f"{name}.log").open("w") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"{name}: pytest exit {result.returncode}", flush=True)
    return result.returncode


if __name__ == "__main__":
    family = sys.argv[1]
    if family == "combined":
        sys.exit(pytest_run("combined-pytest", [NEW, *LEGACY], options=("--tb=short",)))
    if family == "clocks":
        codes = [pytest_run(f"clock-{year}", [NEW], year=year, options=("--tb=short",))
                 for year in (2027, 2028)]
        sys.exit(int(any(codes)))
    if family == "base":
        sys.exit(pytest_run("base-pytest", [NEW], source=REVIEW / "base" / "src",
                            options=("-rA", "--tb=short")))
    raise SystemExit(f"unknown test family: {family}")
