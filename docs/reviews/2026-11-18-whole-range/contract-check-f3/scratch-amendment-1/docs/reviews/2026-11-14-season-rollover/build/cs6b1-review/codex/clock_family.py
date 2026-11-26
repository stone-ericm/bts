"""Replay the new test module under two scoped Python wall clocks."""

import os
import subprocess
import sys

from pytest_family import NEW, REVIEW, ROOT


if __name__ == "__main__":
    result = 0
    for year in (2027, 2028):
        env = {**os.environ, "CS6B1_CLOCK_YEAR": str(year),
               "PYTHONPATH": os.pathsep.join(map(str, (REVIEW, ROOT / "src", ROOT))),
               "PYTEST_PLUGINS": "wall_clock_review,offline_review"}
        with (REVIEW / f"wall-{year}.log").open("w") as log:
            run = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", NEW, "-q", "--tb=short"],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"wall {year}: pytest exit {run.returncode}", flush=True)
        result |= int(run.returncode != 0)
    sys.exit(result)
