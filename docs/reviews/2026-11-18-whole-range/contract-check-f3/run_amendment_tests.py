import os
from pathlib import Path
import subprocess
import sys

review = Path(__file__).resolve().parent
scratch = review / "scratch-amendment-1"
environment = dict(os.environ, TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=os.pathsep.join(map(str, (review, scratch / "src", scratch))))
pattern = r"day_outcome|day-outcome|classify_day|DayOutcome|deliberate_skip|OK_STATES|\bgather\b"
files = subprocess.run(["rg", "-l", pattern, "tests", "scripts"], cwd=scratch,
                       text=True, capture_output=True, check=True, timeout=5)
failed = []
for path in sorted(files.stdout.splitlines()):
    if not Path(path).name.startswith("test_") or not path.endswith(".py"):
        print(f"search hit, not a test: {path}", flush=True)
        continue
    limit = 55 if path == "tests/test_wr_f1_wrong_game.py" else 15
    with (review / "logs-amendment-1" / f"{Path(path).stem}.log").open("w") as log:
        result = subprocess.run(
            ["timeout", "-k", "2s", f"{limit}s", "nice", "-n", "19", sys.executable,
             "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider", "-q", "--tb=short", path],
            env=environment, cwd=scratch, stdout=log, stderr=subprocess.STDOUT, timeout=limit + 4,
        )
    print(f"{result.returncode:3} {path}", flush=True)
    if result.returncode:
        failed.append(path)
print("Files with nonzero exits:", failed, flush=True)
