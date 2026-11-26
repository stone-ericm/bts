"""Run each file found by the contract's literal search in a separate short process."""
import os
from pathlib import Path
import subprocess
import sys

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
PATTERN = ("check_pick_entered|check-pick-entered|_check_pick_entered|pick_entry_status|"
           "entry_receipt|EntryReceipt|qualify|load_units|target_rows|pick_entry_check|_selection")
found = subprocess.run(["rg", "-l", PATTERN, "tests"], check=True, capture_output=True,
                       text=True, timeout=3, cwd=ROOT)
paths = sorted(found.stdout.splitlines())
(REVIEW / "test-files.txt").write_text(f"rg -l '{PATTERN}' tests\n" + "\n".join(paths) + "\n")
env = dict(os.environ, TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1")
env["PYTHONPATH"] = os.pathsep.join(map(str, (REVIEW, REVIEW / "scratch" / "src", ROOT)))
logs = REVIEW / "pytest"
logs.mkdir(exist_ok=True)
for path in paths[int(sys.argv[1]):int(sys.argv[2])]:
    logfile = logs / (path.removeprefix("tests/").replace("/", "__") + ".log")
    with logfile.open("w") as output:
        result = subprocess.run(["nice", "-n", "19", "timeout", "12s", str(ROOT / ".venv/bin/python"),
                                 "-m", "pytest", "-o", "addopts=", "-p", "no:cacheprovider", "-q",
                                 "--tb=short", path], cwd=ROOT, env=env, stdout=output,
                                stderr=subprocess.STDOUT, timeout=14)
    summary = logfile.read_text().strip().splitlines()
    print(f"{result.returncode}\t{path}\t{summary[-1] if summary else '(empty)'}", flush=True)
