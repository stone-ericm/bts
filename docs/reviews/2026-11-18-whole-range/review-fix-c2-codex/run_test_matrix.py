"""Run the agreed combined, parent and mutation suites through the shared wrapper."""

from datetime import datetime, time
from pathlib import Path
import subprocess
import sys
from time import sleep
from zoneinfo import ZoneInfo


REVIEW = Path(__file__).resolve().parent
REPO = REVIEW.parents[1]
NEW_TESTS = "tests/test_wr_c2_recovery.py"
OLD_TESTS = "tests/test_alert_status_callers_cs6b2.py"
COMBINED = (
    NEW_TESTS, "tests/test_lightgbm_optional.py", "tests/test_web_saver.py", OLD_TESTS,
    "tests/test_late_delivery_guard.py", "tests/test_stale_delivery_cs4c.py",
    "tests/test_stale_delivery_cs4c_carbon.py", "tests/test_morning_state_cs6b1.py",
)
MUTATIONS = (
    "wall-key", "wall-filename", "shared-file", "ack-none", "drop-classification",
    "early-classification", "retry-banner", "follow-symlinks", "follow-parent",
)
EASTERN = ZoneInfo("America/New_York")
WRAPPER = Path.home() / "bts-runs" / "yield_run.sh"


def wait_for_runner():
    if datetime.now(EASTERN).time() < time(20, 50):
        raise RuntimeError("Shared wrapper reserved until 20:50 ET")
    announced = False
    while True:
        listing = subprocess.run(["pgrep", "-f", "yield_run.sh"], text=True, capture_output=True, timeout=10)
        if listing.returncode == 1 and not listing.stderr:
            return
        if listing.returncode != 0 or listing.stderr:
            raise RuntimeError(f"Cannot check shared runner: {listing.stderr.strip() or listing.returncode}")
        if not announced:
            print(f"Shared runner active ({listing.stdout.strip().replace(chr(10), ', ')}), waiting.")
            announced = True
        sleep(10)


def run_job(label, source, targets):
    wait_for_runner()
    log = REVIEW / f"{label}.log"
    if log.exists():
        raise FileExistsError(log)
    env = {
        "TZ": "America/New_York",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(source),
        "MPLCONFIGDIR": str(REVIEW / "mpl"),
    }
    args = [
        str(WRAPPER), str(log.relative_to(REPO)), "env",
        *(f"{key}={value}" for key, value in env.items()),
        ".venv/bin/python", "-m", "pytest", "-p", "no:cacheprovider",
        *targets, "-q", "--tb=short",
    ]
    print(f"Starting {label} at {datetime.now(EASTERN):%H:%M:%S ET}.")
    proc = subprocess.Popen(args, cwd=REPO)
    while True:
        try:
            code = proc.wait(timeout=45)
            break
        except subprocess.TimeoutExpired:
            pass
    if code not in {0, 1}:
        raise RuntimeError(f"Wrapper for {label} exited {code}. Inspect {log}.")
    print(f"Finished {label} at {datetime.now(EASTERN):%H:%M:%S ET}, pytest exit {code}.")
    return code


def main():
    jobs = {
        "head-combined": (REPO / "src", COMBINED),
        "base-red": (REVIEW / "base" / "src", (NEW_TESTS, OLD_TESTS)),
    }
    for mutation in MUTATIONS:
        jobs[f"mut-{mutation}"] = (REVIEW / "mutations" / mutation / "src", (NEW_TESTS,))
    if sys.argv[1:]:
        jobs = {label: jobs[label] for label in sys.argv[1:]}
    for label, (source, targets) in jobs.items():
        run_job(label, source, targets)


if __name__ == "__main__":
    main()
