import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from probe_mutations import HERE, REPO, mutation


def check_alarm():
    from probe_support import DAY, bts, health, observed, scratch

    print(f"bts.__file__ = {bts.__file__}", flush=True)
    with scratch() as directory:
        picks = directory / "picks"
        observed(picks)
        os.mkfifo(picks / "account_state/saver_state.json")
        alarms = []

        def raise_timeout(*_):
            alarms.append(True)
            raise TimeoutError("test alarm fired")

        signal.signal(signal.SIGALRM, raise_timeout)
        for mode in (None, DAY.date()):
            signal.alarm(1)
            try:
                assert health.check(picks, now=DAY, historical_day=mode) == []
            finally:
                signal.alarm(0)
            print(f"historical={mode is not None}: check returned normally, alarm count={len(alarms)}", flush=True)
        assert len(alarms) == 2


if __name__ == "__main__" and sys.argv[1:] == ["child"]:
    check_alarm()
elif __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="fifo-reader-", dir=HERE) as directory:
        source = Path(directory) / "src"
        shutil.copytree(REPO / "src", source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        mutation(source, "live-reader")
        env = {**os.environ, "PYTHONPATH": f"{source}:{REPO}"}
        subprocess.run([sys.executable, __file__, "child"], cwd=REPO, env=env, timeout=8, check=True)
        run = subprocess.run([
            sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--tb=no",
            "tests/test_contest_dated_cs4a.py::test_health_observation_saver_without_flag_read",
        ], cwd=REPO, env=env, timeout=10, check=False, capture_output=True, text=True)
        print(run.stdout, end="")
        assert run.returncode == 1 and "28 failed, 56 passed" in run.stdout
