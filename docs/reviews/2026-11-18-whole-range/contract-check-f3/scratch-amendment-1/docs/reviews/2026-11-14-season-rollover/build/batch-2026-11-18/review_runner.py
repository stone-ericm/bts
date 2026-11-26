import datetime as dt
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import offline

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parent.parent
PYTHON = ROOT / ".venv/bin/python"
BASE = REVIEW / "base"
CHANGED_TESTS = (
    "tests/experiment/test_artifacts.py",
    "tests/health/test_park_drag_freshness.py",
    "tests/scripts/test_live_forward_capture_once.py",
    "tests/test_contest_dated_cs4a.py",
    "tests/test_contest_fetch.py",
    "tests/test_entry_intent.py",
    "tests/test_park_drag_producer.py",
    "tests/test_shadow_eval.py",
    "tests/test_skip_policy_shadow.py",
)


def archive(revision, target, *paths):
    payload = subprocess.check_output(["git", "archive", revision, *paths], cwd=ROOT)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as source:
        source.extractall(target, filter="data")


def run(revision, *args, cwd=ROOT, year=None):
    env = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(map(str, (revision / "src", revision, ROOT))),
        "TZ": "America/New_York",
    }
    if year is not None:
        today = dt.date.today()
        env.update(DYLD_INSERT_LIBRARIES=str(REVIEW / "wall_clock.dylib"),
                   BTS_BATCH_CLOCK_OFFSET=str((today.replace(year=year) - today).days * 86400),
                   BTS_BATCH_EXPECT_YEAR=str(year))
    return subprocess.run([str(PYTHON), str(__file__), *args], cwd=cwd, env=env).returncode


def pytest_child(args):
    import pytest
    year = os.environ.get("BTS_BATCH_EXPECT_YEAR")
    if year:
        print(f"wall clock: {dt.datetime.now().isoformat()} / {dt.date.today()}", flush=True)
        assert dt.date.today().year == dt.datetime.now().year == int(year)
    return pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", *args])


def main():
    action, *args = sys.argv[1:]
    if action == "archive":
        archive("3d49308d", BASE, "src", "scripts")
        return 0
    if action == "focused":
        tests = (*CHANGED_TESTS, "tests/test_lightgbm_optional.py", "tests/test_web_saver.py")
        results = [run(ROOT, "pytest", *tests, year=year) for year in (2027, 2028)]
        print(f"Focused exit codes 2027/2028: {results}")
        return int(any(results))
    offline.install()
    if action == "pytest":
        return pytest_child(args)
    if action == "probe":
        from review_probes import probes
        return probes(*args)
    if action == "mutate":
        from review_mutations import mutations
        return mutations()
    raise ValueError(action)


if __name__ == "__main__":
    raise SystemExit(main())
