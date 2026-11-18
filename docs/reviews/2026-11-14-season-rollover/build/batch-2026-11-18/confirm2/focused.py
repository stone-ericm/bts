"""Run the changed tests and both compatibility files under each ET clock."""
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import time

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parent.parent


def run_year(year):
    import pytest
    from sitecustomize import check

    check()
    assert dt.datetime.now().year == dt.date.today().year == time.localtime().tm_year == year
    assert "EST" in time.tzname
    print(f"ET wall clock {dt.datetime.now().astimezone().isoformat()}", flush=True)
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "3d49308d", "HEAD", "--", "tests"],
        text=True, cwd=ROOT,
    ).splitlines()
    return pytest.main(["-o", "addopts=", "-p", "no:cacheprovider", "-q", "--tb=short",
                        "--basetemp", str(REVIEW / f"focused-tmp-{year}"),
                        *changed, "tests/test_lightgbm_optional.py", "tests/test_web_saver.py"])


def main():
    if len(sys.argv) > 1:
        return run_year(int(sys.argv[1]))
    now = dt.datetime.now(dt.timezone.utc)
    codes = []
    for year in (2027, 2028):
        offset = int((now.replace(year=year, month=4, day=3) - now).total_seconds())
        env = os.environ | {
            "TZ": "America/New_York",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(map(str, (REVIEW, ROOT / "src", ROOT))),
            "DYLD_INSERT_LIBRARIES": str(REVIEW / "review_clock.dylib"),
            "BTS_CONFIRM_CLOCK_OFFSET": str(offset),
        }
        codes.append(subprocess.run([sys.executable, __file__, str(year)], cwd=ROOT, env=env).returncode)
    print(f"Focused exit codes (2027, 2028): {codes}", flush=True)
    return int(any(codes))


if __name__ == "__main__":
    raise SystemExit(main())
