"""Run both shifted-clock suites in sequence under the outer yield wrapper."""
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import time

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parent.parent


def main():
    if len(sys.argv) > 1:
        expected = int(sys.argv[1])
        assert dt.datetime.now().year == dt.date.today().year == time.localtime().tm_year == expected
        assert time.tzname[0] == "EST"
        print("ET wall clock:", dt.datetime.now(), dt.date.today(), flush=True)
        import pytest
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", "3d49308d", "HEAD", "--", "tests"],
            text=True, cwd=ROOT,
        ).splitlines()
        return pytest.main(["-o", "addopts=", "-p", "no:cacheprovider", "-q", "--tb=short",
                            "--basetemp", str(REVIEW / f"focused-tmp-{expected}"),
                            *changed, "tests/test_lightgbm_optional.py", "tests/test_web_saver.py"])

    codes = []
    current = dt.datetime.now()
    for year in (2027, 2028):
        offset = int((current.replace(year=year) - current).total_seconds())
        env = os.environ | {
            "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(map(str, (REVIEW, ROOT / "src", ROOT))),
            "DYLD_INSERT_LIBRARIES": str(REVIEW / "review_clock.dylib"),
            "BTS_CONFIRM_CLOCK_OFFSET": str(offset),
        }
        codes.append(subprocess.run([sys.executable, __file__, str(year)], cwd=ROOT, env=env).returncode)
    print("Focused exit codes 2027/2028:", codes, flush=True)
    return int(any(codes))


if __name__ == "__main__":
    raise SystemExit(main())
