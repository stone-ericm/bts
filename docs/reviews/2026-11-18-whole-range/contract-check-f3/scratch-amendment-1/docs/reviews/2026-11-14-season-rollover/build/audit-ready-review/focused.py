import datetime as dt
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from review_runner import PYTHON, REVIEW, ROOT, environment


def main():
    if len(sys.argv) > 1:
        import pytest
        import sitecustomize

        sitecustomize.verify()
        year = int(sys.argv[1])
        assert dt.date.today().year == dt.datetime.now().year == time.localtime().tm_year == year
        assert dt.datetime.now(ZoneInfo('America/New_York')).year == year
        print(f'Process date {dt.datetime.now().isoformat()}', flush=True)
        return pytest.main(['-p', 'no:cacheprovider', '-q', '--tb=short', *sys.argv[2:]])

    tests = subprocess.check_output(['git', 'diff', '--name-only', 'be217307', 'HEAD', '--', 'tests'],
                                    cwd=ROOT, text=True).splitlines()
    tests += ['tests/test_lightgbm_optional.py', 'tests/test_web_saver.py']
    now = dt.datetime.now(dt.timezone.utc)
    results = []
    for year in (2027, 2028):
        env = environment(ROOT) | {
            'DYLD_INSERT_LIBRARIES': str(REVIEW / 'wall_clock.dylib'),
            'AUDIT_READY_CLOCK_OFFSET': str(int((now.replace(year=year) - now).total_seconds())),
        }
        results.append(subprocess.run([str(PYTHON), str(Path(__file__).resolve()), str(year), *tests],
                                      env=env, cwd=ROOT).returncode)
    print(f'Exit codes for 2027 and 2028: {results}')
    return int(any(results))


if __name__ == '__main__':
    raise SystemExit(main())
