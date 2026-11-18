"""Serial pytest matrix for the archived base, scratch mutations and wall clocks."""

from datetime import datetime, timezone
import os
import re
import shutil
import subprocess
import sys

from probe_common import BASE, DATA, HERE, LOOKUP_PATH, ROOT


TEST = "tests/test_bullpen_same_day.py"
EXISTING = [
    "tests/test_lightgbm_optional.py",
    "tests/test_context_features.py",
    "tests/test_park_drag_feature.py",
    "tests/test_swing_features.py",
    "tests/experiment/test_features.py",
    "tests/model/test_predict_feature_coverage.py",
    "tests/scripts/test_determinism_gap_bound.py",
]
COMPUTE = "bts/features/compute.py"
ROLLING = "lambda x: x.rolling(30, min_periods=10).mean()"
BRANCH_START = '        team = "_pitcher_team_id"\n'
BASE_START = "        # Rolling 30-day average per team"
BLOCK_END = '        df.rename(columns={"_pitcher_team_id"'
RUN_PYTEST = """\
from datetime import datetime
import sys
import time
import pandas as pd
import pytest
from bts.features import compute

expected, source, *tests = sys.argv[1:]
assert compute.__file__ == source, (compute.__file__, source)
print('compute:', compute.__file__, flush=True)
clocks = (datetime.now().year, pd.Timestamp.now().year, time.localtime().tm_year)
print('realtime years (datetime / pandas / time):', clocks, flush=True)
assert not expected or clocks == (int(expected),) * 3
sys.exit(pytest.main(['-q', '--tb=short', '-rfEs', '-p', 'no:cacheprovider', *tests]))
"""


def replace(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def mutations():
    original = (ROOT / "src" / COMPUTE).read_text()
    base = (BASE / "src" / COMPUTE).read_text()
    yield "exact", replace(original, "allow_exact_matches=False", "allow_exact_matches=True")
    forward = replace(original, ROLLING, "lambda x: x.shift(1).rolling(30, min_periods=10).mean().ffill()")
    yield "shifted-ffill", replace(forward, "allow_exact_matches=False", "allow_exact_matches=True")
    no_align = replace(original, "known[team].astype(object)", "known[team]")
    yield "dtype", replace(no_align, "history.pop(team).astype(object)", "history.pop(team)")
    team_sort = original.replace('.sort_values("date", kind="stable")',
                                 '.sort_values(keys, kind="stable")')
    assert team_sort != original
    yield "team-first", team_sort
    yield "shifted-exact", (original[:original.index(BRANCH_START)]
                            + base[base.index(BASE_START):base.index(BLOCK_END)]
                            + original[original.index(BLOCK_END):])
    yield "min9", replace(original, ROLLING, "lambda x: x.rolling(30, min_periods=9).mean()")
    yield "window29", replace(original, ROLLING, "lambda x: x.rolling(29, min_periods=10).mean()")
    yield "null-keys", replace(original, "df[keys].dropna().drop_duplicates()",
                               "df[keys].drop_duplicates()")
    yield "always-missing", replace(original, ROLLING, "lambda x: np.nan")


def pytest_run(label, source, *, real=True, year=None, existing=False):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(source),
               BULLPEN_TEST_DATA=str(DATA if real else HERE / "absent-inputs"),
               BULLPEN_TEST_LOOKUP=str(LOOKUP_PATH if real else HERE / "absent-lookup.json"))
    if year:
        env.update(DYLD_INSERT_LIBRARIES=str(HERE / "review_clock.dylib"),
                   BULLPEN_REVIEW_EPOCH=str(int(datetime(year, 11, 18, 18, tzinfo=timezone.utc).timestamp())))
    pytest_args = [TEST, *EXISTING] if existing else [TEST]
    completed = subprocess.run(
        [sys.executable, "-u", "-c", RUN_PYTEST, str(year or ""), str(source / COMPUTE),
         *pytest_args], cwd=ROOT, env=env, capture_output=True, text=True,
    )
    text = completed.stdout + completed.stderr
    (HERE / f"pytest-{label}.txt").write_text(text)
    failed = re.findall(r"^FAILED (\S+)", text, flags=re.MULTILINE)
    summary = [line for line in text.splitlines()
               if re.match(r"^\d+ (passed|failed|skipped|xfailed)", line.strip())]
    print(label, "exit", completed.returncode, *summary, failed, flush=True)
    return completed.returncode, failed, text


def main():
    subprocess.run(["/usr/bin/clang", "-Wall", "-Wextra", "-dynamiclib",
                    str(HERE / "review_clock.c"), "-o", str(HERE / "review_clock.dylib")],
                   check=True)
    status, failed, text = pytest_run("base", BASE / "src")
    assert status == 1 and len(failed) == 4 and "2 passed" in text
    for name, mutated in mutations():
        dest = HERE / "mutants" / name / "src"
        shutil.copytree(ROOT / "src" / "bts", dest / "bts", dirs_exist_ok=True)
        (dest / COMPUTE).write_text(mutated)
        status, failed, text = pytest_run(name, dest)
        assert status == 1 and failed, (name, status, text)
    status, _, text = pytest_run("hidden", ROOT / "src", real=False)
    assert status == 0 and "5 passed, 1 skipped" in text
    statuses = [pytest_run("combined", ROOT / "src", existing=True)[0]]
    for year in (2027, 2028):
        statuses.append(pytest_run(str(year), ROOT / "src", year=year)[0])
    assert statuses == [0, 0, 0]


if __name__ == "__main__":
    main()
