"""Run one June-rollover test with one changed line in a temporary package."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


CASES = {
    "loader": (1355, "slate_date=date", "slate_date=None"),
    "saved": (1362, "slate_date=date", "slate_date=None"),
    "check": (2029, "slate_date=date", "slate_date=None"),
    "in_loop": (3351, "slate_date=date", "slate_date=None"),
    "final": (3504, "slate_date=date", "slate_date=None"),
    "missed_live": (328, "slate_date=date", "slate_date=None"),
    "void": (2633, "fromisoformat(date)", "fromisoformat(None)"),
    "late_initial": (2570, "fromisoformat(date)", "fromisoformat(None)"),
}
RUNNER = """
import pathlib
import sys

from bts import scheduler
import pytest

assert pathlib.Path(scheduler.__file__).resolve() == pathlib.Path(sys.argv[1]).resolve()
print(f"scheduler: {scheduler.__file__}", flush=True)
sys.exit(pytest.main(["-q", "--tb=short", "-p", "no:cacheprovider", sys.argv[2]]))
"""


def main(case):
    line_no, old, new = CASES[case]
    with TemporaryDirectory(prefix="wrk1-r2-date-") as tmp:
        scratch = Path(tmp)
        shutil.copytree("src/bts", scratch / "bts", ignore=shutil.ignore_patterns("__pycache__"))
        source = scratch / "bts/scheduler.py"
        lines = source.read_text().splitlines(keepends=True)
        original = lines[line_no - 1]
        assert original.count(old) == 1, (case, line_no, original)
        lines[line_no - 1] = original.replace(old, new)
        source.write_text("".join(lines))
        print(f"{case}, scheduler.py:{line_no}, {old} -> {new}", flush=True)
        node = f"tests/test_wr_k1_r2.py::test_scheduler_alert_dates[{case}-midnight]"
        result = subprocess.run(
            [sys.executable, "-c", RUNNER, str(source), node],
            env={**os.environ, "PYTHONPATH": os.pathsep.join((str(scratch), str(Path.cwd())))},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=9,
        )
        print(result.stdout, end="")
        assert result.returncode == 1, result.returncode
        assert "assert None == '2027-06-14'" in result.stdout
        assert "1 failed" in result.stdout


if __name__ == "__main__":
    main(sys.argv[1])
