"""Run C31's two individual restorations against scratch copies of src."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


WORK = Path(__file__).resolve().parent
ROOT = WORK.parent.parent
CASES = (
    (
        "delivery-slate",
        '"Write only on the pick\'s ET slate date once prior entries settle or their windows close empty. "',
        'f"Write for the {warning_day.isoformat()} slate only on that ET date, with prior entries settled or absent. "',
        "tests/test_wr_e_texts.py::test_stale_warning_maximum_detail[delivery]",
        "Write for the 2027-07-15 slate only on that ET date",
    ),
    (
        "stored-substring",
        r're.sub(r"\bbelow the\b", "under the", " ".join(value.splitlines()))',
        '" ".join(value.splitlines()).replace("below the", "under the")',
        "tests/test_wr_e_texts.py::test_skip_stored_text_surfaces[longer-word]",
        "under themodel",
    ),
)


def run_one(name, head, replacement, test, expected):
    with TemporaryDirectory(prefix=f"scratch-{name}-", dir=WORK) as scratch:
        src = Path(scratch) / "src"
        shutil.copytree(ROOT / "src", src)
        scheduler = src / "bts/scheduler.py"
        original = scheduler.read_text()
        assert original.count(head) == 1
        scheduler.write_text(original.replace(head, replacement))
        env = os.environ | {"PYTHONPATH": os.pathsep.join((str(src), str(ROOT))),
                            "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1"}
        launcher = (
            "from pathlib import Path\n"
            "import sys\n"
            "import bts.scheduler\n"
            "assert Path(bts.scheduler.__file__).resolve() == Path(sys.argv[1]).resolve()\n"
            "import pytest\n"
            "sys.exit(pytest.main(['-p', 'no:cacheprovider', '-q', sys.argv[2]]))\n"
        )
        result = subprocess.run(["timeout", "15", "nice", "-n", "19", sys.executable, "-c",
                                 launcher, str(scheduler), test], cwd=ROOT, env=env, timeout=18,
                                capture_output=True, text=True)
        output = result.stdout + result.stderr
        (WORK / f"mutation-{name}.log").write_text(output)
        assert result.returncode == 1 and "1 failed" in output and expected in output, output
        print(f"C31 {name}: {test} failed on its expected rendered words (exit 1).")


if __name__ == "__main__":
    for case in CASES:
        run_one(*case)
