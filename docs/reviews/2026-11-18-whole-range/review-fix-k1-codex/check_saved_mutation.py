"""Prove the overnight saved-pick receipt check fails for the surviving change."""

import os
import subprocess
import sys
from pathlib import Path

from probe_mutations import MUTATIONS, ROOT


def main():
    scratch = Path(sys.argv[1])
    filename, old, new = MUTATIONS["saved-wall-date"]
    target = scratch / "src" / "bts" / filename
    base = target.read_text()
    assert base == (ROOT / "src/bts" / filename).read_text() and base.count(old) == 1
    target.write_text(base.replace(old, new))
    try:
        env = os.environ | {"PYTHONPATH": os.pathsep.join(map(str, (scratch / "src", ROOT, Path(__file__).parent)))}
        result = subprocess.run(
            [sys.executable, "-c",
             "from datetime import datetime; from probe_callers import ET, replay; "
             "replay('saved', '2027-06-14', datetime(2027, 6, 15, tzinfo=ET), True)"],
            capture_output=True, text=True, env=env, cwd=ROOT, timeout=10,
        )
        print(result.stdout + result.stderr)
        assert result.returncode != 0 and "assert chosen ==" in result.stderr
        assert str(scratch / "src") in env["PYTHONPATH"]
        print("The saved-pick route selected wall ET in the scratch copy.")
    finally:
        target.write_text(base)


if __name__ == "__main__":
    main()
