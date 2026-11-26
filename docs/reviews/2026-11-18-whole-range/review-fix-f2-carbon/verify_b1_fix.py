"""Verify that test_f2_b1_stat_oserror_regression.py fails on 2e20fdfc and passes with the one-word fix `except (OSError, ValueError, RecursionError):` on an isolated scratch copy."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIG_SRC = REPO_ROOT / "src"
REG_TEST = Path(__file__).resolve().parent / "test_f2_b1_stat_oserror_regression.py"


def main() -> int:
    # 1. Run regression test against unmodified 2e20fdfc -> expect FAIL (rc != 0)
    unmod = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", str(REG_TEST)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    print("=== Unmodified 2e20fdfc ===")
    print(unmod.stdout)
    assert unmod.returncode != 0, "Expected RED on 2e20fdfc"

    # 2. Run regression test + test_wr_f2_saver_flag.py against scratch copy with smallest fix -> expect PASS (rc == 0)
    with tempfile.TemporaryDirectory(prefix="f2_b1_fix_") as tmp:
        scratch_src = Path(tmp) / "src"
        shutil.copytree(ORIG_SRC / "bts", scratch_src / "bts")
        (scratch_src / "sitecustomize.py").write_text(
            f"import sys\n"
            f"from pathlib import Path\n"
            f"orig = Path({str(ORIG_SRC)!r}).resolve()\n"
            f"sys.path = [p for p in sys.path if Path(p or '.').resolve() != orig]\n"
            f"sys.meta_path = [f for f in sys.meta_path if 'editable' not in type(f).__name__.lower()]\n"
            f"sys.path.insert(0, {str(scratch_src.resolve())!r})\n",
            encoding="utf-8",
        )
        cs_file = scratch_src / "bts" / "health" / "contest_state.py"
        text = cs_file.read_text(encoding="utf-8")
        old = "            except (ValueError, RecursionError):"
        new = "            except (OSError, ValueError, RecursionError):"
        assert old in text
        cs_file.write_text(text.replace(old, new, 1), encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(scratch_src.resolve())
        env["TZ"] = "America/New_York"

        fixed = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", str(REG_TEST), "tests/test_wr_f2_saver_flag.py"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )
        print("=== With smallest fix on scratch copy ===")
        print(fixed.stdout)
        assert fixed.returncode == 0, fixed.stdout + fixed.stderr

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
