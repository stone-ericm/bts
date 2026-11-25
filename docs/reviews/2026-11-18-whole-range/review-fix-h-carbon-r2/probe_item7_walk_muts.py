"""Verify the 4 walk.py R4 scratch mutations with sroot/scripts loaded ahead of REPO/scripts."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

from probe_item7_scratch_detail import WALK_MUTATIONS


def main() -> None:
    with tempfile.TemporaryDirectory(dir=HERE, prefix="walk-mut-") as tmp:
        sroot = Path(tmp)
        shutil.copytree(REPO / "scripts", sroot / "scripts")
        shutil.copytree(REPO / "tests", sroot / "tests")
        # Symlink data and src so REPO is not needed ahead of sroot
        (sroot / "data").symlink_to(REPO / "data")
        (sroot / "src").symlink_to(REPO / "src")

        base = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-x", "tests/scripts/season_walk/test_season_walk.py"],
            cwd=sroot, capture_output=True, text=True, timeout=30,
        )
        assert base.returncode == 0, base.stdout + base.stderr
        print("OK: Unmutated walk test baseline passed in scratch tree")

        for name, rel_path, target, replacement, _ in WALK_MUTATIONS:
            fpath = sroot / rel_path
            orig = fpath.read_text()
            fpath.write_text(orig.replace(target, replacement, 1))
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "-x", "--tb=line", "tests/scripts/season_walk/test_season_walk.py"],
                    cwd=sroot, capture_output=True, text=True, timeout=30,
                )
                assert proc.returncode != 0, f"MISSED {name}!"
                line = next((ln for ln in proc.stdout.splitlines() if "FAILED" in ln), proc.stdout.splitlines()[-1])
                print(f"CAUGHT {name}: {line}")
            finally:
                fpath.write_text(orig)


if __name__ == "__main__":
    main()
