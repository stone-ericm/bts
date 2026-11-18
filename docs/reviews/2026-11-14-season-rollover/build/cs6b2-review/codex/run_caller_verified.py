"""Verify caller pairs after normalizing the initial harness's path and clock differences."""

import sys
import tempfile
from pathlib import Path

from run_pytest_review import HERE, ROOT, archive, run


def main():
    cases = str((HERE / "test_caller_verified.py").relative_to(ROOT))
    with tempfile.TemporaryDirectory(prefix="caller-base-", dir=HERE) as cache:
        base = Path(cache) / "base"
        archive("59efa825", base)
        branch_ok = run("caller-branch-verified", [cases], verbose=True)
        base_ok = run("caller-base-verified", [cases], source=base, base=True, verbose=True)
    return 0 if branch_ok and base_ok else 1


if __name__ == "__main__":
    sys.exit(main())
