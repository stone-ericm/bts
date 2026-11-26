"""Sequential pytest family runner v2: creates basetemp.parent before invoking pytest."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    (HERE / "pytest_tmp").mkdir(parents=True, exist_ok=True)
    import run_pytest_families as rpf

    if len(sys.argv) > 1 and sys.argv[1] == "child":
        rpf.child_pytest(sys.argv[2])
        return

    # Ensure child invocations also call this v2 entrypoint (or pytest_tmp already exists!)
    rpf.main()


if __name__ == "__main__":
    main()
