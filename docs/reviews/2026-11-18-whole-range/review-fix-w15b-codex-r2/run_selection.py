"""Run one pytest process under the requested wrapper with a verified import root."""
import sys
from pathlib import Path

import bts
import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
W15 = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "tests").glob("test_wr_w15*.py"))
FOCUSED = [
    *W15,
    "tests/test_probable_cache_review.py",
    "tests/test_leads_small_fixes.py",
    "tests/health/",
    "tests/test_lightgbm_optional.py",
    "tests/test_web_saver.py",
]


def main():
    source = Path(sys.argv[1]).resolve()
    assert Path(bts.__file__).resolve() == source / "bts/__init__.py", bts.__file__
    guard = sys.modules.get("sitecustomize")
    assert guard and Path(guard.__file__).resolve() == HERE.parent / "offline/sitecustomize.py"
    print(f"source: {bts.__file__}", flush=True)
    raise SystemExit(pytest.main(sys.argv[2:] or [
        "-q", "-o", "addopts=", "--tb=short", "-p", "no:cacheprovider",
        "--basetemp", str(HERE / "focused-tmp"), *FOCUSED,
    ]))


if __name__ == "__main__":
    main()
