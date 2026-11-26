"""Run the eleven requested test files in one invocation on the round-two head."""

from importlib.util import find_spec
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run_focused import TESTS

assert Path(find_spec("bts").origin).resolve().is_relative_to(HERE.parents[2] / "src")
assert find_spec("sitecustomize").origin.endswith("offline_python/sitecustomize.py")
raise SystemExit(pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short",
                              *(f"tests/{name}" for name in TESTS)]))
