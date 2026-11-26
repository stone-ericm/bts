"""Run the requested group in one process. Allow only test loopback sockets."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "docs/reviews/2026-11-18-whole-range/review-fix-e-codex/run_suite.py"
spec = importlib.util.spec_from_file_location("offline_suite", GUARD)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
suite.ROOT = ROOT
if __name__ == "__main__":
    sys.exit(suite.main())
