"""Re-run the archived live call fixtures against this checkout and verify import."""
from pathlib import Path
import sys

import bts

ROOT = Path(__file__).resolve().parents[2]
assert Path(bts.__file__).resolve().is_relative_to(ROOT / "src")
sys.path.insert(0, str(ROOT / "docs/reviews/2026-11-18-whole-range/review-fix-g-codex"))
# Use the exact archived fixture by path to avoid loading this runner by name.
import runpy

if __name__ == "__main__":
    archive = Path(sys.path[0]) / "entry_points.py"
    probe = runpy.run_path(str(archive))
    probe["cli_and_shadow"]()
    probe["scheduler_fallbacks"]()
