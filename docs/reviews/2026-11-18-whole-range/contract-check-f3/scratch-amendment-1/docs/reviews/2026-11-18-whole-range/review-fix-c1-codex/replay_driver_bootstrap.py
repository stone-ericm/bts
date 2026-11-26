"""Locate archived drivers after their move from docs to this review directory."""

import importlib
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import replay_drivers


def main():
    source = replay_drivers.HERE / "base/src" if sys.argv[1] == "base" else replay_drivers.ROOT / "src"
    sys.path[:0] = [str(source), str(replay_drivers.DRIVERS)]
    probe = importlib.import_module("probe")
    probe.ROOT = replay_drivers.ROOT
    probe.SLICES = probe.ROOT / "docs/reviews/2026-11-18-whole-range"
    original = runpy.run_path

    def run_path(path, **kwargs):
        if Path(path).name == "probe.py":
            return probe.main()
        return original(path, **kwargs)

    with patch.object(runpy, "run_path", run_path):
        replay_drivers.main()


if __name__ == "__main__":
    main()
