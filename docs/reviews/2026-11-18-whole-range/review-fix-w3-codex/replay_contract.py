"""Replay the archived W3 contract examples on the archived base."""

import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BASE = HERE / "base"
CHECK = BASE / ".codex-review" / "contract-check-w3"
GROUPS = {
    "r1-core": [
        ("replay.py", "original"), ("replay.py", "files"),
        ("replay.py", "replacement"), ("replay.py", "replacement", "--reuse"),
        ("replay.py", "warmup", "--reuse"), ("replay.py", "confirmations", "--reuse"),
        ("replay.py", "reload_cutoff", "--reuse"), ("replay.py", "final_cutoff", "--reuse"),
        ("replay.py", "dated_correction", "--reuse"), ("replay.py", "committed", "--reuse"),
        ("replay.py", "attempted", "--reuse"), ("replay.py", "lineup", "--reuse"),
        ("replay.py", "void", "--reuse"), ("replay.py", "marker_cutoff", "--reuse"),
        ("replay.py", "failed", "--reuse"), ("replay.py", "interrupted", "--reuse"),
    ],
    "r1-edges": [
        ("timeline.py", "skip"), ("timeline.py", "coalesce"), ("timeline.py", "void_pending"),
        ("timeline.py", "rollover", "2027-06-18"),
        ("boundary.py", "2027-06-18", "2026-12-31"), ("ordering.py",),
    ],
    "r2": [
        ("r2/probe.py", "cutoff"), ("r2/decision_cases.py",),
        ("r2/remaining.py", "calendar"), ("r2/remaining.py", "status"),
    ],
}


def main(group):
    environment = os.environ | {"PYTHONPATH": str(BASE / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    failures = 0
    for file, *args in GROUPS[group]:
        print(f"=== {file} {' '.join(args)} ===", flush=True)
        try:
            result = subprocess.run([sys.executable, str(CHECK / file), *args], cwd=BASE,
                                    env=environment, timeout=12, check=False)
        except subprocess.TimeoutExpired:
            print("Replay exceeded its 12-second subprocess bound", flush=True)
            failures += 1
        else:
            print(f"Replay exit {result.returncode}", flush=True)
            failures += bool(result.returncode)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
