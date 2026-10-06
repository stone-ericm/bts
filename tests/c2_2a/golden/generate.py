"""C2 step 2a §5.1: write the golden outputs of the deployed baseline.

Run from a worktree at the baseline commit (f882411) with this directory copied in unchanged:

    TZ=America/New_York OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache \\
        uv run python -m tests.c2_2a.golden.generate --out tests/c2_2a/golden/data

It runs every scenario in `scenarios.SCENARIOS` against the `bts` package of the checkout it runs in, and writes one
`<scenario>.json` observation per scenario plus `manifest.json`: the commit, the sha256 of each harness file (so the
candidate test can prove it runs the same harness), the command, the environment, the package versions and the sha256
of every golden file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

HARNESS_FILES = ("generate.py", "scenarios.py", "world.py", "fakes.py", "__init__.py")
PACKAGES = ("lightgbm", "numpy", "pandas", "pyarrow", "scikit-learn")


def harness_hashes(here: Path) -> dict:
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in HARNESS_FILES}


def environment() -> dict:
    return {"TZ": os.environ.get("TZ"), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "python": platform.python_version(), "packages": {p: version(p) for p in PACKAGES}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated scenario names (for debugging; not a golden run)")
    args = ap.parse_args(argv)
    if os.environ.get("TZ") != "America/New_York" or os.environ.get("OMP_NUM_THREADS") != "1":
        print("refusing: set TZ=America/New_York and OMP_NUM_THREADS=1", file=sys.stderr)
        return 2
    repo = Path.cwd()
    here = Path(__file__).resolve().parent
    from tests.c2_2a.golden import scenarios as S
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.only.split(",") if args.only else list(S.SCENARIOS)
    files = {}
    with tempfile.TemporaryDirectory(prefix="c2-2a-golden-") as tmp:
        for name in names:
            print(f"[golden] {name}", file=sys.stderr, flush=True)
            obs = S.run(name, repo, Path(tmp) / name)
            path = out / f"{name}.json"
            path.write_text(json.dumps(obs, indent=1, sort_keys=True) + "\n")
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    manifest = {"schema": "c2_2a_golden_manifest_v1", "commit": commit, "tracked_tree_clean": not dirty,
                "harness_sha256": harness_hashes(here), "command": " ".join([sys.executable, "-m",
                                                                             "tests.c2_2a.golden.generate"] +
                                                                            (argv or sys.argv[1:])),
                "environment": environment(), "scenarios": names, "files": files, "partial": bool(args.only)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"commit": commit, "scenarios": len(names)}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
