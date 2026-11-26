"""Item 8: Fast scratch mutation runner on an isolated copy of src/ (never touching tracked files)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from probe_5_mutations import HERE, MUTATIONS, REPO

SCRATCH_ROOT = HERE / "scratch_mut_pkg2"
SCRATCH_SRC = SCRATCH_ROOT / "src"


def check_mutant(cli_path: Path, original_cli: str, name: str, target: str, replacement: str, check_triples: bool) -> dict:
    assert target in original_cli, f"Target not found for {name}: {target}"
    cli_path.write_text(original_cli.replace(target, replacement, 1))
    env = {**os.environ, "PYTHONPATH": str(SCRATCH_SRC), "TZ": "America/New_York"}

    # Step 1: fast 80-test suite (excluding FIFO 20s sleep and 7,344-call triples test)
    proc_fast = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-x",
            "--tb=no",
            "-k",
            "not fifo and not ordered_pairs_and_triples",
            "tests/test_wr_f1_wrong_game.py",
            "tests/test_wr_f1_review.py",
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=25,
    )
    killed = proc_fast.returncode != 0
    summary = proc_fast.stdout.strip().splitlines()[-1] if proc_fast.stdout.strip() else ""

    # Step 2: if fast suite passed and check_triples is requested, also run ordered_pairs_and_triples
    if not killed and check_triples:
        proc_trip = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-x",
                "--tb=no",
                "-k",
                "ordered_pairs_and_triples",
                "tests/test_wr_f1_review.py",
            ],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        killed = proc_trip.returncode != 0
        summary += " | triples: " + (proc_trip.stdout.strip().splitlines()[-1] if proc_trip.stdout.strip() else "")

    # Step 3: if author's tests missed it, verify our probe_2c_rows_and_texts.py catches it!
    killed_by_probe = None
    if not killed:
        env2 = {**os.environ, "PYTHONPATH": f"{SCRATCH_SRC}:{HERE}", "TZ": "America/New_York"}
        p2 = subprocess.run(
            [sys.executable, str(HERE / "probe_2c_rows_and_texts.py")],
            cwd=str(REPO),
            env=env2,
            capture_output=True,
            text=True,
            timeout=30,
        )
        killed_by_probe = p2.returncode != 0

    cli_path.write_text(original_cli)
    return {
        "killed_by_author_tests": killed,
        "pytest_summary": summary,
        "killed_by_our_probe": killed_by_probe,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", choices=("killed_fast", "survivor_triples"), required=True)
    args = ap.parse_args()

    out_file = HERE / "log_5_mutations.json"
    existing = json.loads(out_file.read_text()) if out_file.is_file() else {}

    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)
    shutil.copytree(REPO / "src", SCRATCH_SRC)
    cli_path = SCRATCH_SRC / "bts" / "cli.py"
    original_cli = cli_path.read_text()

    try:
        if args.slice == "killed_fast":
            for name, target, replacement in MUTATIONS:
                existing[name] = check_mutant(cli_path, original_cli, name, target, replacement, check_triples=False)
        else:
            # Run ordered_pairs_and_triples on one surviving representative (extra_7_allow_bool_game_pk_isinstance)
            for name, target, replacement in MUTATIONS:
                if not existing.get(name, {}).get("killed_by_author_tests", True) and "extra_7" in name:
                    existing[name] = check_mutant(cli_path, original_cli, name, target, replacement, check_triples=True)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)

    out_file.write_text(json.dumps(existing, indent=2) + "\n")
    print(json.dumps(existing, indent=2))


if __name__ == "__main__":
    main()
