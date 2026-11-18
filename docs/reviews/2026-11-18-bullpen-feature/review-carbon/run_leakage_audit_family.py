"""Stage 2023-2025 inputs (no 2026) and run scripts/leakage_audit.py on base and branch (item 6)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_TREE = HERE / "base_tree"
STAGE = HERE / "audit_stage"

PA_SRC = Path("/Users/eric/projects/bts/data/processed")
LOOKUP_SRC = Path("/Users/eric/bts-runs/inputs/probable_pitcher_lookup.2017-2025.json")


def stage_directory() -> str:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    proc_dir = STAGE / "data" / "processed"
    models_dir = STAGE / "data" / "models"
    proc_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    for year in (2023, 2024, 2025):
        assert year <= 2025
        (proc_dir / f"pa_{year}.parquet").symlink_to(PA_SRC / f"pa_{year}.parquet")

    (models_dir / "probable_pitcher_lookup.json").symlink_to(LOOKUP_SRC)

    # Verify zero 2026 files or raw feeds staged
    all_staged = [str(p.relative_to(STAGE)) for p in STAGE.rglob("*")]
    assert not any("2026" in s for s in all_staged), f"2026 path found in stage: {all_staged}"
    assert not (STAGE / "data" / "raw").exists()
    print(f"Staged files: {sorted(all_staged)}", flush=True)
    return hashlib.sha256(LOOKUP_SRC.read_bytes()).hexdigest()


def run_one(label: str, script_path: Path, src_path: Path, hashseed: str = "1") -> tuple[int, str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TZ"] = "America/New_York"
    env["PYTHONHASHSEED"] = hashseed
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=STAGE,
        env=env,
        capture_output=True,
        text=True,
    )
    print(f"\n=== [{label}] exit={proc.returncode} (hashseed={hashseed}) ===", flush=True)
    print(proc.stdout, end="", flush=True)
    if proc.stderr:
        print(f"--- stderr ---\n{proc.stderr}", flush=True)
    return proc.returncode, proc.stdout, proc.stderr


def prefix_before_audit5(stdout: str) -> str:
    marker = "\n--- AUDIT 5:"
    idx = stdout.find(marker)
    return stdout[:idx] if idx != -1 else stdout


def main():
    sha_before = stage_directory()

    rc_base_orig, out_base_orig, _ = run_one(
        "orig_base_script_on_base_src",
        BASE_TREE / "scripts" / "leakage_audit.py",
        BASE_TREE / "src",
        hashseed="1",
    )
    rc_canary_on_base, out_canary_on_base, _ = run_one(
        "branch_script_on_base_src",
        REPO / "scripts" / "leakage_audit.py",
        BASE_TREE / "src",
        hashseed="1",
    )
    rc_branch_s1, out_branch_s1, _ = run_one(
        "branch_script_on_branch_src_seed1",
        REPO / "scripts" / "leakage_audit.py",
        REPO / "src",
        hashseed="1",
    )
    rc_branch_s999, out_branch_s999, _ = run_one(
        "branch_script_on_branch_src_seed999",
        REPO / "scripts" / "leakage_audit.py",
        REPO / "src",
        hashseed="999",
    )

    sha_after = hashlib.sha256(LOOKUP_SRC.read_bytes()).hexdigest()
    assert sha_before == sha_after, "Lookup file SHA changed!"

    assert rc_base_orig == 0, f"Expected base orig script to exit 0, got {rc_base_orig}"
    assert rc_canary_on_base == 1, f"Expected branch script on base src to exit 1, got {rc_canary_on_base}"
    assert rc_branch_s1 == 0, f"Expected branch script on branch src to exit 0, got {rc_branch_s1}"
    assert rc_branch_s999 == 0, f"Expected branch script seed 999 to exit 0, got {rc_branch_s999}"

    # Check whether existing audit sections (Audits 1, 3, 4) changed between base and branch!
    p_base_orig = prefix_before_audit5(out_base_orig).strip()
    p_canary_base = prefix_before_audit5(out_canary_on_base).strip()
    p_branch_s1 = prefix_before_audit5(out_branch_s1).strip()

    assert p_base_orig == p_canary_base, "Existing audit output differed between base script and branch script on base src!"
    assert p_canary_base == p_branch_s1, (
        f"Existing audit sections changed between base src and branch src!\n"
        f"--- base ---\n{p_canary_base}\n--- branch ---\n{p_branch_s1}"
    )
    assert out_branch_s1 == out_branch_s999, "Leakage audit output was NOT deterministic across PYTHONHASHSEED!"

    print("\n=== SUMMARY ===", flush=True)
    print("1. Base script on base src: exit 0 (passed silently with the defect).", flush=True)
    print("2. Branch script on base src: exit 1 (Audit 5 FAILED on 'all starter' [nan] vs 0.4).", flush=True)
    print("3. Branch script on branch src: exit 0 (Audit 5 PASSED both variants at 0.4).", flush=True)
    print("4. Audits 1, 3, and 4 stdout: byte-for-byte IDENTICAL across base and branch.", flush=True)
    print("5. Determinism across PYTHONHASHSEED=1 vs 999: byte-for-byte IDENTICAL.", flush=True)


if __name__ == "__main__":
    main()
