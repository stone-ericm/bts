#!/usr/bin/env python3
"""Validate byte-identical output across two cloud providers for the same seed.

Given two phase1_seed{N} directories produced by audit_driver.py on different
providers (e.g., Hetzner CPX51 and OCI E5.Flex, same seed), diff every output
file. Any drift means the candidate provider is NOT safe to pool into an audit.

Why this matters:
    Audits pool seeds across providers as if each seed is a deterministic
    input to the pipeline. If Hetzner and OCI produce different LGBM output
    for seed=42, that invariant is violated — pooled estimators become biased,
    confidence intervals become wrong. Validate BEFORE pooling seeds from a
    new provider.

Usage:
    python scripts/validate_provider_determinism.py \\
        --baseline data/hetzner_results/audit_full_48seed_v2/bts-audit-hetzner-1/phase1_seed42 \\
        --candidate data/oci_results/validation/bts-oci-validate-1/phase1_seed42

Exit codes:
    0 = all common files byte-identical; no file-set mismatch. Candidate OK.
    1 = byte drift in at least one file. Candidate NOT equivalent.
    2 = file-set mismatch (baseline-only or candidate-only files). Candidate NOT equivalent.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", type=Path, required=True,
                    help="Reference phase1_seedN directory (e.g. Hetzner output)")
    ap.add_argument("--candidate", type=Path, required=True,
                    help="Candidate phase1_seedN directory (new provider)")
    ap.add_argument("--glob", default="**/*",
                    help="Glob pattern for files to compare (default: **/*)")
    args = ap.parse_args()

    for label, path in [("baseline", args.baseline), ("candidate", args.candidate)]:
        if not path.is_dir():
            print(f"ERROR: {label} dir missing: {path}", file=sys.stderr)
            sys.exit(2)

    baseline_files = {
        p.relative_to(args.baseline): p
        for p in args.baseline.glob(args.glob)
        if p.is_file()
    }
    candidate_files = {
        p.relative_to(args.candidate): p
        for p in args.candidate.glob(args.glob)
        if p.is_file()
    }

    baseline_only = sorted(set(baseline_files) - set(candidate_files))
    candidate_only = sorted(set(candidate_files) - set(baseline_files))
    common = sorted(set(baseline_files) & set(candidate_files))

    if baseline_only:
        print(f"BASELINE-ONLY FILES ({len(baseline_only)}):")
        for p in baseline_only[:50]:
            print(f"  - {p}")
    if candidate_only:
        print(f"CANDIDATE-ONLY FILES ({len(candidate_only)}):")
        for p in candidate_only[:50]:
            print(f"  + {p}")

    match = 0
    mismatch: list[tuple[Path, str, str]] = []
    for rel in common:
        bh = hash_file(baseline_files[rel])
        ch = hash_file(candidate_files[rel])
        if bh == ch:
            match += 1
        else:
            mismatch.append((rel, bh[:12], ch[:12]))

    print()
    print("=== SUMMARY ===")
    print(f"Common files:       {len(common)}")
    print(f"Byte-identical:     {match}")
    print(f"Drift:              {len(mismatch)}")
    print(f"Baseline-only:      {len(baseline_only)}")
    print(f"Candidate-only:     {len(candidate_only)}")

    if mismatch:
        print()
        print("DRIFT DETAIL (up to 50):")
        for rel, bh, ch in mismatch[:50]:
            print(f"  - {rel}  baseline={bh}  candidate={ch}")

    if baseline_only or candidate_only:
        print()
        print("FILE-SET MISMATCH — runs produced different output file sets.")
        print("Candidate is NOT determinism-equivalent.")
        sys.exit(2)
    if mismatch:
        print()
        print("BYTE DRIFT DETECTED. Candidate is NOT determinism-equivalent.")
        print("Do not pool candidate seeds into an audit until resolved.")
        sys.exit(1)
    print()
    print("ALL FILES BYTE-IDENTICAL. Candidate is determinism-equivalent.")
    sys.exit(0)


if __name__ == "__main__":
    main()
