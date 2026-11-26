#!/usr/bin/env python3
"""Rolling OCI audit launcher.

Processes a seeds file in batches. Each batch launches N boxes where N is the
CURRENT OCI standard-e5-core-count quota (divided by 8 OCPU/box), with exactly
one seed per box. When a batch finishes, re-queries quota for the next batch —
a quota bump granted mid-audit is automatically applied to subsequent batches.

Checkpoints via {out_base}/seeds_done.txt, so a crash + restart skips completed
seeds. Each batch's results land in {out_base}/batch_N/{label}-bN-M/phase1_seedS/.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit_rolling_oci.py \\
        --seeds-file scripts/audit_seeds_default48.txt \\
        --experiments scripts/audit_experiments.txt \\
        --out-base data/oci_results/audit_n48 \\
        --label-base bts-oci-audit

Why 1 seed per box instead of queuing multiple seeds per box: keeps the fleet
size fungible. If quota raises from 10 to 60 mid-audit, remaining seeds get
distributed to the bigger fleet. A multi-seed-per-box queue would lock seeds
to specific boxes and waste any quota increase that arrives mid-run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OCI_TENANCY = "ocid1.tenancy.oc1..aaaaaaaatstd3de4jhuyyqfyhqc65ftdswtul56e44ofd2vwyoodzoip7bnq"
OCPU_PER_BOX = 8


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def current_quota_max_boxes() -> int:
    """Query OCI for currently AVAILABLE e5 OCPUs across all 3 us-ashburn ADs,
    return as max 8-OCPU boxes. Uses resource-availability (quota minus current
    usage), not raw quota value, so we don't request capacity we can't actually
    launch. OCIProvider's AD fallback distributes boxes across ADs as needed."""
    total_available = 0
    for ad in ("ILYb:US-ASHBURN-AD-1", "ILYb:US-ASHBURN-AD-2", "ILYb:US-ASHBURN-AD-3"):
        try:
            result = subprocess.run(
                ["oci", "limits", "resource-availability", "get",
                 "--compartment-id", OCI_TENANCY,
                 "--service-name", "compute",
                 "--limit-name", "standard-e5-core-count",
                 "--availability-domain", ad,
                 "--output", "json"],
                capture_output=True, text=True, timeout=30,
            )
            data = json.loads(result.stdout).get("data", {})
            total_available += int(data.get("available", 0))
        except Exception as e:
            log(f"WARNING: availability query for {ad} failed: {e}")
    max_boxes = total_available // OCPU_PER_BOX
    if max_boxes == 0:
        log("WARNING: total availability across ADs is 0; defaulting to 10 boxes")
        return 10
    log(f"OCI availability: {total_available} free OCPUs across 3 ADs -> {max_boxes} boxes")
    return max_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-file", type=Path, required=True)
    ap.add_argument("--experiments", type=Path, required=True)
    ap.add_argument("--out-base", type=Path, required=True)
    ap.add_argument("--label-base", default="bts-oci-rolling")
    ap.add_argument("--max-per-batch", type=int, default=None,
                    help="Override auto-detected batch size. "
                         "If unset, each batch uses current OCI AD-1 quota.")
    ap.add_argument("--deadline-hours", type=float, default=14.0,
                    help="Deadline per BATCH (not total). One seed should finish "
                         "in ~10h; default 14h gives buffer.")
    ap.add_argument("--poll-interval", type=int, default=600)
    args = ap.parse_args()

    args.out_base.mkdir(parents=True, exist_ok=True)
    done_file = args.out_base / "seeds_done.txt"
    done_file.touch()

    all_seeds = [int(x) for x in args.seeds_file.read_text().replace(",", " ").split() if x.strip()]
    done_set = set(int(x) for x in done_file.read_text().split() if x.strip())
    pending = [s for s in all_seeds if s not in done_set]

    log(f"Seeds: total={len(all_seeds)} done={len(done_set)} pending={len(pending)}")
    if not pending:
        log("All seeds already marked done. Exiting.")
        return

    batch_num = 0
    while pending:
        batch_num += 1
        max_boxes = args.max_per_batch if args.max_per_batch else current_quota_max_boxes()
        if max_boxes == 0:
            log("ERROR: quota reports 0 boxes. Not launching empty batch.")
            sys.exit(1)

        batch_size = min(len(pending), max_boxes)
        batch_seeds = pending[:batch_size]
        pending = pending[batch_size:]

        log(f"=== BATCH {batch_num}: {batch_size} boxes, "
            f"seeds={batch_seeds[:5]}{'...' if len(batch_seeds) > 5 else ''} "
            f"(quota max={max_boxes}, remaining after this batch: {len(pending)}) ===")

        batch_seeds_file = args.out_base / f"batch_{batch_num:02d}_seeds.txt"
        batch_seeds_file.write_text("\n".join(str(s) for s in batch_seeds) + "\n")

        batch_out = args.out_base / f"batch_{batch_num:02d}"
        cmd = [
            "uv", "run", "python", "scripts/audit_driver.py",
            "--provider", "oci",
            "--boxes", str(batch_size),
            "--seeds-file", str(batch_seeds_file),
            "--experiments", str(args.experiments),
            "--label", f"{args.label_base}-b{batch_num:02d}",
            "--out", str(batch_out),
            "--deadline-hours", str(args.deadline_hours),
            "--poll-interval", str(args.poll_interval),
        ]
        env = os.environ.copy()
        env["UV_CACHE_DIR"] = "/tmp/uv-cache"

        batch_start = time.time()
        log(f"Launching batch {batch_num}: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env)
        batch_wall_h = (time.time() - batch_start) / 3600

        if result.returncode != 0:
            log(f"Batch {batch_num} FAILED (exit code {result.returncode}, "
                f"wall {batch_wall_h:.2f}h). NOT marking seeds done.")
            log(f"Pending seeds remain: {batch_seeds + pending}")
            sys.exit(result.returncode)

        with open(done_file, "a") as f:
            for s in batch_seeds:
                f.write(f"{s}\n")
        total_done = sum(1 for _ in open(done_file) if _.strip())
        log(f"Batch {batch_num} complete in {batch_wall_h:.2f}h. "
            f"Total done: {total_done}/{len(all_seeds)}")

    log("=== ALL BATCHES COMPLETE ===")


if __name__ == "__main__":
    main()
