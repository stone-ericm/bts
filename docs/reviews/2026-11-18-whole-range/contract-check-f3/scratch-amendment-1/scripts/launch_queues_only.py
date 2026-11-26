#!/usr/bin/env python3
"""One-shot recovery: launch audit seed queues on Vultr boxes.

Use case: the original audit_driver.py died between provision_one and
launch_box_queue (SIGKILL, OOM, or mid-batch crash) — boxes are alive,
most are fully provisioned, but no seed queues are running. This script:

  1. Scans all boxes for (venv installed) AND (no existing audit.log).
  2. For any box missing the venv, runs provision_one to finish it.
  3. Distributes seeds across all boxes (distribute_seeds round-robin).
  4. Calls launch_box_queue in parallel to start the seed bash loop.

Idempotent: skips boxes where an audit.log already exists (queue running).
Does NOT spin up boxes or tear them down — that stays with audit_driver.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_driver import (  # noqa: E402
    Box,
    distribute_seeds,
    launch_box_queue,
    log,
    provision_one,
    ssh_run,
)

OUT = Path("data/vultr_results/audit_ext_n100_v4")
SEEDS_FILE = Path("scripts/audit_seeds_extension_n100.txt")
EXPERIMENTS_FILE = Path("scripts/audit_experiments_v2.txt")


def check_ready(box: Box) -> tuple[Box, str]:
    r = ssh_run(
        box.ipv4,
        "if [ -f /root/audit.log ]; then echo QUEUED; "
        "elif [ -x /root/projects/bts/.venv/bin/bts ]; then echo READY; "
        "else echo NOT_PROVISIONED; fi",
        timeout=15,
    )
    return box, r.stdout.strip()


def main() -> None:
    boxes_raw = json.loads((OUT / "boxes.json").read_text())
    boxes = [
        Box(id=b["id"], name=b["name"], ipv4=b["ipv4"], region=b.get("region", ""))
        for b in boxes_raw
    ]
    seeds = [
        int(x) for x in SEEDS_FILE.read_text().replace(",", " ").split() if x.strip()
    ]
    experiments = EXPERIMENTS_FILE.read_text().strip()
    log(
        f"{len(boxes)} boxes, {len(seeds)} seeds, "
        f"{len(experiments.split(','))} experiments"
    )

    # Phase 1: readiness scan
    log("Scanning box readiness...")
    with cf.ThreadPoolExecutor(max_workers=len(boxes)) as ex:
        checks = list(ex.map(check_ready, boxes))
    status_counts: dict[str, int] = {}
    for _, s in checks:
        status_counts[s] = status_counts.get(s, 0) + 1
    log(f"  readiness: {status_counts}")

    queued = [b for b, s in checks if s == "QUEUED"]
    ready = [b for b, s in checks if s == "READY"]
    not_provisioned = [b for b, s in checks if s == "NOT_PROVISIONED"]

    if queued:
        log(f"  SKIPPING {len(queued)} boxes with existing audit.log: "
            f"{[b.name for b in queued]}")

    # Phase 2: complete provisioning for not-provisioned boxes
    if not_provisioned:
        log(f"Completing provisioning for {len(not_provisioned)} boxes: "
            f"{[b.name for b in not_provisioned]}")
        with cf.ThreadPoolExecutor(max_workers=3) as ex:
            futures = {ex.submit(provision_one, b): b for b in not_provisioned}
            for fut in cf.as_completed(futures):
                b = futures[fut]
                try:
                    name, status = fut.result()
                    log(f"  [{name}] provision: {status}")
                except Exception as e:
                    log(f"  [{b.name}] provision EXCEPTION: {type(e).__name__}: {e}")

        # Re-scan to confirm readiness
        log("Re-scanning after provisioning...")
        with cf.ThreadPoolExecutor(max_workers=len(boxes)) as ex:
            checks = list(ex.map(check_ready, boxes))
        ready = [b for b, s in checks if s == "READY"]
        queued = [b for b, s in checks if s == "QUEUED"]
        still_not = [b for b, s in checks if s == "NOT_PROVISIONED"]
        log(f"  after re-provision: READY={len(ready)}, QUEUED={len(queued)}, "
            f"NOT_PROVISIONED={len(still_not)}")
        if still_not:
            log(f"  ABORT: {len(still_not)} boxes still not provisioned: "
                f"{[b.name for b in still_not]}")
            sys.exit(1)

    # Phase 3: distribute + launch
    launch_targets = ready  # only boxes that are READY (not QUEUED)
    if not launch_targets:
        log("No boxes to launch — all already queued or unable to provision")
        return

    # IMPORTANT: distribute_seeds uses the full `boxes` list for canonical
    # seed distribution. audit_attach.py will call distribute_seeds(boxes, seeds)
    # when retrieving, so we must match that. Launch on the full list, skip any
    # that are already queued up above.
    queues = distribute_seeds(boxes, seeds)
    log("Seed distribution (canonical, matches what audit_attach will expect):")
    for nm, sl in queues.items():
        marker = " (SKIP: already queued)" if nm in {b.name for b in queued} else ""
        log(f"  {nm}: {sl}{marker}")

    log(f"Launching queues on {len(launch_targets)} boxes...")
    with cf.ThreadPoolExecutor(max_workers=len(launch_targets)) as ex:
        futures = {
            ex.submit(launch_box_queue, b, queues[b.name], experiments): b
            for b in launch_targets
        }
        for fut in cf.as_completed(futures):
            b = futures[fut]
            try:
                name, rc, out = fut.result()
                log(f"  [{name}] rc={rc}  {out}")
            except Exception as e:
                log(f"  [{b.name}] launch EXCEPTION: {type(e).__name__}: {e}")

    log("DONE. Next: `python3 scripts/audit_attach.py --provider vultr "
        "--out data/vultr_results/audit_ext_n100_v4 "
        "--seeds-file scripts/audit_seeds_extension_n100.txt "
        "--deadline-hours 72` to poll + retrieve + teardown.")


if __name__ == "__main__":
    main()
