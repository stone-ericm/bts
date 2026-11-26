#!/usr/bin/env python3
"""Attach to an already-provisioned audit run: poll, retrieve, teardown.

Does NOT spin up new boxes. Reads boxes.json from --out, re-derives the
seed queue deterministically from DEFAULT_SEEDS + distribute_seeds(), polls
until all boxes report done or the new deadline hits, then retrieves and
tears down. Use --run-kind profiles for raw backtest/profile audits so
retrieve looks for data/simulation_seed*/ artifacts instead of phase1_seed*/.

Used when the original driver's --deadline-hours is too short and needs to
be extended without losing the running VMs. Preferred recovery workflow:

    # 1. Let audit_driver reach its deadline. It retrieves complete boxes,
    #    tears down boxes whose retrieve is clean, and logs PRESERVED lines for
    #    boxes with incomplete or partial results.
    # 2. Re-attach only to preserved boxes with the original output directory,
    #    seed source, and run kind.
    python3 scripts/audit_attach.py --provider oci --run-kind profiles \\
        --out data/oci_results/phase_c_pooled_policy_profiles_2026-05-07 \\
        --seeds-file scripts/audit_seeds_extension_n100.txt \\
        --only-box bts-phasec-profile-oci-r2-4 --deadline-hours 24

Proactive extension is possible, but it is an operator-controlled exception:

    # 1. SIGKILL the original driver (SIGKILL skips its finally:teardown_all)
    kill -9 <original-driver-pid>
    # 2. Launch this script with a fresh, longer deadline (measured from NOW)
    nohup python3 scripts/audit_attach.py --provider hetzner \\
        --out data/hetzner_results/audit_full_48seed_v2 --seeds 48 \\
        --deadline-hours 192 > /tmp/audit_attach.log 2>&1 &

The seed distribution produced by `distribute_seeds(boxes, DEFAULT_SEEDS[:N])`
is deterministic given the same box order from boxes.json, so `retrieve_one`
will match what's actually on each box.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from audit_driver import (
    Box,
    DEFAULT_SEEDS,
    distribute_seeds,
    log,
    make_provider,
    poll,
    retrieve_profile_one,
    retrieve_one,
    teardown_all,
    teardown_retrieved,
)


def retrieve_fn_for_run_kind(run_kind: str):
    if run_kind == "screen":
        return retrieve_one
    if run_kind == "profiles":
        return retrieve_profile_one
    raise ValueError(f"unknown run kind: {run_kind}")


def select_boxes_for_attach(boxes: list[Box], only_boxes: list[str] | None) -> list[Box]:
    if not only_boxes:
        return boxes
    requested = set(only_boxes)
    by_name = {box.name: box for box in boxes}
    missing = sorted(requested - set(by_name))
    if missing:
        raise ValueError(f"--only-box did not match boxes.json entries: {missing}")
    return [box for box in boxes if box.name in requested]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-kind", choices=["screen", "profiles"], default="screen",
                    help="Original audit_driver run kind; controls retrieve layout")
    ap.add_argument("--provider", choices=["hetzner", "vultr", "oci"], required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output dir containing boxes.json (same --out used by audit_driver)")
    ap.add_argument("--seeds", type=int, default=None,
                    help="Original seed count passed to audit_driver (uses DEFAULT_SEEDS[:N])")
    ap.add_argument("--seeds-file", type=Path, default=None,
                    help="Path to seed list file (overrides --seeds; matches audit_driver --seeds-file)")
    ap.add_argument("--deadline-hours", type=float, required=True,
                    help="Hard cap measured from NOW, not from original start")
    ap.add_argument("--poll-interval", type=int, default=900)
    ap.add_argument("--no-teardown", action="store_true",
                    help="Skip VM teardown after retrieve (debug only)")
    ap.add_argument("--skip-alive-check", action="store_true",
                    help="Skip provider.is_active() preflight check")
    ap.add_argument("--only-box", action="append", default=None,
                    help="Restrict attach to this box name; repeat for multiple boxes")
    args = ap.parse_args()

    boxes_path = args.out / "boxes.json"
    raw = json.loads(boxes_path.read_text())
    all_boxes = [Box(id=b["id"], name=b["name"], ipv4=b["ipv4"], region=b.get("region", ""))
                 for b in raw]
    log(f"Attached to {len(all_boxes)} boxes from {boxes_path}")
    for b in all_boxes:
        log(f"  {b.name} id={b.id} ipv4={b.ipv4}")

    if args.seeds_file:
        raw_seeds = args.seeds_file.read_text().replace(",", " ").split()
        seeds = [int(x) for x in raw_seeds if x.strip()]
        log(f"Seeds: {len(seeds)} from {args.seeds_file}")
    elif args.seeds is not None:
        seeds = DEFAULT_SEEDS[: args.seeds]
        log(f"Seeds: {len(seeds)} from DEFAULT_SEEDS[:{args.seeds}]")
    else:
        raise SystemExit("must specify --seeds OR --seeds-file")
    queues = distribute_seeds(all_boxes, seeds)
    log("Seed distribution (for retrieve):")
    for nm, sl in queues.items():
        log(f"  {nm}: {sl}")
    boxes = select_boxes_for_attach(all_boxes, args.only_box)
    if len(boxes) != len(all_boxes):
        log(f"Restricting attach to {len(boxes)}/{len(all_boxes)} boxes: "
            f"{[box.name for box in boxes]}")
    retrieve_fn = retrieve_fn_for_run_kind(args.run_kind)
    log(f"Run kind: {args.run_kind}")

    provider = make_provider(args.provider)

    if not args.skip_alive_check:
        log("Preflight: verifying boxes still exist on provider...")
        missing = []
        for b in boxes:
            try:
                active, ip = provider.is_active(b.id)
                log(f"  {b.name} active={active} ip={ip or b.ipv4}")
                if not active:
                    missing.append(b.name)
            except Exception as e:
                log(f"  {b.name} is_active() ERROR: {e}")
                missing.append(b.name)
        if missing:
            log(f"WARNING: {len(missing)} boxes not reachable via provider API: {missing}")
            log("Continuing anyway — poll uses SSH directly, not provider API.")

    retrieve_results: dict[str, str] = {}
    exit_code = 0

    try:
        start = time.time()
        deadline_t = start + args.deadline_hours * 3600
        log(f"New deadline: {args.deadline_hours:.1f}h from now "
            f"(= {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(deadline_t))})")
        poll_num = 0
        while time.time() < deadline_t:
            poll_num += 1
            time.sleep(args.poll_interval)
            done_count, lines = poll(boxes)
            elapsed_h = (time.time() - start) / 3600
            log(f"=== POLL #{poll_num} ({elapsed_h:.2f}h, {done_count}/{len(boxes)} done) ===")
            for nm, is_done, s in lines:
                mark = "OK " if is_done else "..."
                log(f"  {mark} {nm}: {s[:140]}")
            if done_count == len(boxes):
                break

        log("=== RETRIEVE ===")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(boxes)) as ex:
            futures = [ex.submit(retrieve_fn, b, args.out, queues[b.name]) for b in boxes]
            for fut in concurrent.futures.as_completed(futures):
                nm, status, errs = fut.result()
                retrieve_results[nm] = status
                if errs:
                    log(f"  [{nm}] {status}  errs={errs[:1]}")
                else:
                    log(f"  [{nm}] {status}")
    finally:
        if args.no_teardown:
            log("--no-teardown set; leaving boxes alive")
        elif not boxes:
            pass
        else:
            selected, deleted = teardown_retrieved(provider, boxes, retrieve_results)
            preserved = len(boxes) - selected
            api_failed = selected - deleted
            log(f"=== TEARDOWN: selected={selected}/{len(boxes)} "
                f"deleted={deleted} preserved={preserved} api_failed={api_failed} ===")
            if preserved > 0:
                exit_code = 1

    log("=== AUDIT ATTACH DONE ===")
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
