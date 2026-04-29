#!/usr/bin/env python3
"""Daily multi-seed pooled training orchestrator.

Provisions N fresh cloud boxes (Phase 1: Vultr; Phase 2: Hetzner), trains one
seed's blend models on each, rsyncs the trained models back to the local
output directory, tears down the boxes, and writes a per-day status JSON.

Per-box command (each in parallel):
  BTS_LGBM_RANDOM_STATE=<seed> BTS_LGBM_DETERMINISTIC=1 \\
    uv run bts preview --date <DATE> \\
                       --models-dir /tmp/seed_<N>/

Output layout (under --out):
  models_pooled/seed_<N>/blend_<DATE>.pkl × N seeds
  <DATE>_status.json — per-seed status, hashes, train_time_s, n_complete

Pattern: borrows provisioning + provider abstraction from audit_driver.py.
Differs in that each box does ONE seed of training (not a queue of seeds),
and the artifact retrieved is the .pkl model files (not Phase 1 diffs).

Usage:
  uv run python scripts/pooled_train_daily.py \\
      --provider vultr \\
      --plan voc-c-16c-32gb-300s-amd \\
      --region fra \\
      --seed-set canonical-n10 \\
      --date 2026-04-30 \\
      --out data/models_pooled/

  # When Hetzner limit increase lands:
  uv run python scripts/pooled_train_daily.py \\
      --provider hetzner \\
      --seed-set canonical-n10 \\
      --date 2026-04-30 \\
      --out data/models_pooled/
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

# Reuse audit_driver's cloud-API + provisioning machinery.
sys.path.insert(0, str(Path(__file__).parent))
from audit_driver import (  # noqa: E402
    Box, Provider, HetznerProvider, VultrProvider, OCIProvider,
    log, make_provider, spinup, wait_cloud_init, ssh_run, teardown_all,
    LOCAL_BTS,
)


def load_seed_set(name: str) -> list[int]:
    """Load seeds from data/seed_sets/<name>.json (canonical-n10 etc)."""
    path = LOCAL_BTS / "data" / "seed_sets" / f"{name}.json"
    if not path.exists():
        available = sorted(p.stem for p in (LOCAL_BTS / "data" / "seed_sets").glob("*.json"))
        raise FileNotFoundError(
            f"Seed set '{name}' not found at {path}. Available: {available}"
        )
    return [int(s) for s in json.loads(path.read_text())["seeds"]]


def provision_one(box: Box) -> tuple[str, str]:
    """Re-implements audit_driver.provision_one — wait_cloud_init + rsync code/data.

    Same pattern; importing the original reaches into a function that's tightly
    bound to audit_driver's globals. This is short enough to copy.
    """
    name = box.name
    if not wait_cloud_init(box):
        return (name, "cloud_init_timeout")

    log(f"  [{name}] rsync src/data/models...")
    rsync = (
        f"rsync -az --delete --exclude='__pycache__' --exclude='.venv' "
        f"--exclude='data/raw' --exclude='data/picks*' --exclude='data/models_*' "
        f"--exclude='data/borderline*' --exclude='data/lambdarank*' "
        f"--exclude='data/screen_postcutover' --exclude='data/simulation_mdp' "
        f"--exclude='data/external/savant' "
        f"{LOCAL_BTS}/ root@{box.ipv4}:/root/projects/bts/"
    )
    r = subprocess.run(rsync, shell=True, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return (name, f"rsync_failed: {r.stderr[-500:]}")

    log(f"  [{name}] uv sync...")
    setup = ssh_run(box.ipv4, (
        "set -eux; cd /root/projects/bts && "
        "apt-get install -y libomp-dev > /dev/null 2>&1 || true; "
        "UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model"
    ), timeout=600)
    if setup.returncode != 0:
        return (name, f"setup_failed: {setup.stderr[-500:]}")
    return (name, "ok")


def launch_train(box: Box, seed: int, date: str) -> tuple[str, int, str]:
    """Kick off `bts preview` in the background on the box and return PID.

    Models land in /root/projects/bts/data/models_pooled/seed_<N>/.
    """
    cmd = (
        f"set -eux; "
        f"cd /root/projects/bts && "
        f"mkdir -p data/models_pooled/seed_{seed} && "
        f"BTS_LGBM_RANDOM_STATE={seed} BTS_LGBM_DETERMINISTIC=1 "
        f"UV_CACHE_DIR=/tmp/uv-cache "
        f"nohup uv run bts preview --date {date} "
        f"  --models-dir data/models_pooled/seed_{seed}/ "
        f"  > /tmp/train_{seed}.log 2>&1 & echo PID=$!; disown"
    )
    r = ssh_run(box.ipv4, cmd, timeout=60)
    pid = -1
    for line in r.stdout.splitlines():
        if line.startswith("PID="):
            pid = int(line.split("=", 1)[1].strip())
    return (box.name, pid, r.stdout.strip() + "|" + r.stderr.strip())


def is_done(box: Box, seed: int, date: str) -> tuple[bool, str]:
    """True iff blend_<DATE>.pkl exists and no python is running for this train."""
    cmd = (
        f"set +e; "
        f"test -f /root/projects/bts/data/models_pooled/seed_{seed}/blend_{date}.pkl && "
        f"  ! pgrep -f 'bts preview.*seed_{seed}' >/dev/null && echo DONE || echo PENDING; "
        f"tail -3 /tmp/train_{seed}.log 2>/dev/null"
    )
    r = ssh_run(box.ipv4, cmd, timeout=30)
    out = r.stdout.strip()
    return ("DONE" in out, out[-300:])


def retrieve_models(box: Box, seed: int, date: str, out_dir: Path) -> tuple[str, str, dict]:
    """rsync the trained .pkl(s) back. Returns (status, msg, sha256-by-file)."""
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    rsync = (
        f"rsync -az "
        f"root@{box.ipv4}:/root/projects/bts/data/models_pooled/seed_{seed}/ "
        f"{seed_dir}/"
    )
    r = subprocess.run(rsync, shell=True, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        return (box.name, f"rsync_failed: {r.stderr[-300:]}", {})

    hashes = {}
    for pkl in seed_dir.glob(f"blend_{date}.pkl"):
        h = hashlib.sha256(pkl.read_bytes()).hexdigest()
        hashes[pkl.name] = h
    if not hashes:
        return (box.name, "no_blend_pkl_in_output", {})
    return (box.name, "ok", hashes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["hetzner", "vultr", "oci"], required=True)
    ap.add_argument("--seed-set", default="canonical-n10",
                    help="Seed manifest from data/seed_sets/<name>.json")
    ap.add_argument("--date", default=None,
                    help="Date to train models for (YYYY-MM-DD). Defaults to tomorrow.")
    ap.add_argument("--out", type=Path, default=LOCAL_BTS / "data" / "models_pooled",
                    help="Output directory for retrieved models")
    ap.add_argument("--poll-interval", type=int, default=300,
                    help="Seconds between completion polls")
    ap.add_argument("--deadline-minutes", type=int, default=180,
                    help="Hard cap on training wallclock before forced teardown")
    args = ap.parse_args()

    seeds = load_seed_set(args.seed_set)
    date = args.date or (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    args.out.mkdir(parents=True, exist_ok=True)

    log(f"Pooled-train: {len(seeds)} seeds for date {date} on {args.provider}")
    log(f"  Seeds: {seeds}")
    log(f"  Output: {args.out}")

    provider = make_provider(args.provider)
    boxes: list[Box] = []
    status: dict[int, dict] = {seed: {"status": "pending"} for seed in seeds}

    try:
        boxes = spinup(provider, len(seeds), f"bts-pooled-train-{date}")
        if len(boxes) < len(seeds):
            log(f"WARN: got {len(boxes)}/{len(seeds)} boxes — proceeding with reduced set")

        seed_for_box: dict[str, int] = {b.name: seeds[i] for i, b in enumerate(boxes)}

        # Provision in parallel (rsync code + uv sync)
        log(f"Provisioning {len(boxes)} boxes (parallelism=3)...")
        ready_boxes: list[Box] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futures = {ex.submit(provision_one, b): b for b in boxes}
            for f in concurrent.futures.as_completed(futures):
                b = futures[f]
                name, msg = f.result()
                if msg == "ok":
                    ready_boxes.append(b)
                else:
                    seed = seed_for_box[b.name]
                    status[seed] = {"status": f"provision_failed: {msg}"}

        log(f"Ready: {len(ready_boxes)}/{len(boxes)} boxes")

        # Launch training (one seed per box)
        for b in ready_boxes:
            seed = seed_for_box[b.name]
            log(f"  [{b.name}] launch seed={seed}")
            name, pid, msg = launch_train(b, seed, date)
            if pid > 0:
                status[seed].update({"status": "running", "pid": pid, "box": b.name})
            else:
                status[seed]["status"] = f"launch_failed: {msg}"

        # Poll until all done or deadline reached
        import time
        start = time.time()
        deadline = start + args.deadline_minutes * 60
        while time.time() < deadline:
            time.sleep(args.poll_interval)
            n_done = 0
            for b in ready_boxes:
                seed = seed_for_box[b.name]
                if status[seed]["status"] not in ("running",):
                    n_done += 1
                    continue
                done, last_log = is_done(b, seed, date)
                if done:
                    status[seed]["status"] = "done"
                    n_done += 1
                else:
                    status[seed]["last_log_tail"] = last_log
            elapsed = (time.time() - start) / 60
            log(f"  poll: {n_done}/{len(ready_boxes)} done ({elapsed:.1f}min elapsed)")
            if n_done >= len(ready_boxes):
                break

        # Retrieve models
        log("Retrieving models...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futures = {}
            for b in ready_boxes:
                seed = seed_for_box[b.name]
                if status[seed]["status"] == "done":
                    futures[ex.submit(retrieve_models, b, seed, date, args.out)] = (b, seed)
            for f in concurrent.futures.as_completed(futures):
                b, seed = futures[f]
                name, msg, hashes = f.result()
                if msg == "ok":
                    status[seed].update({"status": "complete", "hashes": hashes})
                else:
                    status[seed]["status"] = f"retrieve_failed: {msg}"

    finally:
        if boxes:
            log("Teardown all boxes...")
            teardown_all(provider, boxes)

    # Write status JSON
    n_complete = sum(1 for s in status.values() if s["status"] == "complete")
    status_path = args.out / f"{date}_status.json"
    status_path.write_text(json.dumps({
        "date": date,
        "provider": args.provider,
        "seed_set": args.seed_set,
        "n_seeds": len(seeds),
        "n_complete": n_complete,
        "per_seed": status,
    }, indent=2, default=str))
    log(f"\nDone: {n_complete}/{len(seeds)} seeds complete. Status → {status_path}")

    # Exit non-zero if fewer than 80% complete (matches health check WARN threshold)
    return 0 if n_complete >= int(0.8 * len(seeds)) else 1


if __name__ == "__main__":
    sys.exit(main())
