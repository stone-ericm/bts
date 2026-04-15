#!/usr/bin/env python3
"""Launch baseline-only × N seeds on Hetzner to populate Option 7's pooled bins.

This is the compute-side companion to bts.simulate.pooled_policy. For each
seed, it runs scripts/rebuild_policy.py with BTS_LGBM_RANDOM_STATE=$SEED,
captures the resulting data/simulation/backtest_*.parquet profiles, and
retrieves them to local data/hetzner_results/pooled_bins_run/.

Context: memory/project_bts_2026_04_15_audit_state.md (Option 7 plan).

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/baseline_profiles_hetzner.py \\
        --boxes 4 --seeds 16

Output layout:
    data/hetzner_results/pooled_bins_run/
        bts-pooled-1/
            audit.log
            simulation_seed1/
                backtest_2021.parquet ... backtest_2025.parquet
            simulation_seed7/
                ...
        bts-pooled-2/
            ...

After retrieval, feed the simulation_seed* dirs into scripts/rebuild_mdp_policy_pooled.py:

    uv run python scripts/rebuild_mdp_policy_pooled.py \\
        --seed-dirs data/hetzner_results/pooled_bins_run/*/simulation_seed* \\
        --out data/models/mdp_policy_pooled_v1.npz
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from audit_driver import (  # type: ignore  # local helper reuse
    Box,
    ENV,
    HetznerProvider,
    LOCAL_BTS,
    SSH_OPTS,
    distribute_seeds,
    log,
    provision_one,
    spinup,
    ssh_run,
    teardown_all,
    wait_cloud_init,
)


def provision_with_scripts(box: Box) -> tuple[str, str]:
    """Wrap audit_driver.provision_one to ALSO rsync scripts/.

    The upstream provisioner rsyncs src/, data/processed/, data/models/,
    pyproject.toml, and uv.lock — but not scripts/, because the experiment
    runner path invokes things through the bts CLI entry point. For this
    driver we need rebuild_policy.py and its helpers (arch_eval.py,
    phase7_same_game_double.py) on the box.
    """
    name, status = provision_one(box)
    if status != "ready":
        return (name, status)
    ssh_arg = "ssh " + " ".join(SSH_OPTS)
    r = subprocess.run(
        ["rsync", "-az", "-e", ssh_arg,
         f"{LOCAL_BTS}/scripts/",
         f"root@{box.ipv4}:/root/projects/bts/scripts/"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        return (name, f"scripts_rsync_failed: {r.stderr[:200]}")
    return (name, "ready+scripts")


# Non-overlapping with audit_driver.DEFAULT_SEEDS; chosen deterministically.
DEFAULT_SEEDS = [
    1, 7, 42, 137, 1024, 8192, 65536, 524288, 999983,
    2041769, 2273360, 3167584, 3911096, 6426836, 6762547, 6992150,
]


def launch_baseline_queue(box: Box, seeds: list[int]) -> tuple[str, int, str]:
    """Shell-script queue that runs rebuild_policy.py once per seed and
    captures data/simulation into a per-seed directory.

    Uses absolute paths to avoid any cwd drift. Emits per-seed progress to
    /root/baseline.log and a final /root/baseline.done marker on completion.
    """
    ip = box.ipv4
    name = box.name
    seed_list_str = " ".join(str(s) for s in seeds)
    BTS = "/root/projects/bts"

    cmd = f"""
rm -f /root/baseline.log /root/baseline.done
nohup bash -c '
set +e
for SEED in {seed_list_str}; do
  echo "=== seed=$SEED starting at $(date) ===" >> /root/baseline.log
  rm -rf {BTS}/data/simulation
  mkdir -p {BTS}/data/simulation
  cd {BTS}
  BTS_LGBM_RANDOM_STATE=$SEED {ENV} \\
    uv run python scripts/rebuild_policy.py >> /root/baseline.log 2>&1
  rc=$?
  mv {BTS}/data/simulation {BTS}/data/simulation_seed$SEED
  echo "=== seed=$SEED done at $(date) rc=$rc ===" >> /root/baseline.log
done
echo "queue done at $(date)" > /root/baseline.done
' > /dev/null 2>&1 &
disown
echo "launched seeds={seed_list_str}"
"""
    r = ssh_run(ip, cmd, timeout=30)
    return (name, r.returncode, r.stdout.strip())


def poll_baseline(boxes: list[Box]) -> tuple[int, list[tuple]]:
    done_count = 0
    lines = []
    for box in boxes:
        q = r"""
if [ -f /root/baseline.done ]; then
  echo DONE
  cat /root/baseline.done
else
  grep -c '=== seed=.* done' /root/baseline.log 2>/dev/null || echo 0
  tail -1 /root/baseline.log 2>/dev/null | head -c 140
fi
"""
        r = ssh_run(box.ipv4, q, timeout=15)
        is_done = "DONE" in r.stdout
        if is_done:
            done_count += 1
        lines.append((box.name, is_done, r.stdout.strip()[:160]))
    return done_count, lines


def retrieve_baseline(box: Box, out_root: Path, seeds: list[int]) -> tuple[str, str, list[str]]:
    """rsync simulation_seedN dirs + logs back to the local output root.

    Uses the root@{ip}: prefix explicitly — this was the 2026-04-14 retrieve
    bug that lost Phase 1 profiles.
    """
    name = box.name
    ip = box.ipv4
    box_out = out_root / name
    box_out.mkdir(parents=True, exist_ok=True)
    ssh_arg = "ssh " + " ".join(SSH_OPTS)
    errors: list[str] = []

    for remote, local in [
        ("/root/baseline.log", box_out / "baseline.log"),
        ("/root/baseline.done", box_out / "baseline.done"),
    ]:
        r = subprocess.run(
            ["rsync", "-az", "-e", ssh_arg, f"root@{ip}:{remote}", str(local)],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            errors.append(f"{remote}: {r.stderr[:120]}")

    for seed in seeds:
        remote = f"/root/projects/bts/data/simulation_seed{seed}/"
        local = box_out / f"simulation_seed{seed}"
        local.mkdir(exist_ok=True)
        r = subprocess.run(
            ["rsync", "-az", "-e", ssh_arg, f"root@{ip}:{remote}", str(local) + "/"],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            errors.append(f"simulation_seed{seed}: {r.stderr[:120]}")

    return (name, "ok" if not errors else "partial", errors[:3])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", type=int, default=4,
                    help="Max Hetzner CPX51 boxes to request (graceful if fewer)")
    ap.add_argument("--seeds", type=int, default=16,
                    help="Number of seeds to run (first N from DEFAULT_SEEDS)")
    ap.add_argument("--label", default="bts-pooled",
                    help="Box name prefix")
    ap.add_argument("--out", type=Path,
                    default=LOCAL_BTS / "data" / "hetzner_results" / "pooled_bins_run",
                    help="Local output directory")
    ap.add_argument("--poll-interval", type=int, default=600,
                    help="Poll interval in seconds")
    ap.add_argument("--deadline-hours", type=float, default=12.0,
                    help="Hard cap on total runtime")
    args = ap.parse_args()

    seeds = DEFAULT_SEEDS[: args.seeds]
    log(f"Baseline-only run: {len(seeds)} seeds on up to {args.boxes} Hetzner boxes")
    log(f"  seeds: {seeds}")

    provider = HetznerProvider()
    args.out.mkdir(parents=True, exist_ok=True)

    boxes: list[Box] = []
    try:
        boxes = spinup(provider, args.boxes, args.label)
        if not boxes:
            log("No boxes spun up — aborting")
            return

        (args.out / "boxes.json").write_text(json.dumps(
            [{"id": b.id, "name": b.name, "ipv4": b.ipv4, "region": b.region} for b in boxes],
            indent=2,
        ))

        log(f"Provisioning {len(boxes)} boxes (parallel, src+scripts)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(boxes)) as ex:
            futures = [ex.submit(provision_with_scripts, b) for b in boxes]
            for fut in concurrent.futures.as_completed(futures):
                name, status = fut.result()
                log(f"  [{name}] {status}")

        queues = distribute_seeds(boxes, seeds)
        log("Seed distribution:")
        for nm, sl in queues.items():
            log(f"  {nm}: {sl}")

        log("Launching baseline queues...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(boxes)) as ex:
            futures = [ex.submit(launch_baseline_queue, b, queues[b.name]) for b in boxes]
            for fut in concurrent.futures.as_completed(futures):
                nm, rc, out = fut.result()
                log(f"  [{nm}] rc={rc}  {out}")

        start = time.time()
        deadline_t = start + args.deadline_hours * 3600
        poll_num = 0
        while time.time() < deadline_t:
            poll_num += 1
            time.sleep(args.poll_interval)
            done_count, lines = poll_baseline(boxes)
            elapsed_h = (time.time() - start) / 3600
            log(f"=== POLL #{poll_num} ({elapsed_h:.2f}h, {done_count}/{len(boxes)} done) ===")
            for nm, is_done, s in lines:
                mark = "OK " if is_done else "..."
                log(f"  {mark} {nm}: {s[:140]}")
            if done_count == len(boxes):
                break

        log("=== RETRIEVE ===")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(boxes)) as ex:
            futures = [ex.submit(retrieve_baseline, b, args.out, queues[b.name]) for b in boxes]
            for fut in concurrent.futures.as_completed(futures):
                nm, status, errs = fut.result()
                if errs:
                    log(f"  [{nm}] {status}  errs={errs[:1]}")
                else:
                    log(f"  [{nm}] {status}")
    finally:
        if boxes:
            teardown_all(provider, boxes)

    log("=== BASELINE DRIVER DONE ===")


if __name__ == "__main__":
    main()
