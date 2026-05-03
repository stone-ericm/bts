#!/usr/bin/env python3
"""v2_5_run_ablations.py — Launch harness ablation runs on 4 Vultr boxes.

Reads /tmp/v2.5/instances.tsv (cell_label TAB instance_id TAB ip) and fires
one SSH+nohup run per box with the correct mode flags for that cell label.

Cell flag mapping (ablation: bits are params_mode, rho_pair_mode, policy_mode):
    000: params=fold-local  rho=per-bin   policy=per-fold   (v2 baseline — NOT run here)
    111: params=pooled      rho=scalar    policy=global     (v1 baseline — NOT run here)
    010: params=pooled      rho=per-bin   policy=global
    001: params=pooled      rho=scalar    policy=per-fold
    011: params=pooled      rho=per-bin   policy=per-fold
    101: params=fold-local  rho=scalar    policy=per-fold

Bit encoding:
    bit 2 (MSB): params_mode   0=fold-local 1=pooled
    bit 1:       rho_pair_mode 0=per-bin    1=scalar
    bit 0 (LSB): policy_mode   0=per-fold   1=global

Usage:
    python scripts/v2_5_run_ablations.py [--dry-run]

Outputs:
    Each box writes:
        /root/projects/bts/data/validation/falsification_harness_v2.5_cell{CELL}.json
        /root/projects/bts/data/validation/falsification_harness_v2.5_cell{CELL}_heatmap.json
    Logs:
        /root/v2.5_{CELL}.log
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# ---- Cell → harness flag mapping ----
# Bit encoding: bit2=params_mode(1=pooled), bit1=rho_pair_mode(1=scalar), bit0=policy_mode(1=global)
CELL_FLAGS: dict[str, tuple[str, str, str]] = {
    "010": ("pooled",      "per-bin", "global"),
    "001": ("pooled",      "scalar",  "per-fold"),
    "011": ("pooled",      "per-bin", "per-fold"),
    "101": ("fold-local",  "scalar",  "per-fold"),
}

KNOWN_CELLS = set(CELL_FLAGS.keys())

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=60",
    "-o", "ServerAliveCountMax=5",
]

ENV_PREFIX = "PATH=/root/.local/bin:$PATH UV_CACHE_DIR=/tmp/uv-cache"

# Production run parameters (match v2's production run).
N_BOOTSTRAP = 300
N_PERMUTATIONS = 300
PA_N_BOOTSTRAP = 300
N_FINAL = 20000

INSTANCES_TSV = Path("/tmp/v2.5/instances.tsv")


def read_instances(tsv: Path) -> list[tuple[str, str, str]]:
    """Parse instances.tsv → list of (cell_label, instance_id, ip)."""
    if not tsv.exists():
        raise FileNotFoundError(
            f"{tsv} not found. Run scripts/v2_5_provision.sh first."
        )
    rows = []
    for line in tsv.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"Malformed instances.tsv line: {line!r}")
        cell, inst_id, ip = parts[0], parts[1], parts[2]
        rows.append((cell, inst_id, ip))
    return rows


def build_harness_cmd(cell: str, params_mode: str, rho_pair_mode: str, policy_mode: str) -> str:
    """Build the full shell command to run the harness on the remote box."""
    output_path = f"data/validation/falsification_harness_v2.5_cell{cell}.json"
    log_path = f"/root/v2.5_{cell}.log"
    cmd = (
        f"cd /root/projects/bts && {ENV_PREFIX} "
        f"nohup uv run python scripts/run_falsification_harness.py "
        f"--profiles-glob 'data/simulation/profiles_seed*_season*.parquet' "
        f"--pa-glob 'data/simulation/pa_predictions_seed*_season*.parquet' "
        f"--output {output_path} "
        f"--n-bootstrap {N_BOOTSTRAP} "
        f"--n-permutations {N_PERMUTATIONS} "
        f"--pa-n-bootstrap {PA_N_BOOTSTRAP} "
        f"--n-final {N_FINAL} "
        f"--params-mode {params_mode} "
        f"--rho-pair-mode {rho_pair_mode} "
        f"--policy-mode {policy_mode} "
        f"> {log_path} 2>&1 & "
        f"echo $!"
    )
    return cmd


def launch_cell(cell: str, ip: str, dry_run: bool = False) -> int | None:
    """SSH to box and fire the harness with nohup. Returns the remote PID or None (dry-run)."""
    if cell not in CELL_FLAGS:
        raise ValueError(f"Unknown cell {cell!r}. Known: {sorted(CELL_FLAGS)}")
    params_mode, rho_pair_mode, policy_mode = CELL_FLAGS[cell]
    cmd = build_harness_cmd(cell, params_mode, rho_pair_mode, policy_mode)

    print(
        f"  [{cell}] ip={ip} params={params_mode} rho_pair={rho_pair_mode} policy={policy_mode}"
    )
    if dry_run:
        print(f"    [dry-run] would SSH: {cmd}")
        return None

    proc = subprocess.run(
        ["ssh", *SSH_OPTS, f"root@{ip}", cmd],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        print(f"  ERROR launching cell {cell}: {proc.stderr.strip()[:300]}", file=sys.stderr)
        return None

    pid_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        pid = int(pid_line)
    except ValueError:
        pid = None
    print(f"    remote PID: {pid}")
    return pid


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("[dry-run] v2.5_run_ablations.py — no SSH calls will be made")
        # Validate flag mapping for all cells
        for cell, (pm, rm, poly) in sorted(CELL_FLAGS.items()):
            print(f"  cell {cell}: params={pm} rho_pair={rm} policy={poly}")
            cmd = build_harness_cmd(cell, pm, rm, poly)
            # Sanity: all three mode flags appear in command
            assert f"--params-mode {pm}" in cmd, f"params-mode missing in cmd for {cell}"
            assert f"--rho-pair-mode {rm}" in cmd, f"rho-pair-mode missing in cmd for {cell}"
            assert f"--policy-mode {poly}" in cmd, f"policy-mode missing in cmd for {cell}"
        print("[dry-run] All flag mappings validated. OK.")
        return

    instances = read_instances(INSTANCES_TSV)
    if not instances:
        print("ERROR: instances.tsv is empty", file=sys.stderr)
        sys.exit(1)

    # Validate all expected cells are present
    found_cells = {row[0] for row in instances}
    missing = KNOWN_CELLS - found_cells
    if missing:
        print(f"WARNING: Missing cells in instances.tsv: {sorted(missing)}", file=sys.stderr)

    print(f"Launching {len(instances)} ablation run(s) in parallel...")
    t0 = time.time()

    procs: list[tuple[str, str, subprocess.Popen]] = []
    for cell, inst_id, ip in instances:
        params_mode, rho_pair_mode, policy_mode = CELL_FLAGS.get(cell, ("?", "?", "?"))
        if cell not in CELL_FLAGS:
            print(f"  SKIP unknown cell {cell!r}", file=sys.stderr)
            continue
        pid = launch_cell(cell, ip, dry_run=False)
        if pid is None:
            print(f"  WARNING: cell {cell} failed to launch", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nAll launches fired in {elapsed:.1f}s.")
    print("Runs are executing in the background on each box (~22 min each).")
    print("")
    print("To monitor progress:")
    while_ifs = "while IFS=$'\\t' read -r CELL INST IP; do"
    print(f"  {while_ifs}")
    print(f"    echo \"=== $CELL $IP ===\"")
    print(f"    ssh -o StrictHostKeyChecking=no root@$IP 'tail -5 /root/v2.5_\"$CELL\".log' 2>/dev/null")
    print(f"  done < /tmp/v2.5/instances.tsv")
    print("")
    print("When done:")
    print("  bash scripts/v2_5_retrieve.sh")


if __name__ == "__main__":
    main()
