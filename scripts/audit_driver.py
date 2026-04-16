#!/usr/bin/env python3
"""Unified multi-seed audit driver for BTS experiment screening.

Replaces the ad-hoc drivers in /tmp/claude/ that bit us on 2026-04-14/15.
Supports both Hetzner and Vultr as providers via simple dispatch. Handles
graceful degradation when the requested box count isn't available.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit_driver.py \\
        --provider vultr --boxes 15 --seeds 20 \\
        --experiments scripts/audit_experiments.txt

    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit_driver.py \\
        --provider hetzner --boxes 4 --seeds 20 \\
        --experiments scripts/audit_experiments.txt \\
        --label hetzner-only-audit

Fixes 4 bugs from the 2026-04-14/15 overnight runs:
    1. rsync paths missing root@{ip}: prefix (Phase 1 profile retrieve)
    2. Silent degradation when boxes < requested (Vultr 3/20 issue)
    3. Only-teardown-on-full-retrieve (Hetzner zombie risk)
    4. Relative paths in shell launches (cwd drift ambiguity)

Requires:
    - hetzner-cloud-token or vultr-api-token in macOS Keychain
    - SSH pubkey at ~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub
    - /Users/stone/projects/bts/{src,data/processed,data/models,pyproject.toml,uv.lock}
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


LOCAL_BTS = Path("/Users/stone/projects/bts")
SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=10",
]
ENV = "PATH=/root/.local/bin:$PATH UV_CACHE_DIR=/tmp/uv-cache"
USER_DATA_RAW = """#cloud-config
packages: [rsync, git, libgomp1, python3, ca-certificates, curl]
runcmd:
  - curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
  - env HOME=/root sh /tmp/uv-install.sh
  - mkdir -p /root/projects/bts/data
  - touch /root/cloud-init-done
"""


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

@dataclass
class Box:
    id: str
    name: str
    ipv4: str = ""
    region: str = ""


class Provider:
    """Abstract provider interface. Each provider implements create/delete/list."""

    name: str = "?"

    def create(self, label: str) -> Box:
        raise NotImplementedError

    def get(self, box_id: str) -> dict:
        raise NotImplementedError

    def delete(self, box_id: str) -> None:
        raise NotImplementedError

    def is_active(self, box_id: str) -> tuple[bool, str]:
        """Return (active, ipv4). ipv4 is empty string if not yet assigned."""
        raise NotImplementedError


def _keychain(service: str) -> str:
    return subprocess.check_output(
        ["security", "find-generic-password", "-s", service, "-w"],
        text=True,
    ).strip()


class HetznerProvider(Provider):
    name = "hetzner"
    BASE = "https://api.hetzner.cloud/v1"
    SSH_KEY_ID = 110611534
    IMAGE = "ubuntu-24.04"
    SERVER_TYPE = "cpx51"
    LOCATION_FALLBACKS = ["hil", "hel1", "nbg1", "fsn1", "ash"]

    def __init__(self):
        self.token = _keychain("hetzner-cloud-token")

    def _api(self, method: str, path: str, body=None):
        url = self.BASE + path
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                b = r.read()
                return json.loads(b) if b else {}
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Hetzner HTTP {e.code}: {e.read().decode()[:300]}") from e

    def create(self, label: str) -> Box:
        last_err = None
        for loc in self.LOCATION_FALLBACKS:
            try:
                r = self._api("POST", "/servers", {
                    "name": label,
                    "server_type": self.SERVER_TYPE,
                    "image": self.IMAGE,
                    "location": loc,
                    "ssh_keys": [self.SSH_KEY_ID],
                    "user_data": USER_DATA_RAW,
                    "start_after_create": True,
                })
                srv = r["server"]
                return Box(id=str(srv["id"]), name=srv["name"],
                           ipv4=srv["public_net"]["ipv4"]["ip"], region=loc)
            except RuntimeError as e:
                last_err = e
        raise RuntimeError(f"All Hetzner locations exhausted: {last_err}")

    def is_active(self, box_id: str) -> tuple[bool, str]:
        r = self._api("GET", f"/servers/{box_id}")
        srv = r.get("server", {})
        active = srv.get("status") == "running"
        ip = srv.get("public_net", {}).get("ipv4", {}).get("ip", "")
        return (active, ip)

    def delete(self, box_id: str) -> None:
        self._api("DELETE", f"/servers/{box_id}")


class VultrProvider(Provider):
    name = "vultr"
    BASE = "https://api.vultr.com/v2"
    IMAGE_OS_NAME = "Ubuntu 24.04 LTS x64"
    PLAN_FALLBACKS = ["voc-c-16c-32gb-300s-amd", "voc-c-16c-32gb-500s-amd"]
    REGION_FALLBACKS = ["fra", "ams", "sto", "lhr", "cdg", "waw"]

    def __init__(self):
        self.token = _keychain("vultr-api-token")
        self._os_id = None
        self._plan_id = None
        self._ssh_key_id = None

    def _api(self, method: str, path: str, body=None):
        url = self.BASE + path
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                b = r.read()
                return json.loads(b) if b else {}
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Vultr HTTP {e.code}: {e.read().decode()[:300]}") from e

    def _resolve_once(self):
        if self._os_id is not None:
            return
        r = self._api("GET", "/os?per_page=500")
        for o in r.get("os", []):
            if self.IMAGE_OS_NAME == (o.get("name") or "") and o.get("arch") == "x64":
                self._os_id = o["id"]
                break
        if self._os_id is None:
            raise RuntimeError(f"Vultr: no OS named {self.IMAGE_OS_NAME}")
        r = self._api("GET", "/plans?per_page=500")
        available = {p["id"] for p in r.get("plans", [])}
        for pid in self.PLAN_FALLBACKS:
            if pid in available:
                self._plan_id = pid
                break
        if self._plan_id is None:
            raise RuntimeError("Vultr: no matching plan")
        # SSH key upload / lookup
        pub_path = os.path.expanduser("~/.ssh/id_ed25519.pub")
        if not os.path.exists(pub_path):
            pub_path = os.path.expanduser("~/.ssh/id_rsa.pub")
        pubkey = open(pub_path).read().strip()
        r = self._api("GET", "/ssh-keys")
        for k in r.get("ssh_keys", []):
            if k.get("ssh_key", "").strip() == pubkey:
                self._ssh_key_id = k["id"]
                break
        if self._ssh_key_id is None:
            r = self._api("POST", "/ssh-keys", {"name": "bts-audit-claude", "ssh_key": pubkey})
            self._ssh_key_id = r["ssh_key"]["id"]

    def create(self, label: str) -> Box:
        self._resolve_once()
        user_data_b64 = base64.b64encode(USER_DATA_RAW.encode()).decode()
        last_err = None
        for region in self.REGION_FALLBACKS:
            try:
                r = self._api("POST", "/instances", {
                    "region": region,
                    "plan": self._plan_id,
                    "os_id": self._os_id,
                    "label": label,
                    "sshkey_id": [self._ssh_key_id],
                    "user_data": user_data_b64,
                    "backups": "disabled",
                    "tag": "bts-audit",
                })
                inst = r["instance"]
                return Box(id=inst["id"], name=label,
                           ipv4=inst.get("main_ip", ""), region=region)
            except RuntimeError as e:
                last_err = e
        raise RuntimeError(f"All Vultr regions exhausted: {last_err}")

    def is_active(self, box_id: str) -> tuple[bool, str]:
        r = self._api("GET", f"/instances/{box_id}")
        inst = r.get("instance", {})
        active = inst.get("status") == "active"
        ip = inst.get("main_ip", "") or ""
        return (active and ip not in ("0.0.0.0", ""), ip)

    def delete(self, box_id: str) -> None:
        self._api("DELETE", f"/instances/{box_id}")


def make_provider(name: str) -> Provider:
    if name == "hetzner":
        return HetznerProvider()
    if name == "vultr":
        return VultrProvider()
    raise ValueError(f"unknown provider: {name}")


# ---------------------------------------------------------------------------
# Driver logic
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ssh_run(ip: str, cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *SSH_OPTS, f"root@{ip}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def spinup(provider: Provider, requested: int, label_prefix: str) -> list[Box]:
    """Attempt to create `requested` boxes. Returns those that actually came up.

    Graceful: if provider refuses past N, log it clearly and return the N.
    """
    log(f"Requesting {requested} {provider.name} boxes...")
    created: list[Box] = []
    for i in range(1, requested + 1):
        name = f"{label_prefix}-{i}"
        try:
            box = provider.create(name)
            created.append(box)
            log(f"  created {name} id={box.id} region={box.region}")
        except RuntimeError as e:
            log(f"  create {name} FAILED: {str(e)[:250]}")
            break

    if len(created) < requested:
        log(f"Got {len(created)}/{requested} boxes — continuing with actual count")

    # Wait for all created boxes to be active + have an IP
    log("Waiting for active status + IP assignment...")
    deadline = time.time() + 300
    pending = {b.id for b in created}
    while pending and time.time() < deadline:
        time.sleep(5)
        still_pending = set()
        for box in created:
            if box.id not in pending:
                continue
            try:
                active, ip = provider.is_active(box.id)
                if active:
                    box.ipv4 = ip
                    log(f"  {box.name} active ip={ip}")
                else:
                    still_pending.add(box.id)
            except Exception as e:
                log(f"  {box.name} status poll error: {e}")
                still_pending.add(box.id)
        pending = still_pending

    return [b for b in created if b.ipv4]


def wait_cloud_init(box: Box) -> bool:
    deadline = time.time() + 600
    while time.time() < deadline:
        r = ssh_run(box.ipv4, "test -f /root/cloud-init-done && echo DONE", timeout=15)
        if "DONE" in r.stdout:
            return True
        time.sleep(10)
    return False


def provision_one(box: Box) -> tuple[str, str]:
    name = box.name
    ip = box.ipv4
    if not wait_cloud_init(box):
        return (name, "cloud_init_timeout")

    log(f"  [{name}] rsync src/data/models...")
    ssh_arg = "ssh " + " ".join(SSH_OPTS)
    for src, dst in [
        (f"{LOCAL_BTS}/src/", "/root/projects/bts/src/"),
        (f"{LOCAL_BTS}/data/processed/", "/root/projects/bts/data/processed/"),
        (f"{LOCAL_BTS}/data/models/", "/root/projects/bts/data/models/"),
    ]:
        r = subprocess.run(
            ["rsync", "-az", "-e", ssh_arg, src, f"root@{ip}:{dst}"],
            capture_output=True, text=True, timeout=900,
        )
        if r.returncode != 0:
            return (name, f"rsync_failed: {r.stderr[:200]}")

    r = subprocess.run(
        ["rsync", "-az", "-e", ssh_arg,
         f"{LOCAL_BTS}/pyproject.toml", f"{LOCAL_BTS}/uv.lock",
         f"{LOCAL_BTS}/README.md",
         f"root@{ip}:/root/projects/bts/"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        return (name, f"rsync_configs: {r.stderr[:200]}")

    ssh_run(ip, "mkdir -p /root/projects/bts/data/validation /root/projects/bts/experiments/results/phase1 /root/projects/bts/data/experiments")

    log(f"  [{name}] uv sync --extra model...")
    r = ssh_run(ip, f"cd /root/projects/bts && {ENV} uv sync --extra model 2>&1 | tail -3", timeout=900)
    return (name, "ready")


def distribute_seeds(boxes: list[Box], seeds: list[int]) -> dict[str, list[int]]:
    """Round-robin seeds across boxes (interleaved for magnitude diversity)."""
    queues: dict[str, list[int]] = {b.name: [] for b in boxes}
    names = [b.name for b in boxes]
    for i, s in enumerate(seeds):
        queues[names[i % len(names)]].append(s)
    return queues


def launch_box_queue(box: Box, seeds: list[int], exp_list: str) -> tuple[str, int, str]:
    """Launch a shell-based seed queue on one box.

    Uses ABSOLUTE paths to eliminate cwd drift. Each seed runs:
      1. bts experiment screen --subset exp_list (baseline + all experiments)
      2. mv experiments/results/phase1 → phase1_seed$SEED for per-seed isolation
    """
    ip = box.ipv4
    name = box.name
    seed_list_str = " ".join(str(s) for s in seeds)

    BTS = "/root/projects/bts"
    cmd = f"""
rm -f /root/audit.log /root/audit.done
nohup bash -c '
set +e
for SEED in {seed_list_str}; do
  echo "=== seed=$SEED starting at $(date) ===" >> /root/audit.log
  rm -rf {BTS}/experiments/results/phase1
  mkdir -p {BTS}/experiments/results/phase1
  cd {BTS}
  BTS_LGBM_RANDOM_STATE=$SEED {ENV} \\
    uv run bts experiment screen --subset "{exp_list}" \\
    --test-seasons 2024,2025 >> /root/audit.log 2>&1
  rc=$?
  mv {BTS}/experiments/results/phase1 {BTS}/experiments/results/phase1_seed$SEED
  echo "=== seed=$SEED done at $(date) rc=$rc ===" >> /root/audit.log
done
echo "queue done at $(date)" > /root/audit.done
' > /dev/null 2>&1 &
disown
echo "launched seeds={seed_list_str}"
"""
    r = ssh_run(ip, cmd, timeout=30)
    return (name, r.returncode, r.stdout.strip())


def poll(boxes: list[Box]) -> tuple[int, list[tuple]]:
    done_count = 0
    lines = []
    for box in boxes:
        q = r"""
if [ -f /root/audit.done ]; then
  echo DONE
  cat /root/audit.done
else
  grep -c '=== seed=.* done' /root/audit.log 2>/dev/null
  tail -1 /root/audit.log 2>/dev/null | head -c 140
fi
"""
        r = ssh_run(box.ipv4, q, timeout=15)
        is_done = "DONE" in r.stdout
        if is_done:
            done_count += 1
        lines.append((box.name, is_done, r.stdout.strip()[:160]))
    return done_count, lines


def retrieve_one(box: Box, out_root: Path, seeds: list[int]) -> tuple[str, str, list[str]]:
    """Rsync results off a box. Uses absolute paths AND includes the root@ip:
    prefix (the 2026-04-14 Phase 1 retrieve failed because this prefix was
    missing from profile-dir paths)."""
    name = box.name
    ip = box.ipv4
    box_out = out_root / f"{name}"
    box_out.mkdir(parents=True, exist_ok=True)
    ssh_arg = "ssh " + " ".join(SSH_OPTS)
    errors: list[str] = []

    # Fixed-name files
    for remote, local in [
        ("/root/audit.log", box_out / "audit.log"),
        ("/root/audit.done", box_out / "audit.done"),
    ]:
        r = subprocess.run(
            ["rsync", "-az", "-e", ssh_arg, f"root@{ip}:{remote}", str(local)],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            errors.append(f"{remote}: {r.stderr[:120]}")

    # Per-seed experiment result directories
    for seed in seeds:
        remote = f"/root/projects/bts/experiments/results/phase1_seed{seed}/"
        local = box_out / f"phase1_seed{seed}"
        local.mkdir(exist_ok=True)
        r = subprocess.run(
            ["rsync", "-az", "-e", ssh_arg, f"root@{ip}:{remote}", str(local) + "/"],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            errors.append(f"phase1_seed{seed}: {r.stderr[:120]}")

    return (name, "ok" if not errors else "partial", errors[:3])


def teardown_all(provider: Provider, boxes: list[Box]) -> None:
    """Unconditional teardown. Called from finally blocks to guarantee cleanup
    even if the main flow fails. Fixes the 2026-04-14 Hetzner retrieve-cascade
    bug where partial rsync → no teardown → zombie boxes."""
    log("=== TEARDOWN ===")
    for box in boxes:
        try:
            provider.delete(box.id)
            log(f"  deleted {box.name}")
        except Exception as e:
            log(f"  FAILED to delete {box.name}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = [
    # Block 1: original 20 seeds (Phase 1 + audit_driver v1)
    1, 7, 42, 137, 1024, 8192, 65536, 524288, 999983,
    2041769, 2273360, 3167584, 3911096, 6426836, 6762547,
    6992150, 7777777, 7886470, 7973654, 9884564,
    # Block 2: extended seeds (pooled-policy Track C)
    11, 23, 73, 199, 2048, 16384, 131072, 1048576, 999979,
    4083538, 4546720, 6335168, 7822192, 12853672, 13525094, 13984300,
    # Block 3: hash-derived expansion for n=48+ audits
    5573027, 181677080, 147270468, 189192333, 33619740, 244383382,
    9934256, 82054048, 262244879, 124613597, 85415891, 120749711,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["hetzner", "vultr"], required=True)
    ap.add_argument("--boxes", type=int, required=True, help="Max boxes to request (graceful if provider gives fewer)")
    ap.add_argument("--seeds", type=int, default=20, help="Number of seeds to run (first N from DEFAULT_SEEDS)")
    ap.add_argument("--experiments", type=Path, required=True, help="Path to a file with comma-separated experiment names")
    ap.add_argument("--label", default=None, help="Box name prefix (default: bts-audit-{provider})")
    ap.add_argument("--out", type=Path, default=LOCAL_BTS / "data" / "hetzner_results" / "audit_run", help="Local output directory")
    ap.add_argument("--poll-interval", type=int, default=900, help="Poll interval in seconds")
    ap.add_argument("--deadline-hours", type=float, default=168.0, help="Hard cap on total runtime")
    args = ap.parse_args()

    experiments = args.experiments.read_text().strip()
    if not experiments:
        log("ERROR: experiments file empty")
        sys.exit(1)
    seeds = DEFAULT_SEEDS[: args.seeds]
    log(f"Audit: {len(seeds)} seeds × {len(experiments.split(','))} experiments "
        f"on up to {args.boxes} {args.provider} boxes")

    label_prefix = args.label or f"bts-audit-{args.provider}"
    provider = make_provider(args.provider)
    args.out.mkdir(parents=True, exist_ok=True)

    boxes: list[Box] = []
    try:
        boxes = spinup(provider, args.boxes, label_prefix)
        if not boxes:
            log("No boxes spun up — aborting")
            return

        # Save box state for debugging / manual recovery
        (args.out / "boxes.json").write_text(json.dumps(
            [{"id": b.id, "name": b.name, "ipv4": b.ipv4, "region": b.region} for b in boxes],
            indent=2,
        ))

        log(f"Provisioning {len(boxes)} boxes (parallel)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(boxes)) as ex:
            futures = [ex.submit(provision_one, b) for b in boxes]
            for fut in concurrent.futures.as_completed(futures):
                name, status = fut.result()
                log(f"  [{name}] {status}")

        queues = distribute_seeds(boxes, seeds)
        log("Seed distribution:")
        for nm, sl in queues.items():
            log(f"  {nm}: {sl}")

        log("Launching queues...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(boxes)) as ex:
            futures = [ex.submit(launch_box_queue, b, queues[b.name], experiments) for b in boxes]
            for fut in concurrent.futures.as_completed(futures):
                nm, rc, out = fut.result()
                log(f"  [{nm}] rc={rc}  {out}")

        # Poll
        start = time.time()
        deadline_t = start + args.deadline_hours * 3600
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

        # Retrieve
        log("=== RETRIEVE ===")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(boxes)) as ex:
            futures = [ex.submit(retrieve_one, b, args.out, queues[b.name]) for b in boxes]
            for fut in concurrent.futures.as_completed(futures):
                nm, status, errs = fut.result()
                if errs:
                    log(f"  [{nm}] {status}  errs={errs[:1]}")
                else:
                    log(f"  [{nm}] {status}")
    finally:
        # Unconditional teardown, always. This is the key fix for the
        # 2026-04-14 Hetzner retrieve-cascade bug.
        if boxes:
            teardown_all(provider, boxes)

    log("=== AUDIT DRIVER DONE ===")


if __name__ == "__main__":
    main()
