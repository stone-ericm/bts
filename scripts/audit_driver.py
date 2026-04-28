#!/usr/bin/env python3
"""Unified multi-seed audit driver for BTS experiment screening.

Replaces the ad-hoc drivers in /tmp/claude/ that bit us on 2026-04-14/15.
Supports Hetzner Cloud, Vultr, and Oracle Cloud Infrastructure (OCI) as
providers via simple dispatch. Handles graceful degradation when the
requested box count isn't available.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit_driver.py \\
        --provider vultr --boxes 15 --seeds 20 \\
        --experiments scripts/audit_experiments.txt

    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit_driver.py \\
        --provider hetzner --boxes 4 --seeds 20 \\
        --experiments scripts/audit_experiments.txt \\
        --label hetzner-only-audit

    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit_driver.py \\
        --provider oci --boxes 1 --seeds 1 \\
        --seeds-file scripts/audit_seeds_determinism.txt \\
        --experiments scripts/audit_experiments.txt \\
        --label bts-oci-validate

Fixes 4 bugs from the 2026-04-14/15 overnight runs:
    1. rsync paths missing root@{ip}: prefix (Phase 1 profile retrieve)
    2. Silent degradation when boxes < requested (Vultr 3/20 issue)
    3. Only-teardown-on-full-retrieve (Hetzner zombie risk)
    4. Relative paths in shell launches (cwd drift ambiguity)

Requires:
    - Provider credentials (one of):
        hetzner-cloud-token (macOS Keychain)
        vultr-api-token (macOS Keychain)
        ~/.oci/config (via `oci setup config`) OR keychain entries:
            oci-tenancy-ocid, oci-user-ocid, oci-fingerprint,
            oci-api-private-key, oci-region
        plus keychain entry `oci-subnet-ocid` for a public subnet.
    - SSH pubkey at ~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub
    - /Users/stone/projects/bts/{src,data/processed,data/models,pyproject.toml,uv.lock}
    - For --provider oci: `uv add oci` (oci-python-sdk) and a service limit
      increase for VM.Standard.E5.Flex AMD OCPUs in the home region.
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

# OCI Ubuntu images default the SSH key to user `ubuntu`. The driver's
# SSH calls are hardcoded to `root@{ip}`, so we copy the authorized_keys
# to root and enable key-only root SSH before touching cloud-init-done.
USER_DATA_OCI = """#cloud-config
packages: [rsync, git, libgomp1, python3, ca-certificates, curl]
runcmd:
  - mkdir -p /root/.ssh
  - cp /home/ubuntu/.ssh/authorized_keys /root/.ssh/authorized_keys
  - chmod 700 /root/.ssh
  - chmod 600 /root/.ssh/authorized_keys
  - sed -i 's/^#\\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
  - systemctl restart ssh
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
    """Fetch a secret by service name, falling back to env var on non-macOS hosts.

    Tries macOS Keychain via `security` first. On Linux (bts-hetzner, Pi5),
    `security` is absent or errors, so we fall back to an env var named
    BTS_SECRET_<SERVICE_UPPER_WITH_UNDERSCORES>. Example: service
    "hetzner-cloud-token" -> env var BTS_SECRET_HETZNER_CLOUD_TOKEN.
    """
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-w"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # non-macOS or hung — fall through to env var

    env_name = "BTS_SECRET_" + service.upper().replace("-", "_")
    val = os.environ.get(env_name)
    if val:
        return val
    raise RuntimeError(
        f"No secret for {service!r}: tried macOS Keychain (service={service!r}) "
        f"and env var {env_name!r}"
    )


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
        last_err = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    b = r.read()
                    return json.loads(b) if b else {}
            except urllib.error.HTTPError as e:
                # Retry 5xx transient errors; fail hard on 4xx client errors.
                if 500 <= e.code < 600 and attempt < 2:
                    last_err = e
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Vultr HTTP {e.code}: {e.read().decode()[:300]}") from e
            except (TimeoutError, urllib.error.URLError) as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
        raise RuntimeError(f"Vultr API failed after 3 attempts: {last_err}") from last_err

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


class OCIProvider(Provider):
    """Oracle Cloud Infrastructure — AMD EPYC Genoa (Zen 4) flex shapes.

    Same microarchitecture family as Hetzner CPX51 and Vultr vhp-8c, which
    gives the highest prior that LightGBM output is byte-identical (required
    for pooling seeds across providers in audits).

    Shape: VM.Standard.E5.Flex @ 8 OCPU / 32 GB. 1 OCPU = 2 vCPU (HT pair)
    on AMD, so 8 OCPU ≈ 16 vCPU, matching Hetzner CPX51.

    Prerequisites:
      - `uv add oci` (oci-python-sdk installed)
      - ~/.oci/config (via `oci setup config`) OR keychain entries:
        oci-tenancy-ocid, oci-user-ocid, oci-fingerprint,
        oci-api-private-key, oci-region
      - Keychain entry `oci-subnet-ocid` pointing to a public subnet in the
        home region (create via OCI Console → Networking → VCN Wizard).
      - Service limit increase for VM.Standard.E5.Flex AMD OCPUs — new
        accounts default to 0 and you can't launch without filing a ticket.
    """
    name = "oci"
    SHAPE = "VM.Standard.E5.Flex"  # AMD EPYC Genoa Zen 4
    OCPUS = 8.0       # 8 OCPU × 2 vCPU/OCPU = 16 vCPU — matches Hetzner CPX51
    MEM_GB = 32.0

    def __init__(self):
        try:
            import oci
        except ImportError as e:
            raise RuntimeError(
                "OCI provider requires oci-python-sdk. Install with: "
                "UV_CACHE_DIR=/tmp/uv-cache uv add oci"
            ) from e
        self._oci = oci

        config_path = os.path.expanduser("~/.oci/config")
        if os.path.exists(config_path):
            self.config = oci.config.from_file(config_path)
        else:
            self.config = {
                "tenancy": _keychain("oci-tenancy-ocid"),
                "user": _keychain("oci-user-ocid"),
                "fingerprint": _keychain("oci-fingerprint"),
                "key_content": _keychain("oci-api-private-key"),
                "region": _keychain("oci-region"),
            }
        oci.config.validate_config(self.config)

        try:
            self.compartment_id = _keychain("oci-compartment-ocid")
        except subprocess.CalledProcessError:
            self.compartment_id = self.config["tenancy"]
        self.subnet_id = _keychain("oci-subnet-ocid")

        self.compute = oci.core.ComputeClient(self.config)
        self.network = oci.core.VirtualNetworkClient(self.config)
        self.identity = oci.identity.IdentityClient(self.config)
        # Composite ops waits on WorkRequest status (quota accounting signal),
        # not just instance lifecycle_state (which fires early). Required per
        # Oracle SDK docs for reliable sequential launches under quota pressure.
        self.composite = oci.core.ComputeClientCompositeOperations(self.compute)
        # Opt LimitExceeded into the retry strategy — it's NOT in the default
        # retry set. Without this opt-in, launch fails immediately on 400
        # LimitExceeded instead of retrying with backoff.
        self.retry_strategy = oci.retry.RetryStrategyBuilder(
            max_attempts=15,
            total_elapsed_time_seconds=1800,
            retry_base_sleep_time_seconds=8,
            retry_max_wait_between_calls_seconds=120,
            service_error_retry_config={400: ["LimitExceeded"]},
            service_error_retry_on_any_5xx=True,
            backoff_type=oci.retry.BACKOFF_FULL_JITTER_EQUAL_ON_THROTTLE_VALUE,
        ).get_retry_strategy()

        self._image_id: str | None = None
        self._ad_fallbacks: list[str] = []

    def _resolve_once(self):
        if self._image_id is not None:
            return
        ads = self.identity.list_availability_domains(
            compartment_id=self.config["tenancy"]
        ).data
        if not ads:
            raise RuntimeError("OCI: no availability domains returned")
        self._ad_fallbacks = [ad.name for ad in ads]

        images = self.compute.list_images(
            compartment_id=self.compartment_id,
            operating_system="Canonical Ubuntu",
            operating_system_version="24.04",
            shape=self.SHAPE,
            sort_by="TIMECREATED",
            sort_order="DESC",
        ).data
        for img in images:
            lbl = (img.display_name or "").lower()
            if "aarch64" in lbl or "arm" in lbl:
                continue
            self._image_id = img.id
            break
        if self._image_id is None:
            raise RuntimeError("OCI: no Ubuntu 24.04 x86_64 image found for E5.Flex")

    def create(self, label: str) -> Box:
        self._resolve_once()
        oci = self._oci

        pub_path = os.path.expanduser("~/.ssh/id_ed25519.pub")
        if not os.path.exists(pub_path):
            pub_path = os.path.expanduser("~/.ssh/id_rsa.pub")
        pubkey = open(pub_path).read().strip()
        user_data_b64 = base64.b64encode(USER_DATA_OCI.encode()).decode()

        last_err = None
        for ad in self._ad_fallbacks:
            try:
                launch = oci.core.models.LaunchInstanceDetails(
                    availability_domain=ad,
                    compartment_id=self.compartment_id,
                    display_name=label,
                    shape=self.SHAPE,
                    shape_config=oci.core.models.LaunchInstanceShapeConfigDetails(
                        ocpus=self.OCPUS,
                        memory_in_gbs=self.MEM_GB,
                    ),
                    source_details=oci.core.models.InstanceSourceViaImageDetails(
                        image_id=self._image_id,
                    ),
                    create_vnic_details=oci.core.models.CreateVnicDetails(
                        subnet_id=self.subnet_id,
                        assign_public_ip=True,
                    ),
                    metadata={
                        "ssh_authorized_keys": pubkey,
                        "user_data": user_data_b64,
                    },
                )
                # Wait for WorkRequest SUCCEEDED (not just instance RUNNING).
                # This is the correct signal that quota accounting has released
                # the in-flight OCPUs — polling lifecycle_state fires too early
                # and causes the next launch to see LimitExceeded. SDK retry
                # strategy handles LimitExceeded backoff internally.
                resp = self.composite.launch_instance_and_wait_for_work_request(
                    launch,
                    operation_kwargs={"retry_strategy": self.retry_strategy},
                )
                # Extract instance OCID from the work request's affected resources.
                wr = resp.data
                inst_ocid = None
                for res in wr.resources or []:
                    if getattr(res, "entity_type", "") == "instance":
                        inst_ocid = res.identifier
                        break
                if inst_ocid is None:
                    raise RuntimeError(
                        f"OCI work request {wr.id} succeeded but no instance "
                        f"in resources list"
                    )
                return Box(id=inst_ocid, name=label, ipv4="", region=ad)
            except oci.exceptions.ServiceError as e:
                last_err = e
                code = e.code or ""
                if "OutOfCapacity" in code or e.status in (500, 503):
                    continue  # try next AD
                # LimitExceeded should have been handled by retry_strategy;
                # if we get here with LimitExceeded, all retries exhausted.
                raise RuntimeError(
                    f"OCI launch failed: {e.status} {e.code} {str(e)[:200]}"
                ) from e
        raise RuntimeError(f"All OCI availability domains exhausted: {last_err}")

    def is_active(self, box_id: str) -> tuple[bool, str]:
        inst = self.compute.get_instance(box_id).data
        if inst.lifecycle_state != "RUNNING":
            return (False, "")
        atts = self.compute.list_vnic_attachments(
            compartment_id=self.compartment_id,
            instance_id=box_id,
        ).data
        for att in atts:
            if att.lifecycle_state != "ATTACHED":
                continue
            vnic = self.network.get_vnic(att.vnic_id).data
            if vnic.public_ip:
                return (True, vnic.public_ip)
        return (False, "")

    def delete(self, box_id: str) -> None:
        self.compute.terminate_instance(box_id, preserve_boot_volume=False)


def make_provider(name: str) -> Provider:
    if name == "hetzner":
        return HetznerProvider()
    if name == "vultr":
        return VultrProvider()
    if name == "oci":
        return OCIProvider()
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

    OCI has a tenancy-wide cap of ≤4 concurrent non-RUNNING instances (launches
    past that fail with 400 LimitExceeded). To stay under, before launching box
    N+5 we poll box N for RUNNING state — ensures ≤4 non-RUNNING at any moment.
    """
    log(f"Requesting {requested} {provider.name} boxes...")
    created: list[Box] = []
    # OCI's LimitExceeded workarounds are now handled inside OCIProvider.create():
    # - launch_instance_and_wait_for_work_request waits on quota-accounting signal
    # - retry_strategy opts LimitExceeded into automatic backoff retries
    # No OCI-specific pacing needed in spinup — SDK handles it.
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
            capture_output=True, text=True, timeout=1800,
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
  BTS_LGBM_RANDOM_STATE=$SEED BTS_LGBM_DETERMINISTIC=1 {ENV} \\
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
    """Poll each box for done-ness + last log line.

    Per-box errors (SSH timeout, connection refused, transient network) are
    isolated: the box is reported as not-done with an error marker and polling
    continues with the remaining boxes. A single hung box can no longer kill
    the entire driver — see 2026-04-25 09:36 ET incident where audit_attach
    crashed on a TimeoutExpired from one box and abandoned the other 25.
    """
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
        try:
            r = ssh_run(box.ipv4, q, timeout=15)
            is_done = "DONE" in r.stdout
            stdout_short = r.stdout.strip()[:160]
        except subprocess.TimeoutExpired:
            log(f"  poll: {box.name} ({box.ipv4}) SSH timeout — will retry next poll")
            is_done = False
            stdout_short = "ssh-timeout"
        except Exception as e:
            log(f"  poll: {box.name} ({box.ipv4}) SSH error: {type(e).__name__}: {e}")
            is_done = False
            stdout_short = f"ssh-error: {type(e).__name__}"
        if is_done:
            done_count += 1
        lines.append((box.name, is_done, stdout_short))
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


def teardown_all(provider: Provider, boxes: list[Box]) -> int:
    """Unconditional teardown. Called from finally blocks to guarantee cleanup
    even if the main flow fails. Fixes the 2026-04-14 Hetzner retrieve-cascade
    bug where partial rsync → no teardown → zombie boxes.

    Returns the count of boxes whose provider.delete() call did not raise.
    """
    log("=== TEARDOWN ===")
    deleted = 0
    for box in boxes:
        try:
            provider.delete(box.id)
            log(f"  deleted {box.name}")
            deleted += 1
        except Exception as e:
            log(f"  FAILED to delete {box.name}: {e}")
    return deleted


def teardown_retrieved(
    provider: Provider,
    boxes: list[Box],
    retrieve_results: dict[str, str],
) -> tuple[int, int]:
    """Tear down only the boxes whose retrieve_results[box.name] == "ok".

    Preserved boxes are logged with name + ipv4 + status for manual recovery.
    Unrecognized keys in retrieve_results (names that don't match any box in
    `boxes`) are also logged — signals a caller bug, not a safety concern.

    Returns (selected, deleted):
      - selected: count of boxes where retrieve_results[name] == "ok"
                  (i.e., passed the data-integrity gate; handed to teardown_all)
      - deleted:  count of boxes whose provider.delete() call didn't raise
                  (inherited from teardown_all; `selected - deleted` = API-failed)

    Callers use `preserved = len(boxes) - selected` to know how many boxes
    were held back for data-integrity. That's the signal that drives the
    non-zero exit code.

    Any box name missing from retrieve_results is treated as "not-attempted"
    and preserved. Default-to-preserve makes the helper safe to call even if
    the retrieve loop was interrupted partway through.
    """
    if retrieve_results is None:
        raise TypeError("retrieve_results must be a dict, got None")

    box_names = {b.name for b in boxes}
    for key in retrieve_results:
        if key not in box_names:
            log(f"  unrecognized key in retrieve_results: {key} — caller bug?")

    candidates: list[Box] = []
    for box in boxes:
        status = retrieve_results.get(box.name)
        if status == "ok":
            candidates.append(box)
        else:
            status_str = status if status is not None else "not-attempted"
            log(f"  PRESERVED {box.name} ip={box.ipv4} retrieve_status={status_str}")

    deleted = teardown_all(provider, candidates) if candidates else 0
    return len(candidates), deleted


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
    ap.add_argument("--provider", choices=["hetzner", "vultr", "oci"], required=True)
    ap.add_argument("--boxes", type=int, required=True, help="Max boxes to request (graceful if provider gives fewer)")
    ap.add_argument("--seeds", type=int, default=20, help="Number of seeds to run (first N from DEFAULT_SEEDS)")
    ap.add_argument("--seeds-file", type=Path, default=None,
                    help="Read seed list from file (whitespace or comma separated). Overrides --seeds.")
    ap.add_argument("--vultr-plan", default=None,
                    help="Override Vultr plan id (default: first available in VultrProvider.PLAN_FALLBACKS)")
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
    if args.seeds_file:
        raw = args.seeds_file.read_text().replace(",", " ").split()
        seeds = [int(x) for x in raw if x.strip()]
        log(f"Seeds: {len(seeds)} from {args.seeds_file}")
    else:
        seeds = DEFAULT_SEEDS[: args.seeds]
    if args.vultr_plan and args.provider == "vultr":
        VultrProvider.PLAN_FALLBACKS = [args.vultr_plan] + [
            p for p in VultrProvider.PLAN_FALLBACKS if p != args.vultr_plan
        ]
        log(f"Vultr plan override: {args.vultr_plan} (falls back to {VultrProvider.PLAN_FALLBACKS[1:]})")
    log(f"Audit: {len(seeds)} seeds × {len(experiments.split(','))} experiments "
        f"on up to {args.boxes} {args.provider} boxes")

    label_prefix = args.label or f"bts-audit-{args.provider}"
    provider = make_provider(args.provider)
    args.out.mkdir(parents=True, exist_ok=True)

    boxes: list[Box] = []
    retrieve_results: dict[str, str] = {}
    exit_code = 0
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

        provision_parallelism = min(3, len(boxes))
        log(f"Provisioning {len(boxes)} boxes (parallelism={provision_parallelism}, "
            f"capped to avoid upload-bandwidth saturation)...")
        ready_boxes: list[Box] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=provision_parallelism) as ex:
            futures = {ex.submit(provision_one, b): b for b in boxes}
            for fut in concurrent.futures.as_completed(futures):
                b = futures[fut]
                try:
                    name, status = fut.result()
                    log(f"  [{name}] {status}")
                    if status == "ready":
                        ready_boxes.append(b)
                    else:
                        log(f"  [{name}] NOT READY — dropping from fleet (others continue)")
                except Exception as e:
                    log(f"  [{b.name}] provision EXCEPTION: {type(e).__name__}: {str(e)[:200]} — dropping")
        if not ready_boxes:
            log("No boxes became ready — aborting before seed launch")
            return
        if len(ready_boxes) < len(boxes):
            log(f"Continuing with {len(ready_boxes)}/{len(boxes)} ready boxes")
            # Tear down the unready ones now so they don't eat the monthly fee cap.
            unready = [b for b in boxes if b not in ready_boxes]
            for b in unready:
                try:
                    provider.delete(b.id)
                    log(f"  released unready {b.name}")
                except Exception as e:
                    log(f"  FAILED to release {b.name}: {e}")
            boxes = ready_boxes

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
                retrieve_results[nm] = status
                if errs:
                    log(f"  [{nm}] {status}  errs={errs[:1]}")
                else:
                    log(f"  [{nm}] {status}")
    finally:
        if not boxes:
            pass
        else:
            selected, deleted = teardown_retrieved(provider, boxes, retrieve_results)
            preserved = len(boxes) - selected      # held back for data-integrity
            api_failed = selected - deleted         # provider API transient failures
            log(f"=== TEARDOWN: selected={selected}/{len(boxes)} "
                f"deleted={deleted} preserved={preserved} api_failed={api_failed} ===")
            if preserved > 0:
                exit_code = 1  # data-integrity signal for external monitoring

    log("=== AUDIT DRIVER DONE ===")
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
