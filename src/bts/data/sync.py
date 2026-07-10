"""R2 canonical data sync for BTS cloud deployment.

Provides CLI commands to sync parquets + lookup cache + manifest between
local disk and a Cloudflare R2 bucket. Manifest records per-file SHA-256,
git SHA of producer, and schema version for drift detection.

Environment variables (all required for any R2 operation):
    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET         (defaults to "bts-backup-data")
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

DEFAULT_MANIFEST_KEY = "manifest.json"
PREV_MANIFEST_KEY = "manifest.prev.json"
DEFAULT_BUCKET = "bts-backup-data"
MANIFEST_VERSION = 1


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class R2Client:
    """Thin wrapper around boto3 S3 client configured for Cloudflare R2."""
    bucket: str
    client: object

    @classmethod
    def from_env(cls) -> "R2Client":
        account_id = os.environ.get("R2_ACCOUNT_ID")
        access_key = os.environ.get("R2_ACCESS_KEY_ID")
        secret = os.environ.get("R2_SECRET_ACCESS_KEY")
        bucket = os.environ.get("R2_BUCKET", DEFAULT_BUCKET)
        if not all([account_id, access_key, secret]):
            raise RuntimeError(
                "R2 credentials not set. Required env vars: "
                "R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY"
            )
        client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret,
            region_name="auto",
        )
        return cls(bucket=bucket, client=client)

    def upload_file(self, source: Path, key: str) -> None:
        self.client.upload_file(str(source), self.bucket, key)

    def download_file(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(dest))

    def get_object_json(self, key: str) -> Optional[dict]:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
                return None
            raise

    def put_object_json(self, key: str, data: dict) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(data, indent=2).encode(),
            ContentType="application/json",
        )

    def copy_object(self, src_key: str, dst_key: str) -> None:
        self.client.copy_object(
            Bucket=self.bucket,
            Key=dst_key,
            CopySource={"Bucket": self.bucket, "Key": src_key},
        )

    def delete_object(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def head_object(self, key: str) -> Optional[dict]:
        """Return {'size': ...} if the object exists, else None."""
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
            return {"size": response["ContentLength"]}
        except ClientError as e:
            # HEAD has no error body, so S3-compatible stores answer with
            # "404" or "NotFound" rather than NoSuchKey (Codex review).
            if e.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
                return None
            raise

    def list_objects(self, prefix: str) -> list[dict]:
        """List objects under prefix: [{'key', 'size', 'last_modified'}, ...]."""
        out: list[dict] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                out.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"],
                })
        return out


def read_manifest(client: R2Client, key: str = DEFAULT_MANIFEST_KEY) -> Optional[dict]:
    return client.get_object_json(key)


def write_manifest_atomic(client: R2Client, manifest: dict, key: str = DEFAULT_MANIFEST_KEY) -> None:
    """Write manifest via tmp-key + copy to ensure readers never see a torn state."""
    tmp_key = f"{key}.tmp"
    client.put_object_json(tmp_key, manifest)
    client.copy_object(tmp_key, key)
    client.delete_object(tmp_key)


def _current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _current_git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _storage_key(sha256: str, name: str) -> str:
    """Content-addressed object key (audit F8).

    Uploads under objects/<sha>/<name> never overwrite a key an existing
    manifest references, so a sync interrupted at ANY point leaves the old
    manifest + old objects fully consistent. The manifest flip (atomic PUT)
    is the single commit point.
    """
    return f"objects/{sha256}/{name}"


def sync_to_r2(
    client: R2Client,
    processed_dir: Path,
    models_dir: Path,
) -> dict:
    """Upload changed local files to R2 and write an updated manifest.

    Compares SHA-256 of each eligible local file against the current
    manifest's entry. Only files whose hash differs are uploaded — to a
    content-addressed key (objects/<sha>/<name>), never overwriting bytes
    a prior manifest references (audit F8). Unchanged files keep their
    prior entry verbatim (including legacy pre-F8 keys, so old layouts
    stay restorable without re-upload). The manifest is written last,
    atomically (tmp + copy + delete) — that PUT is the commit point.

    Eligible files:
    - data/processed/pa_*.parquet
    - data/models/probable_pitcher_lookup.json (if present)
    - data/models/mdp_policy.npz (if present)

    Returns the new manifest (the exact one written to R2).
    """
    from bts.data.schema import SCHEMA_VERSION

    current_manifest = read_manifest(client) or {"files": {}}

    new_files: dict[str, dict] = {}

    def process_file(local_path: Path, key: str):
        if not local_path.exists():
            return
        local_sha = sha256_file(local_path)
        size = local_path.stat().st_size
        prior = current_manifest["files"].get(key)

        if prior and prior.get("sha256") == local_sha:
            # Unchanged — keep the prior entry verbatim: uploaded_at for age
            # tracking, and its storage key (or legacy absence) untouched.
            new_files[key] = dict(prior)
            new_files[key]["size"] = size
            print(f"  skip {key} (unchanged)", file=sys.stderr)
        else:
            storage_key = _storage_key(local_sha, local_path.name)
            print(f"  upload {key} -> {storage_key} ({size / 1e6:.1f} MB)",
                  file=sys.stderr)
            client.upload_file(local_path, storage_key)
            new_files[key] = {
                "sha256": local_sha,
                "size": size,
                "uploaded_at": now_iso(),
                "key": storage_key,
            }

    # Parquets
    for parquet in sorted(processed_dir.glob("pa_*.parquet")):
        process_file(parquet, f"parquets/{parquet.name}")

    # Probable pitcher lookup (optional)
    lookup = models_dir / "probable_pitcher_lookup.json"
    if lookup.exists():
        process_file(lookup, "models/probable_pitcher_lookup.json")

    # MDP policy (optional — strategy falls back to heuristic without it)
    mdp_policy = models_dir / "mdp_policy.npz"
    if mdp_policy.exists():
        process_file(mdp_policy, "models/mdp_policy.npz")

    # Safety guard: never silently wipe the manifest. If the new manifest
    # has fewer than half the files the prior manifest had, something is
    # wrong (wrong CWD, failed build, accidental rm). Refuse unless the
    # caller explicitly opts into deletion.
    prior_count = len(current_manifest.get("files", {}))
    new_count = len(new_files)
    print(
        f"  manifest: {prior_count} → {new_count} files",
        file=sys.stderr,
    )
    if prior_count > 0 and new_count < prior_count / 2:
        raise RuntimeError(
            f"Refusing to sync: new manifest would have {new_count} files, "
            f"prior had {prior_count}. This likely means sync_to_r2 was run "
            f"from the wrong directory or local files are missing. "
            f"If this is intentional (e.g., you're decommissioning old data), "
            f"delete the R2 manifest manually first."
        )

    new_manifest = {
        "version": MANIFEST_VERSION,
        "updated_at": now_iso(),
        "updated_by": socket.gethostname(),
        "git_sha": _current_git_sha(),
        "git_branch": _current_git_branch(),
        "schema_version": SCHEMA_VERSION,
        "files": new_files,
    }
    # Preserve the outgoing manifest as manifest.prev.json BEFORE the flip:
    # prune spares objects referenced by either generation, so a restore
    # that loaded the manifest just before this sync keeps its objects
    # (Codex review I2 — one full generation of reader grace).
    if current_manifest.get("files"):
        try:
            client.copy_object(DEFAULT_MANIFEST_KEY, PREV_MANIFEST_KEY)
        except ClientError as e:
            print(f"  warn: could not preserve prev manifest: {e}", file=sys.stderr)
    write_manifest_atomic(client, new_manifest)
    return new_manifest


def sync_from_r2(
    client: R2Client,
    processed_dir: Path,
    models_dir: Path,
    expected_schema_version: Optional[str] = None,
) -> dict:
    """Download files from R2 whose local hash differs from the manifest.

    Verifies pre- and post-download:
    - Manifest branch must be 'main' (refuses to load experiment data)
    - Manifest schema_version must match expected (refuses on drift)
    - Every downloaded file must match its declared SHA-256

    Returns the manifest that was used.
    """
    from bts.data.schema import SCHEMA_VERSION

    manifest = read_manifest(client)
    if manifest is None:
        raise RuntimeError("R2 manifest.json not found — nothing to sync")

    manifest_version = manifest.get("version", 1)
    if manifest_version > MANIFEST_VERSION:
        raise RuntimeError(
            f"R2 manifest version {manifest_version} is newer than "
            f"supported ({MANIFEST_VERSION}). Upgrade bts code to read this manifest."
        )

    if manifest.get("git_branch") != "main":
        raise RuntimeError(
            f"R2 manifest is from branch '{manifest.get('git_branch')}', "
            f"not on main branch. Refusing to sync experiment data."
        )

    expected = expected_schema_version or SCHEMA_VERSION
    if manifest.get("schema_version") != expected:
        raise RuntimeError(
            f"Schema version mismatch: worker expects {expected}, "
            f"R2 manifest has {manifest.get('schema_version')}. "
            f"The producer at git_sha {manifest.get('git_sha', 'unknown')} is "
            f"out of sync with the current code. Fix: on the producer, run "
            f"'bts data build && bts data sync-to-r2'."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    for key, meta in manifest["files"].items():
        if key.startswith("parquets/"):
            dest = processed_dir / key[len("parquets/"):]
        elif key.startswith("models/"):
            dest = models_dir / key[len("models/"):]
        else:
            print(f"  skip unknown key {key}", file=sys.stderr)
            continue

        if dest.exists() and sha256_file(dest) == meta["sha256"]:
            print(f"  skip {key} (already local)", file=sys.stderr)
            continue

        # Legacy pre-F8 entries have no storage key: the object lives at the
        # manifest's logical key itself.
        storage_key = meta.get("key", key)
        print(f"  download {storage_key} ({meta['size'] / 1e6:.1f} MB)", file=sys.stderr)

        # Verify in a temp path, then atomically replace — a failed or
        # corrupt download must never destroy a good local file (audit F8).
        # Pid-suffixed so concurrent restores can't swap each other's
        # verified bytes through a shared temp name (Codex review I6).
        part = dest.with_name(f"{dest.name}.{os.getpid()}.part")
        try:
            client.download_file(key=storage_key, dest=part)
            actual = sha256_file(part)
            if actual != meta["sha256"]:
                raise RuntimeError(
                    f"Checksum mismatch for {storage_key}: "
                    f"expected {meta['sha256']}, got {actual}"
                )
            os.replace(part, dest)
        finally:
            part.unlink(missing_ok=True)

    return manifest


import tarfile


def verify_manifest(
    client: R2Client,
    expected_schema_version: Optional[str] = None,
    stale_hours: int = 48,
) -> dict:
    """Read-only check of R2 manifest state. Returns a report dict.

    Used by `bts data verify-manifest` CLI and by the tripwire mode that
    runs periodically. Does not modify any local or remote state.
    """
    from bts.data.schema import SCHEMA_VERSION

    manifest = read_manifest(client)
    if manifest is None:
        return {"exists": False, "stale": True}

    manifest_version = manifest.get("version", 1)
    version_supported = manifest_version <= MANIFEST_VERSION

    expected = expected_schema_version or SCHEMA_VERSION
    updated_at_str = manifest.get("updated_at")
    age_hours: Optional[float] = None
    if updated_at_str:
        updated_at = datetime.fromisoformat(updated_at_str)
        age_seconds = (datetime.now(timezone.utc) - updated_at).total_seconds()
        age_hours = age_seconds / 3600
        stale = age_hours > stale_hours
    else:
        stale = True  # Undated manifest treated as stale

    # Object-level verification (audit F8): manifest age says nothing about
    # whether the referenced bytes are actually present and sized right.
    objects_missing: list[str] = []
    objects_size_mismatch: list[str] = []
    for key, meta in manifest.get("files", {}).items():
        storage_key = meta.get("key", key)
        head = client.head_object(storage_key)
        if head is None:
            objects_missing.append(storage_key)
        elif meta.get("size") is not None and head["size"] != meta["size"]:
            objects_size_mismatch.append(storage_key)

    return {
        "exists": True,
        "version": manifest_version,
        "version_supported": version_supported,
        "branch": manifest.get("git_branch"),
        "git_sha": manifest.get("git_sha"),
        "schema_version": manifest.get("schema_version"),
        "schema_version_match": manifest.get("schema_version") == expected,
        "updated_at": updated_at_str,
        "updated_by": manifest.get("updated_by"),
        "age_hours": age_hours,
        "stale": stale,
        "n_files": len(manifest.get("files", {})),
        "objects_missing": objects_missing,
        "objects_size_mismatch": objects_size_mismatch,
        "objects_ok": not objects_missing and not objects_size_mismatch,
    }


def prune_unreferenced(
    client: R2Client,
    min_age_days: float = 7.0,
) -> dict:
    """Delete content-addressed objects the current manifest no longer references.

    Only scans the objects/ prefix — legacy-layout keys, manifest.json and
    the raw-archive tarball are structurally untouchable. The age guard keeps
    anything younger than min_age_days so an in-flight sync's fresh uploads
    (objects exist, manifest not yet flipped) can never be collected.
    """
    referenced = set()
    for manifest_key in (DEFAULT_MANIFEST_KEY, PREV_MANIFEST_KEY):
        manifest = client.get_object_json(manifest_key) or {"files": {}}
        referenced |= {
            meta["key"] for meta in manifest["files"].values() if meta.get("key")
        }

    now = datetime.now(timezone.utc)
    deleted: list[str] = []
    kept_recent: list[str] = []
    for obj in client.list_objects("objects/"):
        if obj["key"] in referenced:
            continue
        age_days = (now - obj["last_modified"]).total_seconds() / 86400
        if age_days < min_age_days:
            kept_recent.append(obj["key"])
            continue
        print(f"  prune {obj['key']} (unreferenced, {age_days:.1f}d old)",
              file=sys.stderr)
        client.delete_object(obj["key"])
        deleted.append(obj["key"])

    return {
        "deleted": sorted(deleted),
        "kept_recent": sorted(kept_recent),
        "n_referenced": len(referenced),
    }


def archive_historical_raw(
    client: R2Client,
    raw_dir: Path,
    tarball_key: str,
    exclude_seasons: set[int],
) -> None:
    """Build a tarball of historical raw JSON seasons and upload to R2.

    Excludes current season (passed explicitly) so the archive is a
    stable snapshot of historical data that rarely needs refresh.
    """
    print(f"Building tarball of historical raw data from {raw_dir}...", file=sys.stderr)

    # Use a temp file on disk (safer than in-memory for large archives)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with tarfile.open(tmp_path, "w:gz") as tar:
            for season_dir in sorted(raw_dir.iterdir()):
                if not season_dir.is_dir():
                    continue
                try:
                    season = int(season_dir.name)
                except ValueError:
                    continue
                if season in exclude_seasons:
                    print(f"  skip season {season} (excluded)", file=sys.stderr)
                    continue
                print(f"  add season {season}", file=sys.stderr)
                tar.add(season_dir, arcname=season_dir.name)

        size_mb = tmp_path.stat().st_size / 1e6
        print(f"Uploading {tarball_key} ({size_mb:.1f} MB) to R2...", file=sys.stderr)
        client.upload_file(tmp_path, tarball_key)
        print("  done", file=sys.stderr)
    finally:
        tmp_path.unlink(missing_ok=True)
