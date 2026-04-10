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
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
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


def sync_to_r2(
    client: R2Client,
    processed_dir: Path,
    models_dir: Path,
) -> dict:
    """Upload changed local files to R2 and write an updated manifest.

    Compares SHA-256 of each eligible local file against the current
    manifest's entry. Only files whose hash differs are uploaded.
    Unchanged files keep their original uploaded_at timestamp.
    Writes the manifest last, atomically (tmp + copy + delete).

    Eligible files:
    - data/processed/pa_*.parquet
    - data/models/probable_pitcher_lookup.json (if present)

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
            # Unchanged — keep original uploaded_at for accurate age tracking
            new_files[key] = {
                "sha256": local_sha,
                "size": size,
                "uploaded_at": prior["uploaded_at"],
            }
            print(f"  skip {key} (unchanged)", file=sys.stderr)
        else:
            print(f"  upload {key} ({size / 1e6:.1f} MB)", file=sys.stderr)
            client.upload_file(local_path, key)
            new_files[key] = {
                "sha256": local_sha,
                "size": size,
                "uploaded_at": now_iso(),
            }

    # Parquets
    for parquet in sorted(processed_dir.glob("pa_*.parquet")):
        process_file(parquet, f"parquets/{parquet.name}")

    # Probable pitcher lookup (optional)
    lookup = models_dir / "probable_pitcher_lookup.json"
    if lookup.exists():
        process_file(lookup, "models/probable_pitcher_lookup.json")

    new_manifest = {
        "version": MANIFEST_VERSION,
        "updated_at": now_iso(),
        "updated_by": socket.gethostname(),
        "git_sha": _current_git_sha(),
        "git_branch": _current_git_branch(),
        "schema_version": SCHEMA_VERSION,
        "files": new_files,
    }
    write_manifest_atomic(client, new_manifest)
    return new_manifest
