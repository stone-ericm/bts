# BTS Cloud Migration — Plan 02: R2 Canonical Data Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the R2 data layer as the canonical store for parquets and the incremental lookup cache. Add schema versioning derived from `PA_COLUMNS` so workers can verify compatibility before loading data. Produce `bts data sync-to-r2`, `sync-from-r2`, `verify-manifest`, and `archive-historical-raw` CLI commands.

**Architecture:** Cloudflare R2 bucket holds parquets + lookup cache + a `manifest.json` index file that records per-file SHA-256, git SHA of producer, and schema version. Uploads are incremental (changed files only) with atomic manifest swap via temp-file + copy-object. Downloads verify checksums before accepting. Schema version auto-derived from `PA_COLUMNS` via SHA-256 hash.

**Tech Stack:** Python 3.12, `boto3` (S3-compatible client for R2), Click for CLI, hashlib, tarfile for historical raw archive.

**Dependencies on other plans:** None. Independently testable using moto (S3 mock) for unit tests and a real R2 bucket for integration smoke tests.

**Parent spec:** `docs/superpowers/specs/2026-04-09-bts-cloud-migration-design.md` (§ R2 canonical data layer, § Schema versioning)

---

## File Structure

- Create `src/bts/data/sync.py` — R2 client + sync logic + manifest handling, ~350 lines
- Modify `src/bts/data/schema.py` — add `SCHEMA_VERSION` derived from `PA_COLUMNS`
- Modify `src/bts/data/build.py` — add runtime assert that output columns match `PA_COLUMNS`
- Modify `src/bts/cli.py` — register four new commands
- Modify `pyproject.toml` — add `boto3` as main dep (needed by the Fly runtime)
- Create `tests/test_sync.py` — ~200 lines
- Create `tests/test_schema_version.py` — ~50 lines
- Create `tests/test_build_schema_assert.py` — ~80 lines

---

### Task 1: Add boto3 dependency and SCHEMA_VERSION

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/bts/data/schema.py`
- Create: `tests/test_schema_version.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_schema_version.py`:

```python
"""Tests for SCHEMA_VERSION derivation."""
import hashlib

from bts.data.schema import PA_COLUMNS, SCHEMA_VERSION


def test_schema_version_is_12_char_hex():
    assert len(SCHEMA_VERSION) == 12
    assert all(c in "0123456789abcdef" for c in SCHEMA_VERSION)


def test_schema_version_matches_pa_columns_hash():
    expected = hashlib.sha256("\n".join(PA_COLUMNS).encode()).hexdigest()[:12]
    assert SCHEMA_VERSION == expected


def test_schema_version_is_deterministic():
    # Re-import and verify the same value
    from bts.data import schema as s1
    from bts.data import schema as s2
    assert s1.SCHEMA_VERSION == s2.SCHEMA_VERSION


def test_schema_version_changes_when_columns_change():
    # Simulate column addition
    mutated = PA_COLUMNS + ["new_column"]
    new_version = hashlib.sha256("\n".join(mutated).encode()).hexdigest()[:12]
    assert new_version != SCHEMA_VERSION
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_schema_version.py -v
```

Expected: All tests FAIL with `ImportError: cannot import name 'SCHEMA_VERSION'`.

- [ ] **Step 3: Add SCHEMA_VERSION to schema.py**

Modify `src/bts/data/schema.py`. At the bottom of the file, add:

```python
import hashlib

# Auto-derived schema version. Any change to PA_COLUMNS (addition, removal,
# rename, reorder) produces a new version. Workers compare their expected
# SCHEMA_VERSION against the one in R2 manifest.json and refuse to load
# parquets on mismatch. This makes schema drift detectable at sync time.
SCHEMA_VERSION = hashlib.sha256("\n".join(PA_COLUMNS).encode()).hexdigest()[:12]
```

- [ ] **Step 4: Add boto3 to pyproject.toml**

Modify `pyproject.toml`. In the `[project]` dependencies list, add `"boto3"` (it's needed for R2 sync which is part of the main runtime, not an optional extra).

- [ ] **Step 5: Sync dependencies and run tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_schema_version.py -v
```

Expected: All four tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/bts/data/schema.py tests/test_schema_version.py
git commit -m "feat(data): add SCHEMA_VERSION derived from PA_COLUMNS"
```

---

### Task 2: Runtime assert in `build_season`

**Files:**
- Modify: `src/bts/data/build.py`
- Create: `tests/test_build_schema_assert.py`

- [ ] **Step 1: Read current build_season implementation to understand structure**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -c "from bts.data.build import build_season; help(build_season)"
```

You need to see where the DataFrame is constructed before `.to_parquet()` is called. The assert goes immediately before the write.

- [ ] **Step 2: Write the failing test**

Create `tests/test_build_schema_assert.py`:

```python
"""Tests for build_season runtime schema assertion."""
import json
from pathlib import Path

import pandas as pd
import pytest

from bts.data.schema import PA_COLUMNS
from bts.data.build import assert_columns_match_schema


def test_assert_passes_when_columns_match():
    df = pd.DataFrame({col: [] for col in PA_COLUMNS})
    # Should not raise
    assert_columns_match_schema(df)


def test_assert_fails_when_column_missing():
    cols = [c for c in PA_COLUMNS if c != "is_hit"]
    df = pd.DataFrame({col: [] for col in cols})
    with pytest.raises(RuntimeError, match=r"missing columns.*is_hit"):
        assert_columns_match_schema(df)


def test_assert_fails_when_extra_column():
    cols = PA_COLUMNS + ["unexpected_col"]
    df = pd.DataFrame({col: [] for col in cols})
    with pytest.raises(RuntimeError, match=r"extra columns.*unexpected_col"):
        assert_columns_match_schema(df)


def test_assert_error_message_includes_pa_columns_hint():
    df = pd.DataFrame({col: [] for col in PA_COLUMNS[:-2]})
    with pytest.raises(RuntimeError, match=r"Update PA_COLUMNS"):
        assert_columns_match_schema(df)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_build_schema_assert.py -v
```

Expected: Tests FAIL with `ImportError: cannot import name 'assert_columns_match_schema'`.

- [ ] **Step 4: Implement the assert function**

Add to `src/bts/data/build.py` near the top (after imports, before `build_season`):

```python
from bts.data.schema import PA_COLUMNS


def assert_columns_match_schema(df):
    """Assert that a DataFrame's columns exactly match PA_COLUMNS.

    Fails loudly with a specific diff if not. This is the schema
    guard that prevents silent drift: any change to build logic
    that affects the parquet's columns MUST also update PA_COLUMNS
    in src/bts/data/schema.py or this assert fires.
    """
    actual = set(df.columns)
    expected = set(PA_COLUMNS)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise RuntimeError(
        f"Schema drift in build_season:\n"
        f"  missing columns: {missing}\n"
        f"  extra columns: {extra}\n"
        f"Update PA_COLUMNS in src/bts/data/schema.py to match "
        f"the new schema, which will also update SCHEMA_VERSION."
    )
```

Then, inside the existing `build_season` function, immediately before the `df.to_parquet(output_path)` call, add:

```python
    assert_columns_match_schema(df)
    df = df[PA_COLUMNS]  # Enforce deterministic column order
```

- [ ] **Step 5: Run tests to verify**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_build_schema_assert.py -v
```

Expected: All four tests PASS.

- [ ] **Step 6: Regression test — make sure existing build still works**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -v -k "build"
```

Expected: All existing build-related tests PASS. If any fail, your assert logic may be rejecting a legitimate DataFrame — investigate before proceeding.

- [ ] **Step 7: Commit**

```bash
git add src/bts/data/build.py tests/test_build_schema_assert.py
git commit -m "feat(build): assert DataFrame columns match PA_COLUMNS before write"
```

---

### Task 3: R2 client factory and basic operations

**Files:**
- Create: `src/bts/data/sync.py`
- Create: `tests/test_sync.py`

- [ ] **Step 1: Install moto dev dependency for S3 mocking**

Modify `pyproject.toml`. Under `[dependency-groups]` or `[project.optional-dependencies]` (whichever your project uses for dev deps), add `"moto[s3]"` to the dev group.

Run:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --all-extras
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_sync.py`:

```python
"""Tests for R2 sync module using moto to mock S3."""
import json
import os
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from bts.data.sync import (
    R2Client,
    sha256_file,
    read_manifest,
    write_manifest_atomic,
    DEFAULT_MANIFEST_KEY,
)


@pytest.fixture(autouse=True)
def aws_credentials(monkeypatch):
    """Mocked AWS credentials for moto tests."""
    monkeypatch.setenv("R2_ACCOUNT_ID", "testaccount")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "testkey")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "testsecret")
    monkeypatch.setenv("R2_BUCKET", "test-bucket")


@pytest.fixture
def mock_bucket():
    with mock_aws():
        client = boto3.client(
            "s3",
            endpoint_url="https://testaccount.r2.cloudflarestorage.com",
            aws_access_key_id="testkey",
            aws_secret_access_key="testsecret",
            region_name="auto",
        )
        client.create_bucket(Bucket="test-bucket")
        yield client


def test_sha256_file(tmp_path):
    f = tmp_path / "sample.bin"
    f.write_bytes(b"hello world")
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert sha256_file(f) == expected


def test_r2client_uploads_and_downloads_file(mock_bucket, tmp_path):
    client = R2Client.from_env()
    source = tmp_path / "source.txt"
    source.write_text("hello r2")

    client.upload_file(source, key="test/source.txt")
    downloaded = tmp_path / "downloaded.txt"
    client.download_file(key="test/source.txt", dest=downloaded)
    assert downloaded.read_text() == "hello r2"


def test_write_and_read_manifest(mock_bucket, tmp_path):
    client = R2Client.from_env()
    manifest = {
        "version": 1,
        "updated_at": "2026-04-09T12:00:00Z",
        "git_sha": "abc123",
        "git_branch": "main",
        "schema_version": "0123456789ab",
        "updated_by": "test-host",
        "files": {
            "parquets/pa_2026.parquet": {
                "sha256": "deadbeef" * 8,
                "size": 1024,
                "uploaded_at": "2026-04-09T12:00:00Z",
            },
        },
    }
    write_manifest_atomic(client, manifest)
    loaded = read_manifest(client)
    assert loaded == manifest


def test_read_manifest_returns_none_when_missing(mock_bucket):
    client = R2Client.from_env()
    assert read_manifest(client) is None


def test_write_manifest_atomic_cleans_up_tmp(mock_bucket, tmp_path):
    client = R2Client.from_env()
    write_manifest_atomic(client, {"version": 1, "files": {}})
    # After write, the tmp key should not exist
    s3 = mock_bucket
    objs = s3.list_objects_v2(Bucket="test-bucket")
    keys = {obj["Key"] for obj in objs.get("Contents", [])}
    assert DEFAULT_MANIFEST_KEY in keys
    assert f"{DEFAULT_MANIFEST_KEY}.tmp" not in keys
```

- [ ] **Step 3: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sync.py -v
```

Expected: All tests FAIL with `ModuleNotFoundError: No module named 'bts.data.sync'`.

- [ ] **Step 4: Write the implementation**

Create `src/bts/data/sync.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sync.py -v
```

Expected: All five tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/bts/data/sync.py tests/test_sync.py pyproject.toml uv.lock
git commit -m "feat(sync): add R2Client + manifest read/write with atomic swap"
```

---

### Task 4: `sync-to-r2` — upload local data with incremental diff

**Files:**
- Modify: `src/bts/data/sync.py`
- Modify: `tests/test_sync.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sync.py`:

```python
def test_sync_to_r2_uploads_parquets_and_writes_manifest(mock_bucket, tmp_path, monkeypatch):
    # Create fake local parquets and lookup
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    models_dir.mkdir()

    (processed_dir / "pa_2017.parquet").write_bytes(b"fake-2017-data")
    (processed_dir / "pa_2026.parquet").write_bytes(b"fake-2026-data")
    (models_dir / "probable_pitcher_lookup.json").write_text('{"a": 1}')

    from bts.data.sync import sync_to_r2
    client = R2Client.from_env()
    manifest = sync_to_r2(
        client=client,
        processed_dir=processed_dir,
        models_dir=models_dir,
    )

    assert manifest["version"] == 1
    assert "parquets/pa_2017.parquet" in manifest["files"]
    assert "parquets/pa_2026.parquet" in manifest["files"]
    assert "models/probable_pitcher_lookup.json" in manifest["files"]
    assert manifest["schema_version"]  # Non-empty

    # Verify files are actually in the bucket
    s3 = mock_bucket
    obj = s3.get_object(Bucket="test-bucket", Key="parquets/pa_2017.parquet")
    assert obj["Body"].read() == b"fake-2017-data"


def test_sync_to_r2_skips_unchanged_files(mock_bucket, tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    models_dir.mkdir()

    parquet_path = processed_dir / "pa_2026.parquet"
    parquet_path.write_bytes(b"same-data")
    (models_dir / "probable_pitcher_lookup.json").write_text('{"a": 1}')

    from bts.data.sync import sync_to_r2
    client = R2Client.from_env()

    # First sync uploads everything
    manifest1 = sync_to_r2(client=client, processed_dir=processed_dir, models_dir=models_dir)
    first_uploaded_at = manifest1["files"]["parquets/pa_2026.parquet"]["uploaded_at"]

    # Second sync with no changes should preserve original uploaded_at
    manifest2 = sync_to_r2(client=client, processed_dir=processed_dir, models_dir=models_dir)
    assert manifest2["files"]["parquets/pa_2026.parquet"]["uploaded_at"] == first_uploaded_at
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sync.py::test_sync_to_r2_uploads_parquets_and_writes_manifest -v
```

Expected: FAIL with `ImportError: cannot import name 'sync_to_r2'`.

- [ ] **Step 3: Write the implementation**

Append to `src/bts/data/sync.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sync.py -v
```

Expected: All tests in test_sync.py PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/data/sync.py tests/test_sync.py
git commit -m "feat(sync): add sync_to_r2 with incremental upload + atomic manifest"
```

---

### Task 5: `sync-from-r2` — download with checksum verification

**Files:**
- Modify: `src/bts/data/sync.py`
- Modify: `tests/test_sync.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sync.py`:

```python
def test_sync_from_r2_downloads_and_verifies_checksums(mock_bucket, tmp_path):
    # Populate R2 with a known parquet and a manifest
    client = R2Client.from_env()
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    models_dir.mkdir()

    # Upload directly via mock_bucket to avoid using sync_to_r2
    fake_parquet = b"fake-parquet-bytes"
    expected_sha = hashlib.sha256(fake_parquet).hexdigest()
    mock_bucket.put_object(
        Bucket="test-bucket",
        Key="parquets/pa_2026.parquet",
        Body=fake_parquet,
    )

    manifest = {
        "version": 1,
        "updated_at": now_iso(),
        "updated_by": "test",
        "git_sha": "abc",
        "git_branch": "main",
        "schema_version": "must-match",
        "files": {
            "parquets/pa_2026.parquet": {
                "sha256": expected_sha,
                "size": len(fake_parquet),
                "uploaded_at": now_iso(),
            },
        },
    }
    mock_bucket.put_object(Bucket="test-bucket", Key="manifest.json", Body=json.dumps(manifest).encode())

    from bts.data.sync import sync_from_r2
    # Override the expected schema version via monkeypatch so the test passes
    import bts.data.sync as sync_module
    original = sync_module.SCHEMA_VERSION_FALLBACK
    sync_module.SCHEMA_VERSION_FALLBACK = "must-match"
    try:
        sync_from_r2(
            client=client,
            processed_dir=processed_dir,
            models_dir=models_dir,
            expected_schema_version="must-match",
        )
    finally:
        sync_module.SCHEMA_VERSION_FALLBACK = original

    downloaded = processed_dir / "pa_2026.parquet"
    assert downloaded.exists()
    assert downloaded.read_bytes() == fake_parquet


def test_sync_from_r2_rejects_schema_version_mismatch(mock_bucket, tmp_path):
    client = R2Client.from_env()

    manifest = {
        "version": 1,
        "schema_version": "old-version",
        "git_sha": "xyz",
        "git_branch": "main",
        "files": {},
    }
    mock_bucket.put_object(Bucket="test-bucket", Key="manifest.json", Body=json.dumps(manifest).encode())

    from bts.data.sync import sync_from_r2

    with pytest.raises(RuntimeError, match="Schema version mismatch"):
        sync_from_r2(
            client=client,
            processed_dir=tmp_path / "p",
            models_dir=tmp_path / "m",
            expected_schema_version="new-version",
        )


def test_sync_from_r2_rejects_non_main_branch(mock_bucket, tmp_path):
    client = R2Client.from_env()

    manifest = {
        "version": 1,
        "schema_version": "ok",
        "git_sha": "xyz",
        "git_branch": "feature/experiment",
        "files": {},
    }
    mock_bucket.put_object(Bucket="test-bucket", Key="manifest.json", Body=json.dumps(manifest).encode())

    from bts.data.sync import sync_from_r2
    with pytest.raises(RuntimeError, match="not on main branch"):
        sync_from_r2(
            client=client,
            processed_dir=tmp_path / "p",
            models_dir=tmp_path / "m",
            expected_schema_version="ok",
        )
```

Add `import hashlib` to the top of `tests/test_sync.py` if not already present.

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sync.py -v -k "sync_from_r2"
```

Expected: Tests FAIL with `ImportError: cannot import name 'sync_from_r2'`.

- [ ] **Step 3: Write the implementation**

Append to `src/bts/data/sync.py`:

```python
# Module-level fallback used when schema version isn't passed explicitly.
# Re-assigned in tests when needed.
SCHEMA_VERSION_FALLBACK = None  # Set on first import below


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

    if manifest.get("git_branch") != "main":
        raise RuntimeError(
            f"R2 manifest is from branch '{manifest.get('git_branch')}', "
            f"not main. Refusing to sync experiment data."
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

        print(f"  download {key} ({meta['size'] / 1e6:.1f} MB)", file=sys.stderr)
        client.download_file(key=key, dest=dest)

        actual = sha256_file(dest)
        if actual != meta["sha256"]:
            dest.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {key}: expected {meta['sha256']}, got {actual}"
            )

    return manifest
```

- [ ] **Step 4: Run tests to verify**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sync.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/data/sync.py tests/test_sync.py
git commit -m "feat(sync): add sync_from_r2 with checksum + version verification"
```

---

### Task 6: `verify-manifest` and `archive-historical-raw`

**Files:**
- Modify: `src/bts/data/sync.py`
- Modify: `tests/test_sync.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sync.py`:

```python
def test_verify_manifest_reports_age(mock_bucket):
    client = R2Client.from_env()
    manifest = {
        "version": 1,
        "schema_version": "ok",
        "git_branch": "main",
        "git_sha": "abc",
        "updated_at": "2026-04-09T00:00:00+00:00",
        "files": {},
    }
    mock_bucket.put_object(Bucket="test-bucket", Key="manifest.json", Body=json.dumps(manifest).encode())

    from bts.data.sync import verify_manifest
    report = verify_manifest(client=client, expected_schema_version="ok")
    assert report["branch"] == "main"
    assert report["schema_version_match"] is True
    assert "age_hours" in report


def test_verify_manifest_flags_stale(mock_bucket):
    client = R2Client.from_env()
    # Manifest from several days ago
    manifest = {
        "version": 1,
        "schema_version": "ok",
        "git_branch": "main",
        "git_sha": "abc",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "files": {},
    }
    mock_bucket.put_object(Bucket="test-bucket", Key="manifest.json", Body=json.dumps(manifest).encode())

    from bts.data.sync import verify_manifest
    report = verify_manifest(client=client, expected_schema_version="ok", stale_hours=24)
    assert report["stale"] is True


def test_archive_historical_raw(tmp_path, mock_bucket):
    # Create a fake raw directory with historical seasons
    raw_dir = tmp_path / "raw"
    for season in [2017, 2018, 2019]:
        season_dir = raw_dir / str(season)
        season_dir.mkdir(parents=True)
        (season_dir / "game1.json").write_text(f'{{"season": {season}}}')

    # Current season — should be excluded
    current = raw_dir / "2026"
    current.mkdir()
    (current / "game2.json").write_text('{"season": 2026}')

    client = R2Client.from_env()
    from bts.data.sync import archive_historical_raw

    archive_historical_raw(
        client=client,
        raw_dir=raw_dir,
        tarball_key="raw-archive-2017-2025.tar.gz",
        exclude_seasons={2026},
    )

    # Verify the tarball exists in R2
    obj = mock_bucket.get_object(Bucket="test-bucket", Key="raw-archive-2017-2025.tar.gz")
    assert obj["ContentLength"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sync.py -v -k "verify_manifest or archive_historical"
```

Expected: FAIL with `ImportError: cannot import name 'verify_manifest'`.

- [ ] **Step 3: Write the implementation**

Append to `src/bts/data/sync.py`:

```python
import io
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

    expected = expected_schema_version or SCHEMA_VERSION
    updated_at_str = manifest.get("updated_at")
    age_hours: Optional[float] = None
    stale = False
    if updated_at_str:
        updated_at = datetime.fromisoformat(updated_at_str)
        age_seconds = (datetime.now(timezone.utc) - updated_at).total_seconds()
        age_hours = age_seconds / 3600
        stale = age_hours > stale_hours

    return {
        "exists": True,
        "branch": manifest.get("git_branch"),
        "git_sha": manifest.get("git_sha"),
        "schema_version": manifest.get("schema_version"),
        "schema_version_match": manifest.get("schema_version") == expected,
        "updated_at": updated_at_str,
        "updated_by": manifest.get("updated_by"),
        "age_hours": age_hours,
        "stale": stale,
        "n_files": len(manifest.get("files", {})),
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
    Streams the tarball to R2 in memory (fine for ~15 GB with enough RAM,
    or use a temp file for very constrained environments).
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
```

- [ ] **Step 4: Run tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_sync.py -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/data/sync.py tests/test_sync.py
git commit -m "feat(sync): add verify_manifest and archive_historical_raw"
```

---

### Task 7: Register CLI commands

**Files:**
- Modify: `src/bts/cli.py`

- [ ] **Step 1: Add the four new commands**

Add to `src/bts/cli.py` inside the `data` group:

```python
@data.command(name="sync-to-r2")
def data_sync_to_r2():
    """Upload local parquets + lookup cache to R2, atomically updating manifest."""
    from pathlib import Path
    from bts.data.sync import R2Client, sync_to_r2

    client = R2Client.from_env()
    manifest = sync_to_r2(
        client=client,
        processed_dir=Path("data/processed"),
        models_dir=Path("data/models"),
    )
    click.echo(f"Sync complete: {len(manifest['files'])} files, schema={manifest['schema_version']}")


@data.command(name="sync-from-r2")
def data_sync_from_r2():
    """Download parquets + lookup cache from R2, verifying checksums."""
    from pathlib import Path
    from bts.data.sync import R2Client, sync_from_r2

    client = R2Client.from_env()
    manifest = sync_from_r2(
        client=client,
        processed_dir=Path("data/processed"),
        models_dir=Path("data/models"),
    )
    click.echo(
        f"Sync complete: {len(manifest['files'])} files, "
        f"git_sha={manifest.get('git_sha', 'unknown')[:12]}"
    )


@data.command(name="verify-manifest")
def data_verify_manifest():
    """Check R2 manifest state without modifying anything (tripwire mode)."""
    from bts.data.sync import R2Client, verify_manifest

    client = R2Client.from_env()
    report = verify_manifest(client)
    if not report["exists"]:
        click.echo("Manifest not found in R2.", err=True)
        raise SystemExit(2)
    click.echo(f"branch:         {report['branch']}")
    click.echo(f"git_sha:        {report['git_sha']}")
    click.echo(f"schema_version: {report['schema_version']} "
               f"{'OK' if report['schema_version_match'] else 'MISMATCH'}")
    click.echo(f"updated_at:     {report['updated_at']} ({report['age_hours']:.1f}h ago)")
    click.echo(f"n_files:        {report['n_files']}")
    click.echo(f"stale:          {report['stale']}")
    if report['stale'] or not report['schema_version_match']:
        raise SystemExit(1)


@data.command(name="archive-historical-raw")
@click.option("--raw-dir", default="data/raw", type=click.Path(exists=True))
@click.option("--exclude-season", multiple=True, type=int, default=[2026])
@click.option("--tarball-key", default="raw-archive-2017-2025.tar.gz")
def data_archive_historical_raw(raw_dir, exclude_season, tarball_key):
    """One-shot: tar historical raw JSON and upload to R2 as cold archive."""
    from pathlib import Path
    from bts.data.sync import R2Client, archive_historical_raw

    client = R2Client.from_env()
    archive_historical_raw(
        client=client,
        raw_dir=Path(raw_dir),
        tarball_key=tarball_key,
        exclude_seasons=set(exclude_season),
    )
    click.echo(f"Archive uploaded: {tarball_key}")
```

- [ ] **Step 2: Smoke test (requires real R2 credentials)**

```bash
# Set R2 env vars in your shell
export R2_ACCOUNT_ID=$(security find-generic-password -a "claude-cli" -s "cloudflare-r2-bts-backup-account-id" -w)
export R2_ACCESS_KEY_ID=$(security find-generic-password -a "claude-cli" -s "cloudflare-r2-bts-backup-access-key-id" -w)
export R2_SECRET_ACCESS_KEY=$(security find-generic-password -a "claude-cli" -s "cloudflare-r2-bts-backup-secret-access-key" -w)
export R2_BUCKET=bts-backup-data

UV_CACHE_DIR=/tmp/uv-cache uv run bts data sync-to-r2
UV_CACHE_DIR=/tmp/uv-cache uv run bts data verify-manifest
```

Expected: First command uploads all 10 parquets + probable pitcher lookup; second command prints the manifest summary.

**Prerequisite**: you need to have created the R2 bucket `bts-backup-data` in Cloudflare and set up an API token with read/write access. See the migration spec Phase 0 checklist.

- [ ] **Step 3: Commit**

```bash
git add src/bts/cli.py
git commit -m "feat(sync): register sync-to-r2 / sync-from-r2 / verify / archive CLI"
```

---

## Completion criteria for Plan 02

- [ ] All tests pass: `uv run pytest tests/test_sync.py tests/test_schema_version.py tests/test_build_schema_assert.py -v`
- [ ] `SCHEMA_VERSION` is derived from `PA_COLUMNS` and changes only when columns change
- [ ] `build_season` fails loudly if the DataFrame columns don't match `PA_COLUMNS`
- [ ] `bts data sync-to-r2` successfully uploads local parquets to the real R2 bucket
- [ ] `bts data verify-manifest` reports the uploaded manifest accurately
- [ ] `bts data sync-from-r2` works (tested by running it on a clean directory and verifying parquets are downloaded with matching checksums)
- [ ] `bts data archive-historical-raw` can build and upload a tarball (may skip this step if the bucket has a cost concern — it's a one-time operation for Phase 0 of the cutover)

**Next plan:** `03-state.md` — State management (export, regenerate, verify). Independent of this plan.
