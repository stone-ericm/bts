"""Tests for R2 sync module using moto to mock S3."""
import hashlib
import json
import os

# Tell moto to treat the R2 endpoint hostname as an S3-compatible custom
# endpoint and rewrite it to s3.amazonaws.com before interception. This MUST
# be set before `moto` is imported; otherwise moto's URL patterns (which only
# match *.amazonaws.com) won't intercept R2 traffic and tests will fall
# through to the real network and time out.
os.environ.setdefault(
    "MOTO_S3_CUSTOM_ENDPOINTS",
    "testaccount.r2.cloudflarestorage.com",
)

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
    now_iso,
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
        # Bootstrap client uses us-east-1 because moto rewrites the R2
        # endpoint to s3.amazonaws.com; create_bucket without a
        # LocationConstraint requires the us-east-1 region.
        client = boto3.client(
            "s3",
            endpoint_url="https://testaccount.r2.cloudflarestorage.com",
            aws_access_key_id="testkey",
            aws_secret_access_key="testsecret",
            region_name="us-east-1",
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
    sync_from_r2(
        client=client,
        processed_dir=processed_dir,
        models_dir=models_dir,
        expected_schema_version="must-match",
    )

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
