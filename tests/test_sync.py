"""Tests for R2 sync module using moto to mock S3."""
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
