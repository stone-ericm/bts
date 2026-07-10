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

    # Verify files are actually in the bucket at their content-addressed
    # keys (F8: fresh uploads no longer live at the legacy logical key)
    s3 = mock_bucket
    storage_key = manifest["files"]["parquets/pa_2017.parquet"]["key"]
    obj = s3.get_object(Bucket="test-bucket", Key=storage_key)
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


def test_sync_to_r2_refuses_to_wipe_manifest(mock_bucket, tmp_path):
    """Guard: sync_to_r2 refuses if new manifest has < half the prior files."""
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    processed.mkdir()
    models.mkdir()
    # Write 4 fake parquets
    for yr in [2023, 2024, 2025, 2026]:
        (processed / f"pa_{yr}.parquet").write_bytes(f"data-{yr}".encode())

    from bts.data.sync import sync_to_r2
    client = R2Client.from_env()
    # First sync establishes a baseline manifest with 4 entries
    sync_to_r2(client=client, processed_dir=processed, models_dir=models)

    # Simulate wrong-cwd: point at an empty directory
    empty = tmp_path / "empty"
    empty.mkdir()
    empty_models = tmp_path / "empty_models"
    empty_models.mkdir()

    with pytest.raises(RuntimeError, match="Refusing to sync"):
        sync_to_r2(client=client, processed_dir=empty, models_dir=empty_models)


def test_sync_from_r2_rejects_newer_manifest_version(mock_bucket, tmp_path):
    client = R2Client.from_env()
    manifest = {
        "version": 999,  # Pretend future version
        "schema_version": "ok",
        "git_branch": "main",
        "git_sha": "xyz",
        "files": {},
    }
    mock_bucket.put_object(Bucket="test-bucket", Key="manifest.json",
                           Body=json.dumps(manifest).encode())

    from bts.data.sync import sync_from_r2
    with pytest.raises(RuntimeError, match="newer than supported"):
        sync_from_r2(
            client=client,
            processed_dir=tmp_path / "p",
            models_dir=tmp_path / "m",
            expected_schema_version="ok",
        )


# --------------------------------------------------------------------------
# F8 (2026-07-09 audit): the old protocol overwrote stable keys in place
# BEFORE replacing the manifest, so an interrupted sync left the old manifest
# pointing at new bytes — the backup was inconsistent during exactly the
# failure window it exists to survive. Content-addressed keys make uploads
# non-destructive; the manifest flip stays the single atomic commit point.

def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _mk_dirs(tmp_path):
    processed = tmp_path / "processed"; processed.mkdir()
    models = tmp_path / "models"; models.mkdir()
    return processed, models


def test_sync_to_r2_uploads_content_addressed_keys(mock_bucket, tmp_path):
    from bts.data.sync import sync_to_r2
    processed, models = _mk_dirs(tmp_path)
    data = b"fresh-2026-data"
    (processed / "pa_2026.parquet").write_bytes(data)

    manifest = sync_to_r2(client=R2Client.from_env(), processed_dir=processed, models_dir=models)

    entry = manifest["files"]["parquets/pa_2026.parquet"]
    expected_key = f"objects/{_sha(data)}/pa_2026.parquet"
    assert entry["key"] == expected_key
    obj = mock_bucket.get_object(Bucket="test-bucket", Key=expected_key)
    assert obj["Body"].read() == data


def test_sync_to_r2_change_never_overwrites_old_object(mock_bucket, tmp_path):
    from bts.data.sync import sync_to_r2
    processed, models = _mk_dirs(tmp_path)
    pq = processed / "pa_2026.parquet"

    pq.write_bytes(b"version-one")
    sync_to_r2(client=R2Client.from_env(), processed_dir=processed, models_dir=models)
    old_key = f"objects/{_sha(b'version-one')}/pa_2026.parquet"

    pq.write_bytes(b"version-two")
    manifest = sync_to_r2(client=R2Client.from_env(), processed_dir=processed, models_dir=models)

    # old bytes still intact at the old key; manifest points at the new key
    old = mock_bucket.get_object(Bucket="test-bucket", Key=old_key)
    assert old["Body"].read() == b"version-one"
    assert manifest["files"]["parquets/pa_2026.parquet"]["key"] == (
        f"objects/{_sha(b'version-two')}/pa_2026.parquet"
    )


def test_interrupted_sync_leaves_old_manifest_fully_restorable(mock_bucket, tmp_path, monkeypatch):
    """THE F8 scenario: uploads succeed, manifest publish fails → a fresh
    restore from the surviving old manifest must still reproduce the old
    bytes exactly (previously the old manifest pointed at new bytes)."""
    from bts.data import sync as sync_mod
    # Pin the provenance stamp: this test round-trips a manifest built by the
    # real sync_to_r2 through sync_from_r2, whose main-branch refusal would
    # otherwise trip on CI checkouts of the deploy branch (2026-07-10 deploy
    # gate failure). The refusal itself has its own dedicated test.
    monkeypatch.setattr(sync_mod, "_current_git_branch", lambda: "main")
    processed, models = _mk_dirs(tmp_path)
    pq = processed / "pa_2026.parquet"

    pq.write_bytes(b"committed-state")
    sync_mod.sync_to_r2(client=R2Client.from_env(), processed_dir=processed, models_dir=models)

    pq.write_bytes(b"torn-state")
    def boom(*a, **k):
        raise RuntimeError("simulated crash before manifest publish")
    # A scoped MonkeyPatch: undoing the fixture-level `monkeypatch` would
    # also strip the autouse R2 credential env vars.
    mp = pytest.MonkeyPatch()
    mp.setattr(sync_mod, "write_manifest_atomic", boom)
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            sync_mod.sync_to_r2(client=R2Client.from_env(), processed_dir=processed, models_dir=models)
    finally:
        mp.undo()

    restore_p = tmp_path / "restore_p"; restore_m = tmp_path / "restore_m"
    sync_mod.sync_from_r2(
        client=R2Client.from_env(), processed_dir=restore_p, models_dir=restore_m,
        expected_schema_version=None,
    )
    assert (restore_p / "pa_2026.parquet").read_bytes() == b"committed-state"


def test_sync_from_r2_corrupt_download_preserves_existing_file(mock_bucket, tmp_path):
    from bts.data.sync import sync_from_r2
    processed, models = _mk_dirs(tmp_path)
    good = b"known-good-local"
    dest = processed / "pa_2026.parquet"
    dest.write_bytes(good)

    # Manifest claims a sha the stored object does not have
    claimed_sha = _sha(b"what-the-object-should-be")
    mock_bucket.put_object(
        Bucket="test-bucket",
        Key=f"objects/{claimed_sha}/pa_2026.parquet",
        Body=b"corrupted-bytes",
    )
    manifest = {
        "version": 1, "schema_version": "ok", "git_branch": "main", "git_sha": "x",
        "updated_at": now_iso(),
        "files": {"parquets/pa_2026.parquet": {
            "sha256": claimed_sha, "size": 24, "uploaded_at": now_iso(),
            "key": f"objects/{claimed_sha}/pa_2026.parquet",
        }},
    }
    mock_bucket.put_object(Bucket="test-bucket", Key="manifest.json",
                           Body=json.dumps(manifest).encode())

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        sync_from_r2(
            client=R2Client.from_env(), processed_dir=processed, models_dir=models,
            expected_schema_version="ok",
        )
    # the pre-existing good file must survive a failed download
    assert dest.read_bytes() == good
    assert list(processed.glob("*.part")) == []


def test_verify_manifest_checks_referenced_objects(mock_bucket, tmp_path):
    from bts.data.sync import sync_to_r2, verify_manifest
    processed, models = _mk_dirs(tmp_path)
    (processed / "pa_2026.parquet").write_bytes(b"present-bytes")
    sync_to_r2(client=R2Client.from_env(), processed_dir=processed, models_dir=models)

    report = verify_manifest(client=R2Client.from_env(), expected_schema_version=None)
    assert report["objects_ok"] is True
    assert report["objects_missing"] == []

    # now break it: manifest references an object that is gone
    missing_key = f"objects/{_sha(b'present-bytes')}/pa_2026.parquet"
    mock_bucket.delete_object(Bucket="test-bucket", Key=missing_key)
    report = verify_manifest(client=R2Client.from_env(), expected_schema_version=None)
    assert report["objects_ok"] is False
    assert missing_key in report["objects_missing"]


def test_prune_unreferenced_deletes_only_old_unreferenced_objects(mock_bucket, tmp_path):
    from bts.data.sync import sync_to_r2, prune_unreferenced
    processed, models = _mk_dirs(tmp_path)
    (processed / "pa_2026.parquet").write_bytes(b"live-bytes")
    sync_to_r2(client=R2Client.from_env(), processed_dir=processed, models_dir=models)
    referenced_key = f"objects/{_sha(b'live-bytes')}/pa_2026.parquet"

    mock_bucket.put_object(Bucket="test-bucket", Key="objects/deadbeef/orphan.parquet",
                           Body=b"orphaned")
    mock_bucket.put_object(Bucket="test-bucket", Key="raw-archive-2017-2025.tar.gz",
                           Body=b"cold archive outside objects/ - never touched")

    # age guard on (moto objects are brand new): nothing deleted
    report = prune_unreferenced(client=R2Client.from_env(), min_age_days=7)
    assert report["deleted"] == []
    assert "objects/deadbeef/orphan.parquet" in report["kept_recent"]

    # age guard off: orphan goes, referenced + non-objects/ keys stay
    report = prune_unreferenced(client=R2Client.from_env(), min_age_days=0)
    assert report["deleted"] == ["objects/deadbeef/orphan.parquet"]
    keys = {o["Key"] for o in mock_bucket.list_objects_v2(Bucket="test-bucket")["Contents"]}
    assert referenced_key in keys
    assert "raw-archive-2017-2025.tar.gz" in keys
    assert "objects/deadbeef/orphan.parquet" not in keys


def test_prune_spares_previous_manifest_objects(mock_bucket, tmp_path):
    """Codex review I2: a restore that loaded the manifest just before a sync
    replaced it must not have its objects pruned out from under it. Prune
    spares everything referenced by the current OR the immediately previous
    manifest — one full generation of reader grace."""
    from bts.data.sync import sync_to_r2, prune_unreferenced
    processed, models = _mk_dirs(tmp_path)
    pq = processed / "pa_2026.parquet"

    pq.write_bytes(b"gen-one")
    sync_to_r2(client=R2Client.from_env(), processed_dir=processed, models_dir=models)
    gen1_key = f"objects/{_sha(b'gen-one')}/pa_2026.parquet"

    pq.write_bytes(b"gen-two")
    sync_to_r2(client=R2Client.from_env(), processed_dir=processed, models_dir=models)
    gen2_key = f"objects/{_sha(b'gen-two')}/pa_2026.parquet"

    # gen-one is unreferenced by the CURRENT manifest but held by the previous
    report = prune_unreferenced(client=R2Client.from_env(), min_age_days=0)
    assert gen1_key not in report["deleted"]
    keys = {o["Key"] for o in mock_bucket.list_objects_v2(Bucket="test-bucket")["Contents"]}
    assert gen1_key in keys

    pq.write_bytes(b"gen-three")
    sync_to_r2(client=R2Client.from_env(), processed_dir=processed, models_dir=models)

    # now gen-one is two generations back — collectable; gen-two spared as prev
    report = prune_unreferenced(client=R2Client.from_env(), min_age_days=0)
    assert gen1_key in report["deleted"]
    keys = {o["Key"] for o in mock_bucket.list_objects_v2(Bucket="test-bucket")["Contents"]}
    assert gen1_key not in keys
    assert gen2_key in keys


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


def test_prev_manifest_copy_failure_aborts_publish(mock_bucket, tmp_path):
    """Round-3 catch: the round-2 commit CLAIMED this fail-closed behavior but
    the code still warned-and-published. If the outgoing manifest can't be
    preserved as manifest.prev.json, publishing anyway makes the outgoing
    generation immediately pruneable under a concurrent reader — abort the
    sync instead (the next run retries)."""
    from botocore.exceptions import ClientError as BotoClientError
    from bts.data.sync import PREV_MANIFEST_KEY, read_manifest, sync_to_r2
    processed, models = _mk_dirs(tmp_path)
    pq = processed / "pa_2026.parquet"

    pq.write_bytes(b"gen-one")
    client = R2Client.from_env()
    m1 = sync_to_r2(client=client, processed_dir=processed, models_dir=models)

    pq.write_bytes(b"gen-two")
    real_copy = client.copy_object

    def failing_copy(src_key, dst_key):
        if dst_key == PREV_MANIFEST_KEY:
            raise BotoClientError({"Error": {"Code": "InternalError"}}, "CopyObject")
        return real_copy(src_key, dst_key)

    client.copy_object = failing_copy
    with pytest.raises(RuntimeError, match="prev manifest"):
        sync_to_r2(client=client, processed_dir=processed, models_dir=models)

    # the new manifest must NOT have been published
    client.copy_object = real_copy
    current = read_manifest(client)
    assert current["files"]["parquets/pa_2026.parquet"]["sha256"] == \
        m1["files"]["parquets/pa_2026.parquet"]["sha256"]
