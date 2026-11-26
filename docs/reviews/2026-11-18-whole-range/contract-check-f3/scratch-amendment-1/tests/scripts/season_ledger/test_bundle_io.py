import json

import pyarrow as pa
import pytest

from scripts.audit.season_ledger.bundle import BundleEntry, BundleError, open_bundle, write_manifest
from scripts.audit.season_ledger.ids import (load_json_bytes, obs_id, quarantine, sha256_hex, stamp_to_utc, take, typed,
                                             utc_iso)
from scripts.audit.season_ledger.io import build_table, write_table
from tests.scripts.season_ledger.builders import gz, seal_bundle


def test_declared_missing_input_is_accepted(tmp_path):
    seal_bundle(tmp_path, {"picks/2026-05-01.json": b"{}"}, missing=["schedules/2026-05-01.json"])
    _, files = open_bundle(tmp_path)
    assert files == {"picks/2026-05-01.json": b"{}", "schedules/2026-05-01.json": None}


def test_declared_present_but_absent_is_refused(tmp_path):
    seal_bundle(tmp_path, {"picks/a.json": b"{}"})
    (tmp_path / "picks/a.json").unlink()
    with pytest.raises(BundleError, match="declared present but absent"):
        open_bundle(tmp_path)


def test_changed_file_is_refused(tmp_path):
    seal_bundle(tmp_path, {"picks/a.json": b"{}"})
    (tmp_path / "picks/a.json").write_bytes(b'{"x": 1}')
    with pytest.raises(BundleError, match="hash mismatch"):
        open_bundle(tmp_path)


def test_unsafe_path_is_refused(tmp_path):
    with pytest.raises(BundleError, match="unsafe"):
        write_manifest(tmp_path, [BundleEntry(rel_path="../x.json", status="missing")],
                       acquired_at_utc="2026-09-28T16:00:00.000000Z", builder_version="t")


def test_manifest_order_does_not_change_what_compile_reads(tmp_path):
    seal_bundle(tmp_path, {"b.json": b"2", "a.json": b"1"})
    manifest = tmp_path / "manifest.json"
    doc = json.loads(manifest.read_text())
    doc["entries"].reverse()
    manifest.write_text(json.dumps(doc))
    _, files = open_bundle(tmp_path)
    assert list(files) == ["a.json", "b.json"]


def test_duplicate_bytes_at_two_paths_are_two_occurrences():
    digest = sha256_hex(b"same bytes")
    assert obs_id("picks/a.json", "file", digest) != obs_id("picks/archive/a.json", "file", digest)


def test_gzip_and_plain_json_both_load():
    assert load_json_bytes(b'{"a": 1}') == {"a": 1}
    assert load_json_bytes(gz(b'{"a": 1}')) == {"a": 1}
    assert gz(b"x")[4:8] == b"\x00\x00\x00\x00"     # pinned gzip mtime: fixtures are byte-stable (Codex plan r1 #12)
    with pytest.raises(ValueError, match="bad_gzip"):
        load_json_bytes(b"\x1f\x8b" + b"not really gzip")
    with pytest.raises(ValueError, match="invalid_json"):
        load_json_bytes(b'{"a": ')


def test_times_normalize_to_fixed_precision_utc():
    # Review Focus 5: every source format in the snapshot, plus a naive value.
    assert utc_iso("2026-08-20T17:00:00.123456-04:00") == "2026-08-20T21:00:00.123456Z"
    assert utc_iso("2026-05-01T23:05:00Z") == "2026-05-01T23:05:00.000000Z"
    assert utc_iso("2026-05-01T15:00:00+00:00") == "2026-05-01T15:00:00.000000Z"
    assert utc_iso("2026-05-01T15:00:00") is None
    assert utc_iso(None) is None and utc_iso("garbage") is None
    assert utc_iso("2026-05-01T15:00:00.5Z") < utc_iso("2026-05-01T15:00:01Z")   # string order == time order


def test_capture_stamp_to_utc():
    assert stamp_to_utc("20260704T030011Z.json.gz") == "2026-07-04T03:00:11.000000Z"
    assert stamp_to_utc("001_rounds.json.gz") is None


def test_typed_policy_keeps_null_absent_and_wrong_type_distinct():
    values, absent, bad = take({"hits": "1", "atBats": None, "p": 1},
                               {"hits": "int", "atBats": "int", "p": "float", "n": "int"}, prefix="s.")
    assert values == {"hits": None, "atBats": None, "p": 1.0, "n": None}
    assert (absent, bad) == (["s.n"], ["s.hits"])
    assert typed(True, "int") == (None, True) and typed(3, "int") == (3, False) and typed(None, "str") == (None, False)


SCHEMA = pa.schema([("obs_id", pa.string()), ("n", pa.int64()), ("flag", pa.bool_())])


def test_quarantine_keeps_a_parsed_null_distinct_from_no_value():
    # Codex plan r3 #6: a record that parsed as JSON null keeps "null"; only undecodable bytes carry no raw value.
    assert quarantine("picks/x.json", "file", "no_pick_object", raw=None)["record_raw_json"] == "null"
    assert quarantine("picks/x.json", "file", "invalid_json:JSONDecodeError")["record_raw_json"] is None


def test_write_table_is_byte_identical_for_reordered_input(tmp_path):
    rows = [{"obs_id": "b", "n": 2, "flag": None}, {"obs_id": "a", "n": None, "flag": True}]
    write_table(build_table(rows, SCHEMA, sort_keys=["obs_id"], name="t"), tmp_path / "x.parquet")
    write_table(build_table(list(reversed(rows)), SCHEMA, sort_keys=["obs_id"], name="t"), tmp_path / "y.parquet")
    assert (tmp_path / "x.parquet").read_bytes() == (tmp_path / "y.parquet").read_bytes()


def test_build_table_refuses_duplicate_sort_keys_and_unknown_columns():
    with pytest.raises(ValueError, match="duplicate sort keys"):
        build_table([{"obs_id": "a"}, {"obs_id": "a"}], SCHEMA, sort_keys=["obs_id"], name="t")
    with pytest.raises(ValueError, match="not in schema"):
        build_table([{"obs_id": "a", "typo": 1}], SCHEMA, sort_keys=["obs_id"], name="t")
