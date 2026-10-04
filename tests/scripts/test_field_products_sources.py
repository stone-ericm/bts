"""The source freeze (code review r1 F1): every input read once and hashed before outcomes; parsers read only those
bytes; any later content or listing change is detected."""
from __future__ import annotations

import hashlib

import pytest

from scripts.audit.field_products.sources import Stage


def test_staged_bytes_are_served_and_unstaged_paths_are_refused(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"one")
    st = Stage()
    st.add(a, "a")
    a.write_bytes(b"two")
    assert st.read(a) == b"one"                                  # the frozen bytes, not the disk
    with pytest.raises(KeyError, match="not staged"):
        st.read(tmp_path / "b.bin")
    assert st.manifest()["files"]["a"] == {"path": str(a), "sha256": hashlib.sha256(b"one").hexdigest(), "bytes": 3}


def test_missing_inputs_are_recorded_and_read_as_missing(tmp_path):
    st = Stage()
    st.add(tmp_path / "gone.bin", "gone")
    assert st.manifest()["files"]["gone"]["sha256"] is None
    with pytest.raises(FileNotFoundError):
        st.read(tmp_path / "gone.bin")


def test_verify_reports_content_and_listing_changes(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "x.parquet").write_bytes(b"x")
    st = Stage()
    st.add(d / "x.parquet", "x")
    assert st.listing("d", d, "*.parquet") == ["x.parquet"]
    assert st.verify() == []
    (d / "x.parquet").write_bytes(b"changed")
    (d / "y.parquet").write_bytes(b"new")
    problems = st.verify()
    assert any("x: content changed" in p for p in problems) and any("d: listing changed" in p for p in problems)


def test_the_same_path_cannot_be_staged_under_two_keys_with_different_bytes(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"one")
    st = Stage()
    st.add(a, "a")
    a.write_bytes(b"two")
    with pytest.raises(ValueError, match="already staged"):
        st.add(a, "a2")
