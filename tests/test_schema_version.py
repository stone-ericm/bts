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
    from bts.data import schema as s1
    from bts.data import schema as s2
    assert s1.SCHEMA_VERSION == s2.SCHEMA_VERSION


def test_schema_version_changes_when_columns_change():
    mutated = PA_COLUMNS + ["new_column"]
    new_version = hashlib.sha256("\n".join(mutated).encode()).hexdigest()[:12]
    assert new_version != SCHEMA_VERSION
