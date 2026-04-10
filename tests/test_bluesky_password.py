"""Tests for unified Bluesky password helper."""
from unittest.mock import patch

import pytest


def test_new_env_var_is_primary(monkeypatch):
    monkeypatch.delenv("BTS_BLUESKY_APP_PASSWORD", raising=False)
    monkeypatch.delenv("BTS_BLUESKY_PASSWORD", raising=False)
    monkeypatch.delenv("BTS_BLUESKY_DM_PASSWORD", raising=False)

    monkeypatch.setenv("BTS_BLUESKY_APP_PASSWORD", "new-unified-password")

    with patch("bts.posting.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1  # keychain miss
        from bts.posting import get_bluesky_password
        assert get_bluesky_password() == "new-unified-password"


def test_falls_back_to_legacy_bts_bluesky_password(monkeypatch):
    monkeypatch.delenv("BTS_BLUESKY_APP_PASSWORD", raising=False)
    monkeypatch.setenv("BTS_BLUESKY_PASSWORD", "legacy-posting-password")
    monkeypatch.delenv("BTS_BLUESKY_DM_PASSWORD", raising=False)

    with patch("bts.posting.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        from bts.posting import get_bluesky_password
        assert get_bluesky_password() == "legacy-posting-password"


def test_dm_module_uses_shared_helper(monkeypatch):
    monkeypatch.setenv("BTS_BLUESKY_APP_PASSWORD", "shared-password")

    with patch("bts.posting.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        from bts.dm import get_bluesky_dm_password
        assert get_bluesky_dm_password() == "shared-password"


def test_dm_module_falls_back_to_legacy_dm_var(monkeypatch):
    monkeypatch.delenv("BTS_BLUESKY_APP_PASSWORD", raising=False)
    monkeypatch.delenv("BTS_BLUESKY_PASSWORD", raising=False)
    monkeypatch.setenv("BTS_BLUESKY_DM_PASSWORD", "legacy-dm-password")

    with patch("bts.posting.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        from bts.dm import get_bluesky_dm_password
        assert get_bluesky_dm_password() == "legacy-dm-password"
