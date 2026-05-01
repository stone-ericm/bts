"""Tests for auth flow: cookie loading + uid extraction + xSid minting."""
from __future__ import annotations

import json
import re
import subprocess as _subprocess
from unittest.mock import patch, MagicMock

import pytest

from bts.leaderboard.auth import (
    load_session_cookies,
    extract_uid,
    fetch_xsid,
    is_session_valid,
    AuthError,
)


SAMPLE_COOKIES = [
    {"name": "oktaid", "value": "00u7q0ft1NTz5zMUQ356", "domain": ".mlb.com"},
    {"name": "session_id", "value": "abc123", "domain": ".mlb.com"},
]


class TestLoadSessionCookies:
    def test_loads_plain_json_from_keychain(self):
        # mac path: subprocess returns plain JSON
        with patch("bts.leaderboard.auth.subprocess.check_output",
                   return_value=json.dumps(SAMPLE_COOKIES).encode()), \
             patch("bts.leaderboard.auth.sys.platform", "darwin"):
            cookies = load_session_cookies()
        assert cookies["oktaid"] == "00u7q0ft1NTz5zMUQ356"
        assert cookies["session_id"] == "abc123"

    def test_decodes_hex_output_from_security(self):
        # mac path: subprocess returns hex-encoded JSON (when `security` thinks
        # the value contains binary bytes — common with cookie blobs)
        hex_payload = json.dumps(SAMPLE_COOKIES).encode().hex()
        with patch("bts.leaderboard.auth.subprocess.check_output",
                   return_value=hex_payload.encode()), \
             patch("bts.leaderboard.auth.sys.platform", "darwin"):
            cookies = load_session_cookies()
        assert cookies["oktaid"] == "00u7q0ft1NTz5zMUQ356"

    def test_loads_from_pass_on_linux(self):
        with patch("bts.leaderboard.auth.subprocess.check_output",
                   return_value=json.dumps(SAMPLE_COOKIES).encode()), \
             patch("bts.leaderboard.auth.sys.platform", "linux"):
            cookies = load_session_cookies()
        assert cookies["oktaid"] == "00u7q0ft1NTz5zMUQ356"

    def test_raises_auth_error_when_keychain_fails(self):
        with patch("bts.leaderboard.auth.subprocess.check_output",
                   side_effect=_subprocess.CalledProcessError(1, "security")), \
             patch("bts.leaderboard.auth.sys.platform", "darwin"):
            with pytest.raises(AuthError, match="cookie store"):
                load_session_cookies()


class TestExtractUid:
    def test_returns_oktaid_value(self):
        cookies = {"oktaid": "00u123", "other": "x"}
        assert extract_uid(cookies) == "00u123"

    def test_raises_when_oktaid_missing(self):
        with pytest.raises(AuthError, match="oktaid"):
            extract_uid({"other": "x"})


class TestFetchXsid:
    def test_posts_uid_and_platform_returns_xsid(self):
        fake_response = MagicMock(status_code=200)
        fake_response.json.return_value = {
            "success": {"user": {"id": 50311}, "xSid": "abc_1700000000"},
            "errors": [],
        }
        fake_response.raise_for_status = lambda: None
        with patch("bts.leaderboard.auth.httpx.post", return_value=fake_response) as mock_post:
            xsid = fetch_xsid(uid="00u123", cookies={"oktaid": "00u123"})
        assert xsid == "abc_1700000000"
        # Verify POST body shape
        kwargs = mock_post.call_args.kwargs
        assert kwargs["json"] == {"uid": "00u123", "platform": "web"}

    def test_raises_auth_error_on_non_200(self):
        fake = MagicMock(status_code=401, text="Unauthorized")
        with patch("bts.leaderboard.auth.httpx.post", return_value=fake):
            with pytest.raises(AuthError, match="auth/login"):
                fetch_xsid(uid="00u123", cookies={"oktaid": "00u123"})

    def test_raises_auth_error_when_xsid_missing(self):
        fake = MagicMock(status_code=200)
        fake.json.return_value = {"success": {}, "errors": [{"message": "boom"}]}
        fake.raise_for_status = lambda: None
        with patch("bts.leaderboard.auth.httpx.post", return_value=fake):
            with pytest.raises(AuthError, match="xSid"):
                fetch_xsid(uid="00u123", cookies={"oktaid": "00u123"})


class TestIsSessionValid:
    def test_returns_true_when_fetch_xsid_succeeds(self):
        with patch("bts.leaderboard.auth.fetch_xsid", return_value="x_123"):
            assert is_session_valid({"oktaid": "00u123"}) is True

    def test_returns_false_when_auth_error(self):
        with patch("bts.leaderboard.auth.fetch_xsid", side_effect=AuthError("expired")):
            assert is_session_valid({"oktaid": "00u123"}) is False

    def test_returns_false_when_oktaid_missing(self):
        # No oktaid -> can't extract uid -> not valid
        assert is_session_valid({"other": "x"}) is False
