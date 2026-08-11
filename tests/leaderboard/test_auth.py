"""Tests for auth flow: cookie loading + uid extraction + xSid minting."""
from __future__ import annotations

import json
import os
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

    def test_loads_from_file_on_linux_when_pass_not_available(self, tmp_path, monkeypatch):
        """If pass isn't installed, fall back to BTS_LEADERBOARD_COOKIE_FILE."""
        cookie_file = tmp_path / "cookies.json"
        cookie_file.write_text(json.dumps(SAMPLE_COOKIES))
        monkeypatch.setenv("BTS_LEADERBOARD_COOKIE_FILE", str(cookie_file))
        # Force pass to "not installed" by pointing PATH at empty dir
        monkeypatch.setenv("PATH", str(tmp_path))
        with patch("bts.leaderboard.auth.sys.platform", "linux"):
            cookies = load_session_cookies()
        assert cookies["oktaid"] == "00u7q0ft1NTz5zMUQ356"

    def test_default_linux_cookie_file_in_home_dir(self, tmp_path, monkeypatch):
        """Default path is ~/.bts-leaderboard-cookies.json"""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        cookie_file = fake_home / ".bts-leaderboard-cookies.json"
        cookie_file.write_text(json.dumps(SAMPLE_COOKIES))
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.delenv("BTS_LEADERBOARD_COOKIE_FILE", raising=False)
        monkeypatch.setenv("PATH", str(tmp_path))  # no `pass` available
        with patch("bts.leaderboard.auth.sys.platform", "linux"):
            cookies = load_session_cookies()
        assert cookies["oktaid"] == "00u7q0ft1NTz5zMUQ356"

    def test_linux_prefers_pass_when_available(self, tmp_path, monkeypatch):
        """If `pass` IS installed, use it (don't fall back to file)."""
        # Make a fake `pass` executable that prints valid JSON
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_pass = fake_bin / "pass"
        fake_pass.write_text(f"#!/bin/sh\necho '{json.dumps(SAMPLE_COOKIES)}'\n")
        fake_pass.chmod(0o755)
        monkeypatch.setenv("PATH", str(fake_bin))
        with patch("bts.leaderboard.auth.sys.platform", "linux"):
            cookies = load_session_cookies()
        assert cookies["oktaid"] == "00u7q0ft1NTz5zMUQ356"


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


def _ok_login_response():
    fake = MagicMock(status_code=200)
    fake.json.return_value = {
        "success": {"user": {"id": 50311, "username": "stonehengee"}, "xSid": "x_1"},
        "errors": [],
    }
    return fake


def _blank_200_response():
    """The observed 2026-08-11 flap shape: HTTP 200 with an empty (non-JSON) body."""
    fake = MagicMock(status_code=200, text="")
    fake.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
    return fake


class TestFetchLoginSessionTransientRetry:
    """Kill-switch-safe classification (2026-08-11 incident + Codex review):
    retried in-process = transport errors, 5xx, and EMPTY-body 200s only.
    Anything that could be a rejection of *us* — 4xx, redirects, non-empty
    non-JSON 200s (challenge/denial pages), missing xSid — never retries."""

    def _install_sleep_recorder(self, monkeypatch):
        import types
        import bts.leaderboard.auth as auth
        recorded = []
        monkeypatch.setattr(auth, "time",
                            types.SimpleNamespace(sleep=recorded.append),
                            raising=False)
        return recorded

    def test_retries_empty_200_then_succeeds(self, monkeypatch):
        from bts.leaderboard.auth import fetch_login_session
        sleeps = self._install_sleep_recorder(monkeypatch)
        with patch("bts.leaderboard.auth.httpx.post",
                   side_effect=[_blank_200_response(), _ok_login_response()]) as mock_post:
            session = fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert session.xsid == "x_1"
        assert mock_post.call_count == 2
        assert sleeps == [2.0]      # a successful retry still paces itself

    def test_retries_5xx_then_succeeds(self, monkeypatch):
        from bts.leaderboard.auth import fetch_login_session
        sleeps = self._install_sleep_recorder(monkeypatch)
        bad = MagicMock(status_code=503, text="upstream connect error")
        with patch("bts.leaderboard.auth.httpx.post",
                   side_effect=[bad, _ok_login_response()]) as mock_post:
            session = fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert session.xsid == "x_1"
        assert mock_post.call_count == 2
        assert sleeps == [2.0]      # MagicMock headers -> unparseable -> default backoff

    def test_5xx_retry_after_honored_and_capped(self, monkeypatch):
        """Server-directed backoff wins over the 2s default, capped at 30s so a
        huge header can't stall a cron run for minutes."""
        from bts.leaderboard.auth import fetch_login_session
        sleeps = self._install_sleep_recorder(monkeypatch)
        bad = MagicMock(status_code=503, text="maintenance")
        bad.headers = {"Retry-After": "120"}
        with patch("bts.leaderboard.auth.httpx.post",
                   side_effect=[bad, _ok_login_response()]):
            fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert sleeps == [30.0]

    def test_5xx_retry_after_short_value_used_verbatim(self, monkeypatch):
        from bts.leaderboard.auth import fetch_login_session
        sleeps = self._install_sleep_recorder(monkeypatch)
        bad = MagicMock(status_code=503, text="maintenance")
        bad.headers = {"Retry-After": "3"}
        with patch("bts.leaderboard.auth.httpx.post",
                   side_effect=[bad, _ok_login_response()]):
            fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert sleeps == [3.0]

    @pytest.mark.parametrize("header", ["-1", "NaN", "inf"])
    def test_5xx_hostile_retry_after_falls_back_to_default(self, header, monkeypatch):
        """Negative/non-finite Retry-After is server-controlled input and must
        never reach time.sleep, where it raises out of every handler (r2 #4)."""
        from bts.leaderboard.auth import fetch_login_session
        sleeps = self._install_sleep_recorder(monkeypatch)
        bad = MagicMock(status_code=503, text="maintenance")
        bad.headers = {"Retry-After": header}
        with patch("bts.leaderboard.auth.httpx.post",
                   side_effect=[bad, _ok_login_response()]):
            fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert sleeps == [2.0]

    def test_retries_transport_error_then_succeeds(self, monkeypatch):
        import httpx
        from bts.leaderboard.auth import fetch_login_session
        sleeps = self._install_sleep_recorder(monkeypatch)
        with patch("bts.leaderboard.auth.httpx.post",
                   side_effect=[httpx.ConnectError("boom"), _ok_login_response()]) as mock_post:
            session = fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert session.xsid == "x_1"
        assert mock_post.call_count == 2
        assert sleeps == [2.0]

    def test_transient_exhaustion_raises_transient_auth_error(self, monkeypatch):
        from bts.leaderboard.auth import fetch_login_session
        self._install_sleep_recorder(monkeypatch)
        with patch("bts.leaderboard.auth.httpx.post",
                   side_effect=[_blank_200_response() for _ in range(3)]) as mock_post:
            with pytest.raises(AuthError) as ei:
                fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert ei.value.__class__.__name__ == "TransientAuthError"
        assert "3 attempts" in str(ei.value)
        assert mock_post.call_count == 3
        # Chained cause survives for debuggability (Codex review #9).
        assert isinstance(ei.value.__cause__, json.JSONDecodeError)

    def test_non_empty_non_json_200_fails_immediately_without_retry(self, monkeypatch):
        """A 200 with an HTML body can be a WAF/bot-defense challenge page —
        hammering it with retries is exactly what the kill-switch philosophy
        forbids (Codex review #1). Still classified transient (NOT a cookie
        problem), and the snippet stays diagnosable from cron.log."""
        from bts.leaderboard.auth import fetch_login_session, TransientAuthError
        sleeps = self._install_sleep_recorder(monkeypatch)
        html = MagicMock(status_code=200, text="<html>Service Unavailable</html>")
        html.json.side_effect = json.JSONDecodeError("Expecting value", "<", 0)
        with patch("bts.leaderboard.auth.httpx.post", return_value=html) as mock_post:
            with pytest.raises(TransientAuthError) as ei:
                fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert "<html>Service Unavailable</html>" in str(ei.value)
        assert mock_post.call_count == 1
        assert sleeps == []

    def test_undecodable_body_fails_immediately_as_transient(self, monkeypatch):
        """response.json() can raise UnicodeDecodeError (not just JSONDecodeError);
        real httpx renders undecodable bytes as non-empty replacement text, so the
        shape is garbage-page-like: classified transient, never retried (r2 #1)."""
        from bts.leaderboard.auth import fetch_login_session, TransientAuthError
        sleeps = self._install_sleep_recorder(monkeypatch)
        bad = MagicMock(status_code=200, text="�")
        bad.json.side_effect = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        with patch("bts.leaderboard.auth.httpx.post", return_value=bad) as mock_post:
            with pytest.raises(TransientAuthError):
                fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert mock_post.call_count == 1
        assert sleeps == []

    def test_whitespace_only_200_not_retried(self, monkeypatch):
        """Retry is licensed ONLY by a zero-length body (the observed outage
        shape); whitespace bytes are already a non-empty response (r2 #1)."""
        from bts.leaderboard.auth import fetch_login_session, TransientAuthError
        sleeps = self._install_sleep_recorder(monkeypatch)
        bad = MagicMock(status_code=200, text="\r\n")
        bad.json.side_effect = json.JSONDecodeError("Expecting value", "\r\n", 0)
        with patch("bts.leaderboard.auth.httpx.post", return_value=bad) as mock_post:
            with pytest.raises(TransientAuthError):
                fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert mock_post.call_count == 1
        assert sleeps == []

    def test_non_object_json_body_fails_immediately_as_transient(self, monkeypatch):
        """`null`/list bodies are malformed upstream responses: no retry, no
        cookie advice — and never an escaping AttributeError (pre-fix these
        bypassed _fail entirely, so NO DM was sent at all; Codex review #6)."""
        from bts.leaderboard.auth import fetch_login_session, TransientAuthError
        sleeps = self._install_sleep_recorder(monkeypatch)
        fake = MagicMock(status_code=200, text="null")
        fake.json.return_value = None
        with patch("bts.leaderboard.auth.httpx.post", return_value=fake) as mock_post:
            with pytest.raises(TransientAuthError, match="non-object JSON"):
                fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert mock_post.call_count == 1
        assert sleeps == []

    def test_success_null_is_auth_error_with_snippet_not_a_crash(self, monkeypatch):
        """{"success": null} used to AttributeError out of every CLI handler."""
        from bts.leaderboard.auth import fetch_login_session
        sleeps = self._install_sleep_recorder(monkeypatch)
        fake = MagicMock(status_code=200, text='{"success": null, "errors": []}')
        fake.json.return_value = {"success": None, "errors": []}
        with patch("bts.leaderboard.auth.httpx.post", return_value=fake) as mock_post:
            with pytest.raises(AuthError, match="xSid missing") as ei:
                fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert ei.value.__class__.__name__ == "AuthError"
        assert "errors empty" in str(ei.value)      # snippet branch: diagnosable
        assert mock_post.call_count == 1
        assert sleeps == []

    def test_non_dict_user_object_does_not_crash(self, monkeypatch):
        """{"success": {"xSid": ..., "user": [...]}} must not AttributeError out
        of every CLI handler (r2 #2); identity fields degrade to None."""
        from bts.leaderboard.auth import fetch_login_session
        fake = MagicMock(status_code=200)
        fake.json.return_value = {"success": {"xSid": "x_1", "user": [1, 2]},
                                  "errors": []}
        with patch("bts.leaderboard.auth.httpx.post", return_value=fake):
            session = fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert session.xsid == "x_1"
        assert session.user_id is None and session.username is None

    @pytest.mark.parametrize("status", [401, 403])
    def test_4xx_fails_immediately_without_retry(self, status, monkeypatch):
        from bts.leaderboard.auth import fetch_login_session
        sleeps = self._install_sleep_recorder(monkeypatch)
        bad = MagicMock(status_code=status, text="rejected")
        with patch("bts.leaderboard.auth.httpx.post", return_value=bad) as mock_post:
            with pytest.raises(AuthError, match="auth/login returned") as ei:
                fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert ei.value.__class__.__name__ == "AuthError"
        assert mock_post.call_count == 1
        assert sleeps == []

    def test_429_raises_rate_limited_login_error_without_retry(self, monkeypatch):
        """429 is the kill-switch shape at login time: a distinct class so CLIs
        say 'back off' instead of 're-capture cookies' (Codex review #4 — the
        interactive re-capture flow would ADD auth traffic against a limiter)."""
        from bts.leaderboard.auth import fetch_login_session, RateLimitedLoginError
        sleeps = self._install_sleep_recorder(monkeypatch)
        bad = MagicMock(status_code=429, text="slow down")
        with patch("bts.leaderboard.auth.httpx.post", return_value=bad) as mock_post:
            with pytest.raises(RateLimitedLoginError, match="429"):
                fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert mock_post.call_count == 1
        assert sleeps == []

    def test_redirect_3xx_is_auth_error_without_retry(self, monkeypatch):
        """A redirect on the login POST is session-shaped (go re-authenticate),
        so it keeps the cookie-advice path. httpx.post does not follow redirects."""
        from bts.leaderboard.auth import fetch_login_session
        sleeps = self._install_sleep_recorder(monkeypatch)
        bad = MagicMock(status_code=302, text="")
        with patch("bts.leaderboard.auth.httpx.post", return_value=bad) as mock_post:
            with pytest.raises(AuthError, match="returned 302") as ei:
                fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert ei.value.__class__.__name__ == "AuthError"
        assert mock_post.call_count == 1
        assert sleeps == []

    def test_2xx_non_200_fails_immediately_as_transient(self, monkeypatch):
        from bts.leaderboard.auth import fetch_login_session, TransientAuthError
        sleeps = self._install_sleep_recorder(monkeypatch)
        bad = MagicMock(status_code=204, text="")
        with patch("bts.leaderboard.auth.httpx.post", return_value=bad) as mock_post:
            with pytest.raises(TransientAuthError, match="unexpected status 204"):
                fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert mock_post.call_count == 1
        assert sleeps == []

    def test_missing_xsid_fails_immediately_without_retry(self, monkeypatch):
        from bts.leaderboard.auth import fetch_login_session
        sleeps = self._install_sleep_recorder(monkeypatch)
        fake = MagicMock(status_code=200)
        fake.json.return_value = {"success": {}, "errors": [{"message": "bad session"}]}
        with patch("bts.leaderboard.auth.httpx.post", return_value=fake) as mock_post:
            with pytest.raises(AuthError, match="xSid") as ei:
                fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert ei.value.__class__.__name__ == "AuthError"
        assert mock_post.call_count == 1
        assert sleeps == []

    def test_retry_backoff_sequence(self, monkeypatch):
        from bts.leaderboard.auth import fetch_login_session
        sleeps = self._install_sleep_recorder(monkeypatch)
        with patch("bts.leaderboard.auth.httpx.post",
                   side_effect=[_blank_200_response() for _ in range(3)]):
            with pytest.raises(AuthError):
                fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"})
        assert sleeps == [2.0, 4.0]

    @pytest.mark.parametrize("kwargs", [
        {"attempts": 0}, {"attempts": 1.5},
        {"retry_delay": -1.0}, {"retry_delay": float("nan")},
        {"retry_delay": float("inf")},
    ])
    def test_invalid_retry_params_rejected_before_any_request(self, kwargs, monkeypatch):
        """Non-int attempts and negative/non-finite delays fail loudly up front
        instead of leaking TypeError/ValueError mid-loop (r2 #8)."""
        from bts.leaderboard.auth import fetch_login_session
        with patch("bts.leaderboard.auth.httpx.post") as mock_post:
            with pytest.raises(ValueError):
                fetch_login_session(uid="00u123", cookies={"oktaid": "00u123"},
                                    **kwargs)
        assert mock_post.call_count == 0


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
