"""Tests for Bluesky DM notifications."""

import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

from bts.dm import send_dm, get_dm_password


class TestGetDmPassword:
    @patch("bts.posting.subprocess.run")
    def test_keychain_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="test-password\n")
        assert get_dm_password() == "test-password"

    @patch("bts.posting.subprocess.run")
    def test_env_fallback(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        with patch.dict("os.environ", {"BTS_BLUESKY_DM_PASSWORD": "env-pw"}):
            assert get_dm_password() == "env-pw"

    @patch("bts.posting.subprocess.run")
    def test_raises_when_not_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="Bluesky app password not found"):
                get_dm_password()


def _mock_urlopen_responses(responses):
    """Create a side_effect that returns successive mock responses."""
    mocks = []
    for resp in responses:
        m = MagicMock()
        m.read.return_value = json.dumps(resp).encode()
        mocks.append(m)
    return mocks


class TestSendDm:
    @patch("bts.dm.retry_urlopen")
    @patch("bts.dm.get_dm_password", return_value="test-password")
    def test_sends_dm_successfully(self, mock_pw, mock_urlopen):
        mock_urlopen.side_effect = _mock_urlopen_responses([
            # createSession
            {"accessJwt": "jwt-token", "did": "did:plc:bot"},
            # resolveHandle
            {"did": "did:plc:recipient"},
            # getConvoForMembers
            {"convo": {"id": "convo-123"}},
            # sendMessage
            {"id": "msg-456", "sentAt": "2026-04-01T12:00:00Z"},
        ])

        result = send_dm("stonehengee.bsky.social", "Test message")
        assert result == "msg-456"
        assert mock_urlopen.call_count == 4

    @patch("bts.dm.retry_urlopen")
    @patch("bts.dm.get_dm_password", return_value="test-password")
    def test_auth_failure_raises(self, mock_pw, mock_urlopen):
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            url="", code=401, msg="Unauthorized",
            hdrs=None, fp=BytesIO(b"bad auth"),
        )

        with pytest.raises(RuntimeError, match="DM auth failed"):
            send_dm("stonehengee.bsky.social", "Test")
