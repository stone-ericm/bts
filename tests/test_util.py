"""Tests for shared retry utility."""

import pytest
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError, URLError
from bts.util import retry_urlopen


class TestRetryUrlopen:
    @patch("bts.util.urlopen")
    def test_success_on_first_try(self, mock_urlopen):
        mock_response = MagicMock()
        mock_urlopen.return_value = mock_response
        result = retry_urlopen("http://example.com", max_retries=3, delay=0)
        assert result is mock_response
        assert mock_urlopen.call_count == 1

    @patch("bts.util.time.sleep")
    @patch("bts.util.urlopen")
    def test_retries_on_server_error(self, mock_urlopen, mock_sleep):
        mock_response = MagicMock()
        mock_urlopen.side_effect = [
            HTTPError("http://example.com", 500, "Server Error", {}, None),
            mock_response,
        ]
        result = retry_urlopen("http://example.com", max_retries=3, delay=1)
        assert result is mock_response
        assert mock_urlopen.call_count == 2
        mock_sleep.assert_called_once_with(1)

    @patch("bts.util.urlopen")
    def test_no_retry_on_client_error_401(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            "http://example.com", 401, "Unauthorized", {}, None,
        )
        with pytest.raises(HTTPError) as exc_info:
            retry_urlopen("http://example.com", max_retries=3, delay=0)
        assert exc_info.value.code == 401
        assert mock_urlopen.call_count == 1

    @patch("bts.util.urlopen")
    def test_no_retry_on_client_error_404(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            "http://example.com", 404, "Not Found", {}, None,
        )
        with pytest.raises(HTTPError):
            retry_urlopen("http://example.com", max_retries=3, delay=0)
        assert mock_urlopen.call_count == 1

    @patch("bts.util.time.sleep")
    @patch("bts.util.urlopen")
    def test_retries_on_url_error(self, mock_urlopen, mock_sleep):
        mock_response = MagicMock()
        mock_urlopen.side_effect = [
            URLError("Connection refused"),
            mock_response,
        ]
        result = retry_urlopen("http://example.com", max_retries=3, delay=1)
        assert result is mock_response

    @patch("bts.util.time.sleep")
    @patch("bts.util.urlopen")
    def test_raises_after_max_retries(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = URLError("Connection refused")
        with pytest.raises(URLError):
            retry_urlopen("http://example.com", max_retries=3, delay=1)
        assert mock_urlopen.call_count == 3

    @patch("bts.util.time.sleep")
    @patch("bts.util.urlopen")
    def test_backoff_delay(self, mock_urlopen, mock_sleep):
        """Delay increases with each attempt: delay*1, delay*2."""
        mock_urlopen.side_effect = [
            URLError("fail"),
            URLError("fail"),
            MagicMock(),
        ]
        retry_urlopen("http://example.com", max_retries=3, delay=5)
        assert mock_sleep.call_args_list[0][0][0] == 5   # attempt 0: 5*1
        assert mock_sleep.call_args_list[1][0][0] == 10  # attempt 1: 5*2
