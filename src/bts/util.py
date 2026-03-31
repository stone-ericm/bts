"""Shared utilities for BTS automation."""

import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def retry_urlopen(req, timeout=15, max_retries=3, delay=5):
    """urlopen with retry on transient failures.

    Retries on server errors (5xx) and network errors.
    Does NOT retry client errors (400, 401, 403, 404).
    """
    for attempt in range(max_retries):
        try:
            return urlopen(req, timeout=timeout)
        except (HTTPError, URLError) as e:
            if isinstance(e, HTTPError) and e.code in (400, 401, 403, 404):
                raise  # Don't retry client errors
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise
