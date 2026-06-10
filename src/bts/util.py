"""Shared utilities for BTS automation."""

import os
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def atomic_write_text(path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    A temp file in the same directory is fsynced and ``os.replace``d into place,
    so a crash mid-write can never leave a truncated/torn file — a torn pick or
    streak JSON would otherwise crash every loader into a silent crash-loop the
    heartbeat monitor can't see (audit D1). Preserves the caller's exact
    serialization; only the write mechanism changes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


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
