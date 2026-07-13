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


def retry_urlopen(req, timeout=15, max_retries=3, delay=5, idempotent=True):
    """urlopen with retry on transient failures.

    Retries on server errors (5xx) and network errors.
    Does NOT retry client errors (400, 401, 403, 404).

    Set ``idempotent=False`` for requests that CREATE state (e.g. a Bluesky
    post/DM createRecord). A network error after the request was sent can mean
    the server committed but the response was lost — retrying then double-posts.
    Non-idempotent requests raise on the first transient failure instead.
    """
    for attempt in range(max_retries):
        try:
            return urlopen(req, timeout=timeout)
        except (HTTPError, URLError) as e:
            if isinstance(e, HTTPError) and e.code in (400, 401, 403, 404):
                raise  # Don't retry client errors
            if not idempotent:
                raise  # non-idempotent: a lost response may mean it committed
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                raise


def is_regular_season_game(game: dict) -> bool:
    """BTS is a regular-season contest: exhibition, All-Star, and postseason
    schedule entries must never enter pick pipelines (an unfiltered 7/14
    would treat the All-Star Game as a real 1-game slate — 2026-07-12
    incident, round-2 review #3). Lenient on a missing gameType so older
    fixtures keep working; statsapi always sends it.
    """
    return game.get("gameType", "R") == "R"
