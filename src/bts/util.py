"""Shared utilities for BTS automation."""

import io
import os
import stat
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class NotRegularFileError(OSError):
    """The target or opened descriptor is not a regular file."""


class FileReadLimitError(OSError):
    """The file contains more bytes than the caller's read limit."""


class MissingFileTargetError(FileNotFoundError):
    """The target was missing at stat or open during a bounded read."""


def read_regular_file_text(path: Path, *, max_bytes: int | None = None) -> str:
    """Read UTF-8 from a regular file, optionally rejecting inputs above a byte limit."""
    try:
        if max_bytes is not None and not stat.S_ISREG(path.stat().st_mode):
            raise NotRegularFileError(f"{path}: not a regular file")
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except FileNotFoundError as exc:
        if max_bytes is None:
            raise
        raise MissingFileTargetError(*exc.args) from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            if max_bytes is not None:
                raise NotRegularFileError(f"{path}: not a regular file")
            raise OSError(f"{path}: not a regular file")
        if max_bytes is not None:
            data = bytearray()
            while len(data) <= max_bytes:
                chunk = os.read(fd, max_bytes + 1 - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            if len(data) > max_bytes:
                raise FileReadLimitError(f"{path}: exceeds read limit")
            return data.decode("utf-8")
        with io.open(fd, "r", encoding="utf-8", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(fd)


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
