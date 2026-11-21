"""File format shared by the probable-pitcher cache writer and health reader."""

import json
import os
import re
import stat
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo


SOURCE = "probable_pitcher_cache"
SCHEMA = "bts_probable_pitcher_cache_status_v1"
ET = ZoneInfo("America/New_York")
MAX_BYTES = 64 * 1024
COUNTS = ("dropped", "recovered_from_cache", "rescanned", "unusable_feed", "without_feed")
REASONS = ("invalid_entries", "unreadable_cache")
HEX = re.compile(r"[0-9a-f]{64}")
EVIDENCE = re.compile(r"probable_pitcher_lookup\.[0-9]{8}T[0-9]{6}Z\.[0-9a-f]{12}\.json")
FIELDS = {
    "schema_version", "reason", "recorded_at", "recorded_date_et", "cache_path",
    "game_ids", "identity", "preserved", "original_sha256", *COUNTS,
}


def display(value):
    """Show a path or file name read from disk with no backtick, reversibly (W15b)."""
    return str(value).replace("\\", "\\\\").replace("`", "\\x60")


class NonRegularFile(OSError):
    """The opened descriptor is not a regular file."""


@contextmanager
def open_regular(path, *, follow_symlinks=False):
    flags = os.O_RDONLY | os.O_NONBLOCK
    if not follow_symlinks:
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise NonRegularFile(path)
        with os.fdopen(fd, "rb", closefd=False) as stream:
            yield stream
    finally:
        os.close(fd)


def directory_exists(path):
    """Return false for an absent directory and reject other entry types."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(mode):
        raise NotADirectoryError(path)
    return True


def read_record(path):
    with open_regular(path) as stream:
        payload = stream.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("status exceeds size limit")
    row = json.loads(payload.decode("utf-8"))
    if not isinstance(row, dict) or row.keys() != FIELDS or row["schema_version"] != SCHEMA:
        raise ValueError("invalid status schema")
    reason, day, identity = row["reason"], row["recorded_date_et"], row["identity"]
    if (reason not in REASONS or not isinstance(identity, str) or not HEX.fullmatch(identity)
            or path.name != f"{day}.{reason}.{identity[:12]}.json"):
        raise ValueError("invalid status name or identity")
    cache_path, copy, digest = row["cache_path"], row["preserved"], row["original_sha256"]
    if (not isinstance(cache_path, str) or not cache_path
            or (copy is not None and (not isinstance(copy, str) or not EVIDENCE.fullmatch(copy)))
            or (digest is not None and (not isinstance(digest, str) or not HEX.fullmatch(digest)))):
        raise ValueError("invalid cache or evidence metadata")
    timestamp = row["recorded_at"]
    if not isinstance(timestamp, str) or not timestamp.endswith("+00:00"):
        raise ValueError("invalid UTC timestamp")
    at = datetime.fromisoformat(timestamp)
    if at.tzinfo is None or at.astimezone(ET).date().isoformat() != day:
        raise ValueError("timestamp and ET date disagree")

    ids = row["game_ids"]
    if (not isinstance(ids, list) or len(ids) > 10 or any(type(game) is not int for game in ids)
            or ids != sorted(set(ids))):
        raise ValueError("invalid game ID sample")
    dropped, aliases, rescanned, unusable, missing = (row[key] for key in COUNTS)
    if type(rescanned) is not int or rescanned < 0:
        raise ValueError("invalid rescan count")
    if reason == "unreadable_cache":
        if ids or any(value is not None for value in (dropped, aliases, unusable, missing)):
            raise ValueError("invalid whole-cache outcome")
    elif (any(type(value) is not int or value < 0 for value in (dropped, aliases, unusable, missing))
          or not 1 <= len(ids) <= dropped or dropped != aliases + rescanned + unusable + missing):
        raise ValueError("invalid drop counts")
    return row
