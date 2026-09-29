"""Occurrence identity, the typed-field policy and small shared helpers."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_STAMP = re.compile(r"(\d{8}T\d{6})Z")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def obs_id(rel_path: str, locator: str, content_sha256: str) -> str:
    """Identity of one source occurrence: path + locator + content hash, so byte-identical files at
    two paths remain two occurrences (spec §4)."""
    return hashlib.sha256(f"{rel_path}\x00{locator}\x00{content_sha256}".encode()).hexdigest()[:24]


@dataclass
class Parsed:
    rows: list[dict] = field(default_factory=list)
    quarantined: list[dict] = field(default_factory=list)


_NO_RAW = object()     # `raw` omitted: the record never decoded (a parsed JSON null is passed as None)


def quarantine(rel_path: str, locator: str, reason: str, raw=_NO_RAW) -> dict:
    """A quarantined occurrence. Every quarantined record that parsed passes its own parsed value as `raw`
    (kept as JSON in the occurrence table; a parsed null is "null"). Only undecodable bytes (bad gzip, invalid
    JSON, not UTF-8) and path-level refusals omit it; they stay in the sealed bundle, pinned by the manifest."""
    return {"source_path": rel_path, "locator": locator, "reason": reason,
            "record_raw_json": None if raw is _NO_RAW else canonical_json(raw)}


def is_int(value) -> bool:
    """An integer that fits the int64 output columns (bools are not integers here)."""
    return isinstance(value, int) and not isinstance(value, bool) and -(2 ** 63) <= value < 2 ** 63


_KINDS = {"int": is_int, "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
          "bool": lambda v: isinstance(v, bool), "str": lambda v: isinstance(v, str)}


def typed(value, kind: str) -> tuple[object, bool]:
    """Typed-field policy (Interpretation I13): (value, False) when it fits `kind` (a float field accepts
    an int and returns a float); (None, False) for None; (None, True) for a present value of the wrong
    type. The raw value always survives in the record's raw JSON."""
    if value is None:
        return None, False
    if not _KINDS[kind](value):
        return None, True
    return (float(value) if kind == "float" else value), False


def take(record: dict, fields: dict[str, str], prefix: str = "") -> tuple[dict, list[str], list[str]]:
    """Typed values for `fields` ({name: kind}), plus the names that were absent and the names whose value
    had the wrong type (both qualified with `prefix`, e.g. 'pick.')."""
    values, absent, mismatched = {}, [], []
    for name, kind in fields.items():
        if name not in record:
            absent.append(prefix + name)
        value, bad = typed(record.get(name), kind)
        if bad:
            mismatched.append(prefix + name)
        values[name] = value
    return values, absent, mismatched


def joined(names) -> str | None:
    return ",".join(sorted(names)) or None


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_json_bytes(data: bytes):
    """Parse JSON from plain or gzip bytes; raise ValueError('bad_gzip:…' / 'invalid_json:…')."""
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError, zlib.error) as exc:
            raise ValueError(f"bad_gzip:{exc.__class__.__name__}") from exc
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid_json:{exc.__class__.__name__}") from exc


def utc_iso(raw) -> str | None:
    """ISO-8601 with an offset → fixed-precision UTC ('YYYY-MM-DDTHH:MM:SS.ffffffZ'), so string order is
    time order. None, non-strings, unparseable and naive values → None: a naive time is never read in
    the host's zone."""
    if not isinstance(raw, str):
        return None
    try:
        t = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if t.tzinfo is None or t.utcoffset() is None:
        return None
    return t.astimezone(timezone.utc).strftime(UTC_FORMAT)


def stamp_to_utc(name: str) -> str | None:
    """Static-capture file names carry a UTC stamp 'YYYYmmddTHHMMSSZ'."""
    m = _STAMP.search(name)
    if m is None:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).strftime(UTC_FORMAT)
