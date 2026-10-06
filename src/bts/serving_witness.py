"""Serving provenance for each persisted slate (C2 step 2a; design `docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md`).

The containment helpers below are shared by every capture point (design §3.0). Each contains its own failure: a
provenance problem is recorded in a local error list and never replaces, retries or alters the computation.
"""
from __future__ import annotations

import hashlib
import json


def note(errors, msg: str) -> None:
    """Record a provenance error; recording itself never raises."""
    if errors is None:
        return
    try:
        errors.append(msg)
    except Exception:
        pass


def collect(items, item, errors, what: str) -> None:
    """Append to a provenance collector; a failed append is recorded, never raised."""
    if items is None:
        return
    try:
        items.append(item)
    except Exception as e:
        note(errors, f"{what}: collector append failed: {e!r}")


def sha256_or_none(raw, errors, what: str):
    """sha256 hex of the held bytes, or None with a recorded error."""
    try:
        return hashlib.sha256(raw).hexdigest()
    except Exception as e:
        note(errors, f"{what}: sha256 failed: {e!r}")
        return None


def canon_sha256(obj) -> str:
    """sha256 of canonical JSON (sorted keys, no whitespace, UTF-8); a non-finite float raises ValueError."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
