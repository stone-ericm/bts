"""Synthetic source builders for the season-ledger tests. They write the shapes production writes
(bts.picks.save_pick, bts.daily_decision.write_decision, scheduler save_state, the CLI contest-ledger
append); no test reads real data. Gzip output is pinned (mtime=0) so fixtures are byte-stable."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.audit.season_ledger.bundle import BundleEntry, write_manifest
from scripts.audit.season_ledger.ids import sha256_hex


def dumps(obj) -> bytes:
    return json.dumps(obj).encode()


def gz(data: bytes) -> bytes:
    return gzip.compress(data, mtime=0)


def seal_bundle(root, files: dict[str, bytes], missing=(), mtimes: dict[str, str] | None = None) -> None:
    root = Path(root)
    entries = []
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append(BundleEntry(rel_path=rel, status="present", sha256=sha256_hex(data), size=len(data),
                                   source_mtime_utc=(mtimes or {}).get(rel)))
    entries += [BundleEntry(rel_path=rel, status="missing", note="not found at acquisition") for rel in missing]
    write_manifest(root, entries, acquired_at_utc="2026-09-28T16:00:00.000000Z", builder_version="test")
