"""Sealed evidence bundle: manifest writing (acquire) and verification (compile). Spec §3."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from .ids import sha256_hex

MANIFEST_NAME = "manifest.json"
BUNDLE_SCHEMA = "bts_season_ledger_bundle_v1"


class BundleError(Exception):
    pass


@dataclass(frozen=True)
class BundleEntry:
    rel_path: str
    status: str                       # "present" | "missing"
    sha256: str | None = None
    size: int | None = None
    source_path: str | None = None    # relative to the acquisition source root, or the fetch URL
    source_mtime_utc: str | None = None
    note: str | None = None


def check_rel_path(rel: str) -> str:
    p = PurePosixPath(rel)
    if not rel or p.is_absolute() or ".." in p.parts or p.as_posix() != rel:
        raise BundleError(f"unsafe bundle path: {rel!r}")
    return rel


def write_manifest(root: Path, entries: list[BundleEntry], *, acquired_at_utc: str, builder_version: str,
                   source_root: str | None = None) -> Path:
    for e in entries:
        check_rel_path(e.rel_path)
        if e.status not in ("present", "missing"):
            raise BundleError(f"bad status {e.status!r} for {e.rel_path}")
    body = {"schema": BUNDLE_SCHEMA, "acquired_at_utc": acquired_at_utc, "builder_version": builder_version,
            "source_root": source_root,
            "entries": sorted((asdict(e) for e in entries), key=lambda d: d["rel_path"])}
    path = Path(root) / MANIFEST_NAME
    path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    return path


def open_bundle(root: Path) -> tuple[dict, dict[str, bytes | None]]:
    """Verify every declared entry; return (manifest, {rel_path: bytes | None}) sorted by path. A declared
    `missing` entry is valid evidence of absence; a declared-present file that is absent or changed is
    refused. Files the manifest does not declare are ignored."""
    root = Path(root)
    try:
        manifest = json.loads((root / MANIFEST_NAME).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"unreadable manifest: {exc}") from exc
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise BundleError(f"unexpected bundle schema {manifest.get('schema')!r}")
    files: dict[str, bytes | None] = {}
    for e in sorted(manifest["entries"], key=lambda e: e["rel_path"]):
        rel = check_rel_path(e["rel_path"])
        if rel in files:
            raise BundleError(f"duplicate manifest entry {rel}")
        if e["status"] == "missing":
            files[rel] = None
            continue
        path = root / rel
        if not path.is_file():
            raise BundleError(f"declared present but absent: {rel}")
        data = path.read_bytes()
        if sha256_hex(data) != e["sha256"]:
            raise BundleError(f"hash mismatch: {rel}")
        files[rel] = data
    return manifest, files
