"""Durable publication and discovery for producer receipts (watchdog plan P1/P2; producer review r1 C6).

**Publication** (`publish`):
- **Directories:** each missing directory is created one level at a time, and each new entry's parent is fsynced.
- **The file:** a temp file is written and fsynced, renamed to its final `*.json` name, then the directory is
  fsynced.
- **On any failure:** the temp file and any final file are removed, and the removal is fsynced. If removal fails,
  a `<name>.failed` tombstone is written instead. The error is then re-raised, so the caller reports the receipt
  unavailable.

**Discovery** (`discover`) returns only `*.json` files without a tombstone. A failed publication therefore cannot
be read as a published receipt. The exception is a failure that removes nothing and cannot write a tombstone
either; the filesystem then refused every write, which is the stated limit.
"""
from __future__ import annotations

import os
from pathlib import Path

TOMBSTONE = ".failed"


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_dir(path: Path) -> None:
    """Create each missing level and fsync its parent, so every new directory entry is durable."""
    path = Path(path)
    missing = []
    p = path
    while not p.exists():
        missing.append(p)
        p = p.parent
    for d in reversed(missing):
        d.mkdir()
        _fsync_dir(d.parent)


def publish(path: Path, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    renamed = False
    try:
        ensure_dir(path.parent)
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        renamed = True
        _fsync_dir(path.parent)
    except BaseException:
        _withdraw(path, tmp, renamed)
        raise


def _withdraw(path: Path, tmp: Path, renamed: bool) -> None:
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    if not renamed:
        return
    try:
        path.unlink(missing_ok=True)
        _fsync_dir(path.parent)
        return
    except OSError:
        pass
    try:                                    # the final file may survive: mark it failed for every consumer
        with open(path.with_name(path.name + TOMBSTONE), "wb") as f:
            f.write(b"publication failed\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass


def discover(directory: Path) -> list[Path]:
    """Published receipts in one directory: final `*.json` files without a tombstone, oldest name first."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.json")
                  if not p.name.startswith(".") and not p.with_name(p.name + TOMBSTONE).exists())
