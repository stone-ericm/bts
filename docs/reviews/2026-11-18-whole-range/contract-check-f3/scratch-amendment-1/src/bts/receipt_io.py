"""Durable publication and discovery for producer receipts (watchdog plan P1/P2; producer review r1 C6).

**Publication** (`publish`):
- **Directories:** each missing directory is created one level at a time, and each new entry's parent is fsynced.
- **The file:** a temp file is written and fsynced, renamed to its final `*.json` name, then the directory is
  fsynced.
- **On any failure:** the temp file and any final file are removed, and the removal is fsynced. If removal fails,
  a `<name>.failed` tombstone is written instead. The error is then re-raised, so the caller reports the receipt
  unavailable.

**Sealed acceptance (producer review r2 C6):** a receipt counts only once its `<name>.sealed` marker exists. The
marker is written (temp file, fsync, rename) only **after** the receipt's own rename and directory fsync succeeded,
so a visible seal implies a complete, durable receipt.
- A failure at any earlier step leaves no seal, so the receipt is never discoverable. That includes the
  double refusal: the directory fsync fails, the removal fails, and the tombstone write fails.
- A failure while writing the seal does the same.
- If only the seal's own directory fsync fails, the seal is visible, and the receipt behind it is complete and
  durable. A crash can then lose the seal, which only withdraws acceptance (fail closed). Publication is reported as
  succeeded, with the seal's durability unconfirmed.

**Discovery** (`discover`) returns only `*.json` files that are sealed and not tombstoned.
"""
from __future__ import annotations

import os
from pathlib import Path

TOMBSTONE = ".failed"
SEAL = ".sealed"


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
    _seal(path, tmp)


def _seal(path: Path, receipt_tmp: Path) -> None:
    """The receipt is complete and durable: make it count. A failure before the seal is visible withdraws the
    receipt and re-raises; a failure of only the seal's directory fsync is tolerated (acceptance can then only be
    lost, never wrongly gained)."""
    seal = path.with_name(path.name + SEAL)
    seal_tmp = path.with_name(f".{seal.name}.{os.getpid()}.tmp")
    try:
        with open(seal_tmp, "wb") as f:
            f.write(b"sealed\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(seal_tmp, seal)
    except BaseException:
        try:
            seal_tmp.unlink(missing_ok=True)
        except OSError:
            pass
        _withdraw(path, receipt_tmp, True)
        raise
    try:
        _fsync_dir(path.parent)
    except OSError:
        pass


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
        _fsync_dir(path.parent)             # the tombstone's own directory entry (producer review r2 C6)
    except OSError:
        pass


def discover(directory: Path) -> list[Path]:
    """Published receipts in one directory: final `*.json` files without a tombstone, oldest name first."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.json")
                  if not p.name.startswith(".") and p.with_name(p.name + SEAL).exists()
                  and not p.with_name(p.name + TOMBSTONE).exists())
