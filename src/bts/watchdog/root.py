"""The watchdog's owned root, `data/watchdog/` (registration R5).

Every watchdog write goes through `OwnedRoot.child`.
- **Refused:** an absolute or `..` component, a resolved path outside the root, or a symlink anywhere from the root
  down (the root included).
- **Allowed:** directories are created only beneath the root.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
from pathlib import Path


class RootError(RuntimeError):
    pass


class JobBusy(RuntimeError):
    pass


class OwnedRoot:
    NAME = "watchdog"

    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def under(cls, data_dir: Path) -> "OwnedRoot":
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise RootError(f"{data_dir} is not a directory")
        raw = data_dir / cls.NAME
        if raw.is_symlink():
            raise RootError(f"{raw} is a symlink: refused")
        raw.mkdir(exist_ok=True)
        if raw.is_symlink() or not raw.is_dir():
            raise RootError(f"{raw} is not a real directory (symlink refused)")
        return cls(raw.resolve())

    def child(self, *parts: str) -> Path:
        """A path beneath the root. Every existing component from the root down must be a real (non-symlink)
        entry; the result must resolve inside the root."""
        for part in parts:
            p = Path(part)
            if p.is_absolute() or ".." in p.parts:
                raise RootError(f"refused path component {part!r}")
        target = self.path.joinpath(*parts)
        cur = self.path
        for part in Path(*parts).parts:
            cur = cur / part
            if cur.is_symlink():
                raise RootError(f"{cur} is a symlink: refused")
        resolved = target.parent.resolve() / target.name if target.parent.exists() else target
        if not resolved.is_relative_to(self.path):
            raise RootError(f"{target} escapes {self.path}")
        return target

    def ensure_dir(self, *parts: str) -> Path:
        d = self.child(*parts)
        d.mkdir(parents=True, exist_ok=True)
        self.child(*parts)                          # re-check: nothing created may be a symlink
        return d

    @contextlib.contextmanager
    def job_lock(self, job: str):
        """A job singleton (non-blocking): a second run of the same job refuses with JobBusy."""
        self.ensure_dir("locks")
        path = self.child("locks", f"job-{job}.lock")
        with open(path, "a") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise JobBusy(f"watchdog job {job!r} is already running") from None
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def write_atomic(self, path: Path, data: bytes) -> None:
        """Atomic, durable write of a file beneath the root (temp, fsync, replace, directory fsync)."""
        rel = path.relative_to(self.path)
        target = self.child(*rel.parts)
        if len(rel.parts) > 1:
            self.ensure_dir(*rel.parts[:-1])
        tmp = self.child(*rel.parts[:-1], f".{rel.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
