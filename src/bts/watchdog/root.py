"""The watchdog's owned root, `data/watchdog/` (registration R5; W0 review r1 B1).

**Descriptor-anchored, never by pathname:** every write goes through an open directory descriptor for the admitted
root. Each component is walked with descriptor-relative, no-follow opens (`O_DIRECTORY | O_NOFOLLOW`, `dir_fd`), so
a symlink swapped in after admission is refused **before** any mutation. A refusal is ELOOP or ENOTDIR, never a
post-write recheck.
- **Names:** a component is a plain name (letters, digits, `.`, `_`, `-`), never `.`, `..`, empty, or containing
  `/`.
- **Directories:** a missing one is created relative to its parent descriptor, and the parent is fsynced.
- **Files:** written to a unique, exclusively created temp (`O_CREAT | O_EXCL | O_NOFOLLOW`) in the parent
  directory, fsynced, renamed by descriptor (`os.replace` with `src_dir_fd` / `dst_dir_fd`), then the directory is
  fsynced. A failed write removes only its own temp.
- **Locks:** files opened no-follow by descriptor, then flocked.

**What is and is not refused (W0 review r2 qualification):**
- **Refused:** a symlink at any **directory** component (the root included), and a symlink at a lock or read
  leaf. Each is refused by no-follow opens, never followed.
- **Replaced, not followed:** `write_atomic`'s final `os.replace` replaces a destination name that is a symlink with
  the new file, rather than writing through it.
- **Ownership assumption:** descriptor anchoring keeps operations on the admitted directory **inode**; it does not
  pin that inode to its original pathname. The watchdog never renames its own directories. Keeping the admitted
  namespace in place (`data/watchdog` not moved by another actor) is a deployment and ownership assumption. A
  relocation by an outside actor is not a symlink redirection, and is outside the watchdog's control.
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import re
import uuid
from pathlib import Path

NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


class RootError(RuntimeError):
    pass


class JobBusy(RuntimeError):
    pass


def _name(part) -> str:
    if not isinstance(part, str) or part in (".", "..") or not NAME_RE.fullmatch(part):
        raise RootError(f"refused path component {part!r}")
    return part


def _open_dir(name: str, dir_fd: int) -> int:
    try:
        return os.open(name, _DIR, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.EMLINK):
            raise RootError(f"{name} is a symlink or not a directory: refused") from None
        raise


class OwnedRoot:
    NAME = "watchdog"

    def __init__(self, path: Path, fd: int):
        self.path = path                     # for display and reporting only; never an authority for writes
        self._fd = fd

    @classmethod
    def under(cls, data_dir: Path) -> "OwnedRoot":
        data_dir = Path(data_dir)
        try:
            data_fd = os.open(data_dir, os.O_RDONLY | os.O_DIRECTORY)
        except OSError as exc:
            raise RootError(f"{data_dir} is not a directory: {exc}") from None
        try:
            try:
                os.mkdir(cls.NAME, 0o755, dir_fd=data_fd)
                os.fsync(data_fd)
            except FileExistsError:
                pass
            try:
                fd = os.open(cls.NAME, _DIR, dir_fd=data_fd)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.EMLINK):
                    raise RootError(f"{data_dir / cls.NAME} is a symlink: refused") from None
                raise
        finally:
            os.close(data_fd)
        return cls(data_dir.resolve() / cls.NAME, fd)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    # ---- names and directories --------------------------------------------------------------------------------------
    def child(self, *parts: str) -> Path:
        """The display path of validated components (no filesystem access, no authority)."""
        return self.path.joinpath(*[_name(p) for p in parts])

    def _dir(self, parts, *, create: bool) -> int | None:
        """A new descriptor for the directory `parts` beneath the root (caller closes), or None if it is absent and
        create is false."""
        cur = os.dup(self._fd)
        try:
            for part in parts:
                name = _name(part)
                try:
                    nxt = _open_dir(name, cur)
                except FileNotFoundError:
                    if not create:
                        os.close(cur)
                        return None
                    try:
                        os.mkdir(name, 0o755, dir_fd=cur)
                        os.fsync(cur)                       # the new entry is durable in its parent
                    except FileExistsError:
                        pass
                    nxt = _open_dir(name, cur)
                os.close(cur)
                cur = nxt
            return cur
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(cur)
            raise

    def ensure_dir(self, *parts: str) -> Path:
        os.close(self._dir(parts, create=True))
        return self.child(*parts)

    # ---- files ----------------------------------------------------------------------------------------------------
    def write_atomic(self, parts, data: bytes) -> Path:
        *dirs, leaf = [_name(p) for p in parts]
        dfd = self._dir(dirs, create=True)
        tmp = f".{leaf}.{uuid.uuid4().hex}.tmp"
        created = False
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dfd)
            created = True
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, leaf, src_dir_fd=dfd, dst_dir_fd=dfd)
            created = False
            os.fsync(dfd)
        except BaseException:
            if created:
                with contextlib.suppress(OSError):
                    os.unlink(tmp, dir_fd=dfd)
            raise
        finally:
            os.close(dfd)
        return self.child(*parts)

    def read_bytes(self, parts) -> bytes | None:
        *dirs, leaf = [_name(p) for p in parts]
        dfd = self._dir(dirs, create=False)
        if dfd is None:
            return None
        try:
            try:
                fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
            except FileNotFoundError:
                return None
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise RootError(f"{leaf} is a symlink: refused") from None
                raise
            with os.fdopen(fd, "rb") as f:
                return f.read()
        finally:
            os.close(dfd)

    def list_dir(self, parts) -> list[str]:
        dfd = self._dir(parts, create=False)
        if dfd is None:
            return []
        try:
            return sorted(os.listdir(dfd))
        finally:
            os.close(dfd)

    @contextlib.contextmanager
    def lock(self, parts, *, blocking: bool):
        *dirs, leaf = [_name(p) for p in parts]
        dfd = self._dir(dirs, create=True)
        try:
            try:
                fd = os.open(leaf, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644, dir_fd=dfd)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise RootError(f"{leaf} is a symlink: refused") from None
                raise
        finally:
            os.close(dfd)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                raise JobBusy(f"{leaf} is held") from None
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    @contextlib.contextmanager
    def job_lock(self, job: str):
        """A job singleton (non-blocking): a second run of the same job refuses with JobBusy."""
        try:
            with self.lock(("locks", f"job-{_name(job)}.lock"), blocking=False):
                yield
        except JobBusy:
            raise JobBusy(f"watchdog job {job!r} is already running") from None
