"""Write-confinement tracing for the watchdog (registration §4.2 gate 2).

A process-wide audit hook (`sys.addaudithook`) records every write-type syscall event while a trace is active. It
sees writes at the syscall level, so a same-byte rewrite, or a write that is later undone, is caught whatever the
final file contents. Audit hooks cannot be removed, so the hook is installed once and records only inside `tracing`.
"""
import contextvars
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("watchdog_write_trace", default=None)
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_PATH_EVENTS = {"os.remove": (0,), "os.rmdir": (0,), "os.mkdir": (0,), "os.rename": (0, 1), "os.link": (0, 1),
                "os.symlink": (0, 1), "os.truncate": (0,), "os.chmod": (0,), "os.chown": (0,), "os.utime": (0,),
                "shutil.copyfile": (1,), "shutil.rmtree": (0,), "shutil.move": (0, 1)}


def _hook(event, args):
    sink = _ACTIVE.get()
    if sink is None:
        return
    if event == "open":
        path, mode, flags = (list(args) + [None, None, None])[:3]
        writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                 (isinstance(flags, int) and flags & _WRITE_FLAGS)
        if writes and isinstance(path, (str, bytes, os.PathLike)):
            sink.append(("open", os.fsdecode(path)))
    elif event in _PATH_EVENTS:
        for i in _PATH_EVENTS[event]:
            if i < len(args) and isinstance(args[i], (str, bytes, os.PathLike)):
                sink.append((event, os.fsdecode(args[i])))


sys.addaudithook(_hook)


@contextmanager
def tracing():
    sink: list = []
    token = _ACTIVE.set(sink)
    try:
        yield sink
    finally:
        _ACTIVE.reset(token)


def outside(sink, root: Path) -> list:
    """Traced write targets that do not resolve beneath `root`."""
    r = Path(root).resolve()
    bad = []
    for event, p in sink:
        target = Path(p)
        target = (target if target.is_absolute() else Path.cwd() / target)
        try:
            resolved = target.parent.resolve() / target.name
        except OSError:
            resolved = target
        if not resolved.is_relative_to(r):
            bad.append((event, p))
    return bad


@pytest.fixture
def write_trace():
    return tracing
