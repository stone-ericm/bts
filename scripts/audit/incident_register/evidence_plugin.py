"""Observational pytest plugin for W1.5 evidence runs (Codex phase-1 r1 #4, #5).

Load with ``-p scripts.audit.incident_register.evidence_plugin`` and set ``W15_EVIDENCE_OUT`` to
a JSONL path. From INSIDE the pytest process it records:

* ``session_start`` — interpreter, pid, cwd, the first sys.path entries;
* ``collected`` — every collected node id, in pytest's own form (``tests/x.py::Class::test``);
* ``report`` — one row per node per phase: outcome, xfail reason, the exception's class
  (``__qualname__`` + ``__module__``), whether it was an imperative ``pytest.xfail()``, the
  traceback frames (path, line, function) from the test file outward, and the xfail marker's
  ``raises``/``strict``;
* ``imports`` (at session end) — every loaded ``bts`` / ``bts.*`` module's file and sha256.

It never changes outcomes. Nothing is written when ``W15_EVIDENCE_OUT`` is unset.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ENV = "W15_EVIDENCE_OUT"


def _out():
    path = os.environ.get(ENV)
    return Path(path) if path else None


def _write(record: dict) -> None:
    path = _out()
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def pytest_sessionstart(session):
    _write({"kind": "session_start", "executable": sys.executable, "pid": os.getpid(),
            "cwd": os.getcwd(), "sys_path": sys.path[:12]})


def pytest_collection_finish(session):
    _write({"kind": "collected", "nodeids": [item.nodeid for item in session.items]})


def _marker(item) -> dict | None:
    mark = item.get_closest_marker("xfail")
    if mark is None:
        return None
    raises = mark.kwargs.get("raises")
    names = ([r.__qualname__ for r in raises] if isinstance(raises, tuple)
             else [raises.__qualname__] if raises is not None else None)
    return {"raises": names, "strict": mark.kwargs.get("strict"), "reason": mark.kwargs.get("reason")}


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    exc = call.excinfo
    frames = []
    if exc is not None:
        for entry in exc.traceback:
            frames.append({"path": str(entry.path), "line": entry.lineno + 1,
                           "func": entry.name})
    _write({
        "kind": "report", "nodeid": item.nodeid, "when": call.when, "outcome": rep.outcome,
        "wasxfail": getattr(rep, "wasxfail", None),
        "exc_type": exc.type.__qualname__ if exc is not None else None,
        "exc_module": exc.type.__module__ if exc is not None else None,
        "imperative_xfail": bool(exc is not None and isinstance(exc.value, pytest.xfail.Exception)),
        "frames": frames,
        "marker": _marker(item),
    })
    return rep


def pytest_sessionfinish(session, exitstatus):
    mods = {}
    for name, mod in list(sys.modules.items()):
        if name == "bts" or name.startswith("bts."):
            f = getattr(mod, "__file__", None)
            if f and os.path.exists(f):
                mods[name] = {"file": os.path.realpath(f),
                              "sha256": hashlib.sha256(Path(f).read_bytes()).hexdigest()}
    _write({"kind": "imports", "modules": mods, "exitstatus": int(exitstatus)})
