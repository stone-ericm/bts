"""W1.5 evidence observer: a pytest plugin that ``runner`` loads from a trusted absolute file.

It never changes an outcome, and it edits no source: the causal witness comes from
``sys.monitoring`` (PEP 669, Python 3.12), not from hooks compiled into the code under test
(Codex phase-1 r2 #2, #5, #6). Records are JSON lines (``kind`` field) written to the file named in
the config at ``$W15_OBS_CONFIG``:

* ``session_start`` — pid, executable, prefix, cwd, rootdir, inifile, pytest file + version, this
  observer's own file + sha256 as actually loaded, python version, run id;
* ``collect_error`` / ``collected`` (node ids + sha256 of every collected test file);
* ``report`` — every node x phase: outcome, xfail reason, exception module + qualname, whether an
  xfail was imperative, traceback frames (realpath, line, function), the exception message (first
  500 characters), the xfail marker's ``raises`` (module.qualname) and ``strict``;
* for the nodes in ``observe.nodes`` and only during their CALL phase: ``obs_start``/``obs_end``
  around ``entry`` (every start of a declared production entry function), ``branch`` (every
  execution of the declared line), ``boundary`` (every call made FROM production code — any code
  object under ``prod_root`` — to a declared boundary callable, resolved at call time so a mock
  installed mid-test counts; its identity is read from the callee's frame) and ``return`` (return
  values of declared functions). Each carries a global sequence number, the thread, a coroutine
  flag and the stack as ``[qualname, file, line, frame id]``;
* ``observer_error`` (any exception inside the observer; certificates are then refused);
* ``imports`` (loaded ``bts`` modules: file + sha256) and ``session_finish`` (exit status).

Limits (stated in every certificate): calls made from C code into a boundary are not observed, and
identity is unavailable for callables with neither a Python code object nor a mock ``__call__``.
"""
from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import os
import re
import sys
import threading

import pytest

TOOL_ID = 4
CO_ASYNC = inspect.CO_COROUTINE | inspect.CO_ITERABLE_COROUTINE | inspect.CO_ASYNC_GENERATOR
_SEQ = itertools.count(1)
_CONFIG: dict | None = None


def _config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        path = os.environ.get("W15_OBS_CONFIG")
        with open(path, encoding="utf-8") if path else _nullctx() as fh:
            _CONFIG = json.load(fh) if fh else {}
    return _CONFIG


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _write(record: dict) -> None:
    out = _config().get("out")
    if not out:
        return
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=repr) + "\n")


def _sha_file(path) -> str | None:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def _short(value, n: int = 300) -> str:
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - an unprintable value is still an observation
        text = f"<unprintable {type(value).__qualname__}>"
    return text[:n]


def _error(where: str, exc: BaseException) -> None:
    _write({"kind": "observer_error", "where": where, "error": f"{type(exc).__qualname__}: {exc}"[:300]})


# ---------------------------------------------------------------------------- session records

def pytest_configure(config):
    me = os.path.realpath(__file__)
    _write({"kind": "session_start", "pid": os.getpid(), "executable": sys.executable,
            "prefix": os.path.realpath(sys.prefix), "cwd": os.getcwd(),
            "rootdir": os.path.realpath(str(config.rootpath)),
            "inifile": os.path.realpath(str(config.inipath)) if config.inipath else None,
            "pytest_file": os.path.realpath(pytest.__file__), "pytest_version": pytest.__version__,
            "observer_file": me, "observer_sha256": _sha_file(me),
            "xfail_strict_ini": bool(config.getini("xfail_strict")),
            "python": sys.version.split()[0], "run_id": _config().get("run_id")})


def pytest_collectreport(report):
    if report.failed:
        head = str(report.longrepr).strip().splitlines()[-1:] if report.longrepr else [""]
        _write({"kind": "collect_error", "nodeid": report.nodeid, "head": head[0][:200]})


def pytest_collection_finish(session):
    files: dict[str, str | None] = {}
    for item in session.items:
        path = os.path.realpath(str(item.path))
        if path not in files:
            files[path] = _sha_file(path)
    _write({"kind": "collected", "nodeids": [item.nodeid for item in session.items], "files": files})


def _marker(item) -> dict | None:
    mark = item.get_closest_marker("xfail")
    if mark is None:
        return None
    raises = mark.kwargs.get("raises")
    names = None
    if raises is not None:
        seq = raises if isinstance(raises, tuple) else (raises,)
        names = [f"{r.__module__}.{r.__qualname__}" for r in seq]
    return {"raises": names, "strict": mark.kwargs.get("strict"), "condition_args": len(mark.args)}


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    try:
        exc = call.excinfo
        frames, message = [], None
        if exc is not None:
            for entry in exc.traceback:
                frames.append([os.path.realpath(str(entry.path)), entry.lineno + 1, entry.name])
            try:
                message = str(exc.value)[:500]
            except Exception:  # noqa: BLE001
                message = "<unprintable>"
        _write({"kind": "report", "nodeid": item.nodeid, "when": call.when, "outcome": rep.outcome,
                "wasxfail": getattr(rep, "wasxfail", None),
                "exc_module": exc.type.__module__ if exc is not None else None,
                "exc_qualname": exc.type.__qualname__ if exc is not None else None,
                "imperative_xfail": bool(exc is not None and isinstance(exc.value, pytest.xfail.Exception)),
                "frames": frames, "message": message, "marker": _marker(item)})
    except Exception as e:  # noqa: BLE001
        _error("makereport", e)
    return rep


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    observe = _config().get("observe") or {}
    if item.nodeid not in set(observe.get("nodes", [])):
        return (yield)
    mon = _Monitor(item.nodeid, observe, _config().get("prod_root") or "")
    mon.start()
    try:
        return (yield)
    finally:
        mon.stop()
        for rec in mon.events:
            _write(rec)


def pytest_sessionfinish(session, exitstatus):
    mods = {}
    for name, mod in list(sys.modules.items()):
        if name == "bts" or name.startswith("bts."):
            f = getattr(mod, "__file__", None)
            if f:
                mods[name] = {"file": os.path.realpath(f), "sha256": _sha_file(f)}
    _write({"kind": "imports", "modules": mods})
    _write({"kind": "session_finish", "exitstatus": int(exitstatus)})


# ---------------------------------------------------------------------------- the witness

_ACCESSOR = re.compile(r"args\[(\d+)\]|kw:(\w+)|arg0")


def _access(specs, args, kwargs):
    for spec in specs:
        m = _ACCESSOR.fullmatch(spec)
        if not m:
            continue
        if m.group(1) is not None and int(m.group(1)) < len(args):
            return True, args[int(m.group(1))]
        if m.group(2) is not None and m.group(2) in kwargs:
            return True, kwargs[m.group(2)]
    return False, None


def _resolve(binding: str):
    module, _, attr = binding.partition(":")
    obj = sys.modules.get(module)
    for part in attr.split("."):
        if obj is None:
            return None
        obj = getattr(obj, part, None)
    return obj


def _expect_code(callable_):
    """The code object whose PY_START follows a call to ``callable_`` (None when unknown)."""
    from unittest import mock
    if isinstance(callable_, mock.NonCallableMock):
        return mock.CallableMixin.__call__.__code__
    func = getattr(callable_, "__func__", callable_)
    code = getattr(func, "__code__", None)
    if code is None and isinstance(callable_, type):
        code = getattr(getattr(callable_, "__init__", None), "__code__", None)
    return code


class _Monitor:
    def __init__(self, nodeid: str, observe: dict, prod_root: str):
        self.nodeid = nodeid
        self.prod_root = os.path.realpath(prod_root) + os.sep if prod_root else None
        self.entries = {(os.path.realpath(e["file"]), e["qualname"]) for e in observe.get("entries", [])}
        br = observe.get("branch")
        self.branch = (os.path.realpath(br["file"]), int(br["line"])) if br else None
        self.returns = {(os.path.realpath(r["file"]), r["qualname"]) for r in observe.get("returns", [])}
        self.boundaries = [dict(b) for b in observe.get("boundaries", [])]
        self.events: list[dict] = []
        self.instrumented: set = set()
        self.callee_codes: set = set()
        self.pending: dict[int, dict] = {}
        self.paths: dict[str, str] = {}
        self.active = False
        self.mock_call_code = None

    # -- bookkeeping
    def _real(self, filename: str) -> str:
        path = self.paths.get(filename)
        if path is None:
            path = self.paths[filename] = os.path.realpath(filename)
        return path

    def _record(self, kind: str, fields: dict) -> dict:
        rec = {"kind": kind, "seq": next(_SEQ), "node": self.nodeid, "thread": threading.get_ident(),
               "pid": os.getpid(), **fields}
        self.events.append(rec)
        return rec

    def _err(self, where: str, exc: BaseException) -> None:
        self._record("observer_error", {"where": where, "error": f"{type(exc).__qualname__}: {exc}"[:300]})

    def _stack(self, frame):
        out, is_async, n = [], False, 0
        while frame is not None and n < 80:
            code = frame.f_code
            if code.co_flags & CO_ASYNC:
                is_async = True
            out.append([code.co_qualname, self._real(code.co_filename), frame.f_lineno, id(frame)])
            frame, n = frame.f_back, n + 1
        return out, is_async

    def _identity(self, spec: dict, args, kwargs) -> dict:
        ok, value = _access(spec.get("value", []), args, kwargs)
        if not ok:
            return {"value": None, "sha256": None, "category": "unavailable"}
        text = value if isinstance(value, str) else _short(value)
        category = "other"
        for label, pattern in spec.get("classify", []):
            if re.search(pattern, text):
                category = label
                break
        return {"value": _short(value), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "category": category}

    # -- lifecycle
    def start(self) -> None:
        mon = sys.monitoring
        try:
            mon.use_tool_id(TOOL_ID, "w15-observer")
        except ValueError as e:
            self._err("use_tool_id", e)
            return
        from unittest import mock
        self.mock_call_code = mock.CallableMixin.__call__.__code__
        self.callee_codes.add(self.mock_call_code)
        for b in self.boundaries:
            b["_orig"] = _resolve(b["binding"])
            code = _expect_code(b["_orig"]) if b["_orig"] is not None else None
            if code is not None:
                self.callee_codes.add(code)
        ev = mon.events
        mon.register_callback(TOOL_ID, ev.PY_START, self._on_start)
        mon.register_callback(TOOL_ID, ev.LINE, self._on_line)
        mon.register_callback(TOOL_ID, ev.CALL, self._on_call)
        mon.register_callback(TOOL_ID, ev.PY_RETURN, self._on_return)
        self.active = True
        self._record("obs_start", {})
        mon.set_events(TOOL_ID, ev.PY_START)

    def stop(self) -> None:
        if not self.active:
            return
        mon = sys.monitoring
        mon.set_events(TOOL_ID, 0)
        for code in self.instrumented:
            mon.set_local_events(TOOL_ID, code, 0)
        for event in (mon.events.PY_START, mon.events.LINE, mon.events.CALL, mon.events.PY_RETURN):
            mon.register_callback(TOOL_ID, event, None)
        mon.free_tool_id(TOOL_ID)
        mon.restart_events()
        self.active = False
        unresolved = [p["event"]["seq"] for p in self.pending.values()]
        self._record("obs_end", {"pending_identity": unresolved})

    # -- callbacks (never raise; never change control flow)
    def _on_start(self, code, offset):
        try:
            tid = threading.get_ident()
            pend = self.pending.get(tid)
            if pend is not None and code is pend["expect"]:
                frame = sys._getframe(1)
                loc = frame.f_locals
                if code is self.mock_call_code:
                    if loc.get("self") is pend["callable"]:
                        pend["event"]["identity"] = self._identity(pend["spec"], list(loc.get("args", ())),
                                                                   dict(loc.get("kwargs", {})))
                        del self.pending[tid]
                else:
                    names = code.co_varnames[:code.co_argcount]
                    args = [loc.get(n) for n in names]
                    kwargs = {n: loc.get(n) for n in code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]}
                    if code.co_flags & inspect.CO_VARARGS:
                        args += list(loc.get(code.co_varnames[code.co_argcount + code.co_kwonlyargcount], ()))
                    if code.co_flags & inspect.CO_VARKEYWORDS:
                        idx = code.co_argcount + code.co_kwonlyargcount + bool(code.co_flags & inspect.CO_VARARGS)
                        kwargs.update(loc.get(code.co_varnames[idx], {}) or {})
                    pend["event"]["identity"] = self._identity(pend["spec"], args, kwargs)
                    del self.pending[tid]
            if code in self.callee_codes:
                return None
            filename = self._real(code.co_filename)
            if self.prod_root is None or not filename.startswith(self.prod_root):
                return sys.monitoring.DISABLE
            if code not in self.instrumented:
                events = sys.monitoring.events.CALL
                if self.branch and filename == self.branch[0]:
                    events |= sys.monitoring.events.LINE
                if (filename, code.co_qualname) in self.returns:
                    events |= sys.monitoring.events.PY_RETURN
                sys.monitoring.set_local_events(TOOL_ID, code, events)
                self.instrumented.add(code)
            if (filename, code.co_qualname) in self.entries:
                frame = sys._getframe(1)
                stack, is_async = self._stack(frame)
                self._record("entry", {"file": filename, "qualname": code.co_qualname, "frame": id(frame),
                                       "stack": stack[:40], "async": is_async})
        except BaseException as e:  # noqa: BLE001
            self._err("py_start", e)
        return None

    def _on_line(self, code, line):
        try:
            if self.branch is None or line != self.branch[1] or self._real(code.co_filename) != self.branch[0]:
                return sys.monitoring.DISABLE
            stack, is_async = self._stack(sys._getframe(1))
            self._record("branch", {"file": self.branch[0], "line": line, "stack": stack[:40], "async": is_async})
        except BaseException as e:  # noqa: BLE001
            self._err("line", e)
        return None

    def _match(self, callable_):
        for b in self.boundaries:
            current = _resolve(b["binding"])
            for target in (current, b.get("_orig")):
                if target is None:
                    continue
                if callable_ is target or (getattr(callable_, "__func__", None) is not None
                                           and callable_.__func__ is getattr(target, "__func__", target)):
                    return b
        return None

    def _on_call(self, code, offset, callable_, arg0):
        try:
            spec = self._match(callable_)
            if spec is None:
                return None
            stack, is_async = self._stack(sys._getframe(1))
            rec = self._record("boundary", {"name": spec["name"], "caller": stack[0][:3], "stack": stack[:40],
                                            "async": is_async, "identity": None})
            expect = _expect_code(callable_)
            if expect is not None:
                self.callee_codes.add(expect)
                self.pending[threading.get_ident()] = {"event": rec, "callable": callable_, "expect": expect,
                                                       "spec": spec}
            elif "arg0" in spec.get("value", []):
                rec["identity"] = self._identity(spec, [arg0], {})
        except BaseException as e:  # noqa: BLE001
            self._err("call", e)
        return None

    def _on_return(self, code, offset, retval):
        try:
            filename = self._real(code.co_filename)
            if (filename, code.co_qualname) in self.returns:
                frame = sys._getframe(1)
                stack, is_async = self._stack(frame)
                self._record("return", {"file": filename, "qualname": code.co_qualname, "frame": id(frame),
                                        "value": _short(retval), "type": type(retval).__qualname__,
                                        "stack": stack[:40], "async": is_async})
        except BaseException as e:  # noqa: BLE001
            self._err("return", e)
        return None
