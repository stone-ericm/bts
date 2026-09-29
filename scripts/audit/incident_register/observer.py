"""W1.5 evidence observer: a pytest plugin that ``runner`` loads from a trusted absolute file.

The causal witness comes from ``sys.monitoring`` (PEP 669, Python 3.12), not from hooks compiled
into the code under test. The observer must be OBSERVATIONAL (design §9.3 as amended, Codex
phase-1 r3 #1, #2, #7):

* it never runs application-defined code to obtain an observation — no ``repr``/``str``/properties
  of application objects: values are serialized by ``_safe`` from exact primitive types and from a
  plain instance's raw ``__dict__`` (read through the standard C descriptor); anything else is
  recorded as unavailable;
* boundary calls are recorded on the CALLEE side (the start of a boundary mock's ``__call__`` or of
  the real boundary function), so calls made from C (``map``, callbacks) or from other threads are
  still seen; a boundary that is not a Python-observable callable, or a binding replaced by an unseen
  callable, is recorded as a coverage gap;
* the entry's exits (return or exception) are recorded, and threads started during the observed
  call phase that are still alive at its end are counted, so a certificate can require completion
  and refuse an interval with outstanding work.

Records are JSON lines (``kind`` field) written to the file named in the config at
``$W15_OBS_CONFIG``: ``session_start`` (identity + ``src_digest``), ``collect_error``,
``collected`` (node ids + test-file sha256), ``report`` (every node x phase), and for the nodes in
``observe.nodes`` during their call phase ``obs_start`` … ``obs_end`` around ``entry``,
``entry_exit``, ``branch``, ``boundary``, ``boundary_gap``, ``return``; then ``observer_error``,
``imports`` and ``session_finish`` (with ``src_digest`` again).
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
_TYPE_QUALNAME = type.__dict__["__qualname__"]
_TYPE_MODULE = type.__dict__["__module__"]
_TYPE_DICT = type.__dict__["__dict__"]
_TYPE_MRO = type.__dict__["__mro__"]
_GETSET = type(type.__dict__["__dict__"])


def _config() -> dict:
    global _CONFIG
    if _CONFIG is None:
        path = os.environ.get("W15_OBS_CONFIG")
        if path:
            with open(path, encoding="utf-8") as fh:
                _CONFIG = json.load(fh)
        else:
            _CONFIG = {}
    return _CONFIG


def _write(record: dict) -> None:
    out = _config().get("out")
    if not out:
        return
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _sha_file(path) -> str | None:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def src_digest(root: str | None) -> str | None:
    """sha256 over (relative path, bytes) of every .py file under ``root``, sorted."""
    if not root or not os.path.isdir(root):
        return None
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                p = os.path.join(dirpath, name)
                h.update(os.path.relpath(p, root).encode() + b"\0")
                with open(p, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()


def type_name(t) -> str:
    """A class's module + qualname read through type's own C descriptors (no metaclass code)."""
    try:
        return f"{_TYPE_MODULE.__get__(t, type)}.{_TYPE_QUALNAME.__get__(t, type)}"
    except Exception:  # noqa: BLE001
        return "<unnamed>"


def _plain_instance_dict(value):
    """The instance's raw __dict__ when its class uses the standard C __dict__ descriptor, else None."""
    t = type(value)
    for cls in _TYPE_MRO.__get__(t, type):
        ns = _TYPE_DICT.__get__(cls, type)
        if "__dict__" in ns:
            if type(ns["__dict__"]) is not _GETSET:
                return None
            break
    try:
        d = object.__getattribute__(value, "__dict__")
    except Exception:  # noqa: BLE001
        return None
    return d if type(d) is dict else None


def _safe_items(items, key: str, depth: int, *, skip_private: bool = False) -> dict:
    """Copy (str key, value) pairs from a dict's items view: iterating items never looks a key up again,
    so no application ``__eq__``/``__hash__`` runs (Codex phase-1 r4 #2). Anything omitted marks the
    form ``incomplete``."""
    pairs, incomplete = {}, False
    for n, (k, v) in enumerate(items):
        if n >= 50:
            incomplete = True
            break
        if type(k) is not str or (skip_private and k.startswith("_")):
            incomplete = True
            continue
        pairs[k] = _safe(v, depth + 1)
    out = {key: pairs}
    if incomplete:
        out["incomplete"] = True
    return out


def _safe(value, depth: int = 0):
    """JSON-safe copy built without running application code (see module doc). Whatever is omitted — a
    truncated string, sequence or map, a non-string key, a private field, the depth limit, an object
    without a plain ``__dict__`` — marks the form ``incomplete``; an incomplete form never compares
    equal to anything (``acceptance.unwrap``), so a partial value can never witness a whole one."""
    t = type(value)
    if value is None or t is bool or t is int or t is float:
        return value
    if t is str:
        return value if len(value) <= 500 else {"str": value[:500], "length": len(value), "incomplete": True}
    if depth >= 3:
        return {"unavailable": "depth", "incomplete": True}
    if t is tuple or t is list:
        out = {"seq": [_safe(v, depth + 1) for v in value[:50]]}
        if len(value) > 50:
            out.update(length=len(value), incomplete=True)
        return out
    if t is dict:
        return _safe_items(dict.items(value), "map", depth)
    d = _plain_instance_dict(value)
    if d is None:
        return {"type": type_name(t), "unavailable": "no plain __dict__", "incomplete": True}
    out = _safe_items(dict.items(d), "fields", depth, skip_private=True)
    out["type"] = type_name(t)
    return out


def _text(safe_value) -> str:
    return safe_value if type(safe_value) is str else json.dumps(safe_value, sort_keys=True)


def _error(where: str, exc: BaseException) -> None:
    _write({"kind": "observer_error", "where": where, "error_type": type_name(type(exc))})


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
            "src_digest": src_digest(_config().get("prod_root")),
            "python": sys.version.split()[0], "run_id": _config().get("run_id")})


def pytest_collectreport(report):
    if report.failed:
        _write({"kind": "collect_error", "nodeid": report.nodeid})


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
        seq = raises if type(raises) is tuple else (raises,)
        names = [type_name(r) for r in seq]
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
            crash = getattr(getattr(rep, "longrepr", None), "reprcrash", None)
            text = getattr(crash, "message", None)
            message = text[:500] if type(text) is str else None
        name = type_name(exc.type) if exc is not None else None
        _write({"kind": "report", "nodeid": item.nodeid, "when": call.when, "outcome": rep.outcome,
                "wasxfail": getattr(rep, "wasxfail", None),
                "exc_module": name.rsplit(".", 1)[0] if name else None,
                "exc_qualname": _TYPE_QUALNAME.__get__(exc.type, type) if exc is not None else None,
                "imperative_xfail": bool(exc is not None and issubclass(exc.type, pytest.xfail.Exception)),
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
            ns = getattr(mod, "__dict__", None)
            f = ns.get("__file__") if type(ns) is dict else None
            if type(f) is str:
                mods[name] = {"file": os.path.realpath(f), "sha256": _sha_file(f)}
    _write({"kind": "imports", "modules": mods})
    _write({"kind": "session_finish", "exitstatus": int(exitstatus),
            "src_digest": src_digest(_config().get("prod_root"))})


# ---------------------------------------------------------------------------- the witness

_ACCESSOR = re.compile(r"args\[(\d+)\]|kw:(\w+)")


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
    """The object currently bound at ``module:attr[.attr]``, read from namespaces (no getattr hooks)."""
    module, _, attr = binding.partition(":")
    obj = sys.modules.get(module)
    for part in attr.split("."):
        if obj is None:
            return None
        ns = getattr(obj, "__dict__", None)
        obj = ns.get(part) if type(ns) is dict else None
    return obj


def _classify(rules, safe_value) -> str:
    text = _text(safe_value)
    for label, pattern in rules:
        if re.search(pattern, text):
            return label
    return "other"


class _Monitor:
    def __init__(self, nodeid: str, observe: dict, prod_root: str):
        from unittest import mock
        self.mock_base = mock.NonCallableMock
        self.mock_call_code = mock.CallableMixin.__call__.__code__
        self.nodeid = nodeid
        self.prod_root = os.path.realpath(prod_root) + os.sep if prod_root else None
        self.entries = {(os.path.realpath(e["file"]), e["qualname"]) for e in observe.get("entries", [])}
        br = observe.get("branch")
        self.branch = (os.path.realpath(br["file"]), int(br["line"])) if br else None
        self.returns = {(os.path.realpath(r["file"]), r["qualname"]): r.get("classify", [])
                        for r in observe.get("returns", [])}
        self.boundaries = [dict(b) for b in observe.get("boundaries", [])]
        self.events: list[dict] = []
        self.instrumented: set = set()
        self.boundary_codes: dict = {}          # code object -> [(boundary spec, bound receiver or None)]
        self.seen_targets: dict = {}            # id(obj) -> obj: every binding value registered
        self.paths: dict[str, str] = {}
        self.threads_at_start: set = set()
        self.active = False

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
        name = type_name(type(exc))
        # the observer's own failures are builtin exceptions; their text runs no application code
        detail = "".join(__import__("traceback").format_exception(exc))[-800:] if name.startswith("builtins.") else None
        self._record("observer_error", {"where": where, "error_type": name, "detail": detail})

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
            return {"value": None, "category": "unavailable"}
        safe = _safe(value)
        incomplete = type(safe) is dict and bool(safe.get("incomplete"))
        return {"value": safe, "sha256": hashlib.sha256(_text(safe).encode()).hexdigest(),
                "category": "unavailable" if incomplete else _classify(spec.get("classify", []), safe)}

    def _is_mock(self, obj) -> bool:
        return issubclass(type(obj), self.mock_base)

    def _code_of(self, obj):
        """The code object a call to ``obj`` starts, for plain functions and bound methods only."""
        kind = type_name(type(obj))
        if kind == "builtins.method":
            obj = obj.__func__
            kind = type_name(type(obj))
        return obj.__code__ if kind == "builtins.function" else None

    def _register(self, spec: dict, obj) -> None:
        if obj is None:
            self._record("boundary_gap", {"name": spec["name"], "reason": "the declared binding resolves to nothing"})
            return
        if id(obj) in self.seen_targets:
            return
        self.seen_targets[id(obj)] = obj
        if self._is_mock(obj):
            return                                           # every mock call starts mock_call_code
        code = self._code_of(obj)
        if code is None:
            self._record("boundary_gap", {"name": spec["name"], "reason": "not a Python-observable callable",
                                          "type": type_name(type(obj))})
            return
        # a bound method is the boundary only for ITS receiver: other instances share the code object
        # (Codex phase-1 r4 #1.4); method.__self__ is a C member, no application code runs
        receiver = obj.__self__ if type_name(type(obj)) == "builtins.method" else None
        self.boundary_codes.setdefault(code, []).append((spec, receiver))
        if self.active:
            sys.monitoring.restart_events()                  # its PY_START may have been disabled

    def _mock_spec(self, obj):
        if obj is None:
            return None
        for spec in self.boundaries:
            if _resolve(spec["binding"]) is obj or spec.get("_start") is obj:
                return spec
        return None

    # -- lifecycle
    def start(self) -> None:
        mon = sys.monitoring
        try:
            mon.use_tool_id(TOOL_ID, "w15-observer")
        except ValueError as e:
            self._err("use_tool_id", e)
            return
        self.threads_at_start = {t.ident for t in threading.enumerate()}
        # obs_start FIRST: a gap found while registering the boundaries belongs to this interval
        # (Codex phase-1 r4 #1.1: it used to precede obs_start and so fell outside the certificate)
        self._record("obs_start", {})
        for spec in self.boundaries:
            spec["_start"] = _resolve(spec["binding"])
            self._register(spec, spec["_start"])
        ev = mon.events
        mon.register_callback(TOOL_ID, ev.PY_START, self._on_start)
        mon.register_callback(TOOL_ID, ev.LINE, self._on_line)
        mon.register_callback(TOOL_ID, ev.CALL, self._on_call)
        mon.register_callback(TOOL_ID, ev.PY_RETURN, self._on_return)
        mon.register_callback(TOOL_ID, ev.PY_UNWIND, self._on_unwind)
        self.active = True
        mon.set_events(TOOL_ID, ev.PY_START | ev.PY_UNWIND)

    def stop(self) -> None:
        if not self.active:
            return
        mon = sys.monitoring
        mon.set_events(TOOL_ID, 0)
        for code in self.instrumented:
            mon.set_local_events(TOOL_ID, code, 0)
        for event in (mon.events.PY_START, mon.events.LINE, mon.events.CALL, mon.events.PY_RETURN,
                      mon.events.PY_UNWIND):
            mon.register_callback(TOOL_ID, event, None)
        mon.free_tool_id(TOOL_ID)
        mon.restart_events()
        self.active = False
        for spec in self.boundaries:
            now = _resolve(spec["binding"])
            if now is not None and id(now) not in self.seen_targets and not self._is_mock(now):
                self._record("boundary_gap", {"name": spec["name"], "reason": "binding replaced by an unseen callable"})
        me = threading.get_ident()
        alive = [t for t in threading.enumerate() if t.is_alive() and t.ident != me]
        self._record("obs_end", {
            "outstanding_threads": sum(1 for t in alive if t.ident not in self.threads_at_start),
            # a worker that was already running can also do the declared work after the interval
            # (Codex phase-1 r4 #1.3)
            "preexisting_threads_alive": sum(1 for t in alive if t.ident in self.threads_at_start)})

    # -- callbacks (never raise; never run application code)
    def _on_start(self, code, offset):
        spec = None
        try:
            if code is self.mock_call_code:
                frame = sys._getframe(1)
                loc = frame.f_locals
                spec = self._mock_spec(loc.get("self"))
                if spec is not None:
                    extra = loc.get("args", ())
                    kwargs = loc.get("kwargs", {})
                    self._boundary(spec, frame, list(extra) if type(extra) is tuple else [],
                                   dict(kwargs) if type(kwargs) is dict else {})
                return None
            pairs = self.boundary_codes.get(code)
            if pairs is not None:
                spec = pairs[0][0]
                frame = sys._getframe(1)
                loc = frame.f_locals
                positional = code.co_varnames[:code.co_argcount]
                args = [loc.get(n) for n in positional]
                if code.co_flags & inspect.CO_VARARGS:
                    extra = loc.get(code.co_varnames[code.co_argcount + code.co_kwonlyargcount], ())
                    args += list(extra) if type(extra) is tuple else []
                kwargs = {n: loc.get(n) for n in code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]}
                first = args[0] if args else None
                matched = [sp for sp, rcv in pairs if rcv is None or first is rcv]
                if matched:
                    self._boundary(matched[0], frame, args, kwargs)
                else:                                        # same code, another receiver: ambiguous
                    self._boundary(spec, frame, args, kwargs, other_receiver=True)
            filename = self._real(code.co_filename)
            if self.prod_root is None or not filename.startswith(self.prod_root):
                return None if spec is not None else sys.monitoring.DISABLE
            if code not in self.instrumented:
                events = sys.monitoring.events.CALL
                if self.branch and filename == self.branch[0]:
                    events |= sys.monitoring.events.LINE
                key = (filename, code.co_qualname)
                if key in self.returns or key in self.entries:
                    events |= sys.monitoring.events.PY_RETURN       # PY_UNWIND is global-only (3.12)
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

    def _boundary(self, spec: dict, callee_frame, args, kwargs, *, other_receiver: bool = False) -> None:
        stack, is_async = self._stack(callee_frame.f_back)
        identity = ({"value": None, "category": "unavailable", "reason": "another receiver of the boundary's code"}
                    if other_receiver else self._identity(spec, args, kwargs))
        self._record("boundary", {"name": spec["name"], "caller": stack[0][:3] if stack else None,
                                  "stack": stack[:40], "async": is_async, "identity": identity})

    def _on_line(self, code, line):
        try:
            if self.branch is None or line != self.branch[1] or self._real(code.co_filename) != self.branch[0]:
                return sys.monitoring.DISABLE
            stack, is_async = self._stack(sys._getframe(1))
            self._record("branch", {"file": self.branch[0], "line": line, "stack": stack[:40], "async": is_async})
        except BaseException as e:  # noqa: BLE001
            self._err("line", e)
        return None

    def _on_call(self, code, offset, callable_, arg0):
        """Discovery only: at EVERY production call, a declared binding whose current value has not been
        seen is registered before anything is called (restarting disabled events), so a temporary
        rebinding invoked through C (``map``) or restored before the interval ends is still covered
        (Codex phase-1 r4 #1.2); recording stays callee-side."""
        try:
            for spec in self.boundaries:
                current = _resolve(spec["binding"])
                if current is not None and id(current) not in self.seen_targets:
                    self._register(spec, current)
        except BaseException as e:  # noqa: BLE001
            self._err("call", e)
        return None

    def _exit(self, code, retval, how):
        filename = self._real(code.co_filename)
        key = (filename, code.co_qualname)
        if key not in self.entries and key not in self.returns:
            return
        frame = sys._getframe(2)
        if key in self.returns:                             # recorded BEFORE the exit: frame still live
            stack, is_async = self._stack(frame)
            # an exceptional exit is observed only as the exception's type name (no application code)
            safe = _safe(retval) if how == "return" else {"raised": type_name(type(retval))}
            incomplete = type(safe) is dict and bool(safe.get("incomplete"))
            category = "unavailable" if incomplete else _classify(self.returns[key], safe)
            self._record("return", {"file": filename, "qualname": code.co_qualname, "frame": id(frame),
                                    "value": safe, "category": category, "how": how,
                                    "stack": stack[:40], "async": is_async})
        if key in self.entries:
            self._record("entry_exit", {"file": filename, "qualname": code.co_qualname, "frame": id(frame),
                                        "how": how})

    def _on_return(self, code, offset, retval):
        try:
            self._exit(code, retval, "return")
        except BaseException as e:  # noqa: BLE001
            self._err("return", e)
        return None

    def _on_unwind(self, code, offset, exc):
        try:
            self._exit(code, exc, "unwind")
        except BaseException as e:  # noqa: BLE001
            self._err("unwind", e)
        return None
