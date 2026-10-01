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
  still seen;
* a declared binding is TRACKED AT EVERY STORE on its path (CPython dict watchers on the module and
  class namespaces from ``sys.modules`` down, and a function watcher for ``__code__`` replaced in
  place), whoever makes the store, so every callable is registered before anything can call it
  through the binding. Every value a binding holds stays a target of that binding for the rest of the
  interval, registered per binding + callable. A call that cannot be attributed to one binding's
  current value is recorded with an unavailable identity. An unsupported namespace (anything but a
  module or a class), a callable that is not Python-observable, a namespace replaced wholesale, a
  mock boundary whose ``__call__`` is replaced, or a change no watched store explains, is recorded as
  a ``boundary_gap`` for the reviewer (Codex phase-1 r5 #1, #3). Phase 1 issues no absence
  certificate (plan ruling 10), so a missed call can only fail to witness an event;
* a mock boundary's call counts only while the mock's effective ``__call__`` (read from the raw
  namespaces of its class's MRO) is the standard one and its class is the one it was held with; else
  the call's identity is unavailable (Codex phase-1 r7 #3);
* PURITY (Codex phase-1 r7 #4): ``obs_start`` and ``obs_end`` record the trusted bootstrap's
  audit-hook census (``_AUDIT_CENSUS``), whether automatic garbage collection is on (it is switched
  off for the call phase of every observed or quiesced node, so no collection runs an application
  finalizer inside a callback) and any application signal handler; a certificate requires none;
* the entry's exits (return or exception) are recorded.

Records are JSON lines (``kind`` field) written to the file named in the config at
``$W15_OBS_CONFIG``: ``session_start`` (identity + ``src_digest``), ``collect_error``,
``collected`` (node ids + test-file sha256), ``report`` (every node x phase), and for the nodes in
``observe.nodes`` during their call phase ``obs_start`` … ``obs_end`` around ``entry``,
``entry_exit``, ``branch``, ``boundary``, ``boundary_gap``, ``return``; then ``observer_error``,
``imports`` and ``session_finish`` (with ``src_digest`` again).
"""
from __future__ import annotations

import ctypes
import gc
import hashlib
import inspect
import itertools
import json
import os
import re
import signal
import sys
import threading
import types

import pytest

TOOL_ID = 4
CO_ASYNC = inspect.CO_COROUTINE | inspect.CO_ITERABLE_COROUTINE | inspect.CO_ASYNC_GENERATOR
_SEQ = itertools.count(1)
_CONFIG: dict | None = None
_AUDIT_CENSUS: dict | None = None      # set by the trusted bootstrap after loading this module
_TYPE_QUALNAME = type.__dict__["__qualname__"]
_TYPE_MODULE = type.__dict__["__module__"]
_TYPE_DICT = type.__dict__["__dict__"]
_TYPE_MRO = type.__dict__["__mro__"]
_TYPE_FLAGS = type.__dict__["__flags__"]
_HEAPTYPE = 1 << 9                      # Py_TPFLAGS_HEAPTYPE: the class keeps __module__ in its namespace dict
_GETSET = type(type.__dict__["__dict__"])
_NO_PATH = "<no exact-str filename>"    # never absolute, so never under prod_root and never an entry file
_MODULE_DICT = types.ModuleType.__dict__["__dict__"]
_EXC_ARGS = BaseException.__dict__["args"]
_EXC_TB = BaseException.__dict__["__traceback__"]
_FUNCTION, _METHOD = types.FunctionType, types.MethodType     # neither can be subclassed

# CPython's dict and function watchers (3.12 C API, through ctypes; callbacks run with the GIL held).
# Object arguments are declared as addresses so that a dict or function being deallocated is never
# turned into a Python reference.
_API = ctypes.pythonapi
_DICT_WATCH_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
_FUNC_WATCH_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
for _fn, _argtypes in (("PyDict_AddWatcher", [_DICT_WATCH_CB]), ("PyDict_ClearWatcher", [ctypes.c_int]),
                       ("PyDict_Watch", [ctypes.c_int, ctypes.py_object]),
                       ("PyDict_Unwatch", [ctypes.c_int, ctypes.py_object]),
                       ("PyFunction_AddWatcher", [_FUNC_WATCH_CB]), ("PyFunction_ClearWatcher", [ctypes.c_int])):
    getattr(_API, _fn).argtypes = _argtypes
    getattr(_API, _fn).restype = ctypes.c_int
_DICT_ADDED, _DICT_MODIFIED, _DICT_DELETED, _DICT_CLONED, _DICT_CLEARED = 0, 1, 2, 3, 4
_FUNC_MODIFY_CODE = 2


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


def _exact_type_name(t) -> str | None:
    """A class's module + qualname read through type's own C descriptors (no metaclass code), or None when
    either is not an exact str: a class's ``__module__`` may hold any object, and interpolating one runs its
    own ``__format__`` (Codex phase-1 r8 #3)."""
    try:
        if _TYPE_FLAGS.__get__(t, type) & _HEAPTYPE:
            # the C descriptor would look __module__ up in the class namespace, comparing any same-hash
            # application key (Codex phase-1 r9 #1): read it by iteration instead
            ns = _type_ns(t)
            module = _lookup(ns, "__module__")[1] if ns is not None else None   # None when unsupported
        else:
            module = _TYPE_MODULE.__get__(t, type)
        qualname = _TYPE_QUALNAME.__get__(t, type)
    except Exception:  # noqa: BLE001
        return None
    if type(module) is not str or type(qualname) is not str:
        return None
    return module + "." + qualname


def type_name(t) -> str:
    """``_exact_type_name``, or a fixed ``<unnamed>`` for reports."""
    name = _exact_type_name(t)
    return "<unnamed>" if name is None else name


def _raised(t) -> dict:
    """An exceptional exit, observed only as its exception's type name; incomplete when that name is not
    readable as exact strings."""
    name = _exact_type_name(t)
    return {"raised": name} if name is not None else {"raised": "<unnamed>", "incomplete": True}


def _plain_instance_dict(value):
    """The instance's raw __dict__, read by calling the standard C ``__dict__`` descriptor found by iterating
    the MRO's class namespaces, else None: no hashed lookup and no generic attribute lookup, either of which
    would compare a same-hash application key (Codex phase-1 r9 #1)."""
    t = type(value)
    for cls in _TYPE_MRO.__get__(t, type):
        ns = _type_ns(cls)
        if ns is None:
            return None
        found, desc, why = _lookup(ns, "__dict__")
        if why is not None:
            return None
        if found:
            if type(desc) is not _GETSET:
                return None
            try:
                d = desc.__get__(value, t)
            except Exception:  # noqa: BLE001
                return None
            return d if type(d) is dict else None
    return None


def _name(value) -> str:
    """A code object's name when it is an exact str, else a fixed ``<unnamed>``: crafted code may carry a
    str subclass, whose hash and comparison are application code."""
    return value if type(value) is str else "<unnamed>"


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
    name = _exact_type_name(t)
    if name is None:
        return {"unavailable": "type identity not readable as exact strings", "incomplete": True}
    d = _plain_instance_dict(value)
    if d is None:
        return {"type": name, "unavailable": "no plain __dict__", "incomplete": True}
    out = _safe_items(dict.items(d), "fields", depth, skip_private=True)
    out["type"] = name
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
    observed = item.nodeid in set(observe.get("nodes", []))
    if not observed and item.nodeid not in set(_config().get("quiesce", [])):
        return (yield)
    # automatic collection off for the call phase, observed or not (the observer-off twin runs the
    # same way), so no collection can run an application finalizer inside an observer callback
    gc_was = gc.isenabled()
    gc.disable()
    try:
        if not observed:
            return (yield)
        mon = _Monitor(item.nodeid, observe, _config().get("prod_root") or "")
        try:
            try:
                mon.start()
            except BaseException as e:  # noqa: BLE001 - recorded; stop() still releases what start acquired
                mon._err("start", e)
            return (yield)
        finally:
            mon.stop()
            for rec in mon.events:
                _write(rec)
    finally:
        if gc_was:
            gc.enable()


def _import_record(mod) -> dict | None:
    """One bts module's provenance, read without application dispatch (Codex phase-1 r9 #3): an exact
    module, its namespace by the C descriptor, ``__file__`` by iteration over exact-str keys. Anything else
    is recorded unavailable, which the runner refuses."""
    if type(mod) is not types.ModuleType:
        return {"unavailable": "not an exact module"}
    ns = _MODULE_DICT.__get__(mod, types.ModuleType)
    if type(ns) is not dict:
        return {"unavailable": "no plain module namespace"}
    found, f, why = _lookup(ns, "__file__")
    if why is not None:
        return {"unavailable": why}
    return {"file": os.path.realpath(f), "sha256": _sha_file(f)} if found and type(f) is str else None


def pytest_sessionfinish(session, exitstatus):
    mods = {}
    found, modules, why = _lookup(_MODULE_DICT.__get__(sys, types.ModuleType), "modules")
    for name, mod in (list(dict.items(modules)) if found and why is None and type(modules) is dict else []):
        if not issubclass(type(name), str):
            continue
        plain = str.__str__(name)         # an exact copy, made by str's own C slot (no method of a subclass)
        if not (plain == "bts" or plain.startswith("bts.")):
            continue
        rec = (_import_record(mod) if type(name) is str
               else {"unavailable": "a sys.modules key that is not an exact str"})
        if rec is not None:
            mods[plain] = rec
    _write({"kind": "imports", "modules": mods} if found and why is None and type(modules) is dict
           else {"kind": "imports", "modules": {}, "unavailable": why or "sys.modules is not a plain dict"})
    _write({"kind": "session_finish", "exitstatus": int(exitstatus),
            "src_digest": src_digest(_config().get("prod_root"))})


# ---------------------------------------------------------------------------- the witness

_ACCESSOR = re.compile(r"args\[(\d+)\]|kw:(\w+)")


def _access(specs, args, kwargs):
    """The first accessible identity value. A keyword is found by iterating the call's keyword dict, which
    must hold exact-str keys only: membership or subscription would compare a same-hash application key
    (Codex phase-1 r9 #2)."""
    for spec in specs:
        m = _ACCESSOR.fullmatch(spec)
        if not m:
            continue
        if m.group(1) is not None and int(m.group(1)) < len(args):
            return True, args[int(m.group(1))]
        if m.group(2) is not None:
            found, value, _why = _lookup(kwargs, m.group(2))     # not found when the dict is unsupported
            if found:
                return True, value
    return False, None


# ---------------------------------------------------------------------------- binding resolution
# A declared binding ``module:attr[.attr]`` is resolved through NAMESPACE DICTS only, read by C
# descriptors: never an object's own ``__getattribute__``, ``__getattr__`` or properties (Codex
# phase-1 r5 #3). Two shapes are supported, modules and classes. Their namespace dict cannot be
# replaced wholesale, so a dict watcher on it sees every store (an instance's ``__dict__`` can be
# reassigned, which no watcher sees; Codex phase-1 r5 #1). Any other shape on the path is a
# coverage gap.

_NON_STR_KEY = "unsupported namespace: a key that is not an exact str (its hash and __eq__ are application code)"


def _lookup(ns: dict, name: str):
    """(found, value, why) for a str key, found by iterating the dict's items: a hashed lookup would compare
    the key with any same-hash application key and so run its ``__eq__`` (Codex phase-1 r4 #2). A dict
    holding any key that is not an exact str is unsupported (``why``): skipping such a key could miss the
    entry real lookup finds (Codex phase-1 r9, plan ruling 11)."""
    found, value = False, None
    for k, v in dict.items(ns):
        if type(k) is not str:
            return False, None, _NON_STR_KEY
        if not found and k == name:
            found, value = True, v
    return found, value, None


def _type_ns(cls):
    """A class's real namespace dict, behind its read-only mappingproxy (type's C descriptor)."""
    refs = gc.get_referents(_TYPE_DICT.__get__(cls, type))
    return refs[0] if len(refs) == 1 and type(refs[0]) is dict else None


def _namespace(obj):
    """The namespace dict of an exact module (``type(obj) is ModuleType``) or an exact-``type`` class, the
    only shapes watched (Codex phase-1 r6 #1: a ModuleType subclass or a metaclass can answer from
    ``__getattribute__`` while the raw slot never changes). Even for these the raw namespace is not all
    of attribute lookup (a module ``__getattr__`` fallback, an inherited class attribute; Codex phase-1
    r7 #1): a call reached that way is not recorded, which can only miss an event. None for any other
    shape."""
    if type(obj) is types.ModuleType:
        d = _MODULE_DICT.__get__(obj, types.ModuleType)
        return d if type(d) is dict else None
    if type(obj) is type:
        return _type_ns(obj)
    return None


def _walk(keys: list, level: int, obj, chain: list):
    """Resolve ``keys[level + 1:]`` from ``obj`` (the value at ``level``), appending each step to
    ``chain`` as (namespace dict, key). ``keys[0]`` is sys's 'modules' entry, so step 1 looks the module
    up in sys.modules itself. Returns (value, why-unresolved)."""
    for i in range(level + 1, len(keys)):
        if i == 1:
            ns = obj if type(obj) is dict else None
            if ns is None:
                return None, "unsupported namespace: sys.modules is not a plain dict"
        else:
            ns = _namespace(obj)
            if ns is None:
                return None, "unsupported namespace: only exact modules and classes (standard attribute dispatch) are watched"
        chain.append((ns, keys[i]))
        found, obj, why = _lookup(ns, keys[i])
        if why is not None:
            return None, why
        if not found:
            return None, "the declared binding resolves to nothing"
    return obj, None


def _resolve_chain(binding: str):
    """(keys, chain, value, why): ``chain`` lists the steps that decide the value, from sys's OWN
    namespace down: its 'modules' entry is the root, so replacing sys.modules is a watched store too
    (Codex phase-1 r6 #1)."""
    module, _, attr = binding.partition(":")
    keys = ["modules", module] + attr.split(".")
    sys_ns = _MODULE_DICT.__get__(sys, types.ModuleType)
    chain = [(sys_ns, "modules")]
    found, modules, why = _lookup(sys_ns, "modules")
    if why is not None:
        return keys, chain, None, why
    if not found:
        return keys, chain, None, "unsupported namespace: sys.modules is missing"
    value, why = _walk(keys, 0, modules, chain)
    return keys, chain, value, why


def _resolve(binding: str):
    """The object currently bound at ``module:attr[.attr]``, or None."""
    return _resolve_chain(binding)[2]


def _classify(rules, safe_value) -> str:
    text = _text(safe_value)
    for label, pattern in rules:
        if re.search(pattern, text):
            return label
    return "other"


def _complete(safe) -> bool:
    """True when no part of a ``_safe`` form was omitted AT ANY DEPTH (Codex phase-1 r5 #2: only the top
    level was checked, so a nested truncated payload was classified and certified)."""
    if type(safe) is not dict:
        return True
    if safe.get("incomplete") or "unavailable" in safe:
        return False
    if "seq" in safe:
        return all(_complete(v) for v in safe["seq"])
    for key in ("map", "fields"):
        if key in safe:
            return all(_complete(v) for v in safe[key].values())
    return True


class _Monitor:
    def __init__(self, nodeid: str, observe: dict, prod_root: str):
        from unittest import mock
        self.mock_base = mock.NonCallableMock
        self.mock_call_fn = mock.CallableMixin.__dict__["__call__"]
        self.mock_call_code = self.mock_call_fn.__code__
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
        self.state: dict = {}                   # spec name -> {"keys", "chain", "current", "held"}
        self.watch_index: dict = {}             # address of a namespace dict -> [(spec name, chain level)]
        self.call_watch: dict = {}              # address of a mock class's namespace -> {spec names}
        self.watched: dict = {}                 # address -> namespace dict, kept alive while watched
        self.functions: dict = {}               # address of a target function -> (function, [(spec, receiver)])
        self.dict_watcher = self.func_watcher = None
        self._dict_cb = self._func_cb = None    # the ctypes trampolines, kept alive while installed
        self.paths: dict[str, str] = {}
        self.mock_types: dict = {}              # id of a held mock -> its class when first held
        self.tool_acquired = self.started = self.active = False

    # -- bookkeeping
    def _real(self, filename) -> str:
        """A code filename's realpath, cached; ``_NO_PATH`` when it is not an exact str (crafted code may
        carry a str subclass, whose hash and path methods are application code; plan ruling 11)."""
        if type(filename) is not str:
            return _NO_PATH
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
        """Record an observer failure with no I/O and no application code, so recording it cannot fail in
        turn (self-review before r8: formatting read source files, an audited ``open``, and ran ``str`` of
        arbitrary exception arguments): the exception's type name, its plain-str arguments and its
        traceback's (file, line, function) entries, read through C descriptors."""
        try:
            args = _EXC_ARGS.__get__(exc, BaseException)
            tb, frames = _EXC_TB.__get__(exc, BaseException), []
            while tb is not None and len(frames) < 40:
                code = tb.tb_frame.f_code
                frames.append([_name(code.co_filename), tb.tb_lineno, _name(code.co_name)])
                tb = tb.tb_next
            detail = {"args": [a[:200] for a in args if type(a) is str][:3], "frames": frames[-10:]}
        except BaseException:  # noqa: BLE001
            detail = None
        self._record("observer_error", {"where": where, "error_type": type_name(type(exc)), "detail": detail})

    def _gap(self, name: str, reason: str, **extra) -> None:
        self._record("boundary_gap", {"name": name, "reason": reason, **extra})

    def _stack(self, frame):
        out, is_async, n = [], False, 0
        while frame is not None and n < 80:
            code = frame.f_code
            if code.co_flags & CO_ASYNC:
                is_async = True
            out.append([_name(code.co_qualname), self._real(code.co_filename), frame.f_lineno, id(frame)])
            frame, n = frame.f_back, n + 1
        return out, is_async

    def _identity(self, spec: dict, args, kwargs) -> dict:
        ok, value = _access(spec.get("value", []), args, kwargs)
        if not ok:
            return {"value": None, "category": "unavailable"}
        safe = _safe(value)
        return {"value": safe, "sha256": hashlib.sha256(_text(safe).encode()).hexdigest(),
                "category": _classify(spec.get("classify", []), safe) if _complete(safe) else "unavailable"}

    def _is_mock(self, obj) -> bool:
        return issubclass(type(obj), self.mock_base)

    def _effective_call(self, cls):
        """The ``__call__`` a call of an instance of ``cls`` runs: the first entry in the raw namespaces of
        its MRO (read by C descriptors, no application code), or None if unreadable or absent."""
        for c in _TYPE_MRO.__get__(cls, type):
            ns = _type_ns(c)
            if ns is None:
                return None
            found, value, why = _lookup(ns, "__call__")
            if why is not None:
                return None
            if found:
                return value
        return None

    def _mock_verified(self, obj) -> bool:
        """A held mock whose class is still the one it was held with and whose effective ``__call__`` is
        the standard one: only then are the arguments its standard code receives the call's (Codex
        phase-1 r7 #3: a subclass overriding ``__call__`` before observation was admitted)."""
        t = type(obj)
        return t is self.mock_types.get(id(obj)) and self._effective_call(t) is self.mock_call_fn

    def _code_of(self, obj):
        """The code object a call to ``obj`` starts: plain functions and bound methods only, recognised by
        EXACT runtime type (Codex phase-1 r6 #4: a type NAME is assignable application metadata, and a
        fake 'builtins.function' ran an application ``__code__`` property here). FunctionType and
        MethodType cannot be subclassed, and their ``__code__``/``__func__`` are C members."""
        if type(obj) is _METHOD:
            obj = obj.__func__
        return obj.__code__ if type(obj) is _FUNCTION else None

    def _code_shared(self, code) -> int:
        """How many live functions run ``code``: a code object does not identify its function (Codex
        phase-1 r6 #2: a FunctionType clone of the sender, with other globals, was attributed to it)."""
        return sum(1 for r in gc.get_referrers(code) if type(r) is _FUNCTION and r.__code__ is code)

    # -- binding tracking (Codex phase-1 r5 #1): every store on a binding's path is seen when it happens,
    # whoever makes it (production, a test helper, a C callback), so a callable is registered before
    # anything can call it through the binding
    def _watch(self, ns: dict, name: str | None = None, level: int | None = None) -> None:
        if id(ns) not in self.watched:
            self.watched[id(ns)] = ns
            if self.dict_watcher is not None:
                _API.PyDict_Watch(self.dict_watcher, ns)
        if name is not None:
            self.watch_index.setdefault(id(ns), []).append((name, level))

    def _track(self, spec: dict) -> None:
        keys, chain, value, why = _resolve_chain(spec["binding"])
        self.state[spec["name"]] = {"keys": keys, "chain": chain, "current": None, "held": []}
        for level, step in enumerate(chain):
            self._watch(step[0], spec["name"], level)
        if why is not None:
            self._gap(spec["name"], why)
        else:
            self._hold(spec, value)

    def _hold(self, spec: dict, obj) -> None:
        """``obj`` is now bound at ``spec``'s binding. It stays a target of THAT spec for the rest of the
        interval, since a caller may have captured it before a restore. Targets are registered per spec
        + callable, so a callable bound to two boundaries counts for both (Codex phase-1 r5 #1.2)."""
        st = self.state[spec["name"]]
        st["current"] = obj
        if any(o is obj for o in st["held"]):
            return
        st["held"].append(obj)
        if self._is_mock(obj):
            # a standard mock call starts mock_call_code; each call re-checks the effective __call__
            self.mock_types.setdefault(id(obj), type(obj))
            if not self._mock_verified(obj):
                self._gap(spec["name"], "a mock boundary whose effective __call__ is not the standard one")
            for cls in _TYPE_MRO.__get__(type(obj), type):
                ns = _type_ns(cls)
                if ns is not None:
                    self._watch(ns)
                    self.call_watch.setdefault(id(ns), set()).add(spec["name"])
            return
        code = self._code_of(obj)
        if code is None:
            self._gap(spec["name"], "not a Python-observable callable", type=type_name(type(obj)))
            return
        # a bound method is the boundary only for ITS receiver: other instances share the code object
        # (Codex phase-1 r4 #1.4); method.__self__/__func__ are C members, no application code runs
        receiver = obj.__self__ if type(obj) is _METHOD else None
        func = obj.__func__ if receiver is not None else obj
        self.functions.setdefault(id(func), (func, []))[1].append((spec, receiver))   # its __code__ can be swapped
        self._add_code(code, spec, receiver)

    def _add_code(self, code, spec: dict, receiver) -> None:
        pairs = self.boundary_codes.setdefault(code, [])
        if not any(sp is spec and rcv is receiver for sp, rcv in pairs):
            pairs.append((spec, receiver))
            if self.active:
                sys.monitoring.restart_events()          # its PY_START may have been disabled

    def _restep(self, name: str, level: int, found: bool, value) -> None:
        """A store at ``level`` of ``name``'s chain, reported BEFORE the dict changes, with the new value:
        re-resolve the rest of the chain from it, watch any new namespace, and hold the new final value."""
        st = self.state[name]
        spec = next(sp for sp in self.boundaries if sp["name"] == name)
        chain = st["chain"][:level + 1]
        if not found:
            # deleted: the binding holds nothing. A call reached through fallback or inherited lookup is
            # not recorded, which can only miss an event (no absence is certified; plan ruling 10)
            st["chain"], st["current"] = chain, None
            return
        obj, why = _walk(st["keys"], level, value, chain)
        st["chain"] = chain
        for lv in range(level + 1, len(chain)):
            self._watch(chain[lv][0], name, lv)
        if why is not None:
            st["current"] = None
            if why.startswith("unsupported"):
                self._gap(name, why)
            return
        self._hold(spec, obj)

    def _on_dict(self, event, dict_addr, key_addr, new_addr):
        """Dict watcher callback (C trampoline). Addresses, not objects, arrive: a dying dict is never
        touched. Must return 0 and never raise."""
        try:
            hits = self.watch_index.get(dict_addr, ())
            calls = self.call_watch.get(dict_addr, ())
            if not hits and not calls:
                return 0
            per_key = event in (_DICT_ADDED, _DICT_MODIFIED, _DICT_DELETED)
            key = ctypes.cast(key_addr, ctypes.py_object).value if per_key and key_addr else None
            if per_key and type(key) is str:
                if key == "__call__":
                    for name in sorted(calls):
                        self._gap(name, "a mock boundary's __call__ was replaced")
                new = ctypes.cast(new_addr, ctypes.py_object).value if new_addr and event != _DICT_DELETED else None
                d = self.watched.get(dict_addr)
                for name, level in list(hits):
                    chain = self.state[name]["chain"]
                    if level < len(chain) and chain[level][0] is d and chain[level][1] == key:
                        self._restep(name, level, event != _DICT_DELETED, new)
            elif per_key or event in (_DICT_CLONED, _DICT_CLEARED):
                # a store through a key that is not an exact str may change any binding (its hash and
                # __eq__ decide which entry it replaces; Codex phase-1 r9, plan ruling 11), as a clear or
                # clone does
                reason = ("a store with a key that is not an exact str" if per_key
                          else "a watched namespace was replaced wholesale")
                for name in sorted({n for n, _lv in hits} | set(calls)):
                    self._gap(name, reason)
                # nothing on a path through this namespace is current until a per-key store re-resolves it
                # (Codex phase-1 r8 #1: a captured former callable stayed current); the new contents are
                # not read here, before the change
                d = self.watched.get(dict_addr)
                for name, level in list(hits):
                    st = self.state[name]
                    if level < len(st["chain"]) and st["chain"][level][0] is d:
                        st["chain"], st["current"] = st["chain"][:level + 1], None
        except BaseException as e:  # noqa: BLE001
            self._err("dict_watch", e)
        return 0

    def _on_func(self, event, func_addr, new_addr):
        """Function watcher callback: a target function's ``__code__`` replaced in place starts new code,
        which is registered for the same spec + receiver (Codex phase-1 r5 #1)."""
        try:
            if event != _FUNC_MODIFY_CODE:
                return 0
            entry = self.functions.get(func_addr)
            if entry is None:
                return 0
            code = ctypes.cast(new_addr, ctypes.py_object).value if new_addr else None
            if type(code) is types.CodeType:
                for spec, receiver in entry[1]:
                    self._add_code(code, spec, receiver)
        except BaseException as e:  # noqa: BLE001
            self._err("func_watch", e)
        return 0

    def _install_watchers(self) -> bool:
        try:
            self._dict_cb = _DICT_WATCH_CB(self._on_dict)
            self.dict_watcher = _API.PyDict_AddWatcher(self._dict_cb)
            self._func_cb = _FUNC_WATCH_CB(self._on_func)
            self.func_watcher = _API.PyFunction_AddWatcher(self._func_cb)
            return True
        except BaseException as e:  # noqa: BLE001
            self._err("install_watchers", e)
            return False

    def _remove_watchers(self) -> None:
        """Release each acquired watcher resource on its own, so one failure never skips the others
        (Codex phase-1 r6 #6)."""
        if self.dict_watcher is not None:
            for ns in list(self.watched.values()):
                try:
                    _API.PyDict_Unwatch(self.dict_watcher, ns)
                except BaseException as e:  # noqa: BLE001
                    self._err("unwatch", e)
            try:
                _API.PyDict_ClearWatcher(self.dict_watcher)
            except BaseException as e:  # noqa: BLE001
                self._err("clear_dict_watcher", e)
        if self.func_watcher is not None:
            try:
                _API.PyFunction_ClearWatcher(self.func_watcher)
            except BaseException as e:  # noqa: BLE001
                self._err("clear_func_watcher", e)
        self.dict_watcher = self.func_watcher = None

    # -- lifecycle
    def start(self) -> None:
        """Acquire in order; whatever was acquired is released by stop(), which the pytest wrapper always
        calls, even when start() raises (Codex phase-1 r6 #6: a failed start left both CPython watchers
        and the monitoring id installed)."""
        mon = sys.monitoring
        try:
            mon.use_tool_id(TOOL_ID, "w15-observer")
        except ValueError as e:
            self._err("use_tool_id", e)
            return
        self.tool_acquired = True
        # obs_start FIRST: a gap found while registering the boundaries belongs to this interval
        # (Codex phase-1 r4 #1.1: it used to precede obs_start and so fell outside the certificate)
        self._record("obs_start", {"purity": self._purity()})
        self.started = True
        watching = self._install_watchers() if self.boundaries else False
        for spec in self.boundaries:
            self._track(spec)
            if not watching:
                self._gap(spec["name"], "store watching unavailable: a rebinding could go unseen")
        ev = mon.events
        mon.register_callback(TOOL_ID, ev.PY_START, self._on_start)
        mon.register_callback(TOOL_ID, ev.LINE, self._on_line)
        mon.register_callback(TOOL_ID, ev.PY_RETURN, self._on_return)
        mon.register_callback(TOOL_ID, ev.PY_UNWIND, self._on_unwind)
        self.active = True
        mon.set_events(TOOL_ID, ev.PY_START | ev.PY_UNWIND)

    def stop(self) -> None:
        """Release everything start() acquired, whether or not it completed, each step on its own."""
        mon = sys.monitoring
        if self.tool_acquired:
            steps = [lambda: mon.set_events(TOOL_ID, 0)]
            steps += [lambda c=code: mon.set_local_events(TOOL_ID, c, 0) for code in self.instrumented]
            steps += [lambda e=event: mon.register_callback(TOOL_ID, e, None)
                      for event in (mon.events.PY_START, mon.events.LINE, mon.events.PY_RETURN, mon.events.PY_UNWIND)]
            steps += [lambda: mon.free_tool_id(TOOL_ID), mon.restart_events]
            for step in steps:
                try:
                    step()
                except BaseException as e:  # noqa: BLE001
                    self._err("stop", e)
            self.tool_acquired = False
        self.active = False
        self._remove_watchers()
        if not self.started:
            return
        self.started = False
        for spec in self.boundaries:                    # fail closed: a change no watched store explains
            try:
                st = self.state.get(spec["name"])
                _keys, _chain, now, why = _resolve_chain(spec["binding"])
                if st is not None and why is None and now is not st["current"]:
                    self._gap(spec["name"], "the binding changed without a watched store")
            except BaseException as e:  # noqa: BLE001 - stop() must never raise
                self._err("end_check", e)
        self._record("obs_end", {"purity": self._purity()})

    def _purity(self) -> dict | None:
        """What could run application code inside observation (Codex phase-1 r7 #4): audit hooks added
        after the trusted bootstrap (None without its census), automatic garbage collection, and Python
        signal handlers other than the defaults. Never raises: a failure is an observer error."""
        try:
            census = _AUDIT_CENSUS
            handlers = []
            for sig in sorted(signal.valid_signals()):
                h = signal.getsignal(sig)
                if not (h is None or h is signal.SIG_DFL or h is signal.SIG_IGN or h is signal.default_int_handler):
                    handlers.append(int(sig))
            return {"audit_hooks_added": census["hooks_added"] if type(census) is dict else None,
                    "gc_enabled": gc.isenabled(), "signal_handlers": handlers}
        except BaseException as e:  # noqa: BLE001
            self._err("purity", e)
            return None

    # -- callbacks (never raise; never run application code)
    def _matched(self, specs: list, current: list, frame, args, kwargs, *, unverified: str | None = None) -> None:
        """One boundary record per matched spec. A call whose callable is bound to several boundaries,
        is a value the binding held earlier in the interval (not its current one), or is an unverified
        mock call, cannot be attributed: its identity is unavailable, so it never witnesses an event
        (Codex phase-1 r5 #1, r7 #3)."""
        names = [sp["name"] for sp in specs]
        for sp, is_current in zip(specs, current):
            if unverified is not None:
                reason = unverified
            elif len(specs) > 1:
                reason = f"the callable is bound to several boundaries: {names}"
            elif not is_current:
                reason = "a callable the binding held earlier in the interval, not its current value"
            else:
                reason = None
            self._boundary(sp, frame, args, kwargs, reason=reason)

    def _on_start(self, code, offset):
        matched = None
        try:
            if code is self.mock_call_code:
                frame = sys._getframe(1)
                loc = frame.f_locals
                me = loc.get("self")
                matched = [sp for sp in self.boundaries
                           if any(o is me for o in self.state.get(sp["name"], {}).get("held", ()))]
                if matched:
                    extra = loc.get("args", ())
                    kwargs = loc.get("kwargs", {})
                    self._matched(matched, [self.state[sp["name"]]["current"] is me for sp in matched], frame,
                                  list(extra) if type(extra) is tuple else [],
                                  kwargs if type(kwargs) is dict else {},   # not copied: a copy can compare keys
                                  unverified=None if self._mock_verified(me) else
                                  "a mock whose effective __call__ is not the standard one, or whose class changed")
                return None
            pairs = self.boundary_codes.get(code)
            if pairs is not None:
                frame = sys._getframe(1)
                loc = frame.f_locals
                positional = code.co_varnames[:code.co_argcount]
                args = [loc.get(n) for n in positional]
                if code.co_flags & inspect.CO_VARARGS:
                    extra = loc.get(code.co_varnames[code.co_argcount + code.co_kwonlyargcount], ())
                    args += list(extra) if type(extra) is tuple else []
                kwargs = {n: loc.get(n) for n in code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]}
                first = args[0] if args else None
                mine, others = [], []
                for sp, rcv in pairs:
                    bucket = mine if rcv is None or first is rcv else others
                    if not any(s is sp for s in bucket):
                        bucket.append(sp)
                if mine:
                    matched = mine
                    shared = self._code_shared(code)
                    if shared > 1:                           # the code does not say which function ran
                        for sp in mine:
                            self._boundary(sp, frame, args, kwargs,
                                           reason=f"the boundary's code is shared by {shared} functions: the callable is not identified")
                    else:
                        self._matched(mine, [self._current_code(sp) is code and self._receiver_ok(sp, first)
                                             for sp in mine], frame, args, kwargs)
                else:                                        # same code, another receiver: ambiguous
                    matched = others
                    for sp in others:
                        self._boundary(sp, frame, args, kwargs, reason="another receiver of the boundary's code")
            filename = self._real(code.co_filename)
            if self.prod_root is None or not filename.startswith(self.prod_root):
                return None if matched else sys.monitoring.DISABLE
            if code not in self.instrumented:
                events = 0
                if self.branch and filename == self.branch[0]:
                    events |= sys.monitoring.events.LINE
                key = (filename, _name(code.co_qualname))
                if key in self.returns or key in self.entries:
                    events |= sys.monitoring.events.PY_RETURN       # PY_UNWIND is global-only (3.12)
                if events:
                    sys.monitoring.set_local_events(TOOL_ID, code, events)
                self.instrumented.add(code)
            if (filename, _name(code.co_qualname)) in self.entries:
                frame = sys._getframe(1)
                stack, is_async = self._stack(frame)
                self._record("entry", {"file": filename, "qualname": _name(code.co_qualname), "frame": id(frame),
                                       "stack": stack[:40], "async": is_async})
        except BaseException as e:  # noqa: BLE001
            self._err("py_start", e)
        return None

    def _current_code(self, spec: dict):
        cur = self.state[spec["name"]]["current"]
        return None if cur is None or self._is_mock(cur) else self._code_of(cur)

    def _receiver_ok(self, spec: dict, first) -> bool:
        cur = self.state[spec["name"]]["current"]
        return type(cur) is not _METHOD or cur.__self__ is first

    def _boundary(self, spec: dict, callee_frame, args, kwargs, *, reason: str | None = None) -> None:
        stack, is_async = self._stack(callee_frame.f_back)
        identity = ({"value": None, "category": "unavailable", "reason": reason}
                    if reason else self._identity(spec, args, kwargs))
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

    def _exit(self, code, retval, how):
        filename = self._real(code.co_filename)
        key = (filename, _name(code.co_qualname))
        if key not in self.entries and key not in self.returns:
            return
        frame = sys._getframe(2)
        if key in self.returns:                             # recorded BEFORE the exit: frame still live
            stack, is_async = self._stack(frame)
            # an exceptional exit is observed only as the exception's type name (no application code)
            safe = _safe(retval) if how == "return" else _raised(type(retval))
            category = _classify(self.returns[key], safe) if _complete(safe) else "unavailable"
            self._record("return", {"file": filename, "qualname": _name(code.co_qualname), "frame": id(frame),
                                    "value": safe, "category": category, "how": how,
                                    "stack": stack[:40], "async": is_async})
        if key in self.entries:
            self._record("entry_exit", {"file": filename, "qualname": _name(code.co_qualname), "frame": id(frame),
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
