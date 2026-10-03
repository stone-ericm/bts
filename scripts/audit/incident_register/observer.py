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
  class namespaces from ``sys.modules`` down), whoever makes the store, so every callable is registered
  before anything can call it through the binding; a held function's ``__code__`` replaced in place is
  registered at the new code's first start, before its body runs (there is no function watcher: see
  ``_swapped``). Every value a binding holds stays a target of that binding for the rest of the
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
import dis
import json
import opcode
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

# CPython's dict watchers (3.12 C API, through ctypes; callbacks run with the GIL held). Object arguments are
# declared as addresses so that a dict being deallocated is never turned into a Python reference. There is no
# function watcher: CPython calls function watchers for EVERY function's creation and destruction, including a
# temporary freed after a failed C call while its exception is pending, and a ctypes callback that returns
# normally then makes CPython replace that exception with SystemError (own review during r14: the application
# took an except-SystemError path only when observed; a Cython error path built a traceback with no exception
# set and crashed). A dict watcher fires only for the namespaces watched here, which the observer keeps alive.
_API = ctypes.pythonapi
_DICT_WATCH_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
for _fn, _argtypes in (("PyDict_AddWatcher", [_DICT_WATCH_CB]), ("PyDict_ClearWatcher", [ctypes.c_int]),
                       ("PyDict_Watch", [ctypes.c_int, ctypes.py_object]),
                       ("PyDict_Unwatch", [ctypes.c_int, ctypes.py_object])):
    getattr(_API, _fn).argtypes = _argtypes
    getattr(_API, _fn).restype = ctypes.c_int
_DICT_ADDED, _DICT_MODIFIED, _DICT_DELETED, _DICT_CLONED, _DICT_CLEARED = 0, 1, 2, 3, 4
for _fn, _argtypes in (("PyInterpreterState_Get", []), ("PyInterpreterState_ThreadHead", [ctypes.c_void_p]),
                       ("PyThreadState_Next", [ctypes.c_void_p]), ("PyThreadState_Get", [])):
    getattr(_API, _fn).argtypes = _argtypes
    getattr(_API, _fn).restype = ctypes.c_void_p
_INTERP_GET, _THREAD_HEAD, _THREAD_NEXT, _TS_GET = (_API.PyInterpreterState_Get, _API.PyInterpreterState_ThreadHead,
                                                    _API.PyThreadState_Next, _API.PyThreadState_Get)
_CONCURRENT = ("another thread was alive: the observer reads no application object then, since a temporary "
               "reference it held could become the last one and run an application finalizer (plan ruling 12)")


def _alone() -> bool:
    """True when the calling thread is the interpreter's only thread, read from the C thread-state list (no Python
    object is referenced; the GIL is held). Inside the call phase the observer reads application objects only then
    (plan ruling 12, Codex phase-1 r11 #1): while another thread can run, it could drop its own reference to an
    object the observer holds a temporary reference to, and releasing that temporary would run the object's
    finalizer inside observation. When this thread is alone, no other registered thread can change a reference
    count. The sample is a point predicate: a native thread that attaches to the interpreter after it is not
    excluded (Codex phase-1 r13 #4, plan ruling 12).

    New thread states are inserted at the HEAD of the list, so this thread is alone exactly when its own state is
    the head and has no successor. Only this thread's own state (never freed while it runs) and the
    interpreter's head pointer are read, since the GIL can pass between these calls. (Codex phase-1 r14,
    verbatim:) A version that followed the head's successor reported 'alone' while another thread lived. Reading
    an unlinked or freed state is an inferred cause, not a measured unlink/free sequence. The head is read again
    last, so a state inserted meanwhile is seen."""
    try:
        me, interp = _TS_GET(), _INTERP_GET()
        if not me or _THREAD_HEAD(interp) != me:
            return False
        if _THREAD_NEXT(me):
            return False
        return _THREAD_HEAD(interp) == me
    except Exception:  # noqa: BLE001 - unreadable thread states: not alone, so nothing is read (a miss, never false)
        return False


# A live frame's fast locals, read straight from its interpreter frame (CPython 3.12 layout, verified by
# _FAST_LOCALS_OK at import). frame.f_locals is never used: it creates the frame's cached locals dict, or
# REFRESHES an existing one, and a refresh can release the last reference to an application value and run its
# finalizer inside observation, with one thread (Codex phase-1 r12 #1: a generator whose locals inspect had
# read). The frame owns its slots, so a reference taken from one is never the last one.
_PY_FRAME_F_FRAME = 3 * ctypes.sizeof(ctypes.c_void_p)       # PyFrameObject: ob_refcnt, ob_type, f_back, f_frame
_IFRAME_LOCALSPLUS = 9 * ctypes.sizeof(ctypes.c_void_p)      # _PyInterpreterFrame: f_code ... owner, localsplus
_UNREAD = object()
_UNREAD_ARGS = "an argument the observer could not read"
# the prologue a CPython 3.12 function runs before RESUME (where PY_START fires): which local slots MAKE_CELL wrapped
_OP = {name: opcode.opmap[name] for name in ("NOP", "EXTENDED_ARG", "COPY_FREE_VARS", "MAKE_CELL", "RETURN_GENERATOR",
                                             "POP_TOP", "RESUME")}


_ENTRY: dict = {}       # sha256 of the code's bytes -> entry offset or None: owns no application object
_ENTRY_LIMIT = 65536


def _first_resume(code):
    """The byte offset of the code's first RESUME when that RESUME carries the entry argument (0, after any
    EXTENDED_ARG) and nothing can run it again, where a call's PY_START fires; otherwise None, and no start of
    this code is read. A start reported at any other offset is not the start of a call: a later RESUME can carry
    the entry argument in code built by CodeType.replace or types.CodeType, and PY_START then fires again
    mid-call, after the body has changed its locals (Codex phase-1 r15 #1). Nor is a second start at the entry
    offset itself: code whose jumps or exception handlers lead back to the entry RESUME, or anywhere before it,
    re-runs that RESUME mid-call (Codex phase-1 r16 probe), so such code is never read. Compiled code never
    targets its entry. Only raw bytes are read (dis.findlabels reads none of co_consts)."""
    # keyed by a digest of exact bytes the observer owns: neither the code object nor a weak reference to it is
    # kept (Codex phase-1 r17 #2: a weak reference changed the application's own weakref.getweakrefcount)
    raw, table = code.co_code, code.co_exceptiontable
    key = hashlib.sha256(len(raw).to_bytes(8, "little") + raw + table).digest()
    if key in _ENTRY:
        return _ENTRY[key]
    entry, ext = None, 0
    for i in range(0, len(raw) - 1, 2):
        op, arg = raw[i], raw[i + 1] | ext
        if op == _OP["EXTENDED_ARG"]:
            ext = arg << 8
            continue
        ext = 0
        if op == _OP["RESUME"]:
            entry = i if arg == 0 else None
            break
    if entry is not None and _entry_targeted(code, raw, entry):
        entry = None
    if len(_ENTRY) >= _ENTRY_LIMIT:
        _ENTRY.clear()
    _ENTRY[key] = entry
    return entry


def _varint(it, first=None) -> int:
    """One varint of a 3.12 exception table (6 bits a byte, bit 6 continues; bit 7 marks an entry's start)."""
    b = next(it) if first is None else first
    value = b & 63
    while b & 64:
        b = next(it)
        value = (value << 6) | (b & 63)
    return value


def _entry_targeted(code, raw, entry: int) -> bool:
    """True when a jump or an exception handler in ``code`` leads to ``entry`` or before it."""
    if any(target <= entry for target in dis.findlabels(raw)):
        return True
    it = iter(code.co_exceptiontable)
    while True:
        first = next(it, None)
        if first is None:
            return False                       # the table ended cleanly, between entries
        if not first & 128:
            return True                        # an entry without its start marker: the table is not read
        try:
            _varint(it, first)                 # start
            _varint(it)                        # length
            if _varint(it) * 2 <= entry:       # handler
                return True
            _varint(it)                        # depth and lasti
        except StopIteration:
            return True                        # a table that ends mid-entry is not read (Codex r16 probe)


def _prologue_cells(code):
    """The local slots the code's prologue wraps in cells (MAKE_CELL before the first RESUME), or None when the
    prologue holds anything else. CodeType.replace relabels names (and the kinds derived from them) without
    touching the bytecode (Codex phase-1 r14 #1), so a slot is decoded only when these agree with the names. This
    is a prefix-shape check: (Codex phase-1 r15, verbatim) The prefix check verifies cell-layout agreement at the
    supported initial entry offset; unsupported entry paths are not argument witnesses. It proves nothing about
    later control flow or the exception table; a start anywhere but that entry offset is not read (_first_resume)."""
    raw = code.co_code                     # exact bytes: two per instruction, opcode then argument
    cells, ext = set(), 0
    for i in range(0, len(raw) - 1, 2):
        op, arg = raw[i], raw[i + 1] | ext
        if op == _OP["EXTENDED_ARG"]:
            ext = arg << 8
            continue
        ext = 0
        if op == _OP["RESUME"]:
            return cells
        if op == _OP["MAKE_CELL"]:
            cells.add(arg)
        elif op not in (_OP["NOP"], _OP["COPY_FREE_VARS"], _OP["RETURN_GENERATOR"], _OP["POP_TOP"]):
            return None
    return None


def _fast_local(frame, code, index: int, cells: bool):
    """The value in fast-local slot ``index`` of the live ``frame`` running ``code``, or ``_UNREAD`` (an unbound
    slot, an unverifiable frame, or a cell when ``cells`` is False: another thread could change a shared cell)."""
    if not _FAST_LOCALS_OK or type(frame) is not types.FrameType or type(code) is not types.CodeType:
        return _UNREAD
    if not 0 <= index < code.co_nlocals:
        return _UNREAD
    if len(set(code.co_varnames)) != len(code.co_varnames):     # a slot's kind cannot be told by a name two locals
        return _UNREAD                                          # share (Codex phase-1 r13 #1)
    made = _prologue_cells(code)
    named = {i for i in range(code.co_nlocals) if code.co_varnames[i] in code.co_cellvars}
    if made is None or {c for c in made if c < code.co_nlocals} != named:
        return _UNREAD                     # the names do not say what the bytecode wrapped: never guess a slot's kind
    iframe = ctypes.c_void_p.from_address(id(frame) + _PY_FRAME_F_FRAME).value
    if not iframe or ctypes.c_void_p.from_address(iframe).value != id(code):
        return _UNREAD
    slot = ctypes.c_void_p.from_address(iframe + _IFRAME_LOCALSPLUS + index * ctypes.sizeof(ctypes.c_void_p)).value
    if not slot:
        return _UNREAD
    value = ctypes.cast(slot, ctypes.py_object).value
    if index in named:                     # a cell by its name AND by the prologue that made it
        if not cells or type(value) is not types.CellType:
            return _UNREAD
        try:
            return value.cell_contents
        except ValueError:
            return _UNREAD
    return value


def _verify_fast_locals() -> bool:
    """The layout holds for this interpreter: a function, its *args and **kwargs, a generator and a cell argument
    read back the very objects passed. Any mismatch disables the reader (identities read unavailable)."""
    global _FAST_LOCALS_OK
    _FAST_LOCALS_OK = True
    try:
        a, b = object(), object()

        def probe(x, *rest, k=None, **kw):
            frame, code = sys._getframe(), probe.__code__
            return [_fast_local(frame, code, i, True) for i in range(4)]
        got = probe(a, 1, k=b, z=2)

        def gen(x):
            yield _fast_local(sys._getframe(), gen.__code__, 0, True)

        def cell(x):
            def inner():
                return x
            return _fast_local(sys._getframe(), cell.__code__, 0, True), _fast_local(sys._getframe(), cell.__code__, 0, False)
        ok = (got[0] is a and got[1] is b and type(got[2]) is tuple and got[2] == (1,) and type(got[3]) is dict
              and next(gen(a)) is a and cell(a)[0] is a and cell(a)[1] is _UNREAD)
    except Exception:  # noqa: BLE001
        ok = False
    _FAST_LOCALS_OK = ok
    return ok


_FAST_LOCALS_OK = False
_verify_fast_locals()


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


def _code_key(code) -> int:
    """A code object's key in the observer's maps: its id. Hashing a code object hashes its ``co_name`` and
    ``co_consts``, and comparing two compares them (CPython ``code_hash`` / ``code_richcompare``); crafted code
    may make those application objects (plan ruling 11). Each map also holds the code object itself, so its
    id stays unique while it is a key."""
    return id(code)


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


# what differs between ANY two runs of the same failure: an address, a mock's id, pytest's per-session tmp_path number
_RUN_VARIANT = re.compile(r"0x[0-9a-fA-F]+|(?<=id=')\d+(?=')|(?<=/pytest-)\d+(?=/)")


def _message_digest(text) -> str | None:
    """sha256 of a failure message with its run-variant parts blanked, so an observed failure and its unobserved twin
    can be required to fail with the same message (``defence._conformance``; proposed ruling 13, Codex phase-1 r18).
    Read after the call phase, from pytest's own report; None for anything but an exact str."""
    if type(text) is not str:
        return None
    return hashlib.sha256(_RUN_VARIANT.sub("?", text).encode("utf-8", "surrogatepass")).hexdigest()


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
        frames, message, digest = [], None, None
        if exc is not None:
            for entry in exc.traceback:
                frames.append([os.path.realpath(str(entry.path)), entry.lineno + 1, entry.name])
            crash = getattr(getattr(rep, "longrepr", None), "reprcrash", None)
            text = getattr(crash, "message", None)
            message = text[:500] if type(text) is str else None
            digest = _message_digest(text)
        name = type_name(exc.type) if exc is not None else None
        _write({"kind": "report", "nodeid": item.nodeid, "when": call.when, "outcome": rep.outcome,
                "wasxfail": getattr(rep, "wasxfail", None),
                "exc_module": name.rsplit(".", 1)[0] if name else None,
                "exc_qualname": _TYPE_QUALNAME.__get__(exc.type, type) if exc is not None else None,
                "imperative_xfail": bool(exc is not None and issubclass(exc.type, pytest.xfail.Exception)),
                "frames": frames, "message": message, "message_sha256": digest, "marker": _marker(item)})
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


def _import_record(mod) -> dict:
    """One bts module's provenance, read without application dispatch (Codex phase-1 r9 #3): an exact
    module, its namespace by the C descriptor, ``__file__`` by iteration over exact-str keys. Anything else
    is recorded unavailable, which the runner refuses; nothing is omitted (Codex phase-1 r10 #3: a missing
    or non-str ``__file__`` dropped the module from the record, so the gate never saw it)."""
    if type(mod) is not types.ModuleType:
        return {"unavailable": "not an exact module"}
    ns = _MODULE_DICT.__get__(mod, types.ModuleType)
    if type(ns) is not dict:
        return {"unavailable": "no plain module namespace"}
    found, f, why = _lookup(ns, "__file__")
    if why is not None:
        return {"unavailable": why}
    if not found or type(f) is not str:
        return {"unavailable": "no __file__ that is an exact str"}
    return {"file": os.path.realpath(f), "sha256": _sha_file(f)}


def pytest_sessionfinish(session, exitstatus):
    mods, unavailable = {}, None
    found, modules, why = _lookup(_MODULE_DICT.__get__(sys, types.ModuleType), "modules")
    if not found or why is not None or type(modules) is not dict:
        unavailable = why or "sys.modules is not a plain dict"
    else:
        for name, mod in list(dict.items(modules)):
            if type(name) is not str:
                # its hash and __eq__ are application code, so hashed lookup may resolve it as any module
                # name, whatever its text (Codex phase-1 r10 #3): no record can say which modules were imported
                unavailable = "a sys.modules key that is not an exact str"
                break
            if name == "bts" or name.startswith("bts."):
                mods[name] = _import_record(mod)
    _write({"kind": "imports", "modules": mods} if unavailable is None
           else {"kind": "imports", "modules": {}, "unavailable": unavailable})
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
        self.mock_slots = tuple(self.mock_call_code.co_varnames.index(n) for n in ("self", "args", "kwargs"))
        self.nodeid = nodeid
        self.prod_root = os.path.realpath(prod_root) + os.sep if prod_root else None
        self.entries = {(os.path.realpath(e["file"]), e["qualname"]) for e in observe.get("entries", [])}
        br = observe.get("branch")
        self.branch = (os.path.realpath(br["file"]), int(br["line"])) if br else None
        self.returns = {(os.path.realpath(r["file"]), r["qualname"]): r.get("classify", [])
                        for r in observe.get("returns", [])}
        self.boundaries = [dict(b) for b in observe.get("boundaries", [])]
        self.events: list[dict] = []
        self.instrumented: dict = {}            # _code_key -> the instrumented code object
        self.boundary_codes: dict = {}          # _code_key -> (code object, [(boundary spec, bound receiver or None)])
        self.state: dict = {}                   # spec name -> {"keys", "chain", "current", "held"}
        self.watch_index: dict = {}             # address of a namespace dict -> [(spec name, chain level)]
        self.call_watch: dict = {}              # address of a mock class's namespace -> {spec names}
        self.watched: dict = {}                 # address -> namespace dict, kept alive while watched
        self.functions: dict = {}               # address of a target function -> (function, [(spec, receiver)])
        self.dict_watcher = None
        self._dict_cb = None                    # the ctypes trampoline, kept alive while installed
        self.paths: dict[str, str] = {}
        self.mock_types: dict = {}              # id of a held mock -> its class when first held
        self.tool_acquired = self.started = self.active = False
        # one lock for every binding-state transition and every attribution read (Codex phase-1 r10 part 2
        # #1: another thread's invalidation landed between a re-resolution's read and its publication, which
        # then overwrote it). The lock serializes watcher bookkeeping and attribution checks. It does not cover
        # the subsequent C-level store commit; _current_now separately re-resolves the binding (Codex phase-1 r11
        # #2), and since plan ruling 12 those reads run only while this thread is the only one. Reentrant:
        # observer code can start inside a callback.
        self._lock = threading.RLock()

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
        if not _alone():       # ruling 12 (r12 #3): owned fields only; no exception argument or class namespace read
            self._record("observer_error", {"where": where, "error_type": "<unread: another thread was alive>",
                                            "detail": None})
            return
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
        if value is _UNREAD:                 # the observer could not read it: never a value (r13 #1)
            return {"value": None, "category": "unavailable", "reason": _UNREAD_ARGS}
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
        if not _alone():                                 # nothing is read: the binding is never held (missed)
            self.state[spec["name"]] = {"keys": [], "chain": [], "current": None, "held": []}
            self._gap(spec["name"], _CONCURRENT, where="start")
            return
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
        self.functions.setdefault(id(func), (func, []))[1].append((spec, receiver))   # its __code__ can be swapped (_swapped)
        self._add_code(code, spec, receiver)

    def _add_code(self, code, spec: dict, receiver) -> None:
        pairs = self.boundary_codes.setdefault(_code_key(code), (code, []))[1]
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
        # the changed namespace must itself still be supported: a store through an exact key leaves any key that
        # is not an exact str stored there earlier (Codex phase-1 r10 #2: such a store re-resolved the binding
        # while the namespace still held one). Read before the change, whose own key is an exact str.
        _found, _old, why = _lookup(chain[level][0], chain[level][1])
        if why is not None:
            st["chain"], st["current"] = chain, None
            self._gap(name, why)
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
            with self._lock:
                hits = self.watch_index.get(dict_addr, ())
                calls = self.call_watch.get(dict_addr, ())
                if not hits and not calls:
                    return 0
                alone = _alone()
                if not alone:          # cut by address, before any reference to the key or value (r12 #3)
                    d = self.watched.get(dict_addr)
                    for name in sorted({n for n, _lv in hits} | set(calls)):
                        self._gap(name, _CONCURRENT, where="store")
                    for name, level in list(hits):
                        st = self.state[name]
                        if level < len(st["chain"]) and st["chain"][level][0] is d:
                            st["chain"], st["current"] = st["chain"][:level + 1], None
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

    def _swapped(self, code):
        """The boundary entry for ``code`` when a held function's ``__code__`` is now ``code`` (replaced in place
        after it was held, Codex phase-1 r5 #1): registered for the same spec + receiver at the new code's first
        start, before its body runs. ``__code__`` is a C member of an exact function, and the comparison is by
        identity. Read only while this thread is alone (plan ruling 12); a swapped code that first starts while
        another thread lives, or whose start was disabled earlier, is not registered, which can only miss a call."""
        if not _alone():
            return None
        for func, pairs in list(self.functions.values()):
            if func.__code__ is code:
                for spec, receiver in pairs:
                    self._add_code(code, spec, receiver)
        return self.boundary_codes.get(_code_key(code))

    def _install_watchers(self) -> bool:
        try:
            self._dict_cb = _DICT_WATCH_CB(self._on_dict)
            self.dict_watcher = _API.PyDict_AddWatcher(self._dict_cb)
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
        self.dict_watcher = None

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
        with self._lock:
            for spec in self.boundaries:
                self._track(spec)
                if not watching:
                    self._gap(spec["name"], "store watching unavailable: a rebinding could go unseen")
        ev = mon.events
        # the bootstrap census counts every callback registration but these exact objects (Codex phase-1 r16)
        own = [(ev.PY_START, self._on_start), (ev.LINE, self._on_line), (ev.PY_RETURN, self._on_return),
               (ev.PY_UNWIND, self._on_unwind)]
        if type(_AUDIT_CENSUS) is dict and type(_AUDIT_CENSUS.get("own")) is list:
            _AUDIT_CENSUS["own"].extend(f for _event, f in own)
        for event, f in own:
            mon.register_callback(TOOL_ID, event, f)
        self.active = True
        mon.set_events(TOOL_ID, ev.PY_START | ev.PY_UNWIND)

    def stop(self) -> None:
        """Release everything start() acquired, whether or not it completed, each step on its own."""
        mon = sys.monitoring
        if self.tool_acquired:
            steps = [lambda: mon.set_events(TOOL_ID, 0)]
            steps += [lambda c=code: mon.set_local_events(TOOL_ID, c, 0) for code in self.instrumented.values()]
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
                with self._lock:
                    st = self.state.get(spec["name"])
                    if not _alone():
                        self._gap(spec["name"], _CONCURRENT, where="end")
                        continue
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
            # an application trace, profile or monitoring function runs application code inside the observer's
            # callbacks (the dict watcher's are ordinary Python calls) and can set f_lineno to re-run a frame's
            # entry (Codex phase-1 r16): one active now, or any installed since the bootstrap, is recorded
            tracing = (sys.gettrace() is not None or sys.getprofile() is not None or threading.gettrace() is not None
                       or threading.getprofile() is not None
                       or any(sys.monitoring.get_tool(i) is not None for i in range(6) if i != TOOL_ID))
            return {"audit_hooks_added": census["hooks_added"] if type(census) is dict else None,
                    "gc_enabled": gc.isenabled(), "signal_handlers": handlers, "tracing": tracing,
                    "tracers_installed": census.get("tracers_installed") if type(census) is dict else None}
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
            if offset != _first_resume(code):    # not a call's start: nothing is read or recorded here (r15)
                return None
            if code is self.mock_call_code:
                with self._lock:
                    frame = sys._getframe(1)
                    # the mock itself, from its frame's own slot (not a cell): the frame owns it (r12 #1)
                    me = _fast_local(frame, code, self.mock_slots[0], False)
                    matched = [sp for sp in self.boundaries
                               if any(o is me for o in self.state.get(sp["name"], {}).get("held", ()))]
                    if matched and not _alone():                 # ruling 12: no application object is read
                        for sp in matched:
                            self._boundary(sp, frame, [], {}, reason=_CONCURRENT)
                    elif matched:
                        extra = _fast_local(frame, code, self.mock_slots[1], True)
                        kwargs = _fast_local(frame, code, self.mock_slots[2], True)
                        if type(extra) is not tuple or type(kwargs) is not dict:   # unread, never an empty call
                            for sp in matched:
                                self._boundary(sp, frame, [], {}, reason=_UNREAD_ARGS)
                        else:
                            self._matched(matched, [self._current_now(sp) is me for sp in matched], frame,
                                          list(extra), kwargs,                  # not copied: a copy can compare keys
                                          unverified=None if self._mock_verified(me) else
                                          "a mock whose effective __call__ is not the standard one, or whose class changed")
                return None
            entry = self.boundary_codes.get(_code_key(code))
            if entry is None and self.functions:             # a held function's __code__ replaced in place
                with self._lock:
                    entry = self._swapped(code)
            if entry is not None:
                with self._lock:
                    pairs = entry[1]
                    frame = sys._getframe(1)
                    if not _alone():         # ruling 12, before ANY read (r12 #1): every candidate, no receiver
                        matched = []
                        for sp, _rcv in pairs:
                            if not any(s is sp for s in matched):
                                matched.append(sp)
                                self._boundary(sp, frame, [], {}, reason=_CONCURRENT)
                    else:
                        nargs = code.co_argcount + code.co_kwonlyargcount
                        # unread values stay unread: an identity reaching one is unavailable, never None (r13 #1)
                        args = [_fast_local(frame, code, i, True) for i in range(code.co_argcount)]
                        unread = None
                        if code.co_flags & inspect.CO_VARARGS:
                            extra = _fast_local(frame, code, nargs, True)
                            if type(extra) is tuple:
                                args += list(extra)
                            else:                    # never an empty *args: a later accessor would read another argument
                                unread = _UNREAD_ARGS
                        kwargs = {}
                        for i in range(nargs):
                            kwargs[code.co_varnames[i]] = _fast_local(frame, code, i, True)
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
                                self._matched(mine, [self._is_current_call(sp, code, first) for sp in mine], frame, args, kwargs,
                                              unverified=unread)
                        else:                                        # same code, another receiver: ambiguous
                            matched = others
                            for sp in others:
                                self._boundary(sp, frame, args, kwargs, reason="another receiver of the boundary's code")
            filename = self._real(code.co_filename)
            if self.prod_root is None or not filename.startswith(self.prod_root):
                return None if matched else sys.monitoring.DISABLE
            if _code_key(code) not in self.instrumented:
                events = 0
                if self.branch and filename == self.branch[0]:
                    events |= sys.monitoring.events.LINE
                key = (filename, _name(code.co_qualname))
                if key in self.returns or key in self.entries:
                    events |= sys.monitoring.events.PY_RETURN       # PY_UNWIND is global-only (3.12)
                if events:
                    sys.monitoring.set_local_events(TOOL_ID, code, events)
                self.instrumented[_code_key(code)] = code
            if (filename, _name(code.co_qualname)) in self.entries:
                frame = sys._getframe(1)
                stack, is_async = self._stack(frame)
                self._record("entry", {"file": filename, "qualname": _name(code.co_qualname), "frame": id(frame),
                                       "stack": stack[:40], "async": is_async})
        except BaseException as e:  # noqa: BLE001
            self._err("py_start", e)
        return None

    def _current_now(self, spec: dict):
        """The value a call must be to be attributed to ``spec``, or None: the bookkept current value, counted
        only while the binding resolves to it NOW through watched namespaces that are all supported (Codex
        phase-1 r10 part 2 #1). A store's callback runs before its change, so a re-resolution can read a
        namespace another thread is about to change; this read, at the call, sees the change if it landed
        before the call. Called under ``_lock``, and only while this thread is the interpreter's only thread (plan
        ruling 12), so no registered thread has a store pending or changes a namespace while it reads (a native
        thread attaching after the sample is not excluded: Codex phase-1 r13 #4)."""
        cur = self.state[spec["name"]]["current"]
        if cur is None:
            return None
        _keys, chain, value, why = _resolve_chain(spec["binding"])
        if why is not None or value is not cur:
            return None
        for ns, _key in chain:
            if self.watched.get(id(ns)) is not ns:
                return None
        return cur

    def _is_current_call(self, spec: dict, code, first) -> bool:
        """A start of ``code`` is a call of ``spec``'s current value: that value's code, and its receiver when
        it is a bound method (Codex phase-1 r4 #1.4)."""
        cur = self._current_now(spec)
        if cur is None or self._is_mock(cur) or self._code_of(cur) is not code:
            return False
        return type(cur) is not _METHOD or cur.__self__ is first

    def _boundary(self, spec: dict, callee_frame, args, kwargs, *, reason: str | None = None) -> None:
        stack, is_async = self._stack(callee_frame.f_back)
        # the callee itself may be a coroutine or async generator driven from synchronous code (r13 #2)
        is_async = is_async or bool(callee_frame.f_code.co_flags & CO_ASYNC)
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
            if not _alone():                                # ruling 12: the value is not read
                safe = {"unavailable": _CONCURRENT, "incomplete": True}
            else:
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
