"""Gadget audit of the runner's reviewed vocabulary (revision 9). Run from the repository root:
    UV_CACHE_DIR=/tmp/uv-cache uv run python -B docs/audit/2026-10-06-c2-framing-evidence/gadget_audit.py
Breadth-first from every allowed module, following only allowed attribute names (pytest: only its three), five levels
deep, over modules, classes and callables reached statically (inspect.getattr_static). It flags any module reached that
is not an allowed module, any callable from a dangerous module (dynamic import, deserialization, frames, process
control, pytest or pluggy internals), and any dangerous builtin. It does not model instances or call results; the
suite's tests and the reviewed lists bound those."""
import importlib, inspect, sys, types, builtins
sys.path[:0] = [".", "src"]
from scripts.audit.c2_framing import mutant_runner as R
DANGER_MODS = ("pickle", "_pickle", "marshal", "importlib", "_pytest", "pluggy", "runpy", "pkgutil", "inspect", "gc",
               "ctypes", "sys", "signal", "_thread", "threading", "code", "pdb", "bdb", "traceback", "builtins", "os",
               "posix", "subprocess", "shelve", "dill", "cloudpickle", "joblib", "copyreg", "types", "operator",
               "string", "logging.config", "unittest.mock", "pydoc", "atexit", "faulthandler", "multiprocessing")
DANGER_BUILTINS = {getattr(builtins, n) for n in ("eval", "exec", "compile", "getattr", "setattr", "delattr", "vars",
                   "globals", "locals", "__import__", "open", "breakpoint", "exit", "quit", "input")}
start = {}
for m in sorted(R.ALLOWED_MODULES):
    start[m] = importlib.import_module(m)
seen, frontier, flags = {}, [(name, mod, (name,)) for name, mod in start.items()], []
for depth in range(5):
    nxt = []
    for label, obj, path in frontier:
        if id(obj) in seen:
            continue
        seen[id(obj)] = path
        names = R.PYTEST_ATTRIBUTES if obj is start.get("pytest") else R.ALLOWED_ATTRIBUTES
        for a in names:
            try:
                v = inspect.getattr_static(obj, a)
            except AttributeError:
                continue
            except Exception:
                continue
            p = path + (a,)
            mod = v.__name__ if isinstance(v, types.ModuleType) else getattr(v, "__module__", None) or ""
            if isinstance(v, types.ModuleType) and v.__name__ not in R.ALLOWED_MODULES:
                flags.append(("module", ".".join(p), v.__name__))
            elif any(mod == d or mod.startswith(d + ".") for d in DANGER_MODS) and callable(v):
                flags.append(("callable", ".".join(p), f"{mod}.{getattr(v, '__qualname__', '?')}"))
            elif v in DANGER_BUILTINS if isinstance(v, types.BuiltinFunctionType) else False:
                flags.append(("builtin", ".".join(p), repr(v)))
            if isinstance(v, (types.ModuleType, type)) or callable(v):
                nxt.append((a, v, p))
    frontier = nxt
print("objects visited:", len(seen))
for f in sorted(set(flags), key=lambda x: (x[2], x[1])):
    print(*f)
