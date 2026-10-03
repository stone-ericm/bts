"""Static closure screen for the Phase 1 bounded model (plan ruling 10; ``certify.COVERAGE``).

    python closure_screen.py <repo-root> > closure_screen.txt

A warning list, not a proof. It flags selected literal spellings in ``src/bts`` and ``tests`` (the tooling's
own tests excluded), and does not resolve reflection (``getattr`` with a computed name), generated code or a
dependency outside those two roots. Codex's consult and its phase-1 r18 #3 measured those misses. So every
prepared closure also needs an independent source review for these exclusions, with the disposition recorded,
and a certificate still needs its runtime checks. A module the screen names is flagged at its import, so an
alias made there is listed too. The categories are lines that could put application code inside observation or
change what the recorder attributes:

* audit hooks, signal handlers and timers, collector control, finalizers and exit hooks;
* ``__class__`` assignment and ``sys.modules`` stores;
* patches of the standard-library modules the observer itself calls;
* and, under proposed ruling 13, code that can see the observer: lifetime introspection (weak references,
  reference counts, the collector's queries); instrumentation introspection (the ``sys.monitoring`` registry
  and event sets, adaptive or instrumented bytecode, another thread's frames); elapsed time or resource use;
  object addresses.
"""
import re
import sys
from pathlib import Path

OBSERVER_STDLIB = r"(?:os|sys|gc|signal|threading|json|re|hashlib|inspect|itertools|traceback|ctypes|types)"
PATTERNS = {
    "audit hook": r"\baddaudithook\b",
    "signal handler or timer": r"\bsignal\.(?:signal|setitimer|alarm|siginterrupt|pthread_kill|raise_signal)\b|\bos\.kill\b",
    "collector control": r"\bgc\.(?:enable|disable|callbacks|set_threshold|freeze)\b",
    "finalizer or exit hook": r"\bdef __del__\b|\bweakref\.(?:finalize|ref)\b|\batexit\.register\b",
    # proposed ruling 13 (Codex phase-1 r17 #1, r18 #1-#3): behaviour that can see the observer. A module is flagged
    # wherever its name appears, so an import alias is listed at its import
    "lifetime introspection": r"\bweakref\b|\bsys\.getrefcount\b|\bimport gc\b|\bfrom gc import\b|\bgc\.(?:get|is)_\w+",
    "instrumentation introspection": (r"\bsys\.monitoring\b|\bfrom sys import\b[^#\n]*\bmonitoring\b|\bget_tool\b|"
                                      r"\bget_(?:local_)?events\b|\badaptive\s*=|\b_co_code_adaptive\b|\bINSTRUMENTED_|"
                                      r"\bimport dis\b|\bfrom dis import\b|\b_current_frames\b"),
    "elapsed time or resource use": (r"\b(?:monotonic|perf_counter|process_time|thread_time)(?:_ns)?\(|"
                                     r"\btime\.time(?:_ns)?\(|^\s*from time import\b|^\s*import time as\b|"
                                     r"\bresource\.getrusage\b|\btracemalloc\b|\bsys\.getallocatedblocks\b|\btimeit\b"),
    "object address": r"(?<![\w.])id\(",
    "class assignment": r"\.__class__\s*=(?!=)",
    "sys.modules store": r"\bsys\.modules\s*(?:\[[^\]]*\]\s*=(?!=)|=(?!=))|\bsys\.modules\.(?:pop|update|setdefault|clear)\b",
    "stdlib patch": (r"\bpatch(?:\.object)?\(\s*['\"]?" + OBSERVER_STDLIB + r"[.,'\"]|\bmonkeypatch\.(?:setattr|delattr)\(\s*['\"]?"
                     + OBSERVER_STDLIB + r"[.,'\"]"),
}
EXCLUDE = ("tests/scripts/incident_register/",)


def screen(root: Path) -> list[tuple[str, str, int, str]]:
    hits = []
    for base in ("src/bts", "tests"):
        for path in sorted((root / base).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if rel.startswith(EXCLUDE):
                continue
            for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                for label, pat in PATTERNS.items():
                    if re.search(pat, line):
                        hits.append((label, rel, n, line.strip()[:160]))
    return hits


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    hits = screen(root)
    print(f"# closure screen of {root.name}: src/bts and tests (tests/scripts/incident_register excluded)")
    for label in PATTERNS:
        mine = [h for h in hits if h[0] == label]
        print(f"\n## {label}: {len(mine)}")
        for _, rel, n, text in mine:
            print(f"{rel}:{n}: {text}")
