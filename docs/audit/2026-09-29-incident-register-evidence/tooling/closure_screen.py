"""Static closure screen for the Phase 1 bounded model (plan ruling 10; ``certify.COVERAGE``).

    python closure_screen.py <repo-root> > closure_screen.txt

A conservative text screen, not a proof (Codex's consult: aliases, reflection, generated code and native
packages can do the same things; the screen's hits are REVIEWED, and a certificate still needs its
runtime checks). It lists, in ``src/bts`` and ``tests`` (the tooling's own tests excluded), every line
that could put application code inside observation or change what the recorder attributes:

* audit hooks, signal handlers and timers, collector control, finalizers and exit hooks, lifetime introspection
  (weak-reference counts and collections, reference counts, collector referrer queries: proposed ruling 13);
* ``__class__`` assignment and ``sys.modules`` stores;
* patches of the standard-library modules the observer itself calls.
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
    # proposed ruling 13 (Codex phase-1 r17): behaviour that depends on object lifetimes or reference bookkeeping
    "lifetime introspection": (r"\bweakref\.(?:getweakrefcount|getweakrefs|WeakSet|WeakValueDictionary|WeakKeyDictionary|"
                               r"WeakMethod)\b|\bsys\.getrefcount\b|\bgc\.get_(?:referrers|referents|objects)\b"),
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
