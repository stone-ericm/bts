"""The totality map of the revision-4 witness (manager's row C2-2a-sweep-scope: "the totality argument with file:line").

1. Every call site: each statement in the four deployed files that reaches the witness (`_sw.…`, a ledger method, or
   slate's `_take_serving`) must sit directly in a `try` whose one handler is `except Exception` with a body that only
   assigns a constant or name or passes, with no `else` or `finally`. Each such `try` holds one statement, except
   predict_local's first, which holds the witness module's import and the witness's construction (its handler sets all
   three names to None). Anything else is listed as a VIOLATION.
2. Every hook: for each `bts.serving_witness` function or method those sites call, the statements that run OUTSIDE
   any internal `try`. Only a failure of one of these reaches the call site's guard (an unforeseen fault); every other
   statement's failure is handled inside the hook. `return`/`pass` of a constant or name and the docstring are omitted.

    python totality_map.py            # from the repository root; prints the map, exits 1 on a violation
"""
import ast
import sys
from pathlib import Path

DEPLOYED = ("src/bts/model/predict.py", "src/bts/model/calibrate.py", "src/bts/orchestrator.py", "src/bts/slate.py")
WITNESS = "src/bts/serving_witness.py"
LEDGER_METHODS = {"hold", "confirm"}


def _reaches_witness(node) -> set:
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "_sw":
            names.add(n.attr)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "ledger" \
                and n.attr in LEDGER_METHODS:
            names.add("Ledger." + n.attr)
        elif isinstance(n, ast.Name) and n.id == "_take_serving" and isinstance(n.ctx, ast.Load):
            names.add("_take_serving")
    return names


def _trivial_handler(h: ast.ExceptHandler) -> bool:
    if not (isinstance(h.type, ast.Name) and h.type.id == "Exception"):
        return False
    for s in h.body:
        if isinstance(s, ast.Pass):
            continue
        if isinstance(s, ast.Assign) and isinstance(s.value, (ast.Constant, ast.Name)):
            continue
        return False
    return True


def call_sites():
    sites, violations = [], []
    for f in DEPLOYED:
        tree = ast.parse(Path(f).read_text())
        parent = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.stmt) or isinstance(node, (ast.Try, ast.FunctionDef, ast.If, ast.For, ast.With,
                                                                    ast.ClassDef, ast.While)):
                continue
            hooks = _reaches_witness(node)
            if not hooks:
                continue
            p = parent.get(node)
            ok = (isinstance(p, ast.Try) and node in p.body and len(p.handlers) == 1 and not p.orelse
                  and not p.finalbody and _trivial_handler(p.handlers[0]))
            text = ast.get_source_segment(Path(f).read_text(), node).splitlines()[0][:100]
            row = (f"{Path(f).name}:{node.lineno}", sorted(hooks), text,
                   (f"guard {Path(f).name}:{p.lineno}-{p.handlers[0].end_lineno}"
                    + (f" ({len(p.body)} statements)" if len(p.body) > 1 else "")) if ok else "UNGUARDED")
            (sites if ok else violations).append(row)
            # _take_serving is defined in slate.py itself: its definition is not a call site
    return sites, violations


def _outside_try(body, src, out, fname):
    for s in body:
        if isinstance(s, ast.Try):
            continue                                        # handled inside the hook (its handlers are reported apart)
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
            continue                                        # docstring
        if isinstance(s, ast.Pass) or (isinstance(s, ast.Return) and (s.value is None or isinstance(
                s.value, (ast.Constant, ast.Name)))):
            continue
        if isinstance(s, (ast.If, ast.For, ast.While, ast.With)):
            head = ast.get_source_segment(src, s).splitlines()[0][:100]
            out.append(f"{fname}:{s.lineno}  {head}")
            _outside_try(s.body, src, out, fname)
            _outside_try(getattr(s, "orelse", []), src, out, fname)
            continue
        out.append(f"{fname}:{s.lineno}  {ast.get_source_segment(src, s).splitlines()[0][:100]}")


def hooks(called: set):
    src = Path(WITNESS).read_text()
    tree = ast.parse(src)
    defs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defs[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, ast.FunctionDef):
                    defs[f"{node.name}.{m.name}"] = m
    slate_src = Path("src/bts/slate.py").read_text()
    slate_defs = {n.name: n for n in ast.parse(slate_src).body if isinstance(n, ast.FunctionDef)}
    rows = {}
    for name in sorted(called):
        if name == "_take_serving":
            d, s, fn = slate_defs[name], slate_src, "slate.py"
        elif name in defs:
            d, s, fn = defs[name], src, "serving_witness.py"
        else:
            rows[name] = ["(not a function: a class constructor or module attribute)"]
            continue
        out = []
        _outside_try(d.body, s, out, fn)
        rows[f"{name} ({fn}:{d.lineno})"] = out
    return rows


def main() -> int:
    sites, violations = call_sites()
    print(f"## Call sites: {len(sites)} guarded, {len(violations)} violations")
    for r in sites + violations:
        print(f"  {r[0]:24} {r[3]:46} {', '.join(r[1]):34} {r[2]}")
    called = set().union(*(set(r[1]) for r in sites + violations))
    print(f"\n## Hooks: statements outside any internal try ({len(called)} called names)")
    for name, rows in hooks(called).items():
        print(f"  {name}: {'(none)' if not rows else ''}")
        for r in rows:
            print(f"      {r}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
