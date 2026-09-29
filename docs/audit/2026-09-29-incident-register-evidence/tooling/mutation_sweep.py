"""Mutation sweep for the W1.5 evidence tooling (Codex phase-1 r3: 'nonzero return is not a kill').

    python mutation_sweep.py <build-worktree> [LABEL,LABEL,...]

1. BASELINE: the tooling tests must pass cleanly (exit 0, no FAILED/ERROR lines); otherwise stop.
2. For each mutant (one check disabled), rerun the WHOLE suite (no ``-x``) with ``-rfE`` and classify
   from pytest's own summary lines: KILLED only when at least one node FAILED with an ``AssertionError``
   or pytest.raises' "DID NOT RAISE" (the intended failure shapes) and NO node ERRORED anywhere in the run; ERRORED when any ERROR line
   appears (collection/setup/teardown — not evidence the check matters); SURVIVED when everything
   passed. EVERY killing node id is printed, so a kill for the wrong reason is visible.
Known equivalent guard (not listed): the ``completed and`` condition on the defence/replay verdict is
redundant while every exception path records a reason (D11/P7 pin that reason); removing it cannot be
observed today. The source file is restored after every mutant (verified by byte comparison). No bytecode is written
during the sweep and none compiled before it is left to be read (a same-size mutant or restore written
within the same second as the previous compile would otherwise run the stale .pyc).
"""
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

B = Path(sys.argv[1])
D = B / "scripts/audit/incident_register"
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
TESTS = ["tests/scripts/incident_register/"]
# COLUMNS: no summary truncation. PYTHONDONTWRITEBYTECODE: a mutant or restore of the same size written
# within the same second as the previous compile would otherwise run the stale .pyc (seen 2026-09-29).
ENV = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache", "TZ": "America/New_York", "COLUMNS": "1000",
       "PYTHONDONTWRITEBYTECODE": "1"}
# no -x: every mutant runs the WHOLE suite, so "no node errored" and the killer list cover every node
CMD = ["uv", "run", "--with", "jsonschema==4.23.0", "pytest", *TESTS, "-q", "-p", "no:cacheprovider", "-rfE", "--tb=line"]
M = [
 ("runner.py", 'if s["observer_file"] != run.trusted["file"] or s["observer_sha256"] != run.trusted["sha256"]:', 'if False:', "R1 observer identity"),
 ("runner.py", 'if s["prefix"] != os.path.join(wt, ".venv"):', 'if False:', "R2 own-venv prefix"),
 ("runner.py", 'if s["rootdir"] != wt:', 'if False:', "R3 rootdir"),
 ("runner.py", 'why.append("the session did not finish")', 'pass', "R4 session finish"),
 ("runner.py", 'why.append("collection errors")', 'pass', "R5 collection errors"),
 ("runner.py", 'why.append("observer errors")', 'pass', "R6 observer errors"),
 ("runner.py", 'if expected is not None and sorted(nodes) != sorted(expected):', 'if False:', "R7 inventory"),
 ("runner.py", 'if not m["file"].startswith(src):', 'if False:', "R8 foreign imports"),
 ("runner.py", '        if state != "passed":', '        if False:', "R9 per-node state"),
 ("runner.py", 'if run.returncode != want:', 'if False:', "R10 return code"),
 ("runner.py", 'if teardown is None or teardown["outcome"] != "passed":', 'if teardown is None:', "R11 teardown state"),
 ("runner.py", 'if pf.startswith(wt + os.sep) and not pf.startswith(os.path.join(wt, ".venv") + os.sep):', 'if False:', "R12 pytest shadow"),
 ("runner.py", 'why.append("the production src tree changed during the session")', 'pass', "R13 src digest during session"),
 ("runner.py", 'env.update({"W15_OBS_CONFIG": str(cfg), "PYTHONDONTWRITEBYTECODE": "1"})', 'env.update({"W15_OBS_CONFIG": str(cfg)})', "R14 no bytecode in evidence runs"),
 ("runner.py", 'why.append("the production src tree at session start differs from the tree the runner prepared")', 'pass', "R15 src digest at session start"),
 ("certify.py", 'if any(e.get("async") for e in inside):', 'if False:', "C1 async"),
 ("certify.py", 'entries = [e for e in body if e["kind"] == "entry" and e["file"] == entry["file"]', 'entries = [e for e in body if e["kind"] == "entry"', "C2 entry file"),
 ("certify.py", '    if (en["frame"], en["qualname"], en["file"]) not in {(f, q, p) for q, p, _l, f in ev.get("stack", [])}:\n        return False', '    pass', "C3 invocation on stack"),
 ("certify.py", 'common = [(b, s & live_x) for b, s in live_at_branch if b["seq"] < x["seq"] and s & live_x]', 'common = [(b, s) for b, s in live_at_branch]', "C4 common invocation after branch"),
 ("certify.py", '        if qualifying:', '        if False:', "C5 absence qualifying"),
 ("certify.py", 'why.append("positive control: the baseline did not produce the qualifying event")', 'pass', "C6 positive control"),
 ("certify.py", '        if unidentified:', '        if False:', "C7 unidentified calls"),
 ("certify.py", '    if not live_at_branch:\n        why.append', '    if False:\n        why.append', "C8 branch in live invocation"),
 ("certify.py", '        if end.get("outstanding_threads"):', '        if False:', "C9 outstanding threads"),
 ("certify.py", '        if gaps:', '        if False:', "C10 boundary coverage gap"),
 ("certify.py", '        if live_at_branch and not completed:', '        if False:', "C11 invocation completion"),
 ("certify.py", 'r["kind"] in ("entry_exit", "entry") and r["thread"] == en["thread"]:', 'False:', "C12 exit/reuse splits invocations"),
 ("observer.py", '                if spec is not None:\n                    extra = loc.get("args", ())', '                if False:\n                    extra = loc.get("args", ())', "O1 mock callee recorder"),
 ("observer.py", 'return None if spec is not None else sys.monitoring.DISABLE', 'return sys.monitoring.DISABLE', "O2 keep boundary callee enabled"),
 ("observer.py", '            if code.co_flags & CO_ASYNC:', '            if False:', "O3 async flag"),
 ("observer.py", '                sys.monitoring.restart_events()              # its PY_START may have been disabled', '                pass', "O4 restart on rebinding"),
 ("observer.py", '    d = _plain_instance_dict(value)\n    if d is None:', '    return {"repr": repr(value)}\n    if d is None:', "O5 no application repr"),
 ("observer.py", 'outstanding = [t for t in threading.enumerate() if t.ident not in self.threads_at_start and t.is_alive()]', 'outstanding = []', "O6 outstanding thread count"),
 ("observer.py", '            self._record("boundary_gap", {"name": spec["name"], "reason": "not a Python-observable callable",', '            return None and self._record("boundary_gap", {"name": spec["name"], "reason": "not a Python-observable callable",', "O7 gap for C boundaries"),
 ("observer.py", 'safe = _safe(retval) if how == "return" else {"raised": type_name(type(retval))}', 'safe = _safe(retval)', "O8 exceptional exit as type name"),
 ("defence.py", 'why += _assertion_ok(mutant, n, k["_path"], k["_line"], wt)', 'pass', "D1 assertion location"),
 ("defence.py", '        why.append(f"{where}: frozen files changed: {changed[:5]}")', '        pass', "D2 frozen drift"),
 ("defence.py", 'why.append(f"declared killing node {n} was not killed")', 'pass', "D3 not killed"),
 ("defence.py", 'why += [f"certificate {n}: {r}" for r in cert["reasons"]]', 'pass', "D4 certificate reasons"),
 ("defence.py", 'why += [f"mutant: {r}" for r in mg]', 'pass', "D5 mutant gate"),
 ("defence.py", 'why += [f"green: {r}" for r in g]', 'pass', "D6 green gate"),
 ("defence.py", 'why += [f"restored: {r}" for r in rg]', 'pass', "D7 restored gate"),
 ("defence.py", 'if spec["branch"]["path"] not in {e[0] for e in spec["mutation_edits"]}:', 'if False:', "D8 branch in mutated file"),
 ("defence.py", '            why.append(f"{stage}: {n} is {a} observed but {b} unobserved")', '            pass', "D9 observer-off conformance"),
 ("defence.py", '    inside = [f for f in frames if f[0].startswith(root) and not f[0].startswith(venv)]', '    inside = [f for f in frames if f[0].startswith(root)]', "D10 venv frames excluded"),
 ("defence.py", '        why.append(f"aborted: {type(e).__name__}: {e}")     # recorded, then surfaced to the caller', '        pass', "D11 unexpected exception recorded"),
 ("replay.py", '            why += problems', '            pass', "P1 audit problems"),
 ("replay.py", 'if not last or last[0] != s["_path"] or last[1] != s["_line"]:', 'if False:', "P2 replay assertion location"),
 ("replay.py", 'why += [f"red: {r}" for r in runner.gate(red, worktree=worktree, mode="mutant", expected=inventory)]', 'pass', "P3 red gate"),
 ("replay.py", 'if (call.get("exc_module"), call.get("exc_qualname")) != ("builtins", "AssertionError"):', 'if False:', "P4 replay exception type"),
 ("replay.py", '            elif not str(entry.get("reason", "")).strip():', '            elif False:', "P5 reason required"),
 ("replay.py", '        why += _drift(worktree, m0, v0, "after green", src_too=True)       # before any swap or reset', '        pass', "P6 drift after green"),
 ("replay.py", '        why.append(f"aborted: {type(e).__name__}: {e}")     # recorded, then surfaced to the caller', '        pass', "P7 unexpected exception recorded"),
 ("acceptance.py", 'if c.get("imperative_xfail"):', 'if False:', "A1 imperative"),
 ("acceptance.py", 'if marker.get("raises") != [r["exception"]]:', 'if False:', "A2 marker raises"),
 ("acceptance.py", 'if not (strict is True or (strict is None and _ini_strict(marked))):', 'if False:', "A3 strict"),
 ("acceptance.py", 'if f"{u.get(\'exc_module\')}.{u.get(\'exc_qualname\')}" != r["exception"]:', 'if False:', "A4 exception identity"),
 ("acceptance.py", 'if not last or last[0] != oracle_file or last[2] != r["oracle"]["qualname"]:', 'if False:', "A5 oracle frame"),
 ("acceptance.py", 'if f"got the declared bad value {r[\'bad\']}" not in msg or f"required {r[\'required\']}" not in msg:', 'if False:', "A6 message values"),
 ("acceptance.py", '        if not any(e["kind"] == "entry" and e["file"] == entry_file and e["qualname"] == r["entry"]["qualname"]\n                   for e in inside):', '        if False:', "A7 production invocation"),
 ("acceptance.py", 'out["_session"].append("the two runs collected different test-file bytes")', 'pass', "A8 test bytes pair"),
 ("acceptance.py", '        return ("connected", []) if got == actual else \\\n            ("unconnected", [f"connection: {conn[\'qualname\']} returned', '        return ("connected", []) if True else \\\n            ("unconnected", [f"connection: {conn[\'qualname\']} returned', "A9 return connection"),
 ("acceptance.py", '    res["reasons"] += _drift(wt, m0["files"], m0["untracked"], v0, (), "after marked")', '    pass', "A10 closure freeze after marked"),
 ("records.py", 'if onset_lo is not None and w_hi <= onset_lo:', 'if False:', "V1 report before the event"),
 ("records.py", 'elif lat["min_minutes"] > feasible[0] + 0.01 or lat["max_minutes"] < feasible[1] - 0.01:', 'elif False:', "V2 latency within bounds"),
 ("records.py", 'if hashlib.sha256(data).hexdigest() != entry["acceptance_sha256"]:', 'if False:', "V3 acceptance hash binding"),
 ("records.py", '    if not o.get("continuing"):\n        return _span(o.get("onset"))[1]', '    pass', "V4 one-shot 48h window"),
 ("records.py", 'if steps and not any((lo is None or w_hi > lo) and (hi is None or w_lo <= hi + window) for lo, hi in steps):', 'if False:', "V5 fix-step report window"),
 ("records.py", 'if not cites and not steps:', 'if False:', "V6 uncited report"),
 ("records.py", '    return min(ends) if ends else None', '    return max(ends) if ends else None', "V7 earliest end event"),
 ("records.py", 'links = set(o.get("links") or [f["link"] for f in r["fix"]])', 'links = {f["link"] for f in r["fix"]}', "V8 occurrence links"),
 ("owned.py", 'if os.path.realpath(gitdir) == os.path.realpath(common):', 'if False:', "W1 primary checkout"),
 ("owned.py", '        if cur.is_symlink():', '        if False:', "W2 symlinked component"),
 ("owned.py", '            if name.endswith(".pyc"):\n                continue\n            p = os.path.join(dirpath, name)\n            h.update(os.path.relpath(p, base).encode() + b"\\0")\n            if os.path.islink(p):\n                h.update(b"link:" + os.readlink(p).encode())\n            elif os.path.isfile(p):\n                with open(p, "rb") as fh:\n                    for chunk in iter(lambda: fh.read(1 << 20), b""):\n                        h.update(chunk)', '            if name.endswith(".pyc"):\n                continue\n            p = os.path.join(dirpath, name)\n            h.update(os.path.relpath(p, base).encode() + b"\\0")', "W3 venv content hashed"),
 ("owned.py", '                _tree_hash(h, Path(target))', '                pass', "W4 pth trees hashed"),
]


def run():
    p = subprocess.run(CMD, cwd=B, capture_output=True, text=True, env=ENV)
    lines = p.stdout.splitlines()
    failed = [l for l in lines if l.startswith("FAILED ")]
    errored = [l for l in lines if l.startswith("ERROR ")]
    return p.returncode, failed, errored, (lines[-1] if lines else p.stderr[-200:])


def _on_term(signum, frame):
    raise KeyboardInterrupt            # so the finally below restores the mutated file


signal.signal(signal.SIGTERM, _on_term)
for bak in D.glob("*.sweepbak"):       # a previous sweep died mid-mutant: restore before anything else
    shutil.move(bak, bak.with_suffix(""))
    print(f"restored leftover {bak.name}", flush=True)
for cache in [*D.rglob("__pycache__"), *(B / "tests/scripts/incident_register").rglob("__pycache__")]:
    shutil.rmtree(cache)               # no bytecode compiled before the sweep can be read during it
rc, failed, errored, tail = run()
print(f"BASELINE rc={rc} failed={len(failed)} errored={len(errored)} | {tail}", flush=True)
if rc != 0 or failed or errored:
    print("baseline is not clean; stopping", flush=True)
    sys.exit(1)
for fname, old, new, label in M:
    if ONLY and label.split()[0] not in ONLY:
        continue
    f = D / fname
    src = f.read_bytes()
    text = src.decode()
    if text.count(old) != 1:
        print(f"{label}: ANCHOR COUNT {text.count(old)}", flush=True)
        continue
    bak = f.with_name(f.name + ".sweepbak")
    bak.write_bytes(src)
    try:
        f.write_text(text.replace(old, new, 1))
        rc, failed, errored, tail = run()
        # an AssertionError, or pytest.raises' own failure ("Failed: DID NOT RAISE"): the assertion
        # idiom for "the expected refusal did not happen"
        assertion_kills = [l for l in failed if "AssertionError" in l or " - assert " in l or "DID NOT RAISE" in l]
        if errored:
            verdict = "ERRORED"
        elif assertion_kills:
            verdict = "KILLED"
        elif failed:
            verdict = "FAILED-OTHER"
        else:
            verdict = "SURVIVED"
        nodes = [l.split(" - ")[0][len("FAILED "):] for l in (assertion_kills or failed)]
        print(f"{label}: {verdict} | {tail[:80]} | {nodes}", flush=True)
    finally:
        f.write_bytes(src)
        assert f.read_bytes() == src
        bak.unlink()
print("DONE", flush=True)
