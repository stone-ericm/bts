"""Mutation sweep for the W1.5 evidence tooling (Codex phase-1 r3: 'nonzero return is not a kill').

    python mutation_sweep.py <build-worktree> [LABEL,LABEL,...]

1. BASELINE: the tooling tests must pass cleanly (exit 0, no failure, no error); its test count is kept.
2. For each mutant (one check disabled), rerun the WHOLE suite (no ``-x``) and classify from pytest's
   JUnit XML, never from free-text summary lines (Codex phase-1 r4 #10: ``RuntimeError("DID NOT RAISE")``
   passed a substring test). ``classify`` is the rule:
   * KILLED — exit code 1, the baseline's test count, no testcase error, and at least one failure whose
     message has an ASSERTION shape: a rewritten ``assert`` (pytest's message starts with ``assert ``;
     ``assert`` is a keyword, so no exception class can print that prefix), an explicit builtin
     ``AssertionError`` (exact prefix — a look-alike class prints its qualified name), or pytest.raises'
     own ``Failed: DID NOT RAISE``;
   * ERRORED — any testcase error (collection/setup/teardown) or an exit code other than 0/1;
   * INCOMPLETE — a test count different from the baseline's;
   * FAILED-OTHER — failures, none of an assertion shape;
   * SURVIVED — exit 0, no failure.
   EVERY killing node id is printed, so a kill for the wrong reason is visible.
Known equivalent guard (not listed): the ``completed and`` condition on the defence/replay verdict is
redundant while every exception path records a reason (D11/P7 pin that reason); removing it cannot be
observed today. The source file is restored after every mutant (verified by byte comparison). No bytecode
is written during the sweep and none compiled before it is left to be read (a same-size mutant or restore
written within the same second as the previous compile would otherwise run the stale .pyc).
"""
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

TESTS = ["tests/scripts/incident_register/"]
# COLUMNS: no summary truncation. PYTHONDONTWRITEBYTECODE: a mutant or restore of the same size written
# within the same second as the previous compile would otherwise run the stale .pyc (seen 2026-09-29).
ENV = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache", "TZ": "America/New_York", "COLUMNS": "1000",
       "PYTHONDONTWRITEBYTECODE": "1"}
# no -x: every mutant runs the WHOLE suite, so "no node errored" and the killer list cover every node
CMD = ["uv", "run", "--with", "jsonschema==4.23.0", "pytest", *TESTS, "-q", "-p", "no:cacheprovider", "-rfE", "--tb=line"]
M = [
 ('runner.py', 'if s["observer_file"] != run.trusted["file"] or s["observer_sha256"] != run.trusted["sha256"]:', 'if False:', 'R1 observer identity'),
 ('runner.py', 'if s["prefix"] != os.path.join(wt, ".venv"):', 'if False:', 'R2 own-venv prefix'),
 ('runner.py', 'if s["rootdir"] != wt:', 'if False:', 'R3 rootdir'),
 ('runner.py', 'why.append("the session did not finish")', 'pass', 'R4 session finish'),
 ('runner.py', 'why.append("collection errors")', 'pass', 'R5 collection errors'),
 ('runner.py', 'why.append("observer errors")', 'pass', 'R6 observer errors'),
 ('runner.py', 'if expected is not None and sorted(nodes) != sorted(expected):', 'if False:', 'R7 inventory'),
 ('runner.py', 'if not m["file"].startswith(src):', 'if False:', 'R8 foreign imports'),
 ('runner.py', '        if state != "passed":', '        if False:', 'R9 per-node state'),
 ('runner.py', 'if run.returncode != want:', 'if False:', 'R10 return code'),
 ('runner.py', 'if teardown is None or teardown["outcome"] != "passed":', 'if teardown is None:', 'R11 teardown state'),
 ('runner.py', 'if pf.startswith(wt + os.sep) and not pf.startswith(os.path.join(wt, ".venv") + os.sep):', 'if False:', 'R12 pytest shadow'),
 ('runner.py', 'why.append("the production src tree changed during the session")', 'pass', 'R13 src digest during session'),
 ('runner.py', 'env.update({"W15_OBS_CONFIG": str(cfg), "PYTHONDONTWRITEBYTECODE": "1"})', 'env.update({"W15_OBS_CONFIG": str(cfg)})', 'R14 no bytecode in evidence runs'),
 ('runner.py', 'why.append("the production src tree at session start differs from the tree the runner prepared")', 'pass', 'R15 src digest at session start'),
 ('certify.py', 'if any(e.get("async") for e in inside):', 'if False:', 'C1 async'),
 ('certify.py', 'entries = [e for e in body if e["kind"] == "entry" and e["file"] == entry["file"]', 'entries = [e for e in body if e["kind"] == "entry"', 'C2 entry file'),
 ('certify.py', '    if (en["frame"], en["qualname"], en["file"]) not in {(f, q, p) for q, p, _l, f in ev.get("stack", [])}:\n        return False', '    pass', 'C3 invocation on stack'),
 ('certify.py', 'common = [(b, s & live_x) for b, s in live_at_branch if b["seq"] < x["seq"] and s & live_x]', 'common = [(b, s) for b, s in live_at_branch]', 'C4 common invocation after branch'),
 ('certify.py', '        if qualifying:', '        if False:', 'C5 absence qualifying'),
 ('certify.py', 'why.append("positive control: the baseline did not produce the qualifying event")', 'pass', 'C6 positive control'),
 ('certify.py', '        if unidentified:', '        if False:', 'C7 unidentified calls'),
 ('certify.py', '    if not live_at_branch:\n        why.append', '    if False:\n        why.append', 'C8 branch in live invocation'),
 ('certify.py', '        if end.get("outstanding_threads"):', '        if False:', 'C9 outstanding threads'),
 ('certify.py', '        if gaps:', '        if False:', 'C10 boundary coverage gap'),
 ('certify.py', '        if live_at_branch and not completed:', '        if False:', 'C11 invocation completion'),
 ('certify.py', 'r["kind"] in ("entry_exit", "entry") and r["thread"] == en["thread"]:', 'False:', 'C12 exit/reuse splits invocations'),
 ('observer.py', '                if spec is not None:\n                    extra = loc.get("args", ())', '                if False:\n                    extra = loc.get("args", ())', 'O1 mock callee recorder'),
 ('observer.py', 'return None if spec is not None else sys.monitoring.DISABLE', 'return sys.monitoring.DISABLE', 'O2 keep boundary callee enabled'),
 ('observer.py', '            if code.co_flags & CO_ASYNC:', '            if False:', 'O3 async flag'),
 ('observer.py', '    d = _plain_instance_dict(value)\n    if d is None:', '    return {"repr": repr(value)}\n    if d is None:', 'O5 no application repr'),
 ('observer.py', '            self._record("boundary_gap", {"name": spec["name"], "reason": "not a Python-observable callable",', '            return None and self._record("boundary_gap", {"name": spec["name"], "reason": "not a Python-observable callable",', 'O7 gap for C boundaries'),
 ('observer.py', 'safe = _safe(retval) if how == "return" else {"raised": type_name(type(retval))}', 'safe = _safe(retval)', 'O8 exceptional exit as type name'),
 ('defence.py', 'why += _assertion_ok(mutant, n, k["_path"], k["_line"], wt)', 'pass', 'D1 assertion location'),
 ('defence.py', '        why.append(f"{where}: frozen files changed: {changed[:5]}")', '        pass', 'D2 frozen drift'),
 ('defence.py', 'why.append(f"declared killing node {n} was not killed")', 'pass', 'D3 not killed'),
 ('defence.py', 'why += [f"certificate {n}: {r}" for r in cert["reasons"]]', 'pass', 'D4 certificate reasons'),
 ('defence.py', 'why += [f"mutant: {r}" for r in mg]', 'pass', 'D5 mutant gate'),
 ('defence.py', 'why += [f"green: {r}" for r in g]', 'pass', 'D6 green gate'),
 ('defence.py', 'why += [f"restored: {r}" for r in rg]', 'pass', 'D7 restored gate'),
 ('defence.py', 'if spec["branch"]["path"] not in {e[0] for e in spec["mutation_edits"]}:', 'if False:', 'D8 branch in mutated file'),
 ('defence.py', '            why.append(f"{stage}: {n} is {a} observed but {b} unobserved")', '            pass', 'D9 observer-off conformance'),
 ('defence.py', '    inside = [f for f in frames if f[0].startswith(root) and not f[0].startswith(venv)]', '    inside = [f for f in frames if f[0].startswith(root)]', 'D10 venv frames excluded'),
 ('defence.py', '        why.append(f"aborted: {type(e).__name__}: {e}")     # recorded, then surfaced to the caller', '        pass', 'D11 unexpected exception recorded'),
 ('replay.py', '            why += problems', '            pass', 'P1 audit problems'),
 ('replay.py', 'if not last or last[0] != s["_path"] or last[1] != s["_line"]:', 'if False:', 'P2 replay assertion location'),
 ('replay.py', 'why += [f"red: {r}" for r in runner.gate(red, worktree=worktree, mode="mutant", expected=inventory)]', 'pass', 'P3 red gate'),
 ('replay.py', 'if (call.get("exc_module"), call.get("exc_qualname")) != ("builtins", "AssertionError"):', 'if False:', 'P4 replay exception type'),
 ('replay.py', '            elif not str(entry.get("reason", "")).strip():', '            elif False:', 'P5 reason required'),
 ('replay.py', '        why += _drift(worktree, m0, v0, "after green", src_too=True)       # before any swap or reset', '        pass', 'P6 drift after green'),
 ('replay.py', '        why.append(f"aborted: {type(e).__name__}: {e}")     # recorded, then surfaced to the caller', '        pass', 'P7 unexpected exception recorded'),
 ('acceptance.py', 'if c.get("imperative_xfail"):', 'if False:', 'A1 imperative'),
 ('acceptance.py', 'if marker.get("raises") != [r["exception"]]:', 'if False:', 'A2 marker raises'),
 ('acceptance.py', 'if not (strict is True or (strict is None and _ini_strict(marked))):', 'if False:', 'A3 strict'),
 ('acceptance.py', 'if f"{u.get(\'exc_module\')}.{u.get(\'exc_qualname\')}" != r["exception"]:', 'if False:', 'A4 exception identity'),
 ('acceptance.py', 'if not last or last[0] != oracle_file or last[2] != r["oracle"]["qualname"]:', 'if False:', 'A5 oracle frame'),
 ('acceptance.py', 'if f"got the declared bad value {r[\'bad\']}" not in msg or f"required {r[\'required\']}" not in msg:', 'if False:', 'A6 message values'),
 ('acceptance.py', '        if not any(e["kind"] == "entry" and e["file"] == entry_file and e["qualname"] == r["entry"]["qualname"]\n                   for e in inside):', '        if False:', 'A7 production invocation'),
 ('acceptance.py', 'out["_session"].append("the two runs collected different test-file bytes")', 'pass', 'A8 test bytes pair'),
 ('acceptance.py', '    res["reasons"] += _drift(wt, m0["files"], m0["untracked"], v0, (), "after marked")', '    pass', 'A10 closure freeze after marked'),
 ('records.py', 'elif lat["min_minutes"] > feasible[0] + 0.01 or lat["max_minutes"] < feasible[1] - 0.01:', 'elif False:', 'V2 latency within bounds'),
 ('records.py', 'if hashlib.sha256(data).hexdigest() != entry["acceptance_sha256"]:', 'if False:', 'V3 acceptance hash binding'),
 ('owned.py', 'if os.path.realpath(gitdir) == os.path.realpath(common):', 'if False:', 'W1 primary checkout'),
 ('owned.py', '        if cur.is_symlink():', '        if False:', 'W2 symlinked component'),
 ('owned.py', '                _tree_hash(h, Path(target))', '                pass', 'W4 pth trees hashed'),
 ('observer.py', '            sys.monitoring.restart_events()                  # its PY_START may have been disabled', '            pass', 'O4 restart on a new boundary callee'),
 ('observer.py', '"outstanding_threads": sum(1 for t in alive if t.ident not in self.threads_at_start),', '"outstanding_threads": 0,', 'O6 outstanding thread count'),
 ('observer.py', '"preexisting_threads_alive": sum(1 for t in alive if t.ident in self.threads_at_start)})', '"preexisting_threads_alive": 0})', 'O9 pre-existing threads alive'),
 ('observer.py', '        self._record("obs_start", {})\n        for spec in self.boundaries:\n            spec["_start"] = _resolve(spec["binding"])\n            self._register(spec, spec["_start"])', '        for spec in self.boundaries:\n            spec["_start"] = _resolve(spec["binding"])\n            self._register(spec, spec["_start"])\n        self._record("obs_start", {})', 'O10 registration gaps inside the interval'),
 ('observer.py', 'matched = [sp for sp, rcv in pairs if rcv is None or first is rcv]', 'matched = [sp for sp, rcv in pairs]', 'O11 receiver identity'),
 ('observer.py', 'if current is not None and id(current) not in self.seen_targets:', 'if current is callable_ and id(current) not in self.seen_targets:', 'O12 every current binding discovered'),
 ('observer.py', '        return _safe_items(dict.items(value), "map", depth)', '        return {"map": {k: _safe(dict.__getitem__(value, k), depth + 1) for k in list(dict.keys(value))[:50] if type(k) is str}}', 'O13 no key lookups'),
 ('observer.py', '            out.update(length=len(value), incomplete=True)', '            pass', 'O14 truncation is incomplete'),
 ('observer.py', '"category": "unavailable" if incomplete else _classify(spec.get("classify", []), safe)}', '"category": _classify(spec.get("classify", []), safe)}', 'O15 incomplete identity unclassified'),
 ('observer.py', '            category = "unavailable" if incomplete else _classify(self.returns[key], safe)', '            category = _classify(self.returns[key], safe)', 'O16 incomplete return unclassified'),
 ('certify.py', '        if end.get("preexisting_threads_alive"):', '        if False:', 'C13 pre-existing threads make absence unavailable'),
 ('acceptance.py', '        return ("value_match", []) if got == actual else \\\n            ("unmatched", [f"connection: {conn[\'qualname\']} returned', '        return ("value_match", []) if True else \\\n            ("unmatched", [f"connection: {conn[\'qualname\']} returned', 'A9 return value match'),
 ('acceptance.py', '    if not str(conn.get("review", "")).strip():', '    if False:', 'A11 fixture review required'),
 ('acceptance.py', '    if t is not dict or safe.get("incomplete") or "unavailable" in safe:', '    if t is not dict or "unavailable" in safe:', 'A12 incomplete never equal'),
 ('acceptance.py', '            if len({len(v) for v in values}) != 1:', '            if False:', 'A13 zip cardinality'),
 ('defence.py', '            if fa != fb:', '            if False:', 'D12 same failure observed and unobserved'),
 ('defence.py', '    moved = sorted(k for k in set(untracked) | set(m["untracked"]) if untracked.get(k) != m["untracked"].get(k))', '    moved = sorted(set(untracked) ^ set(m["untracked"]))', 'D13 untracked content drift'),
 ('replay.py', '                   if m["untracked"].get(k) != m0["untracked"].get(k) and (src_too or not k.startswith("src/")))', '                   if (k in m["untracked"]) != (k in m0["untracked"]) and (src_too or not k.startswith("src/")))', 'P8 replay untracked content drift'),
 ('owned.py', '            elif os.path.isfile(p):\n                _hash_file(h, p)', '            elif os.path.isfile(p):\n                pass', 'W3 venv content hashed'),
 ('owned.py', '                elif os.path.isfile(target):\n                    _hash_file(h, target)', '                elif os.path.isfile(target):\n                    pass', 'W5 symlink target bytes'),
 ('owned.py', '            target = os.path.realpath(os.path.join(os.path.dirname(pth), line))', '            target = os.path.realpath(line)', 'W6 relative .pth semantics'),
 ('owned.py', '    untracked = {rel: _file_state(root / rel)', '    untracked = {rel: "present"', 'W7 untracked contents in the manifest'),
 ('owned.py', '        for name in sorted(filenames):\n            p = os.path.join(dirpath, name)\n            h.update(os.path.relpath(p, base).encode() + b"\\0")', '        for name in sorted(f for f in filenames if not f.endswith(".pyc")):\n            p = os.path.join(dirpath, name)\n            h.update(os.path.relpath(p, base).encode() + b"\\0")', 'W8 bytecode hashed'),
 ('owned.py', '    key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)', '    key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)', 'W9 digest memo keyed on ctime'),
 ('deploy_runs.py', '    obs.sort(key=lambda p: (p["at"], p.pop("_k")))', '    obs.sort(key=lambda p: p.pop("_k"))', 'DR1 observation time order'),
 ('deploy_runs.py', '        if a["at"] == b["at"] and a["run_id"] != b["run_id"] and not _same(a["sha"], b["sha"]):', '        if False:', 'DR2 equal-time disagreement refused'),
 ('deploy_runs.py', '"sha": r["rolled_back_sha"], "kind": "rolled_back",', '"sha": r["pre_sha"], "kind": "rolled_back",', 'DR3 rollback observation is the logged sha'),
 ('deploy_runs.py', '        anomalies.append("rolled_back_sha_mismatch")', '        pass', 'DR4 rollback sha mismatch flagged'),
 ('records.py', '    if hi is None or (lo is not None and w_hi < lo):           # certainly written before the claim\'s time\n        return False\n    return w_lo <= hi + WINDOW', '    if hi is None:\n        return False\n    return w_lo <= hi + WINDOW', 'V1 report before the claim'),
 ('records.py', '    if hi is None or (lo is not None and w_hi < lo):           # certainly written before the claim\'s time\n        return False\n    return w_lo <= hi + WINDOW', '    if hi is None or (lo is not None and w_hi < lo):           # certainly written before the claim\'s time\n        return False\n    return True', 'V4 48 h after the claim'),
 ('records.py', '    if hi is None or (lo is not None and w_hi < lo):           # certainly written before the claim\'s time\n        return False\n    return w_lo <= hi + WINDOW', '    if lo is not None and w_hi <= lo:\n        return False\n    return hi is None or w_lo <= hi + WINDOW', 'V5 open-ended claim anchors nothing'),
 ('records.py', '        if not citing:', '        if False:', 'V6 uncited report'),
 ('records.py', '            if not _qualified(e, span):', '            if not any(_qualified(e, sp) for _p, sp in citing):', 'V7 each citation on its own'),
 ('records.py', '        for n in o.get("links", []):', '        for n in o.get("links", [])[:0]:', 'V8 occurrence links exist'),
 ('records.py', '    if disp == "observed_incident" and not _witnessed(r, ev):', '    if False:', 'V9 qualified witness required'),
 ('records.py', '_WITNESS_ROLES = ("onset", "observed", "first_machine_detection")', '_WITNESS_ROLES = ("onset", "observed", "first_machine_detection", "operator_awareness")', 'V10 awareness is not a witness'),
 ('records.py', '        for x in fx["expected_failure"]:\n            errs += _ef_binding_errs(rid, x, evidence_root)', '        for x in fx["expected_failure"][:0]:\n            errs += _ef_binding_errs(rid, x, evidence_root)', 'V11 expected-failure binding'),
 ('records.py', '    if publish and bound_claims and evidence_root is None:', '    if False:', 'V12 evidence root required'),
 ('records.py', '        if s_lo is not None and e_hi is not None and e_hi < s_lo:          # Codex phase-1 r4 #6', '        if False:', 'V13 impossible chronology'),
]



def _assertion_shaped(message: str) -> bool:
    return (message.startswith("assert ") or message == "AssertionError" or message.startswith("AssertionError:")
            or message.startswith("Failed: DID NOT RAISE"))


def classify(junit_xml: str, returncode: int, expected_cases: int | None = None) -> tuple[str, list[str]]:
    """(verdict, killing node ids) from a run's JUnit XML — see the module docstring."""
    root = ET.fromstring(junit_xml)
    cases = list(root.iter("testcase"))
    node = lambda c: f"{c.get('classname')}::{c.get('name')}"                  # noqa: E731
    errors = [c for c in cases if c.find("error") is not None]
    failures = [(c, c.find("failure")) for c in cases if c.find("failure") is not None]
    kills = [node(c) for c, f in failures if _assertion_shaped(f.get("message") or "")]
    if errors or returncode not in (0, 1):
        return "ERRORED", [node(c) for c in errors]
    if expected_cases is not None and len(cases) != expected_cases:
        return "INCOMPLETE", []
    if returncode == 1 and kills:
        return "KILLED", kills
    if failures:
        return "FAILED-OTHER", [node(c) for c, _ in failures]
    return ("SURVIVED", []) if returncode == 0 else ("ERRORED", [])


def run(build: Path) -> tuple[int, str, int, str]:
    """(return code, JUnit XML, test count, last output line) of one whole-suite run."""
    with tempfile.TemporaryDirectory() as td:
        xml = Path(td) / "junit.xml"
        p = subprocess.run(CMD + [f"--junitxml={xml}"], cwd=build, capture_output=True, text=True, env=ENV)
        text = xml.read_text() if xml.exists() else "<testsuites/>"
    lines = p.stdout.splitlines()
    return p.returncode, text, len(list(ET.fromstring(text).iter("testcase"))), (lines[-1] if lines else p.stderr[-200:])


def _on_term(signum, frame):
    raise KeyboardInterrupt            # so the finally below restores the mutated file


def main(argv: list[str]) -> int:
    build = Path(argv[1])
    d = build / "scripts/audit/incident_register"
    only = set(argv[2].split(",")) if len(argv) > 2 else None
    signal.signal(signal.SIGTERM, _on_term)
    for bak in d.glob("*.sweepbak"):   # a previous sweep died mid-mutant: restore before anything else
        shutil.move(bak, bak.with_suffix(""))
        print(f"restored leftover {bak.name}", flush=True)
    for cache in [*d.rglob("__pycache__"), *(build / "tests/scripts/incident_register").rglob("__pycache__")]:
        shutil.rmtree(cache)           # no bytecode compiled before the sweep can be read during it
    rc, xml, n_cases, tail = run(build)
    verdict, _ = classify(xml, rc)
    print(f"BASELINE rc={rc} tests={n_cases} verdict={verdict} | {tail}", flush=True)
    if verdict != "SURVIVED":          # a clean baseline: every test passed
        print("baseline is not clean; stopping", flush=True)
        return 1
    for fname, old, new, label in M:
        if only and label.split()[0] not in only:
            continue
        f = d / fname
        src = f.read_bytes()
        text = src.decode()
        if text.count(old) != 1:
            print(f"{label}: ANCHOR COUNT {text.count(old)}", flush=True)
            continue
        bak = f.with_name(f.name + ".sweepbak")
        bak.write_bytes(src)
        try:
            f.write_text(text.replace(old, new, 1))
            rc, xml, _n, tail = run(build)
            verdict, nodes = classify(xml, rc, expected_cases=n_cases)
            print(f"{label}: {verdict} | rc={rc} | {tail[:80]} | {nodes}", flush=True)
        finally:
            f.write_bytes(src)
            assert f.read_bytes() == src
            bak.unlink()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
