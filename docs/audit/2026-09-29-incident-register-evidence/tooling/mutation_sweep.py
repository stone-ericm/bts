"""Mutation sweep for the W1.5 evidence tooling (Codex phase-1 r3: 'nonzero return is not a kill').

    python mutation_sweep.py <build-worktree> [LABEL,LABEL,...]

1. BASELINE: the tooling tests must pass cleanly (exit 0, every node passed: no failure, error or skip);
   its exact node inventory is kept.
2. For each mutant (one check disabled), rerun the WHOLE suite (no ``-x``) and classify from pytest's
   JUnit XML, never from free-text summary lines (Codex phase-1 r4 #10: ``RuntimeError("DID NOT RAISE")``
   passed a substring test). ``classify`` is the rule:
   * KILLED — exit code 1, the baseline's exact node inventory, no testcase error or skip, and at least one failure whose
     message has an ASSERTION shape: a rewritten ``assert`` (pytest's message starts with ``assert ``;
     ``assert`` is a keyword, so no exception class can print that prefix), an explicit builtin
     ``AssertionError`` (exact prefix — a look-alike class prints its qualified name), or pytest.raises'
     own ``Failed: DID NOT RAISE``;
   * ERRORED — any testcase error (collection/setup/teardown) or an exit code other than 0/1;
   * INCOMPLETE — a node inventory different from the baseline's (not only its count; Codex phase-1 r5 #8);
   * SKIPPED — any skipped case (skip or xfail): that node's verdict is unknown, so it is neither a clean
     baseline nor a clean kill (Codex phase-1 r5 #8: an all-skipped run was a clean baseline);
   * FAILED-OTHER — failures, none of an assertion shape;
   * SURVIVED — exit 0, no failure.
   EVERY killing node id is printed, so a kill for the wrong reason is visible.
Known equivalent guards (not listed):
* the ``completed and`` condition on the defence/replay verdict is redundant while every exception path
  records a reason (D11/P7 pin that reason); removing it cannot be observed today;
* the observer's end-of-interval "binding changed without a watched store" gap. Codex r6 reached it by
  replacing sys.modules itself; since r7 the root (sys's own 'modules' entry) is watched too, so that
  probe is seen at the root (test_r6_counterexamples.py::test_a_persistent_sys_modules_replacement_is_seen)
  and no Python path to the guard is known. It stays as defence in depth;
* the "keep every start" rule (r7) is gone with the dispatch re-check it served (plan ruling 10), so O4
  (restart events when a new boundary callee is registered) is listed again, and O31/O32 and the other
  absence-only mutants (C5-C7, C9-C11, C13, O6, O9, O35) are retired with the code they mutated;
* retired O61 (a Mock call's keyword dict read in place, not copied): CPython builds a fresh keyword dict
  compactly, and copying such a dict takes the clone path, which re-inserts nothing and so compares no keys
  (measured: SURVIVED at 962c20f and in Codex phase-1 r10's targeted run). The equivalence is qualified to
  this interpreter (3.12) and the fresh-call shape: copying a SPARSE dict (keys deleted after growth) does
  compare keys (Codex r10, measured). The in-place read stays as defence in depth;
* the observer's lock (``_lock``, Codex phase-1 r10 part 2 #1), in Codex phase-1 r11 #3's words: No deterministic
  full-integration killer for the lock is currently included. O70 and O71 test the call-time guards. The
  integration race checks refusal once the worker's store has completed; it does not establish overlap or an
  atomic namespace snapshot. (Since plan ruling 12 the reads it orders run only while the observed thread is
  alone; O73-O79 test that gate.)
* retired O64 and O65 (a str-subclass sys.modules key: its own unavailable record; its text copied by str's
  slot): since Codex phase-1 r10 #3 any key that is not an exact str makes the whole import record
  unavailable (O69), so neither branch exists;
* retired O22 (namespaces read without application code): with exact module/class types enforced (O29,
  O30) the raw descriptor read and getattr coincide for every accepted object.
Mutants whose file starts with ``../../../`` mutate the sweep's own classifier (its tests import it). The source file is restored after every mutant (verified by byte comparison). No bytecode
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
 ('certify.py', '    if not live_at_branch:\n        why.append', '    if False:\n        why.append', 'C8 branch in live invocation'),
 ('certify.py', 'r["kind"] in ("entry_exit", "entry") and r["thread"] == en["thread"]:', 'False:', 'C12 exit/reuse splits invocations'),
 ('observer.py', '                    elif matched:\n                        extra = loc.get("args", ())', '                    elif False:\n                        extra = loc.get("args", ())', 'O1 mock callee recorder'),
 ('observer.py', 'return None if matched else sys.monitoring.DISABLE', 'return sys.monitoring.DISABLE', 'O2 keep boundary callee enabled'),
 ('observer.py', '            if code.co_flags & CO_ASYNC:', '            if False:', 'O3 async flag'),
 ('observer.py', '    d = _plain_instance_dict(value)\n    if d is None:', '    return {"repr": repr(value)}\n    if d is None:', 'O5 no application repr'),
 ('observer.py', '            self._gap(spec["name"], "not a Python-observable callable", type=type_name(type(obj)))', '            pass', 'O7 gap for C boundaries'),
 ('observer.py', 'safe = _safe(retval) if how == "return" else _raised(type(retval))', 'safe = _safe(retval)', 'O8 exceptional exit as type name'),
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
 ('observer.py', '        self._record("obs_start", {"purity": self._purity()})\n        self.started = True\n        watching = self._install_watchers() if self.boundaries else False\n        with self._lock:\n            for spec in self.boundaries:\n                self._track(spec)\n                if not watching:\n                    self._gap(spec["name"], "store watching unavailable: a rebinding could go unseen")\n', '        self.started = True\n        watching = self._install_watchers() if self.boundaries else False\n        with self._lock:\n            for spec in self.boundaries:\n                self._track(spec)\n                if not watching:\n                    self._gap(spec["name"], "store watching unavailable: a rebinding could go unseen")\n        self._record("obs_start", {"purity": self._purity()})\n', 'O10 registration gaps inside the interval'),
 ('observer.py', '                    bucket = mine if rcv is None or first is rcv else others', '                    bucket = mine', 'O11 receiver identity'),
 ('observer.py', '                        self._restep(name, level, event != _DICT_DELETED, new)', '                        pass', 'O12 every store on a binding path is seen'),
 ('observer.py', '        return _safe_items(dict.items(value), "map", depth)', '        return {"map": {k: _safe(dict.__getitem__(value, k), depth + 1) for k in list(dict.keys(value))[:50] if type(k) is str}}', 'O13 no key lookups'),
 ('observer.py', '            out.update(length=len(value), incomplete=True)', '            pass', 'O14 truncation is incomplete'),
 ('observer.py', '"category": _classify(spec.get("classify", []), safe) if _complete(safe) else "unavailable"}', '"category": _classify(spec.get("classify", []), safe)}', 'O15 incomplete identity unclassified'),
 ('observer.py', '            category = _classify(self.returns[key], safe) if _complete(safe) else "unavailable"', '            category = _classify(self.returns[key], safe)', 'O16 incomplete return unclassified'),
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
 ('deploy_runs.py', '    obs.sort(key=lambda p: (_instant(p["at"])[0], p.pop("_k")))', '    obs.sort(key=lambda p: p.pop("_k"))', 'DR1 observation time order'),
 ('deploy_runs.py', '            if a["run_id"] == b["run_id"] or _same(a["sha"], b["sha"]) or not (fa < cb and fb < ca):\n                continue', '            if True:\n                continue', 'DR2 cross-run disagreement refused'),
 ('deploy_runs.py', '"sha": r["rolled_back_sha"], "kind": "rolled_back",', '"sha": r["pre_sha"], "kind": "rolled_back",', 'DR3 rollback observation is the logged sha'),
 ('deploy_runs.py', '        anomalies.append("rolled_back_sha_mismatch")', '        pass', 'DR4 rollback sha mismatch flagged'),
 ('records.py', "    if hi is None or (lo is not None and w_hi < lo):           # certainly written before the claim's time\n        return False\n    return w_lo <= hi + WINDOW", '    if hi is None:\n        return False\n    return w_lo <= hi + WINDOW', 'V1 report before the claim'),
 ('records.py', "    if hi is None or (lo is not None and w_hi < lo):           # certainly written before the claim's time\n        return False\n    return w_lo <= hi + WINDOW", "    if hi is None or (lo is not None and w_hi < lo):           # certainly written before the claim's time\n        return False\n    return True", 'V4 48 h after the claim'),
 ('records.py', "    if hi is None or (lo is not None and w_hi < lo):           # certainly written before the claim's time\n        return False\n    return w_lo <= hi + WINDOW", '    if lo is not None and w_hi <= lo:\n        return False\n    return hi is None or w_lo <= hi + WINDOW', 'V5 open-ended claim anchors nothing'),
 ('records.py', '        if not citing:', '        if False:', 'V6 uncited report'),
 ('records.py', '            if not _qualified(e, span):', '            if not any(_qualified(e, sp) for _p, sp in citing):', 'V7 each citation on its own'),
 ('records.py', '        for n in o.get("links", []):', '        for n in o.get("links", [])[:0]:', 'V8 occurrence links exist'),
 ('records.py', '    if disp == "observed_incident" and not _witnessed(r, ev):', '    if False:', 'V9 qualified witness required'),
 ('records.py', '_WITNESS_ROLES = ("onset", "observed", "first_machine_detection")', '_WITNESS_ROLES = ("onset", "observed", "first_machine_detection", "operator_awareness")', 'V10 awareness is not a witness'),
 ('records.py', '        for x in fx["expected_failure"]:\n            errs += _ef_binding_errs(rid, "expected-failure", x, evidence_root)', '        for x in fx["expected_failure"][:0]:\n            errs += _ef_binding_errs(rid, "expected-failure", x, evidence_root)', 'V11 expected-failure binding'),
 ('records.py', '    if publish and bound_claims and evidence_root is None:', '    if False:', 'V12 evidence root required'),
 ('records.py', '        if s_lo is not None and e_hi is not None and e_hi < s_lo:          # Codex phase-1 r4 #6', '        if False:', 'V13 impossible chronology'),
 ('records.py', '        for x in fx["characterization"]:\n            errs += _ef_binding_errs(rid, "characterization", x, evidence_root)', '        for x in fx["characterization"][:0]:\n            errs += _ef_binding_errs(rid, "characterization", x, evidence_root)', 'V14 characterization binding'),
 ('records.py', '                    + fx["characterization"])', '                    )', 'V15 characterization needs an evidence root'),
 ('records.py', '        if c not in acc.get("passed_nodes", []):', '        if False:', 'V16 controls passed in the pair'),
 ('records.py', '        if x is None or x["exception"] != entry["exception"]:', '        if x is None or x["exception"].rsplit(".", 1)[-1] != entry["exception"].rsplit(".", 1)[-1]:', 'V17 exact registered exception'),
 ('acceptance.py', '                           if res["verdict"] == "accepted" else [])', '                           if True else [])', 'A14 a rejected pair vouches for no control'),
 ('acceptance.py', '\n                                  if runner.node_state(marked.events, n) == "passed")', ')', 'A15 passed nodes are passes'),
 ('run_expected_failures.py', '"accepted_nodes": [], "passed_nodes": []}', '"accepted_nodes": []}', 'A16 failure artifact carries no controls'),
 ('observer.py', '            elif not is_current:', '            elif False:', 'O17 a former binding value is unattributed'),
 ('observer.py', '            elif len(specs) > 1:', '            elif False:', 'O18 a callable bound to several boundaries is unattributed'),
 ('observer.py', '                if type(code) is types.CodeType:\n                    for spec, receiver in entry[1]:\n                        self._add_code(code, spec, receiver)', '                if False:\n                    for spec, receiver in entry[1]:\n                        self._add_code(code, spec, receiver)', 'O19 a swapped __code__ is registered'),
 ('observer.py', '        if why is not None:\n            self._gap(spec["name"], why)', '        if False:\n            self._gap(spec["name"], why)', 'O20 an unresolvable binding path is a gap'),
 ('observer.py', '                if key == "__call__":', '                if False:', 'O21 a replaced mock __call__ is a gap'),
 ('observer.py', '        return _type_ns(obj)\n    return None\n', '        return _type_ns(obj)\n    return _plain_instance_dict(obj)\n', 'O23 instance namespaces are unsupported'),
 ('observer.py', '        return all(_complete(v) for v in safe["seq"])', '        return True', 'O24 nested sequences are checked'),
 ('observer.py', '            return all(_complete(v) for v in safe[key].values())', '            return True', 'O25 nested maps and fields are checked'),
 ('observer.py', '            elif per_key or event in (_DICT_CLONED, _DICT_CLEARED):', '            elif False:', 'O26 a namespace replaced wholesale, or a store through a key that is not an exact str, is a gap'),
 ('observer.py', '            if not watching:', '            if False:', 'O27 no watchers means a gap'),
 ('owned.py', '                if not (name.isidentifier() and hook.is_file()\n                        and _sha(hook.read_bytes()) in REVIEWED_PTH_IMPORTS.get(name, {})):', '                if False:', 'W10 executable .pth lines refused unless reviewed'),
 ('owned.py', '        elif os.path.isfile(target):\n            _hash_file(h, target)\n        else:\n            h.update(b"dangling")\n        return h.hexdigest()', '        return h.hexdigest()', 'W11 symlink target bytes in the manifest'),
 ('deploy_runs.py', '(?P<ts>\\d{4}-\\d\\d-\\d\\dT\\d\\d:\\d\\d:\\d\\d(?:\\.\\d+)?)Z ', '(?P<ts>\\d{4}-\\d\\d-\\d\\dT\\d\\d:\\d\\d:\\d\\d)(?:\\.\\d+)?Z ', 'DR5 fractions kept'),
 ('deploy_runs.py', '"live_by": _iso(_instant(p["at"])[1])', '"live_by": _iso(_instant(p["at"])[0])', 'DR6 live_by at the end of its unit'),
 ('deploy_runs.py', '            if a["run_id"] == b["run_id"] or _same(a["sha"], b["sha"]) or not (fa < cb and fb < ca):', '            if a["run_id"] == b["run_id"] or _same(a["sha"], b["sha"]) or not (fa == fb):', 'DR7 overlapping disagreement refused'),
 ('deploy_runs.py', '    obs.sort(key=lambda p: (_instant(p["at"])[0], p.pop("_k")))', '    obs.sort(key=lambda p: (p["at"], p.pop("_k")))', 'DR8 ordered by instant'),
 ('records.py', '        if s_lo is not None and e_hi is not None and e_hi < s_lo:\n            if then_name', '        if False:\n            if then_name', 'V18 reversed chronology refused'),
 ('records.py', '        pairs.append(("observed time", seen, "restoration", o.get("restored_verification")))', '        pass', 'V19 observed before restoration'),
 ('records.py', '             ("alert attempt", alert.get("attempted"), "alert confirmation", alert.get("confirmed")),\n', '', 'V20 alert attempted before confirmed'),
 ('../../../docs/audit/2026-09-29-incident-register-evidence/tooling/mutation_sweep.py', '    if skipped:\n        return "SKIPPED", skipped', '    if False:\n        return "SKIPPED", skipped', 'S1 skipped cases are not clean'),
 ('../../../docs/audit/2026-09-29-incident-register-evidence/tooling/mutation_sweep.py', '    if expected_nodes is not None and sorted(_node(c) for c in cases) != sorted(expected_nodes):\n        return "INCOMPLETE", []', '    if expected_nodes is not None and len(cases) != len(expected_nodes):\n        return "INCOMPLETE", []', 'S2 the exact baseline inventory'),
 ('../../../docs/audit/2026-09-29-incident-register-evidence/tooling/mutation_sweep.py', '    except SyntaxError as e:\n        return f"{type(e).__name__}: {e.msg} (line {e.lineno})"', '    except SyntaxError as e:\n        return None', 'S3 a mutant that does not compile is never run'),
 ('observer.py', '    chain = [(sys_ns, "modules")]', '    chain = [({}, "modules")]', 'O28 sys.modules itself is watched'),
 ('observer.py', '    if type(obj) is types.ModuleType:\n        d = _MODULE_DICT', '    if isinstance(obj, types.ModuleType):\n        d = _MODULE_DICT', 'O29 exact module dispatch'),
 ('observer.py', '    if type(obj) is type:\n        return _type_ns(obj)', '    if isinstance(obj, type):\n        return _type_ns(obj)', 'O30 exact class dispatch'),
 ('observer.py', '                    if shared > 1:', '                    if False:', 'O33 shared code is unattributed'),
 ('observer.py', '        if type(obj) is _METHOD:\n            obj = obj.__func__\n        return obj.__code__ if type(obj) is _FUNCTION else None', '        if type_name(type(obj)) == "builtins.method":\n            obj = obj.__func__\n        return obj.__code__ if type_name(type(obj)) == "builtins.function" else None', 'O34 callables by exact type'),
 ('observer.py', '        if self.tool_acquired:\n            steps = [', '        if self.active:\n            steps = [', 'O36 a failed start releases the monitoring id'),
 ('observer.py', '            except BaseException as e:  # noqa: BLE001 - stop() must never raise\n                self._err("end_check", e)', '            except BaseException as e:  # noqa: BLE001 - stop() must never raise\n                raise', 'O37 stop never raises'),
 ('observer.py', '        self.active = False\n        self._remove_watchers()\n        if not self.started:', '        was = self.active\n        self.active = False\n        if was:\n            self._remove_watchers()\n        if not self.started:', 'O38 a failed start releases its watchers'),
 ('owned.py', '                if caches:\n                    raise ClosureRefused', '                if False:\n                    raise ClosureRefused', 'W12 a cached reviewed hook is refused'),
 ('owned.py', '    _git(path, "clean", "-qffdx", "-e", ".venv")\n    purge_hook_caches(path)', '    _git(path, "clean", "-qffdx", "-e", ".venv")', 'W13 reset purges reviewed hook caches'),
 ('owned.py', '                if others:\n                    raise ClosureRefused', '                if False:\n                    raise ClosureRefused', 'W14 another importable form of a hook is refused'),
 ('deploy_runs.py', '        for j in range(i + 1, len(obs)):', '        for j in range(i + 1, min(i + 2, len(obs))):', 'DR9 every overlapping pair checked'),
 ('deploy_runs.py', '            if f <= tt < c:\n                return {"sha": None, "basis": "within_observation_precision"', '            if False:\n                return {"sha": None, "basis": "within_observation_precision"', 'DR10 live_at inside an observation unit'),
 ('observer.py', '                sys.monitoring.restart_events()          # its PY_START may have been disabled', '                pass', 'O4 restart on a new boundary callee'),
 ('certify.py', '    if kind == "absence":\n        raise AbsenceRefused(ABSENCE_REFUSAL)', '    if False:\n        raise AbsenceRefused(ABSENCE_REFUSAL)', 'C14 certify refuses absence'),
 ('defence.py', '    if spec["symptom"].get("kind") == "absence":\n        raise SpecError(certify.ABSENCE_REFUSAL)', '    if False:\n        raise SpecError(certify.ABSENCE_REFUSAL)', 'D14 an absence spec is refused before any run'),
 ('certify.py', '        return [f"observer purity not recorded at {where}"]', '        return []', 'C15 purity must be recorded'),
 ('certify.py', '    if hooks is None:', '    if False:', 'C16 no census, no certificate'),
 ('certify.py', '    elif hooks:', '    elif False:', 'C17 an added audit hook makes observation impure'),
 ('certify.py', '    if p.get("gc_enabled") is not False:', '    if p.get("gc_enabled") is True:', 'C18 collection on or unknown is impure'),
 ('certify.py', '    if p.get("signal_handlers"):', '    if False:', 'C19 a signal handler makes observation impure'),
 ('certify.py', '    why = _purity_errs(starts[0], "obs_start") + _purity_errs(ends[0], "obs_end")', '    why = _purity_errs(ends[0], "obs_end")', 'C20 purity at the start'),
 ('certify.py', '    why = _purity_errs(starts[0], "obs_start") + _purity_errs(ends[0], "obs_end")', '    why = _purity_errs(starts[0], "obs_start")', 'C21 purity at the end'),
 ('runner.py', '        _c["hooks_added"] += 1', '        pass', 'R16 the census counts added hooks'),
 ('runner.py', 'module._AUDIT_CENSUS = _census\n', '', 'R17 the observer gets the census'),
 ('runner.py', '"quiesce": list(quiesce or [])', '"quiesce": []', 'R18 quiesced nodes reach the observer'),
 ('defence.py', 'observe=None, quiesce=quiet, env_extra=env)', 'observe=None, env_extra=env)', 'D15 the observer-off twin is quiesced'),
 ('observer.py', '    gc.disable()\n    try:\n        if not observed:', '    try:\n        if not observed:', 'O39 collection off in the call phase'),
 ('observer.py', '        if gc_was:\n            gc.enable()', '        pass', 'O40 collection back on after the call phase'),
 ('observer.py', '    if not observed and item.nodeid not in set(_config().get("quiesce", [])):', '    if not observed:', 'O41 a quiesced node runs with collection off'),
 ('observer.py', '"audit_hooks_added": census["hooks_added"] if type(census) is dict else None,', '"audit_hooks_added": 0,', 'O42 the census is recorded'),
 ('observer.py', '"gc_enabled": gc.isenabled(),', '"gc_enabled": False,', 'O43 collection state is recorded'),
 ('observer.py', '                if not (h is None or h is signal.SIG_DFL or h is signal.SIG_IGN or h is signal.default_int_handler):', '                if False:', 'O44 signal handlers are recorded'),
 ('observer.py', '            if not self._mock_verified(obj):\n                self._gap(', '            if False:\n                self._gap(', 'O45 an unverified mock is a gap when held'),
 ('observer.py', 'unverified=None if self._mock_verified(me) else', 'unverified=None if True else', 'O46 an unverified mock call is unattributed'),
 ('observer.py', '        return t is self.mock_types.get(id(obj)) and self._effective_call(t) is self.mock_call_fn', '        return self._effective_call(t) is self.mock_call_fn', 'O47 a held mock keeps its class'),
 ('observer.py', '        return t is self.mock_types.get(id(obj)) and self._effective_call(t) is self.mock_call_fn', '        return t is self.mock_types.get(id(obj))', 'O48 a held mock keeps the standard __call__'),
 ('observer.py', '        try:\n            args = _EXC_ARGS.__get__(exc, BaseException)', '        import traceback\n        detail = "".join(traceback.format_exception(exc))[-800:]\n        try:\n            args = _EXC_ARGS.__get__(exc, BaseException)', 'O49 recording an error runs no I/O'),
 ('observer.py', '                        st["chain"], st["current"] = st["chain"][:level + 1], None', '                        pass', 'O50 a namespace clear leaves nothing current'),
 ('observer.py', '    if type(module) is not str or type(qualname) is not str:\n        return None\n    return module + "." + qualname', '    return f"{module}.{qualname}"', 'O51 class metadata joined only as exact strings'),
 ('observer.py', '    if name is None:\n        return {"unavailable": "type identity not readable as exact strings", "incomplete": True}', '    if False:\n        return {"unavailable": "type identity not readable as exact strings", "incomplete": True}', 'O52 an unreadable type identity is incomplete'),
 ('observer.py', '{"raised": "<unnamed>", "incomplete": True}', '{"raised": "<unnamed>"}', 'O53 an unnamed exception exit is incomplete'),
 ('certify.py', '    if bad.get("category") == RESERVED_CATEGORY:\n        why.append', '    if False:\n        why.append', 'C22 the reserved category cannot be requested'),
 ('certify.py', '        return "reason" not in ident and _complete(ident.get("value"))', '        return _complete(ident.get("value"))', 'C23 an unattributed call never witnesses'),
 ('certify.py', '        return "reason" not in ident and _complete(ident.get("value"))', '        return "reason" not in ident', 'C24 an incomplete identity never witnesses'),
 ('certify.py', '    return _complete(x.get("value"))\n', '    return True\n', 'C25 an incomplete return never witnesses'),
 ('certify.py', '    if safe.get("incomplete") or "unavailable" in safe:\n        return False', '    if False:\n        return False', 'C26 completeness reads the omission markers'),
 ('certify.py', '        return all(_complete(v) for v in safe["seq"])', '        return True', 'C27 completeness at every depth'),
 ('defence.py', '        raise SpecError(f"the category {reserved!r} is reserved', '        pass\n        (f"the category {reserved!r} is reserved', 'D16 a spec naming the reserved category is refused'),
 ('defence.py', '            rule[0] == reserved for item in', '            False for item in', 'D17 classify rules are checked for the reserved category'),
 ('observer.py', '        if _TYPE_FLAGS.__get__(t, type) & _HEAPTYPE:', '        if False:', 'O54 a heap class reads __module__ by iteration'),
 ('observer.py', '        if why is not None:\n            return None\n        if found:\n            if type(desc) is not _GETSET:', '        if found:\n            if type(desc) is not _GETSET:', 'O55 an unsupported class namespace ends the instance-dict search'),
 ('observer.py', '            found, value, _why = _lookup(kwargs, m.group(2))     # not found when the dict is unsupported', '            found, value, _why = m.group(2) in kwargs, kwargs.get(m.group(2)), None', 'O56 keyword identities read by iteration'),
 ('observer.py', '        if type(k) is not str:\n            return False, None, _NON_STR_KEY', '        if type(k) is not str:\n            continue', 'O57 a dict with a key that is not an exact str is unsupported'),
 ('observer.py', '            elif per_key or event in (_DICT_CLONED, _DICT_CLEARED):', '            elif event in (_DICT_CLONED, _DICT_CLEARED):', 'O58 a store through a non-str key invalidates'),
 ('observer.py', '        if type(filename) is not str:\n            return _NO_PATH', '        if False:\n            return _NO_PATH', 'O59 a code filename must be an exact str'),
 ('observer.py', '    return value if type(value) is str else "<unnamed>"', '    return value', 'O60 a code name must be an exact str'),
 ('observer.py', '    if type(mod) is not types.ModuleType:\n        return {"unavailable": "not an exact module"}', '    if False:\n        return {"unavailable": "not an exact module"}', 'O62 import records need an exact module'),
 ('observer.py', '    found, f, why = _lookup(ns, "__file__")\n    if why is not None:\n        return {"unavailable": why}', '    found, f, why = _lookup(ns, "__file__")', 'O63 import records need an exact-str namespace'),
 ('runner.py', '        if "file" not in m:\n            why.append', '        if False:\n            why.append', 'R19 unavailable module provenance is refused'),
 ('runner.py', '    if len(imports) == 1 and "unavailable" in imports[0]:\n        why.append', '    if False:\n        why.append', 'R20 an unavailable import record is refused'),
 ('observer.py', '    return id(code)', '    return code', 'O66 code-object maps are keyed by id'),
 ('observer.py', '        _found, _old, why = _lookup(chain[level][0], chain[level][1])\n        if why is not None:', '        _found, _old, why = _lookup(chain[level][0], chain[level][1])\n        if False:', 'O67 a re-resolution re-checks the changed namespace'),
 ('observer.py', '                mods[name] = _import_record(mod)', '                rec = _import_record(mod)\n                if "unavailable" not in rec:\n                    mods[name] = rec', 'O68 an unreadable import record is kept, not dropped'),
 ('observer.py', '                unavailable = "a sys.modules key that is not an exact str"\n                break', '                continue', 'O69 a sys.modules key that is not an exact str makes the record unavailable'),
 ('observer.py', '        if why is not None or value is not cur:\n            return None', '        if False:\n            return None', 'O70 attribution reads the binding at the call'),
 ('observer.py', '            if self.watched.get(id(ns)) is not ns:\n                return None', '            if False:\n                return None', 'O71 attribution needs every namespace on the path watched'),
 ('observer.py', '    if not found or type(f) is not str:\n        return {"unavailable": "no __file__ that is an exact str"}', '    if not found or type(f) is not str:\n        if found:\n            try:\n                os.fspath(f)\n            except TypeError:\n                pass\n        return {"unavailable": "no __file__ that is an exact str"}', 'O72 an unreadable __file__ is never resolved (Codex r10 part 2 FG_PATHLIKE)'),
 ('observer.py', '        if not me or _THREAD_HEAD(interp) != me:\n            return False\n        if _THREAD_NEXT(me):\n            return False\n        return _THREAD_HEAD(interp) == me', '        return True', 'O73 _alone reads the thread states'),
 ('observer.py', '        if not me or _THREAD_HEAD(interp) != me:\n            return False\n', '', "O80 alone needs this thread's state at the head"),
 ('observer.py', '        if _THREAD_NEXT(me):\n            return False\n', '', 'O81 alone needs no successor'),
 ('observer.py', '        return _THREAD_HEAD(interp) == me', '        return True', 'O82 alone re-reads the head last'),
 ('observer.py', '    except Exception:  # noqa: BLE001 - unreadable thread states: not alone, so nothing is read (a miss, never false)\n        return False', '    except Exception:  # noqa: BLE001 - unreadable thread states: not alone, so nothing is read (a miss, never false)\n        return True', 'O83 unreadable thread states are not alone'),
 ('observer.py', '                    if mine and not _alone():                    # ruling 12: no census, no read', '                    if False:                    # ruling 12: no census, no read', 'O74 a function boundary call reads nothing while another thread is alive'),
 ('observer.py', '                    if matched and not _alone():                 # ruling 12: no application object is read', '                    if False:                 # ruling 12: no application object is read', 'O75 a mock boundary call reads nothing while another thread is alive'),
 ('observer.py', '            if not _alone():                                # ruling 12: the value is not read', '            if False:                                # ruling 12: the value is not read', 'O76 a return is not read while another thread is alive'),
 ('observer.py', '                            if alone:\n                                self._restep(', '                            if True:\n                                self._restep(', 'O77 a store while another thread is alive cuts, never re-resolves'),
 ('observer.py', '        if not _alone():                                 # nothing is read: the binding is never held (missed)', '        if False:                                 # nothing is read: the binding is never held (missed)', 'O78 tracking at the start needs the thread alone'),
 ('observer.py', '                    if not _alone():\n                        self._gap(spec["name"], _CONCURRENT, where="end")', '                    if False:\n                        self._gap(spec["name"], _CONCURRENT, where="end")', 'O79 the end check needs the thread alone'),
]



def _assertion_shaped(message: str) -> bool:
    return (message.startswith("assert ") or message == "AssertionError" or message.startswith("AssertionError:")
            or message.startswith("Failed: DID NOT RAISE"))


def _node(case) -> str:
    return f"{case.get('classname')}::{case.get('name')}"


def nodes_of(junit_xml: str) -> list[str]:
    """The run's node inventory, in order."""
    return [_node(c) for c in ET.fromstring(junit_xml).iter("testcase")]


def classify(junit_xml: str, returncode: int, expected_nodes: list[str] | None = None) -> tuple[str, list[str]]:
    """(verdict, killing node ids) from a run's JUnit XML — see the module docstring."""
    root = ET.fromstring(junit_xml)
    cases = list(root.iter("testcase"))
    errors = [c for c in cases if c.find("error") is not None]
    failures = [(c, c.find("failure")) for c in cases if c.find("failure") is not None]
    kills = [_node(c) for c, f in failures if _assertion_shaped(f.get("message") or "")]
    if errors or returncode not in (0, 1):
        return "ERRORED", [_node(c) for c in errors]
    if expected_nodes is not None and sorted(_node(c) for c in cases) != sorted(expected_nodes):
        return "INCOMPLETE", []
    skipped = [_node(c) for c in cases if c.find("skipped") is not None]
    if skipped:
        return "SKIPPED", skipped
    if returncode == 1 and kills:
        return "KILLED", kills
    if failures:
        return "FAILED-OTHER", [_node(c) for c, _ in failures]
    return ("SURVIVED", []) if returncode == 0 else ("ERRORED", [])


def run(build: Path) -> tuple[int, str, int, str]:
    """(return code, JUnit XML, test count, last output line) of one whole-suite run (the node
    inventory is ``nodes_of`` the XML)."""
    with tempfile.TemporaryDirectory() as td:
        xml = Path(td) / "junit.xml"
        p = subprocess.run(CMD + [f"--junitxml={xml}"], cwd=build, capture_output=True, text=True, env=ENV)
        text = xml.read_text() if xml.exists() else "<testsuites/>"
    lines = p.stdout.splitlines()
    return p.returncode, text, len(list(ET.fromstring(text).iter("testcase"))), (lines[-1] if lines else p.stderr[-200:])


def mutant_error(text: str, old: str, new: str, filename: str) -> str | None:
    """Why the mutated text is not a runnable mutant, or None. A mutant that does not compile would make
    every importing module fail to collect: that is a defect of the mutant, not an ERRORED run."""
    if text.count(old) != 1:
        return f"ANCHOR COUNT {text.count(old)}"
    try:
        compile(text.replace(old, new, 1), filename, "exec")
    except SyntaxError as e:
        return f"{type(e).__name__}: {e.msg} (line {e.lineno})"
    return None


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
    tooling = build / "docs/audit/2026-09-29-incident-register-evidence/tooling"   # the classifier's tests import it
    for cache in [*d.rglob("__pycache__"), *(build / "tests/scripts/incident_register").rglob("__pycache__"),
                  *tooling.rglob("__pycache__")]:
        shutil.rmtree(cache)           # no bytecode compiled before the sweep can be read during it
    rc, xml, n_cases, tail = run(build)
    verdict, _ = classify(xml, rc)
    baseline_nodes = nodes_of(xml)
    print(f"BASELINE rc={rc} tests={n_cases} verdict={verdict} | {tail}", flush=True)
    if verdict != "SURVIVED":          # a clean baseline: every node passed (no failure, error or skip)
        print("baseline is not clean; stopping", flush=True)
        return 1
    for fname, old, new, label in M:
        if only and label.split()[0] not in only:
            continue
        f = d / fname
        src = f.read_bytes()
        text = src.decode()
        invalid = mutant_error(text, old, new, str(f))
        if invalid:
            print(f"{label}: INVALID MUTANT | {invalid}", flush=True)
            continue
        bak = f.with_name(f.name + ".sweepbak")
        bak.write_bytes(src)
        try:
            f.write_text(text.replace(old, new, 1))
            rc, xml, _n, tail = run(build)
            verdict, nodes = classify(xml, rc, expected_nodes=baseline_nodes)
            print(f"{label}: {verdict} | rc={rc} | {tail[:80]} | {nodes}", flush=True)
        finally:
            f.write_bytes(src)
            assert f.read_bytes() == src
            bak.unlink()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
