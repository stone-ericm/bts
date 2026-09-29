"""Mutation sweep for the v3 evidence tooling: each mutant disables one check; the new tests must fail."""
import shutil, subprocess, sys
from pathlib import Path
B = Path(sys.argv[1]); D = B / "scripts/audit/incident_register"
TESTS = ["tests/scripts/incident_register/test_runner.py", "tests/scripts/incident_register/test_defence.py",
         "tests/scripts/incident_register/test_replay.py", "tests/scripts/incident_register/test_expected_failure.py", "tests/scripts/incident_register/test_certify.py"]
M = [
 ("runner.py", 'if s["observer_file"] != run.trusted["file"] or s["observer_sha256"] != run.trusted["sha256"]:', 'if False:', "R1 observer identity"),
 ("runner.py", 'if s["prefix"] != os.path.join(wt, ".venv"):', 'if False:', "R2 own-venv prefix"),
 ("runner.py", 'if s["rootdir"] != wt:', 'if False:', "R3 rootdir"),
 ("runner.py", 'why.append("the session did not finish")', 'pass', "R4 session finish"),
 ("runner.py", 'why.append("collection errors")', 'pass', "R5 collection errors"),
 ("runner.py", 'why.append("observer errors")', 'pass', "R6 observer errors"),
 ("runner.py", 'if expected is not None and sorted(nodes) != sorted(expected):', 'if False:', "R7 inventory"),
 ("runner.py", 'if not m["file"].startswith(src):', 'if False:', "R8 foreign imports"),
 ("runner.py", '            if state != "passed":', '            if False:', "R9 per-node state"),
 ("runner.py", 'if run.returncode != want:', 'if False:', "R10 return code"),
 ("runner.py", 'if teardown is None or teardown["outcome"] != "passed":', 'if teardown is None:', "R11 teardown state"),
 ("runner.py", 'if pf.startswith(wt + os.sep) and not pf.startswith(os.path.join(wt, ".venv") + os.sep):', 'if False:', "R12 pytest shadow (defence-in-depth)"),
 ("certify.py", 'if any(e.get("async") for e in inside):', 'if False:', "C1 async"),
 ("certify.py", 'if e["kind"] == "entry" and e["file"] == entry["file"]', 'if e["kind"] == "entry"', "C2 entry file"),
 ("certify.py", '(en["frame"], en["qualname"], en["file"]) in on_stack', 'True', "C3 invocation link"),
 ("certify.py", 'if en is not None and any(e is en and b["seq"] < x["seq"] for b, e in linked_branches):\n                linked = {"entry": en["seq"], "boundary"', 'if en is not None:\n                linked = {"entry": en["seq"], "boundary"', "C4 event after branch, same invocation"),
 ("certify.py", '        if qualifying:', '        if False:', "C5 absence qualifying"),
 ("certify.py", 'why.append("positive control: the baseline did not produce the qualifying event")', 'pass', "C6 positive control"),
 ("certify.py", '        if unidentified:', '        if False:', "C7 unidentified calls"),
 ("certify.py", 'if not linked_branches:\n        why.append', 'if False:\n        why.append', "C8 branch linked"),
 ("defence.py", 'why += _assertion_ok(mutant, n, k["_path"], k["_line"])', 'pass', "D1 assertion location"),
 ("defence.py", '        why.append(f"{where}: frozen files changed: {changed[:5]}")', '        pass', "D2 frozen drift"),
 ("defence.py", 'why.append(f"declared killing node {n} was not killed")', 'pass', "D3 not killed"),
 ("defence.py", 'why += [f"certificate {n}: {r}" for r in cert["reasons"]]', 'pass', "D4 certificate reasons"),
 ("defence.py", 'why += [f"mutant: {r}" for r in mg]', 'pass', "D5 mutant gate"),
 ("defence.py", 'why += [f"green: {r}" for r in g]', 'pass', "D6 green gate"),
 ("defence.py", 'why += [f"restored: {r}" for r in rg]', 'pass', "D7 restored gate (may survive)"),
 ("defence.py", 'if spec["branch"]["path"] not in {e[0] for e in spec["mutation_edits"]}:', 'if False:', "D8 branch in mutated file"),
 ("replay.py", '            why += problems', '            pass', "P1 audit problems"),
 ("replay.py", 'if not last or last[0] != s["_path"] or last[1] != s["_line"]:', 'if False:', "P2 replay assertion location"),
 ("replay.py", 'why += [f"red: {r}" for r in runner.gate(red, worktree=worktree, mode="replay_red", expected=inventory)]', 'pass', "P3 red gate"),
 ("replay.py", 'if (call.get("exc_module"), call.get("exc_qualname")) != ("builtins", "AssertionError"):', 'if False:', "P4 replay exception type"),
 ("replay.py", '            elif not str(entry.get("reason", "")).strip():', '            elif False:', "P5 reason required"),
 ("acceptance.py", 'if c.get("imperative_xfail"):', 'if False:', "A1 imperative"),
 ("acceptance.py", 'if marker.get("raises") != [r["exception"]]:', 'if False:', "A2 marker raises"),
 ("acceptance.py", 'if not (strict is True or (strict is None and _ini_strict(marked))):', 'if False:', "A3 strict"),
 ("acceptance.py", 'if f"{u.get(\'exc_module\')}.{u.get(\'exc_qualname\')}" != r["exception"]:', 'if False:', "A4 exception identity"),
 ("acceptance.py", 'if not last or last[0] != oracle_file or last[2] != r["oracle"]["qualname"]:', 'if False:', "A5 oracle frame"),
 ("acceptance.py", 'if f"got the declared bad value {r[\'bad\']}" not in msg or f"required {r[\'required\']}" not in msg:', 'if False:', "A6 message values"),
 ("acceptance.py", '        if not any(e["kind"] == "entry" and e["file"] == entry_file', '        if False and not any(e["kind"] == "entry" and e["file"] == entry_file', "A7 production invocation"),
 ("acceptance.py", 'out["_session"].append("the two runs collected different test-file bytes")', 'pass', "A8 test bytes pair"),
 ("observer.py", '                    if loc.get("self") is pend["callable"]:', '                    if False:', "O1 mock identity"),
 ("observer.py", 'if self.prod_root is None or not filename.startswith(self.prod_root):', 'if self.prod_root is None:', "O2 prod-root filter"),
 ("observer.py", '            if code.co_flags & CO_ASYNC:', '            if False:', "O3 async flag"),
 ("observer.py", 'for target in (current, b.get("_orig")):', 'for target in (b.get("_orig"),):', "O4 dynamic binding"),
]
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
for fname, old, new, label in M:
    if ONLY and label.split()[0] not in ONLY:
        continue
    f = D / fname; src = f.read_text()
    if src.count(old) != 1:
        print(f"{label}: ANCHOR COUNT {src.count(old)}", flush=True); continue
    shutil.copy(f, f.with_suffix(".bak"))
    try:
        f.write_text(src.replace(old, new, 1))
        p = subprocess.run(["uv", "run", "pytest", *TESTS, "-q", "-x", "-p", "no:cacheprovider"], cwd=B,
                           capture_output=True, text=True, env={**__import__("os").environ, "UV_CACHE_DIR": "/tmp/uv-cache", "TZ": "America/New_York"})
        last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-200:]
        print(f"{label}: {'KILLED' if p.returncode else 'SURVIVED'} | {last[:110]}", flush=True)
    finally:
        shutil.move(f.with_suffix(".bak"), f)
print("DONE", flush=True)
