"""Mutant ledger runner (C2 framing screen; revised after reviews r1 to r5).

Each mutant's named tests are its intended set: before mutating, `pytest --collect-only` on the unmutated source lists
exactly the nodes they select. Outcomes and the session's end come from pytest itself, never from its text output
(r4 R4-2, r5 R5-1): a small plugin loaded with `-p` records, to a private file the runner creates,
- every test report (node id, phase, outcome);
- a failed collection report;
- pytest's interrupt hook (`pytest.exit`, KeyboardInterrupt) and its internal-error hook;
- at `pytest_sessionfinish`, run last, pytest's own exit status and its collected and failed counters. Its absence means
  the session did not finish normally (an abort before it, or another session-finish hook stopping it).
Printed or captured output cannot add any of these.

A mutant is RED only when all of these hold:
- the session-finish record exists, and there is no interrupt, internal error or collection error;
- no setup or teardown failed, and nothing was skipped (r2 R2-4);
- every intended node has its own setup, call and teardown reports, the call PASSED or FAILED, and no node outside the
  intended set ran (r3 R3-4; r5 R5-1: a teardown cut short leaves the node incomplete);
- pytest's counters agree with the reports: collected = the intended nodes, failed = the failed calls, and its exit
  status = the process return code;
- the exit status is 1 and at least one intended node FAILED.
A complete clean pass (exit status 0) is SURVIVED; anything else is INCONCLUSIVE.

Inherited selection and early-stop options cannot apply: `PYTEST_ADDOPTS` is removed, the ini `addopts` are cleared for
both the collection and the run, and a named test list holding any option is INVALID before anything is mutated. The
runner's root (default: this repository) is pytest's rootdir and working directory, so collection stays inside it.

Out of scope, stated: test code that deliberately tampers with the runner's private records file or with pytest's
internals from inside the run. The runner certifies against incomplete, interrupted or misreported execution, not
against a test suite written to attack it.

Every failure is printed; the file is restored by hash after each mutant; the runner exits 1 if any mutant is not RED.
Spec JSON: [{"id", "rule", "file", "old", "new", "tests": [...]}]; "old" must occur exactly once; "file" and "tests" are
relative to the root; "tests" are paths or node ids only."""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PLUGIN = '''import json, os
import pytest
_f = None
def _write(rec):
    _f.write(json.dumps(rec) + "\\n")
    _f.flush()
def pytest_configure(config):
    global _f
    _f = open(os.environ["FRAMING_RUNNER_RECORDS"], "x")
def pytest_collectreport(report):
    if report.failed:
        _write({"collecterror": report.nodeid})
def pytest_runtest_logreport(report):
    _write({"node": report.nodeid, "when": report.when, "outcome": report.outcome})
def pytest_keyboard_interrupt(excinfo):
    _write({"interrupted": excinfo.typename})
def pytest_internalerror(excrepr, excinfo):
    _write({"internalerror": excinfo.typename})
@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    _write({"session": {"exitstatus": int(exitstatus), "testsfailed": int(session.testsfailed),
                        "testscollected": int(session.testscollected)}})
def pytest_unconfigure(config):
    if _f is not None and not _f.closed:
        _f.close()
'''


def _pytest(root: Path, *args) -> list[str]:
    return [sys.executable, "-B", "-m", "pytest", f"--rootdir={root}", "-o", "addopts=", "-p", "no:cacheprovider", *args]


def _env(extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    env.update(TZ="America/New_York", OMP_NUM_THREADS="1", PYTHONPYCACHEPREFIX=tempfile.mkdtemp(),
               PYTHONDONTWRITEBYTECODE="1", **(extra or {}))
    return env


def classify(returncode: int, intended, records: list[dict]) -> tuple[str, list]:
    """The verdict from pytest's own reports and session record (`records`); text output is never consulted."""
    reports = [r for r in records if "node" in r]
    sessions = [r["session"] for r in records if "session" in r]
    phases = {}
    for r in reports:
        phases.setdefault(r["node"], {})[r["when"]] = r["outcome"]
    calls = {n: ph["call"] for n, ph in phases.items() if "call" in ph}
    failed = sorted(n for n, o in calls.items() if o == "failed")
    passed = sorted(n for n, o in calls.items() if o == "passed")
    tag = f"exit {returncode}"
    if any("interrupted" in r for r in records):
        return f"INCONCLUSIVE({tag}, interrupted)", failed
    if any("internalerror" in r for r in records):
        return f"INCONCLUSIVE({tag}, internal error)", failed
    if any("collecterror" in r for r in records):
        return f"INCONCLUSIVE({tag}, collection errors)", failed
    if len(sessions) != 1:
        return f"INCONCLUSIVE({tag}, no normal end)", failed
    if any(r["when"] != "call" and r["outcome"] == "failed" for r in reports):
        return f"INCONCLUSIVE({tag}, errors)", failed
    if any(r["outcome"] == "skipped" for r in reports):
        return f"INCONCLUSIVE({tag}, skipped)", failed
    missing = set(intended) - set(failed) - set(passed)
    if missing:
        return f"INCONCLUSIVE({tag}, {len(missing)} not run)", failed
    incomplete = [n for n in intended if phases[n].get("setup") != "passed" or phases[n].get("teardown") != "passed"]
    if incomplete:
        return f"INCONCLUSIVE({tag}, {len(incomplete)} incomplete)", failed
    unintended = set(phases) - set(intended)
    if unintended:
        return f"INCONCLUSIVE({tag}, {len(unintended)} unintended)", failed
    sess = sessions[0]
    if (sess["testscollected"] != len(intended) or sess["testsfailed"] != len(failed)
            or sess["exitstatus"] != returncode):
        return f"INCONCLUSIVE({tag}, session mismatch)", failed
    if returncode == 1 and failed:
        return "RED", failed
    if returncode == 0 and not failed:
        return "SURVIVED", failed
    return f"INCONCLUSIVE({tag})", failed


def collect(root: Path, tests: list[str]) -> list[str] | None:
    """The nodes the named tests select, on the unmutated source; None when collection fails or selects nothing."""
    r = subprocess.run(_pytest(root, "--collect-only", "-q", *tests), cwd=root, env=_env(), capture_output=True,
                       text=True)
    nodes = [l.strip() for l in r.stdout.splitlines() if "::" in l and not l.startswith(" ")]
    return nodes if r.returncode == 0 and nodes else None


def run_mutant(root: Path, tests: list[str]):
    """Run the named tests with the recording plugin. Returns (returncode, stdout, records)."""
    work = Path(tempfile.mkdtemp(prefix="framing-runner-"))
    (work / "framing_runner_plugin.py").write_text(PLUGIN)
    records_path = work / "records.jsonl"
    path = [str(work)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    env = _env({"FRAMING_RUNNER_RECORDS": str(records_path), "PYTHONPATH": os.pathsep.join(path)})
    r = subprocess.run(_pytest(root, "-q", "-p", "framing_runner_plugin", *tests), cwd=root, env=env,
                       capture_output=True, text=True)
    records = []
    if records_path.is_file():
        for line in records_path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                records.append({"unparsable": line})
    return r.returncode, r.stdout, records


def main(spec: str, only: str | None = None, *, root: Path = REPO) -> int:
    root = Path(root)
    wanted = set(only.split(",")) if only else None
    bad = []
    for m in json.loads(Path(spec).read_text()):
        if wanted and m["id"] not in wanted:
            continue
        options = [t for t in m["tests"] if t.startswith("-")]
        if options:
            print(m["id"], "INVALID: options in the named test list", options, flush=True); bad.append(m["id"]); continue
        f = root / m["file"]; orig = f.read_bytes(); h = hashlib.sha256(orig).hexdigest(); s = orig.decode()
        if s.count(m["old"]) != 1:
            print(m["id"], "ANCHOR x", s.count(m["old"]), flush=True); bad.append(m["id"]); continue
        intended = collect(root, m["tests"])
        if intended is None:
            print(m["id"], "INVALID: the named tests do not collect", flush=True); bad.append(m["id"]); continue
        f.write_text(s.replace(m["old"], m["new"]))
        try:
            rc, out, records = run_mutant(root, m["tests"])
            verdict, failed = classify(rc, intended, records)
            print(m["id"], verdict, (out.strip().splitlines() or ["?"])[-1], f"[{len(intended)} intended]", flush=True)
            for node in failed:
                print("    FAILED " + node[:300], flush=True)
            if verdict != "RED":
                bad.append(m["id"])
        finally:
            f.write_bytes(orig); assert hashlib.sha256(f.read_bytes()).hexdigest() == h
    print("NOT RED:", ",".join(bad) if bad else "none", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
