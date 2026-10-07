"""Mutant ledger runner (C2 framing screen; redesigned after review r6, Eric's row C2-framing-review-r7).

Each mutant's named tests are its intended set: before mutating, `pytest --collect-only` on the unmutated source lists
exactly the nodes they select. The mutated run is a DRIVER process that calls `pytest.main()` in-process and writes its
evidence only after `pytest.main` returns, so an abort anywhere in the session, including unconfiguration, leaves no
evidence (r5 R5-1, r6 R6-1). Outcomes never come from pytest's text output (r4 R4-2). The evidence:
- a recorder plugin (passed to `pytest.main`) keeps every test report (node id, phase, outcome and its expected-failure
  status `wasxfail`, r6 R6-2), failed collection, and pytest's interrupt and internal-error hooks;
- at `pytest_collection_finish`, after every conftest is loaded, the recorder registers a SENTINEL plugin whose hooks are
  `tryfirst` wrappers. Pluggy calls the last-registered tryfirst wrapper outermost, so the sentinel's
  `pytest_sessionfinish` wrapper sees any inner hook or wrapper abort (r6 R6-1), and its `pytest_runtest_call` wrapper
  records what each test call itself raised, independently of the reports built from it;
- whether a plugin registered after the sentinel, which could wrap it (r7 R7-1). There are three layers:
  - every registration after the sentinel, counted as an EVENT through pytest's public `pytest_plugin_registered` hook,
    and kept even if the plugin later unregisters itself;
  - a final census of the registry;
  - the sentinel's own check, each time it runs, that it is the outermost implementation of that hook. That catches an
    outer wrapper even when its registration bypassed pytest's API;
- the intended nodes carrying an xfail, skip or skipif marker.

The driver guarantee, precisely: evidence is written only after `pytest.main` returns. An abort that escapes
`pytest.main` (for example from unconfiguration) leaves no evidence. An abort pytest itself catches (`pytest.exit`,
KeyboardInterrupt, an abort in a session-finish hook) may still leave evidence, and each such case is refused explicitly
by the rules below. The sentinel's ordering claim holds because a later registration is detected by those three layers.

A mutant is RED only when all of these hold:
- the driver's evidence exists and its pytest exit status is the process return code;
- no interrupt, internal error or collection error; the sentinel's session finish completed; no plugin was registered
  after the sentinel, and the sentinel was outermost each time it ran;
- no intended node is marked xfail/skip/skipif, and no report is skipped, xfailed or xpassed (r6 R6-2);
- no setup or teardown failed; every intended node has its own setup, call and teardown reports; no unintended node ran;
- every intended call report's outcome agrees with what the sentinel saw the call do (a rewritten or fabricated report
  is refused);
- pytest's counters agree: collected = the intended nodes, failed = the failed calls, exit status = the return code;
- the exit status is 1 and at least one intended node FAILED.
A complete clean pass (exit status 0) is SURVIVED; anything else is INCONCLUSIVE.

Inherited selection and early-stop options cannot apply: `PYTEST_ADDOPTS` is removed, the ini `addopts` are cleared for
both the collection and the run, and a named test list holding any option is INVALID before anything is mutated. The
runner's root (default: this repository) is pytest's rootdir and working directory, so collection stays inside it.

Out of scope, stated: test code that deliberately tampers with the runner's own objects, its evidence file or pytest's
internals (for example unregistering the sentinel, or editing another report's attributes) from inside the run.

Every failure is printed; the file is restored by hash after each mutant; the runner exits 1 if any mutant is not RED.
Spec JSON: [{"id", "rule", "file", "old", "new", "tests": [...]}]; "old" must occur exactly once; "file" and "tests" are
relative to the root; "tests" are paths or node ids only."""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DRIVER = '''import json, sys
import pytest

class Sentinel:
    def __init__(self, pm):
        self.pm, self.calls, self.finish, self.not_outermost = pm, {}, None, []

    def _outermost(self, caller, where):
        impls = caller.get_hookimpls()
        if not impls or impls[-1].plugin is not self:     # pluggy calls the list's last implementation first
            self.not_outermost.append(where)

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_call(self, item):
        self._outermost(self.pm.hook.pytest_runtest_call, item.nodeid)
        try:
            result = yield
        except BaseException as e:
            self.calls[item.nodeid] = type(e).__name__
            raise
        self.calls[item.nodeid] = None
        return result

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session, exitstatus):
        self._outermost(self.pm.hook.pytest_sessionfinish, "sessionfinish")
        try:
            result = yield
        except BaseException as e:
            self.finish = {"raised": type(e).__name__}
            raise
        self.finish = {"exitstatus": int(session.exitstatus), "testsfailed": int(session.testsfailed),
                       "testscollected": int(session.testscollected)}
        return result

class Recorder:
    def __init__(self):
        self.records, self.sentinel, self.config, self.at_sentinel, self.marked = [], None, None, None, []
        self.late_events = 0

    def pytest_plugin_registered(self, plugin):
        if self.sentinel is not None and plugin is not self.sentinel:
            self.late_events += 1                        # an event: kept even if the plugin later unregisters

    def pytest_configure(self, config):
        self.config = config

    def pytest_collectreport(self, report):
        if report.failed:
            self.records.append({"collecterror": report.nodeid})

    def pytest_collection_finish(self, session):
        self.marked = [i.nodeid for i in session.items
                       if any(i.get_closest_marker(m) for m in ("xfail", "skip", "skipif"))]
        self.sentinel = Sentinel(session.config.pluginmanager)
        session.config.pluginmanager.register(self.sentinel, "framing-runner-sentinel")
        self.at_sentinel = {id(p) for p in session.config.pluginmanager.get_plugins()}

    def pytest_runtest_logreport(self, report):
        self.records.append({"node": report.nodeid, "when": report.when, "outcome": report.outcome,
                             "wasxfail": hasattr(report, "wasxfail")})

    def pytest_keyboard_interrupt(self, excinfo):
        self.records.append({"interrupted": excinfo.typename})

    def pytest_internalerror(self, excrepr, excinfo):
        self.records.append({"internalerror": excinfo.typename})

out, args = sys.argv[1], sys.argv[2:]
rec = Recorder()
rc = int(pytest.main(args, plugins=[rec]))
late = None
if rec.at_sentinel is not None:
    late = rec.late_events + len({id(p) for p in rec.config.pluginmanager.get_plugins()} - rec.at_sentinel)
evidence = {"rc": rc, "records": rec.records, "calls": rec.sentinel.calls if rec.sentinel else {},
            "finish": rec.sentinel.finish if rec.sentinel else None, "late_plugins": late,
            "not_outermost": rec.sentinel.not_outermost if rec.sentinel else None, "marked": rec.marked}
with open(out, "x") as f:
    json.dump(evidence, f)
sys.exit(rc)
'''


def _pytest(root: Path, *args) -> list[str]:
    return [sys.executable, "-B", "-m", "pytest", *_args(root, *args)]


def _args(root: Path, *args) -> list[str]:
    return [f"--rootdir={root}", "-o", "addopts=", "-p", "no:cacheprovider", *args]


def _env(extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    env.update(TZ="America/New_York", OMP_NUM_THREADS="1", PYTHONPYCACHEPREFIX=tempfile.mkdtemp(),
               PYTHONDONTWRITEBYTECODE="1", **(extra or {}))
    return env


def classify(returncode: int, intended, bundle: dict | None) -> tuple[str, list]:
    """The verdict from the driver's evidence (`bundle`); text output is never consulted."""
    tag = f"exit {returncode}"
    if bundle is None:
        return f"INCONCLUSIVE({tag}, no normal end)", []
    records = bundle.get("records") or []
    reports = [r for r in records if "node" in r]
    phases = {}
    for r in reports:
        phases.setdefault(r["node"], {})[r["when"]] = r["outcome"]
    failed = sorted(n for n, ph in phases.items() if ph.get("call") == "failed")
    passed = sorted(n for n, ph in phases.items() if ph.get("call") == "passed")
    calls, finish = bundle.get("calls") or {}, bundle.get("finish")
    if bundle.get("rc") != returncode:
        return f"INCONCLUSIVE({tag}, exit mismatch)", failed
    if any("interrupted" in r for r in records):
        return f"INCONCLUSIVE({tag}, interrupted)", failed
    if any("internalerror" in r for r in records):
        return f"INCONCLUSIVE({tag}, internal error)", failed
    if any("collecterror" in r for r in records):
        return f"INCONCLUSIVE({tag}, collection errors)", failed
    if not isinstance(finish, dict) or "raised" in finish:
        return f"INCONCLUSIVE({tag}, session finish incomplete)", failed
    if bundle.get("late_plugins") != 0:
        return f"INCONCLUSIVE({tag}, plugins registered late)", failed
    if bundle.get("not_outermost") != []:
        return f"INCONCLUSIVE({tag}, sentinel not outermost)", failed
    if set(bundle.get("marked") or []) & set(intended):
        return f"INCONCLUSIVE({tag}, marked xfail or skip)", failed
    if any(r["outcome"] == "skipped" or r.get("wasxfail") for r in reports):
        return f"INCONCLUSIVE({tag}, xfail or skip)", failed
    if any(r["when"] != "call" and r["outcome"] == "failed" for r in reports):
        return f"INCONCLUSIVE({tag}, errors)", failed
    missing = set(intended) - set(failed) - set(passed)
    if missing:
        return f"INCONCLUSIVE({tag}, {len(missing)} not run)", failed
    incomplete = [n for n in intended if phases[n].get("setup") != "passed" or phases[n].get("teardown") != "passed"]
    if incomplete:
        return f"INCONCLUSIVE({tag}, {len(incomplete)} incomplete)", failed
    unintended = set(phases) - set(intended)
    if unintended:
        return f"INCONCLUSIVE({tag}, {len(unintended)} unintended)", failed
    if any(n not in calls or (calls[n] is not None) != (n in failed) for n in intended):
        return f"INCONCLUSIVE({tag}, report mismatch)", failed
    if (finish.get("testscollected") != len(intended) or finish.get("testsfailed") != len(failed)
            or finish.get("exitstatus") != returncode):
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
    """Run the named tests under the driver. Returns (returncode, stdout, evidence or None)."""
    work = Path(tempfile.mkdtemp(prefix="framing-runner-"))
    (work / "framing_runner_driver.py").write_text(DRIVER)
    evidence = work / "evidence.json"
    r = subprocess.run([sys.executable, "-B", str(work / "framing_runner_driver.py"), str(evidence),
                        *_args(root, "-q", *tests)], cwd=root, env=_env(), capture_output=True, text=True)
    bundle = None
    if evidence.is_file():
        try:
            bundle = json.loads(evidence.read_text())
        except ValueError:
            bundle = None
    return r.returncode, r.stdout, bundle


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
            rc, out, bundle = run_mutant(root, m["tests"])
            verdict, failed = classify(rc, intended, bundle)
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
