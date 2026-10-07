"""Mutant ledger runner (C2 framing screen; revision 9 states and enforces the runner's threat model, Eric's row
C2-framing-review-r9).

THREAT MODEL. The runner certifies what pytest did with the named tests under a mutant. Test code that changes pytest's
plugin system (registering, unregistering or wrapping hook implementations, loading plugins), wraps its hooks, or aborts
the session is OUTSIDE the runner's scope, and the runner refuses to run on any suite containing such code. Out of scope
as before: deliberate tampering with the runner's own objects, its evidence file or pytest's internals (private
attributes), and other processes acting on the run from outside (signals are refused below; debuggers are out of scope).

ENFORCEMENT, before anything runs (the gate, `scope_problems`). The suite is every file the run imports as test code:
each named test's module and the `__init__.py` of every package above it (conftests and entry-point plugins never load;
see below). Each file is parsed from its bytes, as Python will (a coding cookie cannot hide code), and must stay within
a reviewed vocabulary, the real suite's own:
- imports only from ALLOWED_MODULES (no relative or star imports; a dotted import needs an alias); `pytest` only as
  `import pytest` and only as `pytest.<one of PYTEST_ATTRIBUTES>`;
- builtins only from ALLOWED_BUILTINS, never rebound; no dunder name except `__file__`, and no dunder binding except the
  methods `__init__` and `__call__`; no name starting with `pytest` other than `pytest` itself (hooks, `pytest_plugins`,
  `pytestmark`);
- attributes only from ALLOWED_ATTRIBUTES (none is a dunder); keywords only from ALLOWED_KEYWORDS, and `**` only to pass
  on the enclosing function's own `**` parameter unchanged; parameters (and so fixtures) only from ALLOWED_PARAMETERS;
- `setattr` calls only as (object, literal allowed name, value) (monkeypatch's string-target form imports by name); no
  `match` statements.
None of the allowed names reaches pytest's plugin manager, configuration, session, nodes or hooks, a frame, dynamic
import, deserialization or an in-process abort (`test_the_reviewed_vocabulary_excludes_known_escape_routes`; the
evidence's `gadget_audit.py` follows every allowed attribute chain from the allowed modules). The suite can start child
processes (subprocess, and the runner and admission code it tests). A child reaches the run only through the operating
system: a signal interrupts or ends it (refused: interrupted, no evidence, or exit mismatch), a file it writes is caught
by the change scan below, and attaching a debugger is out of scope.

ENFORCEMENT of what loads. Every collection and run has an empty configuration file (`-c /dev/null`: no ini setting
applies), `--noconftest`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (no entry-point plugin), no inherited PYTHON* or PYTEST*
variable (no PYTHONPATH, PYTEST_PLUGINS, PYTEST_ADDOPTS), and an interpreter that adds neither its script directory nor
the user site to the import path (`-P -s`) and neither writes nor reads bytecode (`-B`, a cache prefix under
/dev/null where no file can exist). So the only code in a run is pytest and its builtin plugins, the runner, the gated
suite, and the installed and repository code it imports.

ENFORCEMENT that the imported code is the reviewed code (the change scan). Code written to disk during the ledger and
imported later would bypass the gate. From the ledger's start, no file or directory under the root or any directory on
the interpreter's import path may change (ctime, which no process can set back); the mutation targets, which the runner
itself writes, must keep the ctime of the runner's last write while pytest runs. Any change refuses the mutant, and
every mutant after it. (Run the ledger with `python -B`, so the runner itself writes no bytecode there.)

THE RUN. Each mutant's named tests are its intended set: before mutating, `pytest --collect-only` on the unmutated
source lists exactly the nodes they select. The mutated run is a DRIVER process that calls `pytest.main()` in-process
and writes its evidence only after `pytest.main` returns. An abort that escapes `pytest.main` (for example from
unconfiguration) leaves no evidence; an abort pytest itself catches (`pytest.exit`, KeyboardInterrupt, an abort in a
session-finish hook) may leave evidence, and each such case is refused explicitly below. Outcomes never come from
pytest's text output (r4 R4-2). The evidence, defence in depth behind the gate:
- a recorder plugin (passed to `pytest.main`) keeps every test report (node id, phase, outcome and its expected-failure
  status `wasxfail`, r6 R6-2), failed collection, and pytest's interrupt and internal-error hooks;
- at `pytest_collection_finish`, the origin of every registered plugin: each must be pytest's own (`_pytest.*`) or the
  runner's (FOREIGN otherwise, so a conftest, `pytest_plugins` module or entry point that loaded anyway is refused);
- the recorder then registers a SENTINEL plugin whose hooks are `tryfirst` wrappers, outermost when it registers: its
  `pytest_sessionfinish` wrapper sees an inner hook or wrapper abort (r6 R6-1), and its `pytest_runtest_call` wrapper
  records what each test call raised, independently of the reports built from it;
- from then on, every execution of pluggy's code that adds or removes a hook implementation
  (`HookCaller._add_hookimpl`, `HookCaller._remove_plugin`, watched through `sys.monitoring`), whatever API reached it:
  every registration and unregistration runs one of them, so a late wrapper is seen even if it unregisters itself before
  yielding (r8 R8-1);
- the older layers, kept: registrations after the sentinel counted as events through pytest's public
  `pytest_plugin_registered` hook, a final registry census, and the sentinel's own check that it is the last (outermost)
  implementation of its hook each time it runs. That check reads the live registry, so a wrapper that unregisters itself
  before yielding escapes it (r8 R8-1); the watch above does not;
- the intended nodes carrying an xfail, skip or skipif marker.

A mutant is RED only when all of these hold:
- the gate found nothing, and nothing changed under the scanned trees;
- the driver's evidence exists and its pytest exit status is the process return code;
- no interrupt, internal error or collection error; the sentinel's session finish completed; no plugin was registered
  after the sentinel, the sentinel was outermost each time it ran, and no hook implementation was added or removed;
- no intended node is marked xfail/skip/skipif, and no report is skipped, xfailed or xpassed (r6 R6-2);
- no setup or teardown failed; every intended node has its own setup, call and teardown reports; no unintended node ran;
- every intended call report's outcome agrees with what the sentinel saw the call do (a rewritten or fabricated report
  is refused);
- pytest's counters agree: collected = the intended nodes, failed = the failed calls, exit status = the return code;
- no foreign plugin was registered;
- the exit status is 1 and at least one intended node FAILED.
A complete clean pass (exit status 0) is SURVIVED; anything else is INCONCLUSIVE, and a gate refusal is REFUSED.

Inherited selection and early-stop options cannot apply: the environment is scrubbed, the configuration file is empty,
and a named test list holding any option is INVALID before anything is mutated. The runner's
root (default: this repository) is pytest's rootdir and working directory, so collection stays inside it.

Every failure is printed; the file is restored by hash after each mutant; the runner exits 1 if any mutant is not RED.
Spec JSON: [{"id", "rule", "file", "old", "new", "tests": [...]}]; "old" must occur exactly once; "file" and "tests" are
relative to the root; "tests" are files or node ids only."""
import ast, builtins, hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# The reviewed vocabulary: the real suite's own (generated from tests/scripts/c2_framing/test_screen.py, plus the
# in-scope scratch suites' pathlib reads), reviewed against the known escape routes.
ALLOWED_MODULES = frozenset({
    'ast', 'bts.features.compute', 'bts.features.park_drag', 'bts.model.predict', 'bts.validate.scorecard', 'hashlib',
    'json', 'math', 'numpy', 'pandas', 'pathlib', 'pytest', 'scripts.audit.c1', 'scripts.audit.c1.admission',
    'scripts.audit.c2_framing', 'scripts.audit.c2_framing.mutant_runner', 'scripts.audit.c2_framing.screen', 'shutil',
    'subprocess',
})
ALLOWED_BUILTINS = frozenset({
    'Exception', 'SystemExit', 'ValueError', 'all', 'any', 'bool', 'bytes', 'dict', 'enumerate', 'float', 'int',
    'isinstance', 'iter', 'len', 'list', 'next', 'range', 'round', 'set', 'sorted', 'str', 'tuple', 'zip',
})
ALLOWED_ATTRIBUTES = frozenset({
    'ADMISSION_REL', 'ALLOWED_ATTRIBUTES', 'ALLOWED_BUILTINS', 'ALLOWED_KEYWORDS', 'ALLOWED_MODULES',
    'ALLOWED_PARAMETERS', 'BASIS', 'BLEND_CONFIGS', 'DataFrame', 'DateOffset', 'FEATURE_COLS', 'FunctionDef',
    'IDENTITY_KEYS', 'INPUT_NAMES', 'LGB_PARAMS', 'LOOKUP_NAME', 'NA', 'NEW_COL', 'OLD_COL', 'PYTEST_ATTRIBUTES',
    'Path', 'ProvenanceError', 'REGISTER_REL', 'REPO', 'RETRAIN_EVERY', 'ROOKIE_GATE_K', 'RunInvalid', 'SCORING',
    'SEASONS_IN', 'SETTINGS', 'STAGE_ONE_SEEDS', 'TEST_SEASONS', '_BUILTIN_NAMES', '_args',
    '_build_probable_pitcher_lookup', '_env', '_pytest_cmd', '_python', '_t', 'admission_gate', 'aggregate', 'all',
    'any', 'append', 'arg', 'args', 'array', 'as_posix', 'assert_allclose', 'assert_array_equal', 'assign', 'astype',
    'attach_park_drag', 'base_cols', 'blend_configs', 'body', 'changed_since', 'chdir', 'check_settings', 'chmod',
    'classify', 'collect', 'compute_all_features', 'compute_full_scorecard', 'concat', 'copy', 'copytree',
    'cpu_seconds', 'default_rng', 'defaults', 'delenv', 'diff_scorecards', 'disposition', 'drop', 'dropna', 'dumps',
    'encode', 'exists', 'first_unit_stop', 'fixture', 'framing_by', 'freeze_lookup', 'frozen_lookup', 'get',
    'get_table', 'groupby', 'head', 'head_admitted', 'hexdigest', 'iloc', 'index', 'inf', 'install_closed_inputs',
    'interpreter_trees', 'is_dir', 'is_file', 'isclose', 'isin', 'isna', 'isspace', 'items', 'iterdir', 'iterrows',
    'join', 'kw_defaults', 'kwonlyargs', 'launch', 'literal_eval', 'load_inputs', 'loads', 'loc', 'main', 'mark',
    'mean', 'min', 'mkdir', 'name', 'nan', 'notna', 'original_portion_labels', 'out', 'parametrize', 'parent',
    'parents', 'parse', 'pins', 'pins_digest', 'pop', 'raises', 'random', 'read_bytes', 'read_parquet', 'read_text',
    'readouterr', 'relabel', 'relative_to', 'release', 'replace', 'resolve', 'resumed_counts', 'run', 'run_mutant',
    'scan_roots', 'scope_problems', 'seed_summary', 'self_check', 'setattr', 'setenv', 'setitem', 'sha256',
    'sort_index', 'split', 'sqrt', 'st_ctime_ns', 'startswith', 'stat', 'stdout', 'strip', 'suite_files', 'testing',
    'to_datetime', 'to_dict', 'to_numpy', 'to_parquet', 'unique', 'unlink', 'update', 'validate_run', 'with_name',
    'write_bytes', 'write_text',
})
ALLOWED_KEYWORDS = frozenset({
    '_test_out_root', 'actual_hit', 'atol', 'basis', 'budget', 'calls', 'capture_output', 'check', 'code', 'columns',
    'date', 'deterministic', 'dtype', 'equal_nan', 'execute', 'exempt', 'exist_ok', 'extra', 'feature_settings',
    'finish', 'foreign_plugins', 'head', 'id', 'identity', 'ids', 'ignore_index', 'indent', 'index', 'late_plugins',
    'marked', 'match', 'mc_trials', 'name', 'new', 'not_outermost', 'out_root', 'p_57_exact', 'p_at_1_delta',
    'p_game_hit', 'parents', 'passed', 'pid', 'pins', 'plugin', 'plugin_changes', 'raising', 'rank', 'reverse',
    'reviewed_commit', 'root', 'rtol', 'run', 'scoring', 'season', 'season_length', 'sort_keys', 'source', 'start',
    'text', 'units', 'unrecognized_extra', 'walk_forward', 'wasxfail', 'years',
})
ALLOWED_PARAMETERS = frozenset({
    'a', 'args', 'blend_configs', 'body', 'budget', 'bundle', 'call', 'calls', 'canary', 'capsys', 'cmd', 'cwd', 'd',
    'd24', 'd25', 'damage', 'data_dir', 'df', 'expect', 'extra', 'failed_nodes', 'files', 'fn',
    'game_probability_mode', 'gate', 'head', 'identity', 'k', 'key', 'match', 'monkeypatch', 'msg', 'name', 'node',
    'out', 'outcome', 'over', 'p', 'pas', 'passed', 'path', 'pick', 'pins', 'plugin', 'r', 'rc', 'records',
    'register', 'repo', 'retrain_every', 'rows', 'run_name', 'season', 'seed', 'self', 'source', 'stubbed', 'tests',
    'three_runs', 'tmp_path', 'v', 'verdict', 'wasxfail', 'when', 'why',
})
PYTEST_ATTRIBUTES = frozenset({"fixture", "mark", "raises"})
ALLOWED_DUNDER_NAMES = frozenset({"__file__"})
ALLOWED_DUNDER_METHODS = frozenset({"__init__", "__call__"})
_BUILTIN_NAMES = frozenset(dir(builtins))

DRIVER = '''import json, sys, types
import pytest
import pluggy._hooks

MONITOR = 4                                       # a sys.monitoring tool id no other tool in the run uses


def _origin(p):
    if isinstance(p, types.ModuleType):
        return p.__name__
    return p.__module__ if isinstance(p, type) else type(p).__module__


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
        self.late_events, self.armed, self.changes, self.foreign = 0, False, [], None

    def pytest_plugin_registered(self, plugin):
        if self.sentinel is not None and plugin is not self.sentinel:
            self.late_events += 1                        # an event: kept even if the plugin later unregisters

    def pytest_configure(self, config):
        self.config = config

    def pytest_collectreport(self, report):
        if report.failed:
            self.records.append({"collecterror": report.nodeid})

    def pytest_collection_finish(self, session):
        pm = session.config.pluginmanager
        self.marked = [i.nodeid for i in session.items
                       if any(i.get_closest_marker(m) for m in ("xfail", "skip", "skipif"))]
        self.foreign = sorted({o for o in (_origin(p) for p in pm.get_plugins() if p is not None)
                               if o != "__main__" and not o.startswith("_pytest.")})
        self.sentinel = Sentinel(pm)
        pm.register(self.sentinel, "framing-runner-sentinel")
        self.at_sentinel = {id(p) for p in pm.get_plugins()}
        self.armed = True                                # from here on, any hook implementation added or removed

    def pytest_runtest_logreport(self, report):
        self.records.append({"node": report.nodeid, "when": report.when, "outcome": report.outcome,
                             "wasxfail": hasattr(report, "wasxfail")})

    def pytest_keyboard_interrupt(self, excinfo):
        self.records.append({"interrupted": excinfo.typename})

    def pytest_internalerror(self, excrepr, excinfo):
        self.records.append({"internalerror": excinfo.typename})


def _watch(rec):
    """Every run of pluggy's add/remove code once the sentinel is registered, through whatever API."""
    mon = sys.monitoring
    mon.use_tool_id(MONITOR, "framing-runner")

    def started(code, offset):
        if rec.armed:
            rec.changes.append(code.co_qualname)

    mon.register_callback(MONITOR, mon.events.PY_START, started)
    for fn in (pluggy._hooks.HookCaller._add_hookimpl, pluggy._hooks.HookCaller._remove_plugin):
        mon.set_local_events(MONITOR, fn.__code__, mon.events.PY_START)


out, args = sys.argv[1], sys.argv[2:]
rec = Recorder()
_watch(rec)
rc = int(pytest.main(args, plugins=[rec]))
late = None
if rec.at_sentinel is not None:
    late = rec.late_events + len({id(p) for p in rec.config.pluginmanager.get_plugins()} - rec.at_sentinel)
evidence = {"rc": rc, "records": rec.records, "calls": rec.sentinel.calls if rec.sentinel else {},
            "finish": rec.sentinel.finish if rec.sentinel else None, "late_plugins": late,
            "not_outermost": rec.sentinel.not_outermost if rec.sentinel else None, "marked": rec.marked,
            "plugin_changes": list(rec.changes) if rec.armed else None, "foreign_plugins": rec.foreign}
with open(out, "x") as f:
    json.dump(evidence, f)
sys.exit(rc)
'''


def _python() -> list[str]:
    """The interpreter for every collection and run: no bytecode, no script directory or user site on the path."""
    return [sys.executable, "-B", "-P", "-s"]


def _pytest_cmd(root: Path, *args) -> list[str]:
    return [*_python(), "-m", "pytest", *_args(root, *args)]


def _args(root: Path, *args) -> list[str]:
    return [f"--rootdir={root}", "-c", os.devnull, "--noconftest", "-p", "no:cacheprovider", *args]


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "PYTEST"))}
    env.update(TZ="America/New_York", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPYCACHEPREFIX="/dev/null/framing-runner", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
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
    if bundle.get("plugin_changes") != []:
        return f"INCONCLUSIVE({tag}, plugin system changed)", failed
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
    if bundle.get("foreign_plugins") != []:
        return f"INCONCLUSIVE({tag}, foreign plugins)", failed
    if returncode == 1 and failed:
        return "RED", failed
    if returncode == 0 and not failed:
        return "SURVIVED", failed
    return f"INCONCLUSIVE({tag})", failed


# ---------------------------------------------------------------- the gate

def suite_files(root: Path, tests: list[str]) -> tuple[list[Path], list[str]]:
    """The files the run imports as test code: each named test's module, then the __init__.py of every package above
    it (pytest imports them as its packages). A named test must be a file inside the root."""
    root = Path(root).resolve()
    files, problems = [], []
    for t in tests:
        rel = t.split("::")[0]
        p = (root / rel).resolve()
        if root not in p.parents:
            problems.append(f"{rel}: outside the root")
            continue
        if not p.is_file():
            problems.append(f"{rel}: not a file")
            continue
        chain, d = [p], p.parent
        while (d / "__init__.py").is_file():
            if d != root and root not in d.parents:
                problems.append(f"{rel}: its packages reach above the root")
                break
            chain.append((d / "__init__.py").resolve())
            d = d.parent
        files += [f for f in chain if f not in files]
    return files, problems


def _dunder(ident: str) -> bool:
    return ident.startswith("__") and ident.endswith("__")


def _enclosing_function(node, parents):
    node = parents.get(node)
    while node is not None and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        node = parents.get(node)
    return node


def _vocabulary_problems(data: bytes, filename: str) -> list[str]:
    """Every use outside the reviewed vocabulary in one file, parsed from its bytes as Python will."""
    try:
        tree = ast.parse(data, filename=filename)
    except (SyntaxError, ValueError) as e:
        return [f"0: does not parse ({type(e).__name__})"]
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found = []

    def bad(node, why):
        found.append(f"{getattr(node, 'lineno', 0)}: {why}")

    def bind(node, ident):
        if ident.startswith("pytest") or _dunder(ident):
            bad(node, f"name {ident}")        # hooks, pytest_plugins, pytestmark, pytest rebound, module __getattr__
        elif ident in _BUILTIN_NAMES:
            bad(node, f"rebinds builtin {ident}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_MODULES:
                    bad(node, f"import {a.name}")
                if a.name == "pytest":
                    if a.asname not in (None, "pytest"):
                        bad(node, f"pytest imported as {a.asname}")
                    continue                                       # the one binding of the name pytest
                if "." in a.name and a.asname is None:
                    bad(node, f"import {a.name} without an alias")
                bind(node, a.asname or a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                bad(node, "relative import")
            elif node.module not in ALLOWED_MODULES:
                bad(node, f"import {node.module}")
            else:
                names = PYTEST_ATTRIBUTES if node.module == "pytest" else ALLOWED_ATTRIBUTES
                for a in node.names:
                    if f"{node.module}.{a.name}" not in ALLOWED_MODULES and a.name not in names:
                        bad(node, f"import {node.module}.{a.name}")
            for a in node.names:
                if a.name != "*":
                    bind(node, a.asname or a.name)
        elif isinstance(node, ast.Name):
            ident = node.id
            if not isinstance(node.ctx, ast.Load):
                bind(node, ident)
            elif ident == "pytest":
                up = parents.get(node)
                if not (isinstance(up, ast.Attribute) and up.value is node):
                    bad(node, "pytest used other than as pytest.<name>")
            elif _dunder(ident) and ident not in ALLOWED_DUNDER_NAMES:
                bad(node, f"name {ident}")        # __builtins__, __import__, __loader__ (pytest's rewrite hook), ...
            elif ident in _BUILTIN_NAMES and ident not in ALLOWED_BUILTINS:
                bad(node, f"builtin {ident}")
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "pytest":
                if node.attr not in PYTEST_ATTRIBUTES:
                    bad(node, f"pytest.{node.attr}")
            elif node.attr not in ALLOWED_ATTRIBUTES:
                bad(node, f"attribute {node.attr}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not (isinstance(node, ast.FunctionDef) and node.name in ALLOWED_DUNDER_METHODS
                    and isinstance(parents.get(node), ast.ClassDef)):
                bind(node, node.name)
        elif isinstance(node, ast.arg):                     # (no allowed parameter is a builtin, dunder or pytest*)
            if node.arg not in ALLOWED_PARAMETERS:
                bad(node, f"parameter {node.arg}")
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bind(node, node.name)
        elif isinstance(node, ast.Match):
            bad(node, "match statement")
        elif isinstance(node, ast.keyword):
            if node.arg is not None:
                if node.arg not in ALLOWED_KEYWORDS:
                    bad(node, f"keyword {node.arg}")
                continue
            fn = _enclosing_function(node, parents)
            own = fn.args.kwarg.arg if fn is not None and fn.args.kwarg is not None else None
            passed_on = isinstance(node.value, ast.Name) and own is not None and node.value.id == own
            if passed_on:                                   # the parameter is only ever passed on, never rebuilt
                uses = [n for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == own]
                passed_on = all(isinstance(parents.get(n), ast.keyword) and parents[n].arg is None for n in uses)
            if not passed_on:
                bad(node, "keyword splat")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "setattr":
            name = node.args[1] if len(node.args) == 3 else None
            if (len(node.args) != 3 or any(isinstance(a, ast.Starred) for a in node.args)
                    or not (isinstance(name, ast.Constant) and isinstance(name.value, str))):
                bad(node, "setattr needs (object, literal name, value)")
            elif name.value not in ALLOWED_ATTRIBUTES:
                bad(node, f"setattr name {name.value}")
    return found


def scope_problems(root: Path, tests: list[str], overrides: dict | None = None) -> list[str]:
    """Why the named tests' suite is outside the runner's scope (empty: inside). `overrides` maps a resolved file to
    the bytes it will hold in the run (the mutated target, when it is a suite file)."""
    root = Path(root).resolve()
    files, problems = suite_files(root, tests)
    for f in files:
        data = (overrides or {}).get(f)
        data = f.read_bytes() if data is None else data
        problems += [f"{f.relative_to(root).as_posix()}:{p}" for p in _vocabulary_problems(data, str(f))]
    return problems


# ---------------------------------------------------------------- the change scan

def interpreter_trees() -> list[str]:
    """Where the run's interpreter imports code from: every directory on its import path."""
    return sorted({os.path.realpath(p) for p in sys.path if p and os.path.isdir(p)})


def scan_roots(root: Path) -> list[str]:
    out = []
    for c in sorted({os.path.realpath(root), *interpreter_trees()}):
        if not any(c == r or c.startswith(r + os.sep) for r in out):
            out.append(c)
    return out


def changed_since(t0: int, roots, exempt=()) -> list[str]:
    """Every file or directory under the roots whose ctime is t0 or later, or that cannot be listed (fail closed)."""
    skip = {os.path.realpath(e) for e in exempt}
    changed = set()
    for r in roots:
        for dirpath, dirnames, filenames in os.walk(r, onerror=lambda e: changed.add(str(e.filename))):
            for path in [dirpath] + [os.path.join(dirpath, n) for n in dirnames + filenames]:
                if path not in skip and os.lstat(path).st_ctime_ns >= t0:
                    changed.add(path)
    return sorted(changed)


def _ctime(path: Path):
    try:
        return os.stat(path).st_ctime_ns
    except OSError:
        return None


# ---------------------------------------------------------------- collection and runs

def collect(root: Path, tests: list[str]) -> list[str] | None:
    """The nodes the named tests select, on the unmutated source; None when collection fails or selects nothing."""
    r = subprocess.run(_pytest_cmd(root, "--collect-only", "-q", *tests), cwd=root, env=_env(), capture_output=True,
                       text=True)
    nodes = [l.strip() for l in r.stdout.splitlines() if "::" in l and not l.startswith(" ")]
    return nodes if r.returncode == 0 and nodes else None


def run_mutant(root: Path, tests: list[str]):
    """Run the named tests under the driver. Returns (returncode, stdout, evidence or None)."""
    work = Path(tempfile.mkdtemp(prefix="framing-runner-"))
    (work / "framing_runner_driver.py").write_text(DRIVER)
    evidence = work / "evidence.json"
    r = subprocess.run([*_python(), str(work / "framing_runner_driver.py"), str(evidence),
                        *_args(root, "-q", *tests)], cwd=root, env=_env(), capture_output=True, text=True)
    bundle = None
    if evidence.is_file():
        try:
            bundle = json.loads(evidence.read_text())
        except ValueError:
            bundle = None
    return r.returncode, r.stdout, bundle


def main(spec: str, only: str | None = None, *, root: Path = REPO) -> int:
    t0 = time.time_ns()
    root = Path(root).resolve()
    roots = scan_roots(root)
    wanted = set(only.split(",")) if only else None
    entries = json.loads(Path(spec).read_text())
    targets = sorted({(root / m["file"]).resolve() for m in entries})
    last = {t: _ctime(t) for t in targets}            # each target's ctime after the runner's own last write
    bad = []
    for m in entries:
        if wanted and m["id"] not in wanted:
            continue
        options = [t for t in m["tests"] if t.startswith("-")]
        if options:
            print(m["id"], "INVALID: options in the named test list", options, flush=True); bad.append(m["id"]); continue
        f = (root / m["file"]).resolve(); orig = f.read_bytes(); h = hashlib.sha256(orig).hexdigest(); s = orig.decode()
        if s.count(m["old"]) != 1:
            print(m["id"], "ANCHOR x", s.count(m["old"]), flush=True); bad.append(m["id"]); continue
        mutated = s.replace(m["old"], m["new"]).encode()
        problems = scope_problems(root, m["tests"], {f: mutated})
        if problems:
            print(m["id"], "REFUSED: outside the runner's scope:", "; ".join(problems[:8]), flush=True)
            bad.append(m["id"]); continue
        intended = collect(root, m["tests"])
        if intended is None:
            print(m["id"], "INVALID: the named tests do not collect", flush=True); bad.append(m["id"]); continue
        f.write_bytes(mutated)
        last[f] = _ctime(f)
        try:
            rc, out, bundle = run_mutant(root, m["tests"])
            verdict, failed = classify(rc, intended, bundle)
            changed = changed_since(t0, roots, exempt=targets)
            if any(_ctime(t) != last[t] for t in targets):
                verdict = f"INCONCLUSIVE(exit {rc}, target changed during the run)"
            elif changed:
                verdict = f"INCONCLUSIVE(exit {rc}, files changed during the run)"
            print(m["id"], verdict, (out.strip().splitlines() or ["?"])[-1], f"[{len(intended)} intended]", flush=True)
            for node in failed:
                print("    FAILED " + node[:300], flush=True)
            for path in changed[:5]:
                print("    CHANGED " + path, flush=True)
            if verdict != "RED":
                bad.append(m["id"])
        finally:
            f.write_bytes(orig); assert hashlib.sha256(f.read_bytes()).hexdigest() == h
            last[f] = _ctime(f)
    print("NOT RED:", ",".join(bad) if bad else "none", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
