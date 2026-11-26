"""The fresh whole-range review's counterexamples (a new Codex session, Part 1 blind to the earlier rounds:
``docs/audit/2026-09-29-incident-register-codex-fresh-part1.md``), turned to the required direction.

F1: a declared wrong value matched a different JSON value through Python equality (``False == 0``).
F2: the branch anchor only had to be in a mutated FILE, so a certificate linked an unchanged line to the mutation.
F3: the killing failure's class was checked by its module and qualname, which any class can claim.
F5: an inherited PYTHONPYCACHEPREFIX could make an evidence run read stale bytecode from outside the worktree.
"""
from types import SimpleNamespace

import pytest

from scripts.audit.incident_register import acceptance, defence, runner
from scripts.audit.incident_register.certify import certify, same_json
from tests.scripts.incident_register.synth import MOD, TESTS, defended_project, git, spec
from tests.scripts.incident_register.test_certify import ENTRY, F, IN, N, branch, entry, ev, stack, window

DISTINCT = [(False, 0), (True, 1), (0, False), (1, 1.0), ([False], [0]), ({"a": False}, {"a": 0}), ([[1, True]], [[1, 1]]),
            (None, False), ("0", 0)]
DISTINCT_IDS = ["false-0", "true-1", "0-false", "int-float", "nested-list", "nested-map", "zip-pair", "null-false", "str-int"]


@pytest.mark.parametrize("recorded,requested", DISTINCT, ids=DISTINCT_IDS)
def test_a_return_certificate_never_equates_distinct_json_types(recorded, requested):
    """F1 (fresh review): the observer recorded an exact False, the request named 0, and the certificate was accepted.
    JSON false and 0 are different values; so are 1 and 1.0, and so are nested ones."""
    ret = ev("return", 3, file=F, qualname="deliver", frame=10, value=recorded, type=type(recorded).__name__, stack=stack(IN))
    bad = {"file": F, "qualname": "deliver", "value": requested}
    got = certify(window(entry(1, 10), branch(2, IN), ret), node=N, kind="return", entry=ENTRY, bad=bad)
    assert not got["ok"], got
    assert certify(window(entry(1, 10), branch(2, IN), ret), node=N, kind="return", entry=ENTRY,
                   bad=dict(bad, value=recorded))["ok"]


def test_same_json_is_type_exact_and_recursive():
    for a, b in DISTINCT:
        assert not same_json(a, b) and not same_json(b, a), (a, b)
    assert same_json([1, {"a": None, "b": [True, "x", 2.5]}], [1, {"a": None, "b": [True, "x", 2.5]}])
    assert not same_json({"a": 1}, {"a": 1, "b": 2}) and not same_json([1], [1, 2])


def _connection_events(node, f, value):
    pure = {"audit_hooks_added": 0, "gc_enabled": False, "signal_handlers": [], "tracing": False, "tracers_installed": 0}
    stack_of = lambda q, fr: [[q, f, 1, fr]]                                         # noqa: E731
    return [{"kind": "obs_start", "seq": 1, "node": node, "thread": 1, "purity": pure},
            {"kind": "entry", "seq": 2, "node": node, "file": f, "qualname": "grade", "frame": 100, "thread": 1,
             "stack": stack_of("grade", 100)},
            {"kind": "return", "seq": 3, "node": node, "file": f, "qualname": "grade", "value": value, "frame": 100,
             "thread": 1, "stack": stack_of("grade", 100)},
            {"kind": "entry_exit", "seq": 4, "node": node, "file": f, "qualname": "grade", "frame": 100, "thread": 1,
             "how": "return"},
            {"kind": "return", "seq": 5, "node": node, "file": f, "qualname": "read", "value": value, "frame": 200,
             "thread": 1, "stack": stack_of("read", 200)},
            {"kind": "obs_end", "seq": 6, "node": node, "thread": 1, "purity": pure}]


@pytest.mark.parametrize("kind", ["return", "reads"])
@pytest.mark.parametrize("recorded,actual", DISTINCT[:4], ids=DISTINCT_IDS[:4])
def test_a_value_match_never_equates_distinct_json_types(kind, recorded, actual):
    """F1's pair-side twin: the production return or read-back must equal the oracle's actual as JSON, type included."""
    wt, node, f = "/wt", "tests/test_incident.py::test_x", "/wt/src/bts/mod.py"
    conn = ({"kind": "return", "path": "src/bts/mod.py", "qualname": "grade", "review": "reviewed"} if kind == "return" else
            {"kind": "reads", "review": "reviewed", "components": [{"path": "src/bts/mod.py", "qualname": "read"}]})
    reg = {"entry": {"path": "src/bts/mod.py", "qualname": "grade"}, "connection": conn}
    run = SimpleNamespace(events=_connection_events(node, f, recorded))
    want = actual if kind == "return" else [actual]
    status, why = acceptance.connection(run, node, reg, want, wt)
    assert status == "unmatched", (status, why)
    same = recorded if kind == "return" else [recorded]
    assert acceptance.connection(run, node, reg, same, wt)[0] == "value_match"


@pytest.mark.parametrize("field", ["bad_json", "required_json", "actual"])
def test_the_oracle_record_never_equates_distinct_json_types(field):
    """The oracle record's bad, required and actual values are checked against the registry and each other as JSON."""
    reg = {"bad_json": False, "required_json": True}
    rec = {"bad": False, "required": True, "actual": False}
    assert acceptance._oracle_record_errs(reg, rec) == []
    if field == "bad_json":
        reg = dict(reg, bad_json=0)
    elif field == "required_json":
        reg = dict(reg, required_json=1)
    else:
        rec = dict(rec, actual=0)
    assert acceptance._oracle_record_errs(reg, rec), (reg, rec)


# --- F2: the branch anchor must be inside a replacement the mutation made ---------------------------------------

def test_a_branch_anchor_on_an_unchanged_line_is_refused(tmp_path):
    """F2 (fresh review): the edit flips `if late:` in deliver(), but the declared branch is an unchanged line above
    it. The observer records that line, then the alert the flipped branch sends, and the certificate was accepted
    although its "mutated line" never changed. The anchor must now lie inside a replacement's lines."""
    mod = MOD.replace("def deliver(ready=True, late=False):\n    if late:",
                      "def deliver(ready=True, late=False):\n    note = 'unchanged'\n    if late:")
    assert mod != MOD
    _, wt = defended_project(tmp_path, extra={"src/bts/mod.py": mod})
    res = defence.current_defence(wt, spec(branch={"path": "src/bts/mod.py", "text": "note = 'unchanged'"}), tmp_path / "out")
    assert res["verdict"] == "rejected", res
    assert any("not inside any replacement" in r for r in res["reasons"]), res["reasons"]
    assert git(wt, "status", "--porcelain") == ""


def test_edit_spans_follow_each_replacement_through_later_edits():
    before = "a\nb\nc\nd\ne\n"
    # the second edit inserts two lines above the first's replacement, shifting it down
    assert defence._edit_spans(before, [("d\n", "D\n"), ("b\n", "b\nX\nY\n")]) == [(6, 6), (2, 4)]
    assert defence._edit_spans(before, [("b\nc\n", "")]) == [(2, 2)]            # a deletion is its joining line
    assert defence._edit_spans(before, [("c\n", "C1\nC2\n"), ("C2\nd", "Z")]) == [(3, 4)]   # overlap merges


# --- F3: a killing failure is the builtin AssertionError itself, not a class named like it -----------------------

LOOKALIKE = ("class Lookalike(Exception):\n    pass\n\n\nLookalike.__module__ = 'builtins'\n"
             "Lookalike.__qualname__ = Lookalike.__name__ = 'AssertionError'\n\n\n")


def test_a_failure_named_like_assertion_error_is_not_a_kill(tmp_path):
    """F3 (fresh review): a test that raised a heap class claiming module `builtins` and qualname `AssertionError` at the
    declared assertion line passed the killing-failure gate, since the observer recorded only those names. The
    observer now records whether the class IS the builtin AssertionError, and the gate requires it."""
    tests = TESTS.replace("from bts import mod\n", "from bts import mod\n\n\n" + LOOKALIKE, 1).replace(
        '    assert _texts(send) == ["pick: Turner"]  # ASSERT-SEND',
        '    if _texts(send) != ["pick: Turner"]: raise Lookalike(_texts(send))  # ASSERT-SEND')
    assert tests.count("raise Lookalike") == 1
    _, wt = defended_project(tmp_path, extra={"tests/test_mod.py": tests})
    res = defence.current_defence(wt, spec(), tmp_path / "out")
    calls = [e for e in runner.load(tmp_path / "out/mutant.events.jsonl")
             if e["kind"] == "report" and e["when"] == "call" and e["nodeid"].endswith("test_deliver_sends_one_pick")]
    assert calls and (calls[0]["exc_module"], calls[0]["exc_qualname"]) == ("builtins", "AssertionError"), calls
    assert calls[0]["exc_is_assertion"] is False, calls
    assert res["verdict"] == "rejected" and any("not AssertionError" in r for r in res["reasons"]), res["reasons"]


@pytest.mark.parametrize("is_assertion", [True, False, None])
def test_the_kill_gate_reads_the_recorded_class_identity(is_assertion):
    node, f = "tests/t.py::n", "/wt/tests/t.py"
    events = [{"kind": "report", "nodeid": node, "when": "call", "outcome": "failed", "exc_module": "builtins",
               "exc_qualname": "AssertionError", "exc_is_assertion": is_assertion, "frames": [[f, 7, "n"]]}]
    why = defence._assertion_ok(SimpleNamespace(events=events), node, f, 7, "/wt")
    assert (why == []) is (is_assertion is True), why


# --- F5: no relocated bytecode cache from the parent environment -------------------------------------------------

def test_an_inherited_pycache_prefix_never_runs_stale_bytecode(tmp_path, monkeypatch):
    """F5 (fresh review): PYTHONPYCACHEPREFIX was not scrubbed. A valid cache seeded there, then a same-size edit with
    the same mtime, and an evidence run with bytecode writes off still imported the OLD code from the prefix. The
    runner now scrubs it, so the edited source is what runs."""
    import os
    import subprocess
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    _, wt = defended_project(tmp_path)
    prefix, mod = tmp_path / "relocated-cache", wt / "src/bts/mod.py"
    seed = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
    subprocess.run([str(wt / ".venv/bin/python"), "-c", "import bts.mod"], cwd=wt, check=True,
                   env={**seed, "PYTHONPYCACHEPREFIX": str(prefix)})
    assert list(prefix.rglob("mod*.pyc")), "no relocated cache was seeded"
    before = mod.stat()
    mod.write_text(mod.read_text().replace('return "void"', 'return "miss"'))
    os.utime(mod, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert mod.stat().st_size == before.st_size                          # invisible to the pyc check
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(prefix))
    r = runner.run(wt, ["tests/test_mod.py::test_grade", "-q"], tmp_path / "out", "mutant")
    assert r.returncode == 1, "the edited source did not run: the parent environment's relocated cache was read"


@pytest.mark.parametrize("key", ["PYTHONPYCACHEPREFIX", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "W15_OBS_CONFIG"])
def test_a_spec_environment_cannot_set_a_runner_owned_variable(tmp_path, key):
    """A spec's `env` was applied after the runner's own settings, so it could reintroduce a scrubbed variable or
    override bytecode writes or the observer's configuration. Such keys are refused."""
    _, wt = defended_project(tmp_path)
    with pytest.raises(ValueError, match="runner-owned"):
        runner.run(wt, ["tests/test_mod.py::test_grade", "-q"], tmp_path / "out", "g", env_extra={key: "x"})


def test_a_replay_symptom_named_like_assertion_error_is_not_a_symptom(tmp_path):
    """F3 in historical replay: the symptom node at the fix's parent failed with a class claiming the builtin's module
    and name; the replay symptom check read the names. It now reads the recorded class identity."""
    from scripts.audit.incident_register.replay import historical_replay
    from tests.scripts.incident_register.synth import commit
    from tests.scripts.incident_register.test_replay import _mod, rspec
    test = LOOKALIKE + 'from bts import mod\n\n\ndef test_pass_is_void():\n' \
                       '    if mod.grade() != "void": raise Lookalike("miss")  # ASSERT-VOID\n'
    repo, wt = defended_project(tmp_path, extra={"src/bts/mod.py": _mod('"miss"')})
    fix = commit(repo, {"src/bts/mod.py": _mod('"void"'), "tests/test_grade.py": test}, "fix: grade Pass as void")
    res = historical_replay(repo, wt, rspec(fix), tmp_path / "out")
    assert res["verdict"] == "rejected", res
    assert any("not AssertionError" in r for r in res["reasons"]), res["reasons"]
