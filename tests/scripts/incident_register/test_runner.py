"""Trusted observer + session gate + certificates (Codex phase-1 r2 #3, #5, #6, #9)."""
import copy
import os
from pathlib import Path

import pytest

from scripts.audit.incident_register import certify, runner
from tests.scripts.incident_register.synth import DM, defended_project, write

NODE_SEND = "tests/test_mod.py::test_deliver_sends_one_pick"
NODE_ASYNC = "tests/test_mod.py::test_async_delivery"


@pytest.fixture
def project(tmp_path):
    return defended_project(tmp_path)


def observe(wt, nodes, entry="deliver", branch=None):
    cfg = {"nodes": nodes, "entries": [{"file": str(wt / "src/bts/mod.py"), "qualname": entry}],
           "boundaries": [DM], "returns": []}
    if branch:
        cfg["branch"] = {"file": str(wt / "src/bts/mod.py"), "line": branch}
    return cfg


def line_of(wt, text):
    return next(i for i, ln in enumerate((wt / "src/bts/mod.py").read_text().splitlines(), 1) if text in ln)


def test_clean_run_passes_the_gate_and_records_identity(project, tmp_path):
    repo, wt = project
    r = runner.run(wt, ["tests/test_mod.py", "-q"], tmp_path / "out", "green", observe=observe(wt, [NODE_SEND]))
    assert runner.gate(r, worktree=wt, mode="green") == []
    start = next(e for e in r.events if e["kind"] == "session_start")
    assert start["observer_file"] == r.trusted["file"] and start["observer_sha256"] == r.trusted["sha256"]
    assert start["rootdir"] == str(wt) and start["inifile"] == str(wt / "pytest.ini")
    assert not start["observer_file"].startswith(str(wt))


def test_the_recorder_sees_every_production_call_with_identity(project, tmp_path):
    repo, wt = project
    r = runner.run(wt, ["tests/test_mod.py", "-q"], tmp_path / "out", "green",
                   observe=observe(wt, [NODE_SEND], branch=line_of(wt, "if ready:")))
    inside, why = certify.interval(r.events, NODE_SEND)
    assert why == []
    calls = [e for e in inside if e["kind"] == "boundary"]
    assert [(c["name"], c["identity"]["category"], c["identity"]["value"]) for c in calls] == [
        ("dm", "pick", "pick: Turner")]
    assert calls[0]["caller"][0] == "deliver"
    assert [e["kind"] for e in inside if e["kind"] in ("entry", "branch")] == ["entry", "branch"]


def test_calls_from_test_code_are_recorded_with_a_test_caller(project, tmp_path):
    """Callee-side recording sees every call, whoever makes it; the stack says it came from the test
    (absence counts it conservatively; an event certificate still needs the linked entry)."""
    repo, wt = project
    write(wt, "tests/test_extra.py", "from unittest.mock import patch\nfrom bts import transport\n\n"
          "def test_direct():\n    with patch('bts.transport.send') as s:\n        transport.send('eric', 'pick: x')\n"
          "    assert s.called\n")
    r = runner.run(wt, ["tests/test_extra.py", "-q"], tmp_path / "out", "green",
                   observe=observe(wt, ["tests/test_extra.py::test_direct"]))
    inside, _ = certify.interval(r.events, "tests/test_extra.py::test_direct")
    calls = [e for e in inside if e["kind"] == "boundary"]
    assert len(calls) == 1 and calls[0]["caller"][0] == "test_direct"
    assert not any(f[1].startswith(str(wt / "src")) for f in calls[0]["stack"])


@pytest.mark.parametrize("tamper, needle", [
    (lambda s: s.update(observer_sha256="0" * 64), "trusted copy"),
    (lambda s: s.update(prefix="/usr"), "own venv"),
    (lambda s: s.update(rootdir="/elsewhere"), "rootdir"),
])
def test_gate_rejects_identity_mismatches(project, tmp_path, tamper, needle):
    repo, wt = project
    r = runner.run(wt, ["tests/test_mod.py", "-q"], tmp_path / "out", "green")
    bad = copy.deepcopy(r)
    tamper(next(e for e in bad.events if e["kind"] == "session_start"))
    assert any(needle in x for x in runner.gate(bad, worktree=wt, mode="green"))


def test_a_planted_pytest_module_in_the_worktree_is_not_imported(project, tmp_path):
    repo, wt = project
    write(wt, "pytest.py", "raise SystemExit('shadow pytest imported')\n")
    r = runner.run(wt, ["tests/test_mod.py", "-q"], tmp_path / "out", "green")
    assert runner.gate(r, worktree=wt, mode="green") == []


def test_a_foreign_bts_package_is_detected(project, tmp_path):
    repo, wt = project
    write(wt, "bts/__init__.py", "")
    write(wt, "bts/mod.py", "def grade():\n    return 'void'\ndef run(r):\n    return 'done'\n"
          "def deliver(*a, **k):\n    return 'done'\n")
    r = runner.run(wt, ["tests/test_mod.py", "-q"], tmp_path / "out", "green")
    assert any("outside the worktree src" in x or "no bts modules" in x
               for x in runner.gate(r, worktree=wt, mode="green"))


def test_collection_error_rejects_the_run(project, tmp_path):
    """r2 #3: `--continue-on-collection-errors` baseline was accepted."""
    repo, wt = project
    write(wt, "tests/test_broken.py", "import does_not_exist\n")
    r = runner.run(wt, ["tests", "-q", "--continue-on-collection-errors"], tmp_path / "out", "green")
    assert "collection errors" in runner.gate(r, worktree=wt, mode="green")


def test_teardown_error_rejects_even_a_mutant_run(project, tmp_path):
    """r2 #3: an unrelated node's teardown error coexisted with `accepted`."""
    repo, wt = project
    write(wt, "tests/conftest.py", "import pytest\n\n@pytest.fixture(autouse=True)\ndef boom(request):\n"
          "    yield\n    if request.node.name == 'test_unrelated':\n        raise ValueError('teardown')\n")
    r = runner.run(wt, ["tests/test_mod.py", "-q"], tmp_path / "out", "mutant")
    assert any("test_unrelated: teardown_error" in x for x in runner.gate(r, worktree=wt, mode="mutant"))


def test_imperative_xfail_is_not_green(project, tmp_path):
    """r2 #3: an imperative-XFAIL killing baseline was accepted."""
    repo, wt = project
    write(wt, "tests/test_x.py", "import pytest\nfrom bts import mod\n\ndef test_x():\n"
          "    if mod.grade() == 'void':\n        pytest.xfail('hides the baseline')\n    assert False\n")
    r = runner.run(wt, ["tests/test_x.py", "-q"], tmp_path / "out", "green")
    assert any("test_x: xfail" in x for x in runner.gate(r, worktree=wt, mode="green"))


def test_inventory_must_match(project, tmp_path):
    repo, wt = project
    r = runner.run(wt, ["tests/test_mod.py", "-q"], tmp_path / "out", "green")
    assert any("inventory differs" in x for x in runner.gate(r, worktree=wt, mode="green", expected=["x::y"]))


def test_output_inside_the_worktree_is_refused(project):
    repo, wt = project
    with pytest.raises(ValueError, match="outside the worktree"):
        runner.run(wt, ["tests/test_mod.py"], wt / "evidence", "green")


def test_async_path_is_unavailable(project, tmp_path):
    """r2 #5: an actual `await` inside an async entry received ok=True."""
    repo, wt = project
    r = runner.run(wt, ["tests/test_mod.py", "-q"], tmp_path / "out", "green",
                   observe=observe(wt, [NODE_ASYNC], entry="adeliver"))
    inside, why = certify.interval(r.events, NODE_ASYNC)
    assert any("asynchronous" in x for x in why)
    got = certify.certify(r.events, node=NODE_ASYNC, kind="event",
                          entry={"file": str(wt / "src/bts/mod.py"), "qualname": "adeliver"},
                          bad={"boundary": "dm", "category": "pick"})
    assert not got["ok"] and any("asynchronous" in x for x in got["reasons"])


def test_observer_edits_nothing_and_leaves_no_files(project, tmp_path):
    repo, wt = project
    from scripts.audit.incident_register import owned
    m0 = owned.manifest(wt)
    runner.run(wt, ["tests/test_mod.py", "-q"], tmp_path / "out", "green", observe=observe(wt, [NODE_SEND]))
    assert owned.manifest(wt) == m0


def test_a_session_that_never_finishes_is_rejected(project, tmp_path):
    repo, wt = project
    write(wt, "tests/test_exit.py", "import os\n\ndef test_bail():\n    os._exit(0)\n")
    r = runner.run(wt, ["tests/test_exit.py", "-q"], tmp_path / "out", "green")
    assert "the session did not finish" in runner.gate(r, worktree=wt, mode="green")


def test_observer_errors_reject_the_run(project, tmp_path):
    repo, wt = project
    broken = dict(DM, classify=[["pick", "("]])          # an invalid pattern raises inside the observer
    cfg = {"nodes": [NODE_SEND], "entries": [], "boundaries": [broken], "returns": []}
    r = runner.run(wt, ["tests/test_mod.py", "-q"], tmp_path / "out", "green", observe=cfg)
    assert "observer errors" in runner.gate(r, worktree=wt, mode="green")


def test_a_return_code_the_nodes_do_not_explain_is_rejected(project, tmp_path):
    repo, wt = project
    write(wt, "tests/conftest.py", "import pytest\n\n@pytest.hookimpl(trylast=True)\n"
          "def pytest_sessionfinish(session, exitstatus):\n    session.exitstatus = 5\n")
    r = runner.run(wt, ["tests/test_mod.py", "-q"], tmp_path / "out", "green")
    assert r.returncode == 5
    assert "return code 5, expected 0" in runner.gate(r, worktree=wt, mode="green")


def test_gate_rejects_a_pytest_imported_from_the_worktree(project, tmp_path):
    repo, wt = project
    r = runner.run(wt, ["tests/test_mod.py", "-q"], tmp_path / "out", "green")
    bad = copy.deepcopy(r)
    next(e for e in bad.events if e["kind"] == "session_start")["pytest_file"] = str(wt / "pytest.py")
    assert "pytest was imported from the worktree (shadowed)" in runner.gate(bad, worktree=wt, mode="green")


SCEN = '''import threading
from bts import transport

released = threading.Event()
seen = {"repr": False}


class Value:
    def __repr__(self):
        seen["repr"] = True
        return "VALUE"


def via_map():
    list(map(transport.send, ["eric"], ["pick: Turner"]))
    return "done"


def via_worker():
    def later():
        released.wait(5)
        transport.send("eric", "pick: late")
    threading.Thread(target=later).start()
    return "defer"


def value():
    return Value()


def plain():
    transport.send("eric", "pick: plain")
    return "done"


def long_send():
    transport.send("eric", "pick: " + "x" * 600)
    return "y" * 600


class HardError(Exception):
    pass


def login(blank):
    if blank:
        raise HardError("not JSON")
    return "session"
'''
SCEN_TESTS = '''from unittest.mock import patch

from bts import scen, transport


def test_map():
    with patch("bts.transport.send") as send:
        scen.via_map()
    assert send.call_count == 1


import pytest


@pytest.fixture
def release_in_teardown():
    yield
    scen.released.set()


def test_worker(release_in_teardown):
    with patch("bts.transport.send") as send:
        scen.via_worker()
        assert send.call_count == 0


def test_repr_untouched():
    scen.value()
    assert scen.seen["repr"] is False


def helper(recipient, text):
    return "helper"


def test_rebound_helper(monkeypatch):
    helper("x", "warm-up call from the test")        # its PY_START gets disabled as non-production
    monkeypatch.setattr(transport, "send", helper)
    scen.plain()


def test_rebound_helper_twice(monkeypatch):
    monkeypatch.setattr(transport, "send", helper)
    scen.plain()
    scen.plain()


def test_c_boundary(monkeypatch):
    monkeypatch.setattr(transport, "send", print)
    scen.plain()


def test_long():
    with patch("bts.transport.send"):
        scen.long_send()


def test_raises():
    import pytest
    with pytest.raises(scen.HardError):
        scen.login(True)
'''


@pytest.fixture
def scen(tmp_path):
    return defended_project(tmp_path, extra={"src/bts/scen.py": SCEN, "tests/test_scen.py": SCEN_TESTS})


def _obs(wt, node, entry, returns=()):
    return {"nodes": [f"tests/test_scen.py::{node}"],
            "entries": [{"file": str(wt / "src/bts/scen.py"), "qualname": entry}],
            "boundaries": [DM], "returns": [{"file": str(wt / "src/bts/scen.py"), "qualname": q} for q in returns]}


def _inside(r, node):
    inside, why = certify.interval(r.events, f"tests/test_scen.py::{node}")
    assert not [w for w in why if "observer error" in w], why
    return inside


def test_a_send_made_from_c_is_recorded(scen, tmp_path):
    """r3 #1 measured: map(transport.send, ...) was invisible to the call-site recorder."""
    repo, wt = scen
    r = runner.run(wt, ["tests/test_scen.py::test_map", "-q"], tmp_path / "o", "g", observe=_obs(wt, "test_map", "via_map"))
    calls = [e for e in _inside(r, "test_map") if e["kind"] == "boundary"]
    assert [c["identity"]["category"] for c in calls] == ["pick"]


def test_a_real_run_records_a_pure_observation(scen, tmp_path):
    """Codex phase-1 r7 #4: the trusted bootstrap's census is present and counts no audit hook, automatic
    collection is off at both ends of the call phase, and no application signal handler is installed."""
    repo, wt = scen
    r = runner.run(wt, ["tests/test_scen.py::test_map", "-q"], tmp_path / "o", "g", observe=_obs(wt, "test_map", "via_map"))
    marks = [e for e in r.events if e["kind"] in ("obs_start", "obs_end")]
    assert [e["purity"] for e in marks] == [{"audit_hooks_added": 0, "gc_enabled": False, "signal_handlers": []}] * 2
    assert certify.interval(r.events, "tests/test_scen.py::test_map")[1] == []


def test_observing_a_return_runs_no_application_code(scen, tmp_path):
    """r3 #2 measured: repr() of the observed return value flipped a failing test to passing."""
    repo, wt = scen
    r = runner.run(wt, ["tests/test_scen.py::test_repr_untouched", "-q"], tmp_path / "o", "g",
                   observe=_obs(wt, "test_repr_untouched", "value", returns=["value"]))
    assert runner.gate(r, worktree=wt, mode="green") == []
    ret = [e for e in _inside(r, "test_repr_untouched") if e["kind"] == "return"][0]
    assert ret["value"] == {"type": "bts.scen.Value", "fields": {}}


def test_a_boundary_rebound_to_a_disabled_helper_is_still_recorded(scen, tmp_path):
    """r3 #7 measured: the helper's PY_START was disabled before it became the boundary."""
    repo, wt = scen
    r = runner.run(wt, ["tests/test_scen.py::test_rebound_helper", "-q"], tmp_path / "o", "g",
                   observe=_obs(wt, "test_rebound_helper", "plain"))
    calls = [e for e in _inside(r, "test_rebound_helper") if e["kind"] == "boundary"]
    assert [(c["identity"]["value"], c["identity"]["category"]) for c in calls] == [("pick: plain", "pick")]


def test_a_c_implemented_boundary_is_a_coverage_gap(scen, tmp_path):
    repo, wt = scen
    r = runner.run(wt, ["tests/test_scen.py::test_c_boundary", "-q", "-s"], tmp_path / "o", "g",
                   observe=_obs(wt, "test_c_boundary", "plain"))
    assert [e["reason"] for e in _inside(r, "test_c_boundary") if e["kind"] == "boundary_gap"]


def test_a_source_change_during_the_session_is_rejected(project, tmp_path):
    """r3 #4: imports were hashed only at session end; the loaded code must match the prepared tree."""
    repo, wt = project
    write(wt, "tests/test_z_mutate.py", "from pathlib import Path\nimport bts\n\ndef test_z():\n"
          "    p = Path(bts.__file__).with_name('mod.py')\n    p.write_text(p.read_text() + '\\n# drift\\n')\n")
    r = runner.run(wt, ["tests/test_mod.py", "tests/test_z_mutate.py", "-q"], tmp_path / "out", "green")
    assert "the production src tree changed during the session" in runner.gate(r, worktree=wt, mode="green")


def test_an_exceptional_exit_is_a_classified_return_event(scen, tmp_path):
    """8/11-a's symptom is the entry RAISING the wrong error: recorded as a return-kind event whose
    value is only the exception's type name (read through type's own descriptors)."""
    repo, wt = scen
    obs = _obs(wt, "test_raises", "login")
    obs["returns"] = [{"file": str(wt / "src/bts/scen.py"), "qualname": "login",
                       "classify": [["hard", '"raised": "bts.scen.HardError"']]}]
    r = runner.run(wt, ["tests/test_scen.py::test_raises", "-q"], tmp_path / "o", "g", observe=obs)
    rets = [e for e in _inside(r, "test_raises") if e["kind"] == "return"]
    assert [(x["value"], x["category"], x["how"]) for x in rets] == [({"raised": "bts.scen.HardError"}, "hard", "unwind")]


def test_a_same_size_edit_with_the_same_mtime_is_what_runs(project, tmp_path, monkeypatch):
    """Stale bytecode (seen in a manual scratch run, 2026-09-29): Python trusts a .pyc whose recorded
    source size and mtime match, so a mutant or a restore of the same size written within the same
    second as the last compile would run the OLD code. Evidence runs write no bytecode, so the edited
    source is what runs."""
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)       # only the runner's own setting counts
    repo, wt = project
    mod = wt / "src/bts/mod.py"
    first = runner.run(wt, ["tests/test_mod.py::test_grade", "-q"], tmp_path / "out", "green")
    assert first.returncode == 0
    before = mod.stat()
    mod.write_text(mod.read_text().replace('return "void"', 'return "miss"'))
    os.utime(mod, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert mod.stat().st_size == before.st_size                          # invisible to the pyc check
    second = runner.run(wt, ["tests/test_mod.py::test_grade", "-q"], tmp_path / "out", "mutant")
    assert second.returncode == 1, "the edited source did not run (stale bytecode)"
    assert [p for p in wt.rglob("*.pyc") if ".venv" not in p.parts] == []


def test_a_source_change_before_the_session_starts_is_rejected(project, tmp_path):
    """The loaded tree must be the one the runner prepared: a conftest that edits src at import time
    changes it after the runner's digest and before the observer's session_start."""
    repo, wt = project
    write(wt, "tests/conftest.py", "from pathlib import Path\n\np = Path(__file__).parent.parent / 'src/bts/mod.py'\n"
          "p.write_text(p.read_text() + '\\n# drift at import\\n')\n")
    r = runner.run(wt, ["tests/test_mod.py", "-q"], tmp_path / "out", "green")
    assert "the production src tree at session start differs from the tree the runner prepared" in \
        runner.gate(r, worktree=wt, mode="green")


def test_a_real_boundary_called_twice_is_recorded_twice(project, tmp_path):
    """A real (unmocked) boundary function in production code, called twice, is recorded twice."""
    repo, wt = project
    write(wt, "tests/test_twice.py", "from bts import mod\n\n\ndef test_twice():\n"
          "    assert mod.deliver(True, late=True) == 'done'\n")
    node = "tests/test_twice.py::test_twice"
    r = runner.run(wt, ["tests/test_twice.py", "-q"], tmp_path / "out", "green", observe=observe(wt, [node]))
    assert runner.gate(r, worktree=wt, mode="green") == []
    inside, why = certify.interval(r.events, node)
    assert why == []
    calls = [(e["identity"]["category"], e["identity"]["value"]) for e in inside if e["kind"] == "boundary"]
    assert calls == [("alert", "BTS health CRITICAL: late"), ("pick", "pick: Turner")]


def test_a_boundary_outside_production_stays_enabled_after_its_first_call(scen, tmp_path):
    """Sweep O2: a boundary whose code lives OUTSIDE the production tree (here a test helper the binding
    was rebound to) must not have its start event disabled after its first recorded call, or every
    later call would be invisible to an absence claim."""
    repo, wt = scen
    r = runner.run(wt, ["tests/test_scen.py::test_rebound_helper_twice", "-q"], tmp_path / "o", "g",
                   observe=_obs(wt, "test_rebound_helper_twice", "plain"))
    calls = [e for e in _inside(r, "test_rebound_helper_twice") if e["kind"] == "boundary"]
    assert [(c["identity"]["value"], c["identity"]["category"]) for c in calls] == [("pick: plain", "pick")] * 2


def test_an_incompletely_observed_value_is_never_classified(scen, tmp_path):
    """A boundary text or return longer than the serializer keeps is 'unavailable', never a category
    read off its prefix (Codex phase-1 r4 #2): 'pick: xxx…' must not count as a pick."""
    repo, wt = scen
    r = runner.run(wt, ["tests/test_scen.py::test_long", "-q"], tmp_path / "o", "g",
                   observe=_obs(wt, "test_long", "long_send", returns=["long_send"]))
    inside = _inside(r, "test_long")
    assert [e["identity"]["category"] for e in inside if e["kind"] == "boundary"] == ["unavailable"]
    assert [e["category"] for e in inside if e["kind"] == "return"] == ["unavailable"]
