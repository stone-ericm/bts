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
        ("dm", "pick", "'pick: Turner'")]
    assert calls[0]["caller"][0] == "deliver"
    assert [e["kind"] for e in inside if e["kind"] in ("entry", "branch")] == ["entry", "branch"]


def test_calls_from_test_code_are_not_production_events(project, tmp_path):
    repo, wt = project
    write(wt, "tests/test_extra.py", "from unittest.mock import patch\nfrom bts import transport\n\n"
          "def test_direct():\n    with patch('bts.transport.send') as s:\n        transport.send('eric', 'pick: x')\n"
          "    assert s.called\n")
    r = runner.run(wt, ["tests/test_extra.py", "-q"], tmp_path / "out", "green",
                   observe=observe(wt, ["tests/test_extra.py::test_direct"]))
    inside, _ = certify.interval(r.events, "tests/test_extra.py::test_direct")
    assert not [e for e in inside if e["kind"] == "boundary"]


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
