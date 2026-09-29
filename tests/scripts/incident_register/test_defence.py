"""current_defence end to end on the synthetic project (Codex phase-1 r2 #1–#3, #5; r1 #2, #3)."""
import json
from pathlib import Path

import pytest

from scripts.audit.incident_register.defence import current_defence
from tests.scripts.incident_register.synth import defended_project, git, spec, write

SEND = "tests/test_mod.py::test_deliver_sends_one_pick"
COUNT = "tests/test_mod.py::test_run_sends_once_and_returns_done"


@pytest.fixture
def project(tmp_path):
    return defended_project(tmp_path)


def run(project, tmp_path, **over):
    repo, wt = project
    res = current_defence(wt, spec(**over), tmp_path / "out")
    assert json.loads((tmp_path / "out" / "acceptance.json").read_text())["verdict"] == res["verdict"]
    assert git(wt, "status", "--porcelain") == ""                       # always restored
    return res


def test_absence_certificate_is_accepted(project, tmp_path):
    res = run(project, tmp_path)
    assert res["verdict"] == "accepted", res["reasons"]
    assert res["stages"]["mutant"]["kills"] == [SEND, COUNT]
    assert res["certificates"][SEND]["ok"]
    assert (tmp_path / "out" / "mutant.patch").read_text().count("if not ready:") == 1


def test_event_certificate_is_accepted(project, tmp_path):
    res = run(project, tmp_path,
              mutation_edits=[["src/bts/mod.py", "    if late:\n", "    if not late:\n"]],
              branch={"path": "src/bts/mod.py", "text": "if not late:"},
              symptom={"kind": "event", "boundary": "dm", "category": "alert"})
    assert res["verdict"] == "accepted", res["reasons"]


def test_return_certificate_is_accepted(project, tmp_path):
    res = run(project, tmp_path,
              mutation_edits=[["src/bts/mod.py", 'def grade():\n    return "void"', 'def grade():\n    return "miss"']],
              branch={"path": "src/bts/mod.py", "text": 'return "miss"'},
              entry={"path": "src/bts/mod.py", "qualname": "grade"},
              returns=[{"path": "src/bts/mod.py", "qualname": "grade"}],
              symptom={"kind": "return", "path": "src/bts/mod.py", "qualname": "grade", "value": "miss"},
              killing=[{"node": "tests/test_mod.py::test_grade",
                        "assertion": {"path": "tests/test_mod.py", "text": "# ASSERT-GRADE"}}])
    assert res["verdict"] == "accepted", res["reasons"]


def test_expected_value_mutation_is_refused_even_when_allowed(project, tmp_path):
    """r2 #2 measured: an oracle-only mutant was `accepted` once the test was in allowed_paths."""
    repo, wt = project
    before = (wt / "tests/test_mod.py").read_text()
    res = run(project, tmp_path, allowed_paths=["src/bts/mod.py", "tests/test_mod.py"],
              mutation_edits=[["tests/test_mod.py", '== ["pick: Turner"]', '== ["pick: Other"]']],
              branch={"path": "tests/test_mod.py", "text": "ASSERT-SEND"})
    assert res["verdict"] == "rejected"
    assert any("src/bts" in r or "mutated file" in r for r in res["reasons"]), res["reasons"]
    assert (wt / "tests/test_mod.py").read_text() == before


def test_traversal_edit_is_refused_before_writing(project, tmp_path):
    """r2 #1 measured: `../outside.txt` changed and survived an accepted run."""
    repo, wt = project
    outside = wt.parent / "outside.txt"
    outside.write_text("untouched")
    edits = spec()["mutation_edits"] + [["../outside.txt", "untouched", "CHANGED"]]
    res = run(project, tmp_path, mutation_edits=edits)
    assert res["verdict"] == "rejected" and outside.read_text() == "untouched"


def test_primary_checkout_is_never_used(project, tmp_path):
    repo, wt = project
    write(repo, "keep_me.txt", "sentinel")
    res = current_defence(repo, spec(), tmp_path / "out")
    assert res["verdict"] == "rejected" and any("primary checkout" in r for r in res["reasons"])
    assert (repo / "keep_me.txt").read_text() == "sentinel"


def test_absence_with_a_real_send_elsewhere_is_rejected(project, tmp_path):
    """r2 #5 measured: the callee lacked a boundary but run() still sent the pick; `accepted`."""
    edits = [["src/bts/mod.py",
              '    if ready:\n        transport.send("eric", "pick: Turner")\n    return "done"',
              '    if not ready:\n        transport.send("eric", "pick: Turner")\n    return "defer"']]
    res = run(project, tmp_path, mutation_edits=edits,
              killing=[{"node": COUNT, "assertion": {"path": "tests/test_mod.py", "text": "# ASSERT-RESULT"}}])
    assert res["verdict"] == "rejected"
    assert any("qualifying 'dm' call(s) occurred" in r for r in res["reasons"]), res["reasons"]


def test_killing_failure_at_an_undeclared_assertion_is_rejected(project, tmp_path):
    res = run(project, tmp_path,
              killing=[{"node": COUNT, "assertion": {"path": "tests/test_mod.py", "text": "# ASSERT-RESULT"}}])
    assert res["verdict"] == "rejected"
    assert any("not the declared assertion" in r for r in res["reasons"]), res["reasons"]


def test_declared_killing_node_that_survives_is_rejected(project, tmp_path):
    res = run(project, tmp_path,
              killing=[{"node": "tests/test_mod.py::test_unrelated",
                        "assertion": {"path": "tests/test_mod.py", "text": "assert 1 + 1 == 2"}}])
    assert res["verdict"] == "rejected" and any("was not killed" in r for r in res["reasons"])


def test_absence_without_a_positive_control_is_rejected(project, tmp_path):
    """The baseline never produces a 'pick' in this node, so absence proves nothing."""
    res = run(project, tmp_path, symptom={"kind": "absence", "boundary": "dm", "category": "alert"})
    assert res["verdict"] == "rejected" and any("positive control" in r for r in res["reasons"])


def test_mutated_line_outside_the_entry_invocation_is_rejected(project, tmp_path):
    res = run(project, tmp_path, entry={"path": "src/bts/mod.py", "qualname": "grade"})
    assert res["verdict"] == "rejected"
    assert any("never executed inside a declared entry invocation" in r or "never invoked" in r
               for r in res["reasons"]), res["reasons"]


def test_a_test_that_writes_a_tracked_file_is_rejected(project, tmp_path):
    repo, wt = project
    write(repo, "tests/data.txt", "v1\n")
    write(repo, "tests/test_writer.py", "from pathlib import Path\n\ndef test_w():\n"
          "    Path(__file__).with_name('data.txt').write_text('v2\\n')\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "writer")
    res = run(project, tmp_path, baseline=git(repo, "rev-parse", "HEAD"),
              tests=["tests/test_mod.py", "tests/test_writer.py", "-q"])
    assert res["verdict"] == "rejected" and any("frozen files changed" in r for r in res["reasons"])


def test_collection_error_in_the_baseline_is_rejected(project, tmp_path):
    repo, wt = project
    write(repo, "tests/test_broken.py", "import does_not_exist\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "broken")
    res = run(project, tmp_path, baseline=git(repo, "rev-parse", "HEAD"),
              tests=["tests", "-q", "--continue-on-collection-errors"])
    assert res["verdict"] == "rejected" and any("collection errors" in r for r in res["reasons"])


def test_malformed_spec_still_writes_a_rejected_acceptance(project, tmp_path):
    repo, wt = project
    bad = spec()
    del bad["killing"]
    res = current_defence(wt, bad, tmp_path / "out")
    assert res["verdict"] == "rejected" and (tmp_path / "out" / "acceptance.json").exists()


def test_mutant_only_teardown_error_is_rejected(tmp_path):
    """r2 #3 probe 2 through the pipeline: an unrelated node's teardown fails only under the mutant."""
    conftest = ("import pytest\nfrom bts import mod\n\n@pytest.fixture(autouse=True)\ndef guard(request):\n"
                "    yield\n    if request.node.name == 'test_unrelated' and 'not ready' in open(mod.__file__).read():\n"
                "        raise ValueError('mutant-only teardown error')\n")
    project = defended_project(tmp_path, extra={"tests/conftest.py": conftest})
    res = run(project, tmp_path)
    assert res["verdict"] == "rejected"
    assert any(r.startswith("mutant:") and "teardown_error" in r for r in res["reasons"]), res["reasons"]


FLAKY = '''from pathlib import Path

COUNTER = Path(__file__).resolve().parents[2] / "flaky_counter.txt"   # outside the worktree


def test_flaky():
    n = int(COUNTER.read_text()) + 1 if COUNTER.exists() else 1
    COUNTER.write_text(str(n))
    assert n != FAIL_ON_RUN
'''


@pytest.mark.parametrize("fail_on, stage", [(1, "green"), (2, "green_unobserved"), (5, "restored")])
def test_a_stage_only_failure_is_rejected(tmp_path, fail_on, stage):
    """Each stage's gate counts on its own: a failure seen only in GREEN or only in RESTORED rejects."""
    project = defended_project(tmp_path, extra={"tests/test_flaky.py": f"FAIL_ON_RUN = {fail_on}\n" + FLAKY})
    res = run(project, tmp_path, tests=["tests/test_mod.py", "tests/test_flaky.py", "-q"])
    assert res["verdict"] == "rejected"
    assert any(r.startswith(f"{stage}:") and "test_flaky" in r for r in res["reasons"]), res["reasons"]


def test_branch_anchor_outside_the_mutated_files_is_refused():
    from scripts.audit.incident_register.defence import SpecError, _check_spec
    with pytest.raises(SpecError, match="mutated file"):
        _check_spec(spec(branch={"path": "src/bts/transport.py", "text": "return"}))


def test_assertion_anchor_by_line_and_text(tmp_path):
    """A repeated assertion text is addressed by its frozen line number plus the text on it."""
    from scripts.audit.incident_register.defence import SpecError, anchor_line
    f = tmp_path / "t.py"
    f.write_text("assert x\nassert y\nassert x\n")
    assert anchor_line(f, "assert x", line=3) == 3
    with pytest.raises(SpecError, match="matches 2 lines"):
        anchor_line(f, "assert x")
    with pytest.raises(SpecError, match="line 2"):
        anchor_line(f, "assert x", line=2)


def test_return_certificate_by_category(project, tmp_path):
    res = run(project, tmp_path,
              mutation_edits=[["src/bts/mod.py", 'def grade():\n    return "void"', 'def grade():\n    return "miss"']],
              branch={"path": "src/bts/mod.py", "text": 'return "miss"'},
              entry={"path": "src/bts/mod.py", "qualname": "grade"},
              returns=[{"path": "src/bts/mod.py", "qualname": "grade", "classify": [["miss", "^miss$"]]}],
              symptom={"kind": "return", "path": "src/bts/mod.py", "qualname": "grade", "category": "miss"},
              killing=[{"node": "tests/test_mod.py::test_grade",
                        "assertion": {"path": "tests/test_mod.py", "text": "# ASSERT-GRADE", "line": 12}}])
    assert res["verdict"] == "accepted", res["reasons"]


def test_kill_through_a_mock_assertion_is_located_at_the_test_line(tmp_path):
    """A kill raised inside unittest.mock (assert_called_once_with) is located at the innermost
    frame INSIDE the worktree — the test's own line — not at the stdlib frame."""
    extra = {"tests/test_mockassert.py": (
        "from unittest.mock import patch\nfrom bts import mod\n\n\ndef test_pick_sent_once():\n"
        "    with patch('bts.transport.send') as send:\n        mod.run(True)\n"
        "    send.assert_called_once_with('eric', 'pick: Turner')  # ASSERT-MOCK\n")}
    project = defended_project(tmp_path, extra=extra)
    res = run(project, tmp_path, tests=["tests/test_mod.py", "tests/test_mockassert.py", "-q"],
              killing=[{"node": "tests/test_mockassert.py::test_pick_sent_once",
                        "assertion": {"path": "tests/test_mockassert.py", "text": "# ASSERT-MOCK"}}])
    assert res["verdict"] == "accepted", res["reasons"]


def test_an_assert_inside_production_code_is_not_an_observable_kill(tmp_path):
    project = defended_project(tmp_path)
    repo, wt = project
    edits = spec()["mutation_edits"] + [["src/bts/mod.py", '    return "done"\n\n\ndef run',
                                         '    assert False, "production assert"\n    return "done"\n\n\ndef run']]
    res = run(project, tmp_path, mutation_edits=edits)
    assert res["verdict"] == "rejected" and any("not the declared assertion" in r for r in res["reasons"])


def test_kill_through_a_venv_helper_is_located_at_the_test_line(tmp_path):
    """An AssertionError raised in a package inside the worktree's own .venv (like pandas.testing)
    is located at the test's line, not at the venv frame."""
    extra = {"tests/test_helper.py": (
        "import w15_helper\nfrom unittest.mock import patch\nfrom bts import mod\n\n\ndef test_helper_check():\n"
        "    with patch('bts.transport.send') as send:\n        mod.run(True)\n"
        "    w15_helper.check_sent(send)  # ASSERT-HELPER\n")}
    project = defended_project(tmp_path, extra=extra)
    repo, wt = project
    site = next((wt / ".venv" / "lib").glob("python3*/site-packages"))
    (site / "w15_helper.py").write_text("def check_sent(send):\n    assert send.call_count == 1, send.call_count\n")
    res = run(project, tmp_path, tests=["tests/test_mod.py", "tests/test_helper.py", "-q"],
              killing=[{"node": "tests/test_helper.py::test_helper_check",
                        "assertion": {"path": "tests/test_helper.py", "text": "# ASSERT-HELPER"}}])
    assert res["verdict"] == "accepted", res["reasons"]


def test_observer_dependent_behaviour_is_rejected_by_the_unobserved_control(tmp_path):
    """Design §9.3 as amended: a node that behaves differently when observed cannot certify."""
    extra = {"tests/test_observer_aware.py": (
        "import sys\nfrom unittest.mock import patch\nfrom bts import mod\n\n\ndef test_aware():\n"
        "    with patch('bts.transport.send') as send:\n        mod.run(True)\n"
        "    assert sys.monitoring.get_tool(4) is not None  # passes only while observed\n"
        "    assert [c.args[1] for c in send.call_args_list] == ['pick: Turner']  # ASSERT-AWARE\n")}
    project = defended_project(tmp_path, extra=extra)
    res = run(project, tmp_path, tests=["tests/test_mod.py", "tests/test_observer_aware.py", "-q"],
              killing=[{"node": "tests/test_observer_aware.py::test_aware",
                        "assertion": {"path": "tests/test_observer_aware.py", "text": "# ASSERT-AWARE"}}])
    assert res["verdict"] == "rejected"
    assert any(r.startswith("green_unobserved:") for r in res["reasons"]), res["reasons"]


ASSERT_RESULT = [{"node": COUNT, "assertion": {"path": "tests/test_mod.py", "text": "# ASSERT-RESULT"}}]
SEND_THEN_DONE = '    if ready:\n        transport.send("eric", "pick: Turner")\n    return "done"'


def test_absence_is_not_certified_over_a_send_made_from_c(project, tmp_path):
    """r3 #1 measured: list(map(transport.send, ...)) sent for real; absence was accepted."""
    edits = [["src/bts/mod.py", SEND_THEN_DONE,
              '    if ready:\n        list(map(transport.send, ["eric"], ["pick: Turner"]))  # W1.5 mutant: via C\n'
              '    return "defer"']]
    res = run(project, tmp_path, mutation_edits=edits, branch={"path": "src/bts/mod.py", "text": "W1.5 mutant: via C"},
              killing=ASSERT_RESULT)
    assert res["verdict"] == "rejected"
    assert any("qualifying 'dm' call(s) occurred" in r for r in res["reasons"]), res["reasons"]


def test_absence_is_not_certified_while_a_worker_is_outstanding(project, tmp_path):
    """r3 #1 measured: a worker started at the branch sent after the call phase; absence was accepted."""
    edits = [["src/bts/mod.py", SEND_THEN_DONE,
              '    if ready:\n        import threading  # W1.5 mutant: deferred to a worker\n'
              '        threading.Thread(target=lambda: (threading.Event().wait(2), transport.send("eric", "pick: Turner"))).start()\n'
              '    return "defer"']]
    res = run(project, tmp_path, mutation_edits=edits,
              branch={"path": "src/bts/mod.py", "text": "W1.5 mutant: deferred to a worker"}, killing=ASSERT_RESULT)
    assert res["verdict"] == "rejected"
    assert any("still alive" in r for r in res["reasons"]), res["reasons"]


def test_an_unexpected_exception_never_writes_an_accepted_artifact(project, tmp_path, monkeypatch):
    """Self-review 2026-09-29: the verdict was computed in ``finally`` as 'accepted when no reason was
    recorded', so an exception type outside the handled tuple, raised before any reason, wrote
    ``accepted`` to acceptance.json while it propagated."""
    from scripts.audit.incident_register import runner

    def boom(*a, **k):
        raise RuntimeError("runner exploded")

    monkeypatch.setattr(runner, "run", boom)
    repo, wt = project
    with pytest.raises(RuntimeError):
        current_defence(wt, spec(), tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "acceptance.json").read_text())
    assert saved["verdict"] == "rejected"
    assert any(r.startswith("aborted: RuntimeError") for r in saved["reasons"]), saved["reasons"]
    assert git(wt, "status", "--porcelain") == ""
