import os

import pytest

from scripts.audit.incident_register import witness
from scripts.audit.incident_register.witness import connected, hook, load


def entry_fn(reach_branch: bool, *, fire_boundary: bool = True):
    hook("entry")
    if not reach_branch:
        return "early"
    _decision()
    if fire_boundary:
        _boundary()
    hook("done")
    return "ran"


def _decision():
    hook("branch")


def _boundary():
    hook("boundary")


@pytest.fixture
def wpath(tmp_path, monkeypatch):
    path = tmp_path / "witness.jsonl"
    monkeypatch.setenv(witness.ENV, str(path))
    return path


def _node():
    return os.environ["PYTEST_CURRENT_TEST"].rsplit(" (", 1)[0]


def test_hook_is_a_noop_without_the_env(tmp_path, monkeypatch):
    monkeypatch.delenv(witness.ENV, raising=False)
    hook("anything")
    assert list(tmp_path.iterdir()) == []


def test_hook_records_the_caller_stack_and_node(wpath):
    entry_fn(True)
    recs = load(wpath)
    assert [r["tag"] for r in recs] == ["entry", "branch", "boundary", "done"]
    branch = recs[1]
    assert branch["stack"][0][0] == "_decision"
    assert branch["stack"][1][0] == "entry_fn"
    assert branch["node"].endswith("(call)")


def test_connected_boundary_within_one_invocation(wpath):
    entry_fn(True)
    got = connected(load(wpath), node=_node(), entry="entry_fn", branch_tag="branch",
                    boundary_tag="boundary")
    assert got["ok"], got


def test_connected_completion_for_a_missing_event(wpath):
    entry_fn(True, fire_boundary=False)
    recs = load(wpath)
    assert connected(recs, node=_node(), entry="entry_fn", branch_tag="branch",
                     completion_tag="done")["ok"]
    assert not connected(recs, node=_node(), entry="entry_fn", branch_tag="branch",
                         boundary_tag="boundary")["ok"]


def test_disconnected_direct_call_is_rejected(wpath):
    """r2 counterexample: the entry returns early, then the test calls the helper directly."""
    entry_fn(False)
    _decision()
    _boundary()
    got = connected(load(wpath), node=_node(), entry="entry_fn", branch_tag="branch",
                    boundary_tag="boundary")
    assert not got["ok"] and "not under" in got["reason"]


def test_two_entry_invocations_are_rejected(wpath):
    entry_fn(False)
    entry_fn(True)
    got = connected(load(wpath), node=_node(), entry="entry_fn", branch_tag="branch",
                    boundary_tag="boundary")
    assert not got["ok"] and "need exactly 1" in got["reason"]


def test_records_from_another_node_do_not_count(wpath):
    entry_fn(True)
    got = connected(load(wpath), node="tests/other.py::test_x", entry="entry_fn",
                    branch_tag="branch", boundary_tag="boundary")
    assert not got["ok"]


def test_exactly_one_second_tag_kind_required():
    with pytest.raises(ValueError):
        connected([], node="n", entry="e", branch_tag="b")
    with pytest.raises(ValueError):
        connected([], node="n", entry="e", branch_tag="b", boundary_tag="x", completion_tag="y")
