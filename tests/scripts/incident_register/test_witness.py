import importlib.util
import os

import pytest

from scripts.audit.incident_register import witness
from scripts.audit.incident_register.witness import certify, hook, load, positive_control

HERE = os.path.realpath(__file__)
RUN = "run-1"


def entry_fn(order=("branch", "boundary", "done"), reach=True):
    hook("entry")
    if not reach:
        return
    for tag in order:
        {"branch": _decision, "boundary": _boundary, "done": _done}[tag]()


def _decision():
    hook("branch")


def _boundary():
    hook("boundary")


def _done():
    hook("done")


@pytest.fixture
def wpath(tmp_path, monkeypatch):
    path = tmp_path / "witness.jsonl"
    monkeypatch.setenv(witness.ENV, str(path))
    monkeypatch.setenv(witness.RUN_ENV, RUN)
    return path


def _node():
    return os.environ["PYTEST_CURRENT_TEST"].rsplit(" (", 1)[0]


def _cert(path, kind, **kw):
    args = dict(node=_node(), run=RUN, entry="entry_fn", entry_file=HERE, branch_file=HERE, kind=kind)
    args.update(kw)
    return certify(load(path), **args)


def test_hook_is_a_noop_without_the_env(tmp_path, monkeypatch):
    monkeypatch.delenv(witness.ENV, raising=False)
    hook("anything")
    assert list(tmp_path.iterdir()) == []


def test_records_carry_sequence_process_and_file(wpath):
    entry_fn(("branch",))
    recs = load(wpath)
    assert [r["tag"] for r in recs] == ["entry", "branch"]
    assert recs[0]["seq"] < recs[1]["seq"]
    assert recs[1]["stack"][0][1] == HERE and recs[1]["run"] == RUN


def test_event_certificate_in_order(wpath):
    entry_fn(("branch", "boundary"))
    assert _cert(wpath, "event")["ok"]


def test_boundary_before_branch_is_rejected(wpath):
    """Codex phase-1 r1 #3.1: entry → boundary → branch must not certify."""
    entry_fn(("boundary", "branch"))
    got = _cert(wpath, "event")
    assert not got["ok"] and "after the branch" in got["reasons"][0]


def test_absence_certificate_requires_an_empty_event_set(wpath):
    entry_fn(("branch", "done"))
    assert _cert(wpath, "absence")["ok"]


def test_absence_with_an_event_is_rejected(wpath):
    """Codex phase-1 r1 #3.2: a completion tag alone is not proof that nothing was sent."""
    entry_fn(("branch", "boundary", "done"))
    got = _cert(wpath, "absence")
    assert not got["ok"] and any("event occurred" in r for r in got["reasons"])


def test_same_name_entry_from_another_file_is_rejected(wpath, tmp_path):
    """Codex phase-1 r1 #3.3: a substitute function (e.g. from a conftest) is not the entry."""
    other = tmp_path / "impostor.py"
    other.write_text("from scripts.audit.incident_register.witness import hook\n"
                     "def entry_fn():\n    hook('entry')\n    hook('branch')\n    hook('boundary')\n")
    spec = importlib.util.spec_from_file_location("impostor", other)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.entry_fn()
    got = _cert(wpath, "event")
    assert not got["ok"] and "not 'entry_fn'" in got["reasons"][0]


def test_disconnected_direct_call_is_rejected(wpath):
    entry_fn(reach=False)
    _decision()
    _boundary()
    assert not _cert(wpath, "event")["ok"]


def test_two_entry_invocations_are_rejected(wpath):
    entry_fn(reach=False)
    entry_fn(("branch", "boundary"))
    got = _cert(wpath, "event")
    assert not got["ok"] and "need exactly 1" in got["reasons"][0]


def test_other_node_or_run_do_not_count(wpath):
    entry_fn(("branch", "boundary"))
    assert not certify(load(wpath), node="tests/x.py::test_y", run=RUN, entry="entry_fn",
                       entry_file=HERE, branch_file=HERE, kind="event")["ok"]
    assert not _cert(wpath, "event", run="run-2")["ok"]


def test_positive_control_needs_the_event(wpath):
    entry_fn(("branch", "boundary", "done"))
    assert positive_control(load(wpath), node=_node(), run=RUN, entry="entry_fn", entry_file=HERE)["ok"]


def test_positive_control_without_the_event_is_rejected(wpath):
    entry_fn(("branch", "done"))
    got = positive_control(load(wpath), node=_node(), run=RUN, entry="entry_fn", entry_file=HERE)
    assert not got["ok"] and "did not occur" in got["reasons"][0]


def test_kind_is_validated():
    with pytest.raises(ValueError):
        certify([], node="n", run=RUN, entry="e", entry_file=HERE, branch_file=HERE, kind="other")
