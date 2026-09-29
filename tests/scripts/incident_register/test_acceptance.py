"""Acceptance rules against REAL pytest 9.0.2 reports produced by the evidence plugin.

Each harness runs in a pytester subprocess with the plugin loaded, so the reports and the
import record come from the process that actually ran the tests. The cases include every
counterexample Codex measured in phase-1 r1 (#4, #5).
"""
from pathlib import Path

import pytest

from scripts.audit.incident_register.acceptance import (
    accept_expected_failure,
    accept_green,
    accept_killed,
    imports_under,
    load,
    node_state,
)

pytest_plugins = ("pytester",)
ROOT = Path(__file__).resolve().parents[3]          # the repo root holding scripts/

HARNESS = '''
import pytest

class Dedicated(Exception):
    pass

def _oracle(actual, required, bad):
    if actual == required:
        return
    if actual == bad:
        raise Dedicated(f"bad {actual!r}")
    raise AssertionError(f"unexpected {actual!r}")

mark = pytest.mark.xfail(strict=True, raises=Dedicated, reason="meta")

@pytest.fixture
def teardown_breaks():
    yield
    raise ValueError("unrelated teardown failure")

@pytest.fixture
def dedicated_in_setup():
    raise Dedicated("raised during setup")

@mark
def test_intended():
    _oracle("miss", "void", "miss")

@mark
def test_wrong_location():
    raise Dedicated("fixture body failed before any production invocation")

@mark
def test_imperative():
    pytest.xfail("no oracle")

@mark
def test_teardown_fails(teardown_breaks):
    _oracle("miss", "void", "miss")

@mark
def test_in_setup(dedicated_in_setup):
    pass

@pytest.mark.xfail(strict=False, raises=Dedicated)
def test_not_strict():
    _oracle("miss", "void", "miss")

def test_passes():
    assert True

def test_fails():
    assert 1 == 2
'''


def _run(pytester, monkeypatch, files: dict[str, str], *args) -> list[dict]:
    out = pytester.path / "evidence.jsonl"
    monkeypatch.setenv("W15_EVIDENCE_OUT", str(out))
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    for name, text in files.items():
        path = pytester.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    pytester.runpytest_subprocess("-p", "scripts.audit.incident_register.evidence_plugin",
                                  "-p", "no:cacheprovider", *args)
    return load(out)


def _nid(name):
    return f"test_h.py::{name}"


@pytest.fixture
def harness_events(pytester, monkeypatch):
    return _run(pytester, monkeypatch, {"test_h.py": HARNESS})


def _accept(events, name):
    return accept_expected_failure(events, _nid(name), exc_class="Dedicated",
                                   oracle_func="_oracle", test_file="test_h.py")


def test_intended_oracle_failure_is_accepted(harness_events):
    ok, why = _accept(harness_events, "test_intended")
    assert ok, why


@pytest.mark.parametrize("name,needle", [
    ("test_wrong_location", "not in _oracle"),
    ("test_imperative", "imperative"),
    ("test_teardown_fails", "teardown"),
    ("test_in_setup", "setup"),
    ("test_not_strict", "not strict"),
])
def test_codex_counterexamples_are_rejected(harness_events, name, needle):
    ok, why = _accept(harness_events, name)
    assert not ok
    assert any(needle in w for w in why), why


def test_node_states_cover_every_phase(harness_events):
    assert node_state(harness_events, _nid("test_passes")) == "passed"
    assert node_state(harness_events, _nid("test_fails")) == "failed"
    assert node_state(harness_events, _nid("test_teardown_fails")) == "teardown_error"
    assert node_state(harness_events, _nid("test_in_setup")) == "setup_xfail"
    assert node_state(harness_events, _nid("test_intended")) == "xfail"
    assert node_state(harness_events, "test_h.py::test_absent") == "missing"


def test_green_and_kill_acceptance(harness_events):
    ok, _ = accept_green(harness_events, [_nid("test_passes"), _nid("test_intended")],
                         expected_xfail={_nid("test_intended")})
    assert ok
    ok, why = accept_green(harness_events, [_nid("test_passes"), _nid("test_fails")])
    assert not ok and "test_fails" in why[0]
    assert accept_killed(harness_events, _nid("test_fails"))[0]
    assert not accept_killed(harness_events, _nid("test_teardown_fails"))[0]


def test_import_identity_is_observed_inside_pytest(pytester, monkeypatch):
    """Codex r1 #4: a conftest that puts a foreign `bts` first must be caught."""
    events = _run(pytester, monkeypatch, {
        "src/bts/__init__.py": "WHERE = 'src'\n",
        "foreign/bts/__init__.py": "WHERE = 'foreign'\n",
        "conftest.py": ("import sys, pathlib\n"
                        "sys.path.insert(0, str(pathlib.Path(__file__).parent / 'foreign'))\n"),
        "test_i.py": "import bts\n\ndef test_x():\n    assert bts.WHERE\n",
    })
    ok, offenders = imports_under(events, pytester.path / "src")
    assert not ok and "foreign" in offenders[0]
    assert imports_under(events, pytester.path / "foreign")[0]
