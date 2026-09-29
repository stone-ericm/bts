"""Meta-tests for the W1.5 expected-failure scheme (design §9.7) under the installed pytest.

These pin how pytest 9.0.2 itself REPORTS each shape (in-process ``inline_run`` reports). They
are documentation of pytest's behaviour, not the acceptance rule: pytest converts a matching
``raises=`` exception to XFAIL in ANY phase, setup included. The register's acceptance rule is
``scripts/audit/incident_register/acceptance.accept_expected_failure`` (call phase only, raised
inside the oracle, strict marker, clean setup and teardown, no imperative xfail), tested against
real plugin reports in ``tests/scripts/incident_register/test_acceptance.py``.
"""
from __future__ import annotations

import pytest

pytest_plugins = ("pytester",)

HARNESS = '''
import pytest

class IncidentExpectedFailure(Exception):
    pass

class Dedicated(IncidentExpectedFailure):
    pass

def _oracle(actual, required, bad):
    if actual == required:
        return
    if actual == bad:
        raise Dedicated(f"bad {actual!r}")
    raise AssertionError(f"unexpected {actual!r}")

mark = pytest.mark.xfail(strict=True, raises=Dedicated, reason="meta")

@pytest.fixture
def broken_setup():
    raise ValueError("unrelated setup failure")

@pytest.fixture
def dedicated_in_setup():
    raise Dedicated("raised during setup")

@mark
def test_intended():
    _oracle("miss", "void", "miss")

@mark
def test_fixed_behaviour():
    _oracle("void", "void", "miss")

@mark
def test_other_mismatch():
    _oracle(None, "void", "miss")

@mark
def test_unrelated_call_assertion():
    assert 1 == 2

@mark
def test_unrelated_setup_error(broken_setup):
    _oracle("miss", "void", "miss")

@mark
def test_dedicated_raised_in_setup(dedicated_in_setup):
    pass
'''


def _reports(pytester, *args):
    pytester.makepyfile(test_meta=HARNESS)
    rec = pytester.inline_run(*args)
    by_node: dict[str, list] = {}
    for rep in rec.getreports("pytest_runtest_logreport"):
        by_node.setdefault(rep.nodeid.split("::")[-1], []).append(rep)
    return rec, by_node


def _final(reps):
    """The report that decides the node: the first non-passed report, else the call report."""
    for rep in reps:
        if rep.outcome != "passed":
            return rep
    return next(r for r in reps if r.when == "call")


def test_meta_outcomes_under_the_marker(pytester):
    _rec, by_node = _reports(pytester)
    intended = _final(by_node["test_intended"])
    assert (intended.when, intended.outcome) == ("call", "skipped") and intended.wasxfail

    fixed = _final(by_node["test_fixed_behaviour"])          # XPASS(strict) -> failed
    assert (fixed.when, fixed.outcome) == ("call", "failed")
    assert "XPASS(strict)" in str(fixed.longrepr)

    other = _final(by_node["test_other_mismatch"])           # a changed defect is not absorbed
    assert (other.when, other.outcome) == ("call", "failed")
    assert not getattr(other, "wasxfail", "")

    unrelated = _final(by_node["test_unrelated_call_assertion"])
    assert (unrelated.when, unrelated.outcome) == ("call", "failed")
    assert not getattr(unrelated, "wasxfail", "")

    setup_err = _final(by_node["test_unrelated_setup_error"])  # ERROR in the summary, not XFAIL
    assert (setup_err.when, setup_err.outcome) == ("setup", "failed")
    assert not getattr(setup_err, "wasxfail", "")

    # pytest DOES convert a dedicated exception raised in setup into XFAIL (no phase check in
    # _pytest/skipping.py) — which is exactly why the register's rule rejects non-call XFAILs.
    in_setup = _final(by_node["test_dedicated_raised_in_setup"])
    assert (in_setup.when, in_setup.outcome) == ("setup", "skipped") and in_setup.wasxfail


def test_meta_summary_counts(pytester):
    pytester.makepyfile(test_meta=HARNESS)
    result = pytester.runpytest()
    result.assert_outcomes(xfailed=2, failed=3, errors=1)


def test_meta_runxfail_shows_the_dedicated_exception_in_call(pytester):
    _rec, by_node = _reports(pytester, "--runxfail")
    intended = _final(by_node["test_intended"])
    assert (intended.when, intended.outcome) == ("call", "failed")
    assert "Dedicated" in str(intended.longrepr) and "bad 'miss'" in str(intended.longrepr)
    assert not getattr(intended, "wasxfail", "")
