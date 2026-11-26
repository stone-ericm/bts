"""pytest plugin for the strict mutation sweep (fresh whole-range review F4).

For every failed call phase it records whether the exception IS the builtin ``AssertionError`` or pytest's own
failed expectation from ``pytest.raises`` (``Failed: DID NOT RAISE``), keyed exactly as pytest's JUnit XML names the
test case (``classname::name``). The sweep classifies a kill from these records, never from message text: a class
can claim the builtin's module and name, and a message can start with ``assert ``.

Loaded with ``-p sweep_kill_identity``; writes JSON lines to ``$W15_SWEEP_IDENTITY`` when it is set.
"""
import json
import os

import pytest
from _pytest.junitxml import mangle_test_address


def junit_key(nodeid: str) -> str:
    """The ``classname::name`` pytest's JUnit XML gives this node."""
    names = mangle_test_address(nodeid)
    return ".".join(names[:-1]) + "::" + names[-1]


def kind_of(excinfo) -> str:
    if excinfo.type is AssertionError:
        return "assertion"
    if excinfo.type is pytest.fail.Exception and str(excinfo.value).startswith("DID NOT RAISE"):
        return "did_not_raise"
    return "other"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    out = os.environ.get("W15_SWEEP_IDENTITY")
    if out and call.when == "call" and call.excinfo is not None:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"node": junit_key(item.nodeid), "kind": kind_of(call.excinfo)}) + "\n")
    return rep
