"""Check restoration around each authored test, including fixture teardown."""

import os
import signal
import sys

import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    if item.path.name not in {"test_alert_status_cs6b2.py", "test_alert_status_callers_cs6b2.py"}:
        return (yield)

    from bts import strategy

    def state():
        timer = signal.getitimer(signal.ITIMER_REAL)
        return (id(strategy._mdp_cache), os.getcwd(), sys.getrecursionlimit(),
                sys.get_int_max_str_digits(), signal.getsignal(signal.SIGALRM), bool(timer[0]), timer[1])

    before = state()
    result = yield
    after = state()
    assert before == after, (item.nodeid, before, after)
    return result
