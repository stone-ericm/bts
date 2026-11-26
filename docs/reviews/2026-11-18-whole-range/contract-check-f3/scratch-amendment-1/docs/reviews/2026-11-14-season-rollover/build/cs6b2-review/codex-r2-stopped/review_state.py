"""Capture unexpected DM calls and check process state across test teardown."""

import os
import signal
import sys
from pathlib import Path

import pytest


def pytest_configure(config):
    from bts import dm
    from bts.health import alert

    def unexpected_send(*args, **kwargs):
        pytest.fail(f"DM reached the review transport guard: {args!r}")

    dm.send_dm = alert.send_dm = unexpected_send


def state(strategy):
    active, interval = signal.getitimer(signal.ITIMER_REAL)
    umask = os.umask(0)
    os.umask(umask)
    return (
        os.getcwd(), os.open, Path.stat, umask, sys.getrecursionlimit(),
        sys.get_int_max_str_digits(), signal.getsignal(signal.SIGALRM),
        bool(active), interval, id(strategy._mdp_cache),
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item):
    if "cs6b2" not in item.path.name:
        return (yield)
    from bts import strategy

    before = state(strategy)
    result = yield
    after = state(strategy)
    assert before == after, (item.nodeid, before, after)
    return result
