"""Pytest plugin to verify that CS6b-2 test modules leave zero process-wide state changed."""

from __future__ import annotations

import os
import signal
import sys

import pytest

AUDITED_MODULES = {
    "test_alert_status_cs6b2.py",
    "test_alert_status_callers_cs6b2.py",
    "test_alert_status_cs6b2_review.py",
}

_snapshots: dict[str, dict] = {}


def _take_snapshot() -> dict:
    from bts import strategy

    return {
        "cwd": os.getcwd(),
        "recursion_limit": sys.getrecursionlimit(),
        "sigalrm": signal.getsignal(signal.SIGALRM),
        "itimer_real": signal.getitimer(signal.ITIMER_REAL),
        "mdp_cache_is_none": strategy._mdp_cache is None,
        "mdp_cache_id": id(strategy._mdp_cache),
    }


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    mod_name = item.fspath.basename
    prev_mod = getattr(item.session, "_last_audited_mod", None)
    if mod_name in AUDITED_MODULES and mod_name not in _snapshots:
        _snapshots[mod_name] = _take_snapshot()
    yield
    next_mod = nextitem.fspath.basename if nextitem is not None else None
    if mod_name in AUDITED_MODULES and next_mod != mod_name:
        before = _snapshots.pop(mod_name)
        after = _take_snapshot()
        assert after["cwd"] == before["cwd"], f"{mod_name} leaked cwd: {before['cwd']} -> {after['cwd']}"
        assert after["recursion_limit"] == before["recursion_limit"], (
            f"{mod_name} leaked recursionlimit: {before['recursion_limit']} -> {after['recursion_limit']}"
        )
        assert after["sigalrm"] == before["sigalrm"], f"{mod_name} leaked SIGALRM handler"
        assert after["itimer_real"] == (0.0, 0.0), f"{mod_name} left ITIMER_REAL armed: {after['itimer_real']}"
        assert after["mdp_cache_is_none"] == before["mdp_cache_is_none"] and (
            after["mdp_cache_id"] == before["mdp_cache_id"]
        ), f"{mod_name} leaked strategy._mdp_cache: {before} -> {after}"
    item.session._last_audited_mod = mod_name
