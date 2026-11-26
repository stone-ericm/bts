"""Pytest plugin for Round-2 review:
1. Blocks outward socket connects and non-local DNS (allows AF_UNIX and loopback).
2. Snapshots process-wide state before and after every test item (SIGALRM handler,
   ITIMER_REAL timer, os.open, os.fstat, os.ftruncate, Path.stat, Path.open,
   builtins.open, io.open, umask, cwd, sys.getrecursionlimit, sys.get_int_max_str_digits,
   and bts.strategy._mdp_cache) and records any leak.
3. Writes a machine-readable JSON summary to `BTS_SUMMARY_OUT` when set.
"""

from __future__ import annotations

import builtins
import io
import json
import os
import signal
import socket
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_GETADDRINFO = socket.getaddrinfo
_REAL_GETHOSTBYNAME = socket.gethostbyname
_REAL_GETHOSTBYADDR = socket.gethostbyaddr

try:
    _LOCAL_NAMES = {socket.gethostname(), socket.getfqdn()}
except Exception:
    _LOCAL_NAMES = set()


def _allowed_host(host: object) -> bool:
    if host is None:
        return True
    h = host.decode("ascii", "ignore") if isinstance(host, bytes) else str(host)
    return h in _LOOPBACK_HOSTS or h.startswith("127.") or h in _LOCAL_NAMES


def _install_network_guard() -> None:
    def _connect(self: socket.socket, address: object) -> object:
        if self.family == socket.AF_UNIX:
            return _REAL_CONNECT(self, address)
        host = address[0] if isinstance(address, tuple) and address else None
        if not _allowed_host(host):
            raise RuntimeError(f"review network guard blocked connect: {address!r}")
        return _REAL_CONNECT(self, address)

    def _connect_ex(self: socket.socket, address: object) -> int:
        if self.family == socket.AF_UNIX:
            return _REAL_CONNECT_EX(self, address)
        host = address[0] if isinstance(address, tuple) and address else None
        if not _allowed_host(host):
            raise RuntimeError(f"review network guard blocked connect_ex: {address!r}")
        return _REAL_CONNECT_EX(self, address)

    def _getaddrinfo(host: object, port: object, *a: object, **kw: object) -> object:
        if not _allowed_host(host):
            raise RuntimeError(f"review network guard blocked DNS: {host!r}")
        return _REAL_GETADDRINFO(host, port, *a, **kw)

    def _gethostbyname(hostname: str) -> str:
        if not _allowed_host(hostname):
            raise RuntimeError(f"review network guard blocked gethostbyname: {hostname!r}")
        return _REAL_GETHOSTBYNAME(hostname)

    def _gethostbyaddr(ip: str) -> object:
        if not _allowed_host(ip):
            raise RuntimeError(f"review network guard blocked gethostbyaddr: {ip!r}")
        return _REAL_GETHOSTBYADDR(ip)

    socket.socket.connect = _connect  # type: ignore[assignment]
    socket.socket.connect_ex = _connect_ex  # type: ignore[assignment]
    socket.getaddrinfo = _getaddrinfo  # type: ignore[assignment]
    socket.gethostbyname = _gethostbyname  # type: ignore[assignment]
    socket.gethostbyaddr = _gethostbyaddr  # type: ignore[assignment]


_install_network_guard()


def _current_umask() -> int:
    prev = os.umask(0o022)
    os.umask(prev)
    return prev


def _snapshot_state() -> dict[str, object]:
    itimer_val, itimer_int = signal.getitimer(signal.ITIMER_REAL)
    strat = sys.modules.get("bts.strategy")
    mdp_id = id(strat._mdp_cache) if strat is not None and hasattr(strat, "_mdp_cache") else id(None)
    return {
        "sigalrm": repr(signal.getsignal(signal.SIGALRM)),
        "itimer_active": (itimer_val != 0.0 or itimer_int != 0.0),
        "os_open": id(os.open),
        "os_fstat": id(os.fstat),
        "os_ftruncate": id(os.ftruncate),
        "path_stat": id(Path.stat),
        "path_open": id(Path.open),
        "builtins_open": id(builtins.open),
        "io_open": id(io.open),
        "umask": _current_umask(),
        "cwd": os.getcwd(),
        "recursionlimit": sys.getrecursionlimit(),
        "int_max_str_digits": sys.get_int_max_str_digits(),
        "mdp_cache_id": mdp_id,
    }


_COUNTS = {"passed": 0, "failed": 0, "skipped": 0}
_FAILED_ITEMS: list[dict[str, str]] = []
_LEAKS: list[dict[str, object]] = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    before = _snapshot_state()
    yield
    after = _snapshot_state()
    diffs = {}
    for k, v_before in before.items():
        v_after = after[k]
        if v_before != v_after:
            if k == "mdp_cache_id" and "cs6b2" not in item.nodeid:
                continue
            diffs[k] = {"before": v_before, "after": v_after}
    if diffs:
        _LEAKS.append({"nodeid": item.nodeid, "diffs": diffs})


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call":
        if report.passed:
            _COUNTS["passed"] += 1
        elif report.failed:
            _COUNTS["failed"] += 1
            _FAILED_ITEMS.append({
                "nodeid": report.nodeid,
                "longrepr": str(report.longrepr)[-600:],
            })
        elif report.skipped:
            _COUNTS["skipped"] += 1
    elif report.when in ("setup", "teardown"):
        if report.failed:
            _COUNTS["failed"] += 1
            _FAILED_ITEMS.append({
                "nodeid": f"{report.nodeid} ({report.when})",
                "longrepr": str(report.longrepr)[-600:],
            })
        elif report.skipped and report.when == "setup":
            _COUNTS["skipped"] += 1


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out_path = os.environ.get("BTS_SUMMARY_OUT")
    if out_path:
        et_now = datetime.now(ZoneInfo("America/New_York"))
        payload = {
            "exitstatus": int(exitstatus),
            "et_now": et_now.isoformat(),
            "et_year": et_now.year,
            "counts": _COUNTS,
            "failed_items": _FAILED_ITEMS,
            "state_leaks": _LEAKS,
        }
        Path(out_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
