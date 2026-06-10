"""Systemic guard against the unwrapped-long-sleep bug class (audit O1).

scheduler.py runs under systemd `WatchdogSec=1800`; any `time.sleep()` that
isn't fed by notify_watchdog SIGABRT-kills the daemon, and any sleep that
doesn't refresh the heartbeat trips the external check_heartbeat monitor. Five
bugs of this class shipped Apr 2026. Long sleeps must go through the approved
wrappers, which ping the watchdog (and, where needed, keep the heartbeat fresh).
"""
import ast
from pathlib import Path

_SCHEDULER = Path(__file__).resolve().parents[1] / "src/bts/scheduler.py"

# Functions permitted to call bare time.sleep() — they ARE the wrappers.
_ALLOWED = {"_watchdog_ping_sleep", "_poll_interval_sleep"}


def _enclosing_function(funcs, lineno):
    best = None
    for f in funcs:
        if f.lineno <= lineno <= (f.end_lineno or f.lineno):
            if best is None or f.lineno > best.lineno:
                best = f
    return best.name if best else "<module>"


def test_no_unwrapped_time_sleep_in_scheduler():
    tree = ast.parse(_SCHEDULER.read_text())
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    offenders = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "sleep"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "time"):
            fn = _enclosing_function(funcs, node.lineno)
            if fn not in _ALLOWED:
                offenders.append((fn, node.lineno))

    assert not offenders, (
        f"Unwrapped time.sleep() in scheduler.py at {offenders}. Long sleeps must "
        f"go through _watchdog_ping_sleep / _poll_interval_sleep (which ping "
        f"systemd's WatchdogSec) or the daemon is SIGABRT-killed mid-sleep."
    )
