"""Offline checks that the focused assertions detect representative regressions."""

from contextlib import ExitStack
from datetime import date, datetime, time, timedelta
import logging
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from bts.health import contest_state, runner
from bts.health.alert import Alert


ROOT = Path(__file__).resolve().parents[2]
clock_tests = runpy.run_path(str(ROOT / "tests/test_wr_c1_clock.py"))
history_tests = runpy.run_path(str(ROOT / "tests/test_wr_c1_history.py"))
original_path = sys.path.copy()
try:
    probe = SimpleNamespace(**runpy.run_path(str(
        ROOT / "docs/reviews/2026-11-18-whole-range/contract-check-c1/probe.py"
    )))
finally:
    sys.path[:] = original_path


def check_rejected(name, test, args, overrides):
    with TemporaryDirectory(prefix="wr-c1-mutant-") as temp:
        with ExitStack() as context:
            for obj, field, value in overrides:
                context.enter_context(patch.object(obj, field, value))
            try:
                test(Path(temp), *args)
            except AssertionError:
                print(f"Rejected {name}")
            else:
                raise AssertionError(f"Survived {name}")


clock = clock_tests["test_mode_sample_and_last_et_microsecond"]
history = history_tests["test_every_historical_contest_reason_is_exact_and_command_free"]
grace = history_tests["test_historical_contest_uses_et_seasons_and_utc_grace"]
wire = history_tests["test_historical_warn_pointer_survives_attention_decoration"]

logging.disable(logging.CRITICAL)
check_rejected("whole-second endpoint", clock, (date(2027, 6, 30),), [
    (runner, "time", SimpleNamespace(max=time(23, 59, 59))),
])

check_rejected("no UTC grace", grace, (probe, None, 2), [
    (contest_state, "timedelta", lambda **kwargs: timedelta()),
])

historical_message = contest_state.historical_message
check_rejected("command in historical alert", history, (probe,), [
    (contest_state, "historical_message",
     lambda day, reason: historical_message(day, reason + " bts set-contest-streak")),
])

attention = runner.apply_warn_attention_policy


def misplaced_label(*args, **kwargs):
    critical, warn = attention(*args, **kwargs)
    return critical, [
        Alert(item.level, item.source, item.message.replace(" (", " [", 1), item.incident_key)
        for item in warn
    ]


check_rejected("attention label after pointer", wire, (probe, False), [
    (runner, "apply_warn_attention_policy", misplaced_label),
])
