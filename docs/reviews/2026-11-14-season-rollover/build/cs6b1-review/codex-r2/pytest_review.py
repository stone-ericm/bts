"""Run the review's serial pytest families under the supplied yield wrapper."""

from contextlib import ExitStack
import datetime as calendar
import os
from pathlib import Path
import subprocess
import sys
import time
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo

from review_support import REVIEW, ROOT, archive, environment

AUTHOR = "tests/test_morning_state_cs6b1.py"
LEAD = "tests/test_morning_state_cs6b1_review.py"
CARBON = "tests/test_morning_state_cs6b1_carbon.py"
MORNING = [AUTHOR, LEAD, CARBON]
COMBINED = MORNING + [
    "tests/test_lightgbm_optional.py",
    "tests/test_web_saver.py",
    "tests/test_saved_pick_paths_cs5a.py",
    "tests/test_stale_delivery_cs4c.py",
    "tests/test_stale_delivery_cs4c_carbon.py",
]


class Observe:
    def __init__(self, pytest, year):
        self.year = year
        self.pytest_runtest_protocol = pytest.hookimpl(wrapper=True)(observe_cache)

    def pytest_collection_finish(self, session):
        if not self.year:
            return
        instant = calendar.datetime(self.year, 7, 15, 12, tzinfo=ZoneInfo("America/New_York"))

        class Clock(calendar.datetime):
            @classmethod
            def now(cls, tz=None):
                return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

            @classmethod
            def utcnow(cls):
                return instant.astimezone(calendar.UTC).replace(tzinfo=None)

        class Date(calendar.date):
            @classmethod
            def today(cls):
                return instant.date()

        bindings = ExitStack()
        session.config.add_cleanup(bindings.close)
        for name, module in tuple(sys.modules.items()):
            if module is None or not name.startswith(("bts.", "tests.")):
                continue
            for attribute, value in tuple(vars(module).items()):
                if value is calendar.datetime:
                    bindings.enter_context(patch.object(module, attribute, Clock))
                elif value is calendar.date:
                    bindings.enter_context(patch.object(module, attribute, Date))
        bindings.enter_context(patch.object(time, "time", instant.timestamp))
        bindings.enter_context(patch.object(time, "time_ns", lambda: int(instant.timestamp() * 1e9)))
        print(f"Simulated Python wall clock {instant.isoformat()}", flush=True)


def model_cache():
    strategy = sys.modules.get("bts.strategy")
    return getattr(strategy, "_mdp_cache", None)


def observe_cache(item, nextitem):
    before = model_cache()
    result = yield
    after = model_cache()
    if before is not after and "test_morning_state_cs6b1" in item.nodeid:
        print(f"MDP cache after teardown: {item.nodeid}, {type(before).__name__} -> {type(after).__name__}")
    return result


def pytest_child(year, tests):
    import pytest
    from bts import scheduler

    assert Path(scheduler.__file__).is_relative_to(os.environ["CS6B1R2_SOURCE"])
    return pytest.main(["-p", "no:cacheprovider", "-q", "-rA", "--tb=short", "-s", *tests],
                       plugins=[Observe(pytest, year)])


def run(name, source, tests, *, year=0, expect_failure=None):
    env = environment(source, name)
    log = REVIEW / f"{name}.log"
    with log.open("w") as output:
        result = subprocess.run([sys.executable, __file__, "child", str(year), *tests],
                                cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    rendered = log.read_text()
    summary = [line.strip("= ") for line in rendered.splitlines()
               if line.startswith("=") and " in " in line]
    print(f"{name} rc={result.returncode}: {summary[-1:]}", flush=True)
    assert not Path(env["CS6B1R2_DENIED"]).exists(), name
    expected_rc = 1 if expect_failure else 0
    assert result.returncode == expected_rc, (name, result.returncode, summary)
    if expect_failure:
        assert any(line.startswith("FAILED ") and expect_failure in line
                   for line in rendered.splitlines()), name
        assert "KeyboardInterrupt:" not in "\n".join(summary), name


def replace_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after, 1)


def mutants(source):
    for name, function, clause, target in (
        ("failure-stderr", "def _report_morning_check_failure", "except Exception:", "failure_print"),
        ("advisory-stderr", "def _print_morning_check", "except Exception as error:", "advisory_print"),
    ):
        start = source.index(function)
        end = source.index("\n\ndef ", start + 1)
        text = replace_once(source[start:end], clause, clause.replace("Exception", "BaseException"))
        yield name, source[:start] + text + source[end:], target
    yield "outer", replace_once(
        source, "except Exception as error:\n        _report_morning_check_failure(error)\n\n    # 4.",
        "except BaseException as error:\n        _report_morning_check_failure(error)\n\n    # 4.",
    ), "test_an_interrupt_during_the_morning_work_propagates"
    yield "decision-recovery", replace_once(
        source, 'except Exception:  # a malformed record is "nothing found" (review B2)',
        "except (OSError, ValueError):",
    ), "nested_decision"
    yield "pick-recovery", replace_once(source, "except Exception:\n        daily = None\n",
        "except (OSError, ValueError, TypeError, KeyError, AttributeError):\n        daily = None\n",
    ), "nested_pick"
    yield "dispatch-context", replace_once(source, "    if refused is not None:\n", ""), (
        "test_a_failed_send_does_not_republish_the_refusal"
    )
    for name, marker, target in (
        ("day-critical", "contest_morning_check_dm_delivery_status.json", "[refusal]"),
        ("day-warn", "contest_stale_reading_dm_delivery_status.json", "[stale]"),
    ):
        start = source.index("    refused = None")
        end = source.index("\ndef ", start)
        original = f'status_path=health_dir / "{marker}",'
        start = source.index(original, start, end) + len(original)
        altered = source[:start] + source[start:].replace("\n" + (
            "                " if name == "day-critical" else "            "
        ) + "now_et_date=now.date(),", "", 1)
        assert altered != source
        yield name, altered, target
    yield "double-read", replace_once(
        source, "    record = automatic_reading_record(state, now)\n",
        "    state = load_decision_streak_state(picks_dir, require_contest_state=True, now=now)\n"
        "    record = automatic_reading_record(state, now)\n",
    ), "test_an_accepted_morning_reads_the_state_once"


def matrix():
    with TemporaryDirectory(prefix="pytest-", dir=REVIEW) as scratch:
        scratch = Path(scratch)
        old = archive("0e0c49a3", scratch / "round1")
        run("round1-red", old, MORNING, expect_failure="test_a_failed_send_does_not_republish_the_refusal")
        scratch_head = archive("7efc3cea", scratch / "mutants")
        path = scratch_head / "bts/scheduler.py"
        clean = path.read_text()
        for name, mutation, target in mutants(clean):
            path.write_text(mutation)
            run(f"mutation-{name}", scratch_head, MORNING, expect_failure=target)
    for year in (2027, 2028):
        run(f"wall-{year}", ROOT / "src", MORNING, year=year)


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "child":
        sys.exit(pytest_child(int(args[0]), args[1:]))
    if mode == "matrix":
        matrix()
    else:
        assert mode == "combined"
        run("combined-detail", ROOT / "src", COMBINED)
