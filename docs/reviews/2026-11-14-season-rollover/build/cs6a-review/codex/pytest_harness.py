"""Run the touched tests and isolated source mutations with audit and clock controls."""

import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import local_guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / "base"
BED = HERE / "testbed"
LIVE = "tests/test_live_gate_cs6a.py"
ENTRY = "tests/test_entry_intent.py"
OUTCOME = "tests/test_day_outcome.py"
TOUCHED = (
    LIVE, ENTRY, "tests/test_cli_integration.py", "tests/test_cli_schedule_date.py",
    "tests/test_season_contract_cs4b2_edges.py",
)


def child():
    import pandas
    import pytest

    clock = os.environ.get("CS6A_REVIEW_CLOCK")
    if clock:
        old_datetime, old_date = datetime.datetime, datetime.date
        fixed = old_datetime.fromisoformat(clock).astimezone(ZoneInfo("America/New_York"))

        class DateTimeMeta(type):
            def __instancecheck__(cls, instance):
                return isinstance(instance, old_datetime)

        class DateMeta(type):
            def __instancecheck__(cls, instance):
                return isinstance(instance, old_date)

        class DateTime(old_datetime, metaclass=DateTimeMeta):
            @classmethod
            def now(cls, tz=None):
                return fixed.replace(tzinfo=None) if tz is None else fixed.astimezone(tz)

            @classmethod
            def utcnow(cls):
                return fixed.astimezone(datetime.timezone.utc).replace(tzinfo=None)

            @classmethod
            def today(cls):
                return cls.now()

        class Date(old_date, metaclass=DateMeta):
            @classmethod
            def today(cls):
                return fixed.date()

        datetime.datetime, datetime.date = DateTime, Date
        time.time = lambda: fixed.timestamp()
        time.time_ns = lambda: int(fixed.timestamp() * 1_000_000_000)
        print(f"Wall clock {datetime.datetime.now(ZoneInfo('America/New_York')).isoformat()}", flush=True)

    # The repo's cron fixture supplies an explicit environment. Preserve its
    # throwaway HOME/PATH while carrying the guard and tested source into uv.
    real_popen = subprocess.Popen

    def popen(args, *rest, **kwargs):
        if (isinstance(args, (list, tuple)) and len(args) > 1 and
                str(args[1]).endswith("/cron-setup-hetzner.sh")):
            env = dict(kwargs["env"])
            assert env["HOME"] != str(Path.home())
            assert (Path(env["PATH"].split(":")[0]) / "crontab").is_file()
            env.update(PYTHONPATH=os.environ["PYTHONPATH"], PYTHONDONTWRITEBYTECODE="1",
                       TZ="America/New_York")
            kwargs["env"] = env
        return real_popen(args, *rest, **kwargs)

    subprocess.Popen = popen
    print(f"Source search path {os.environ['PYTHONPATH']}", flush=True)
    rc = pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", "-rA", *sys.argv[2:]])
    print(f"Network audit events {local_guard.network!r}", flush=True)
    return int(rc)


def invoke(name, root, source, args, *, clock=None):
    env = os.environ | {"PYTHONPATH": os.pathsep.join(map(str, (HERE, source, root)))}
    if clock:
        env["CS6A_REVIEW_CLOCK"] = clock
    with (HERE / f"pytest-{name}.log").open("w") as log:
        run = subprocess.run([sys.executable, __file__, "child", *args], cwd=root, env=env,
                             stdout=log, stderr=subprocess.STDOUT)
    print(f"{name} exit={run.returncode}", flush=True)
    return run.returncode


def variant(name, filename, original, replacement, tests, *, index=1):
    path = BED / filename
    initial = path.read_text()
    before, sep, after = initial.partition(original)
    for _ in range(index - 1):
        prefix, sep, after = after.partition(original)
        before += original + prefix
    assert sep, (name, original)
    path.write_text(before + replacement + after)
    try:
        return invoke(name, BED, BED / "src", tests)
    finally:
        path.write_text(initial)


def parent():
    status = {}
    for year in (2027, 2028):
        status[str(year)] = invoke(str(year), ROOT, ROOT / "src", TOUCHED,
                                   clock=f"{year}-07-15T12:00:00-04:00")
    status["day-outcome-controls"] = invoke("day-outcome-controls", ROOT, ROOT / "src",
                                           (OUTCOME, "tests/test_day_outcome_entry_sequence.py"))
    example = BED / "config" / "orchestrator.example.toml"
    example_head = example.read_bytes()
    example.write_bytes((BASE / "config" / example.name).read_bytes())
    try:
        status["base"] = invoke("base", BED, BASE / "src", (
            LIVE, ENTRY, "tests/test_cli_schedule_date.py",
            "tests/test_cli_integration.py::TestBtsRun",
            "tests/test_season_contract_cs4b2_edges.py::test_real_selection_fallbacks_orchestrator_and_clean_outcome",
        ))
    finally:
        example.write_bytes(example_head)

    cli = "src/bts/cli.py"
    entry = "src/bts/entry_intent.py"
    for name, filename, original, replacement, tests, index in (
        ("no-schedule-gate", cli, "problem = live_gate_problem(config)\n", "problem = None\n", [LIVE], 1),
        ("no-orch-gate", cli, 'problem = live_gate_problem(config, delivery_override="public")',
         "problem = None", [LIVE], 1),
        ("no-entry-gate", cli, "result = check_entry_intent(config)",
         "result = check_entry_intent(config, include_live_gate=False)", [LIVE, ENTRY], 1),
        ("no-resolver-gate", entry, "    if include_live_gate:", "    if False:", [LIVE, ENTRY], 1),
        ("ignore-intent", entry, '    if isinstance(intent, str) and intent == "enter":',
         "    if False:", [LIVE], 1),
        ("ignore-delivery", entry, '    if delivery in ("dm", "public"):',
         "    if False:", [LIVE], 1),
        ("ignore-override", cli, 'live_gate_problem(config, delivery_override="public")',
         "live_gate_problem(config)", [LIVE], 1),
        ("ignore-scheduler-shape", entry, 'return "[scheduler] is not a table"',
         "return None", [LIVE], 1),
        ("ignore-health-shape", entry, 'return "[health_checks] is not a table"',
         "return None", [LIVE], 1),
        *((f"optional-run-{n}", cli, "load_decision_streak_state(picks_path, require_contest_state=True)",
           "load_decision_streak_state(picks_path)", [LIVE], n) for n in (1, 2, 3)),
        ("reload-orchestrator", "src/bts/orchestrator.py", "    if config is None:",
         "    if True:", [LIVE], 1),
        ("gate-day-outcome", "src/bts/day_outcome.py", "check_entry_intent(config, include_live_gate=False)",
         "check_entry_intent(config)", [LIVE, ENTRY, OUTCOME, "tests/test_day_outcome_entry_sequence.py"], 1),
        ("example-scheduler-false", "config/orchestrator.example.toml", "contest_state_required = true\n",
         "contest_state_required = false\n", [LIVE], 1),
        ("example-health-false", "config/orchestrator.example.toml", "contest_state_expected = true\n",
         "contest_state_expected = false\n", [LIVE], 1),
    ):
        status[name] = variant(name, filename, original, replacement, tests, index=index)
    good = {key: 0 for key in ("2027", "2028", "day-outcome-controls")}
    good.update({key: 1 for key in status if key not in good})
    print(status, flush=True)
    return int(status != good)


if __name__ == "__main__":
    raise SystemExit(child() if len(sys.argv) > 1 and sys.argv[1] == "child" else parent())
