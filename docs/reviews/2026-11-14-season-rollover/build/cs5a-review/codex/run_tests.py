"""Run the touched suite at two clocks, test the archived base, and mutate copies."""

import datetime as calendar
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from offline import forbid_network

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parent.parent
NEW = ["tests/test_pick_chosen_on_cs5a.py", "tests/test_saved_pick_paths_cs5a.py"]
TOUCHED = NEW + ["tests/test_live_gate_cs6a.py", "tests/test_live_gate_cs6a_review.py",
                 "tests/test_season_contract_cs4b2_edges.py"]
COEXIST = ["tests/test_lightgbm_optional.py", "tests/test_web_saver.py"]


def run_pytest(source, year, arguments):
    import pytest

    sys.path[:0] = [str(source), str(ROOT)]
    forbid_network(loopback=True)
    if year:
        from zoneinfo import ZoneInfo

        # Load extension modules holding references to CPython's original date
        # class before installing the Python-facing wall clock.
        import pandas  # noqa: F401

        real_datetime = calendar.datetime
        real_date = calendar.date
        wall = real_datetime(int(year), 7, 15, 18, 35, tzinfo=ZoneInfo("America/New_York"))

        class ReviewDateTime(real_datetime):
            @classmethod
            def now(cls, tz=None):
                when = wall.astimezone(tz)
                return cls.fromisoformat(when.isoformat() if tz is not None else when.replace(tzinfo=None).isoformat())

            @classmethod
            def utcnow(cls):
                return cls.now(calendar.timezone.utc).replace(tzinfo=None)

            @classmethod
            def today(cls):
                return cls.now()

        class ReviewDate(real_date):
            @classmethod
            def today(cls):
                return cls.fromisoformat(wall.astimezone().date().isoformat())

        calendar.datetime = ReviewDateTime
        calendar.date = ReviewDate
        print(f"Pinned clock {calendar.datetime.now().isoformat()}", flush=True)

    import bts

    assert Path(bts.__file__).resolve().is_relative_to(source.resolve()), bts.__file__
    return pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", *arguments])


SCHEDULER = "bts/scheduler.py"
CLI = "bts/cli.py"
CORE = "bts/orchestrator.py"
PICKS = "bts/picks.py"
SCHEDULER_JUDGE = "refusal = None if fresh_choice else saved_pick_refusal(daily, decision_state)"
CATCH_UP_JUDGE = "refusal = saved_pick_refusal(result.daily, decision_state)"
UNLOCKED_JUDGE = "refusal = saved_pick_refusal(daily, decision_state)"
LIFECYCLE = "tests/test_saved_pick_paths_cs5a.py"
MUTATIONS = [
    ("scheduler-no-judge", SCHEDULER, SCHEDULER_JUDGE, "refusal = None", "uncovered_preview or same_cycle"),
    ("run-no-locked-judge", CLI, CATCH_UP_JUDGE, "refusal = None", "legacy_public_reuse"),
    ("run-no-unlocked-judge", CLI, UNLOCKED_JUDGE, "refusal = None", "legacy_public_reuse"),
    ("orchestrate-no-locked-judge", CORE, CATCH_UP_JUDGE, "refusal = None", "legacy_public_reuse"),
    ("orchestrate-no-unlocked-judge", CORE, UNLOCKED_JUDGE, "refusal = None", "legacy_public_reuse"),
    ("scheduler-no-decision-check", SCHEDULER, "selection is not None and selection.decision is not None",
     "selection is not None", "same_cycle"),
    ("scheduler-no-identity-check", SCHEDULER, " and selection.pick_result.daily is daily", "", "same_cycle or overrun"),
    ("judge-accepts-optional", PICKS, 'elif not record["required"]:', 'elif False:', "pure_matcher or uncovered_preview"),
    ("judge-dated-kind-only", PICKS, '(live_kind == "dated" and record["dated"] == live_dated)',
     '(live_kind == "dated")', "pure_matcher or failed_refresh"),
    ("judge-compares-automatic-streak", PICKS, 'live_kind == "automatic" or (live_kind',
     '(live_kind == "automatic" and record["streak"] == decision_state.streak) or (live_kind',
     "pure_matcher or automatic_gap or failed_refresh"),
    ("core-stamps-reuse", CORE, "sel.decision is not None and sel.pick_result is not None", "sel.pick_result is not None",
     "real_selection_reuse"),
    ("run-stamps-reuse", CLI, "if sel.decision is not None:\n        daily.chosen_on = chosen_on_record(decision_state, required=True)",
     "if True:\n        daily.chosen_on = chosen_on_record(decision_state, required=True)", "legacy_public_reuse"),
    ("preview-overwrites-reuse", CLI, "if sel.decision is not None:\n        daily.chosen_on = chosen_on_record(decision_state, required=False)",
     "if True:\n        daily.chosen_on = chosen_on_record(decision_state, required=False)", "preview_resaves"),
    ("validator-accepts-boolean-counts", PICKS,
     "or type(streak) is not int or streak < 0", "or not isinstance(streak, int) or streak < 0",
     "pure_matcher or invalid_record"),
    ("raising-alert-propagates", SCHEDULER, "except Exception as alert_error:\n                print(f\"  Contest-state alert failed: {alert_error}\", file=sys.stderr)",
     "except Exception:\n                raise", "non_object_health"),
    ("old-missed-pick-text", SCHEDULER, "if _contest_state_required(config):\n        timing =", "if False:\n        timing =", "missed_pick"),
]


def test_batch():
    guard_directory = REVIEW / "matplotlib"
    guard_directory.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(guard_directory)
    os.environ["PYTHONHASHSEED"] = "0"
    results = {}

    def run(label, source, year, args):
        invocation = [sys.executable, str(Path(__file__).resolve()), "child", str(source), str(year), *args]
        with (REVIEW / f"{label}.log").open("w") as output:
            result = subprocess.run(invocation, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=False)
        results[label] = result.returncode
        print(f"{label}: {result.returncode}", flush=True)

    run("clock-2027", ROOT / "src", 2027, TOUCHED)
    run("clock-2028", ROOT / "src", 2028, TOUCHED)
    run("together", ROOT / "src", 0, TOUCHED + COEXIST)
    run("new-on-base", REVIEW / "base" / "src", 0, ["-rA", *NEW])
    with tempfile.TemporaryDirectory(dir=REVIEW) as temp:
        copy = Path(temp) / "src"
        shutil.copytree(ROOT / "src", copy, ignore=shutil.ignore_patterns("__pycache__"))
        for label, file, old, new, selector in MUTATIONS:
            target = copy / file
            original = target.read_text()
            assert original.count(old) == 1, (label, original.count(old))
            target.write_text(original.replace(old, new))
            try:
                run(f"mutation-{label}", copy, 0, ["-k", selector, *NEW])
            finally:
                target.write_text(original)
    if any(results[name] for name in ("clock-2027", "clock-2028", "together")):
        return 1
    return 0


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "child":
        raise SystemExit(run_pytest(Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]))
    if sys.argv[1:] == ["coexist"]:
        raise SystemExit(run_pytest(ROOT / "src", 0, TOUCHED + COEXIST))
    raise SystemExit(test_batch())
