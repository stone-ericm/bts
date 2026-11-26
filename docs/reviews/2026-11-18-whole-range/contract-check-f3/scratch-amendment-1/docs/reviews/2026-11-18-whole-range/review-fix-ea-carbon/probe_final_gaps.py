"""Close final two checks:
1. Real morning-route vs delivery-route interrupt containment on 63669c42 and HEAD.
2. Check whether boolean-streak and negative-streak mutations in format_skip_summary
   survive across ALL skip test files (not just test_wr_e_texts.py).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_SRC = HERE / "base_63669c42" / "src"

INTERRUPT_WORKER = """
import contextlib, io, json, os, sys, tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, os.environ["REPO_ROOT"])
sys.path.insert(0, os.environ["TARGET_SRC"])

import bts
from bts import contest_state, scheduler
from bts.health import alert
from scripts.audit.season_walk import walk

ET = ZoneInfo("America/New_York")
now = datetime(2027, 7, 15, 10, 0, tzinfo=ET)
st_dec = contest_state.DecisionStreakState(
    10, True, True, "contest", "fresh", 0, True,
    observation_kind="automatic", contest_recorded_at="2027-07-14T00:00:00Z",
)
rec = contest_state.automatic_reading_record(st_dec, now)
out = {"bts_file": str(Path(bts.__file__).resolve()), "cases": {}}

for exc_cls in (RuntimeError, KeyboardInterrupt, SystemExit):
    called = {"deliv": 0, "morn": 0}
    def raise_deliv(*a, exc_cls=exc_cls, **k):
        called["deliv"] += 1
        raise exc_cls("deliv boom")
    def raise_morn(*a, exc_cls=exc_cls, **k):
        called["morn"] += 1
        raise exc_cls("morn boom")

    with tempfile.TemporaryDirectory() as tmp:
        pdir = Path(tmp) / "picks"
        pdir.mkdir()
        deliv_escaped = None
        with walk.offline(now), \\
                patch.object(scheduler, "_now_et", lambda: now), \\
                patch.object(alert, "dispatch_dm_for_health_alerts", raise_deliv), \\
                contextlib.redirect_stderr(io.StringIO()):
            try:
                scheduler._warn_stale_contest_reading(rec, {"bluesky": {"dm_recipient": "op.test"}}, pdir)
            except BaseException as e:
                deliv_escaped = type(e).__name__

        morn_escaped = None
        cfg = walk._config(pdir)
        cfg["bluesky"] = {"dm_recipient": "op.test"}
        cfg["health_checks"]["enabled"] = False
        cfg["orchestrator"]["heartbeat_path"] = str(Path(tmp) / ".hb")
        game = {
            "gamePk": 70001, "gameType": "R", "gameDate": "2027-07-15T23:10:00Z",
            "teams": {"away": {"team": {"id": 1}}, "home": {"team": {"id": 2}}},
        }
        with walk.offline(now), \\
                patch.object(scheduler, "_now_et", lambda: now), \\
                patch.object(scheduler, "fetch_schedule", lambda d: [game] if d == "2027-07-15" else []), \\
                patch.object(scheduler, "_next_day_wakeup", lambda *a: now + timedelta(days=1)), \\
                patch.object(contest_state, "load_decision_streak_state", lambda *a, **k: st_dec), \\
                patch.object(alert, "dispatch_dm_for_health_alerts", raise_morn), \\
                patch.object(scheduler, "_watchdog_ping_sleep", lambda _s: None), \\
                patch.object(scheduler, "_idle_until_next_wakeup", lambda *a: None), \\
                contextlib.redirect_stderr(io.StringIO()):
            try:
                scheduler.run_day("2027-07-15", cfg)
            except BaseException as e:
                morn_escaped = type(e).__name__
        assert called == {"deliv": 1, "morn": 1}, called
        out["cases"][exc_cls.__name__] = {
            "delivery_escaped": deliv_escaped,
            "morning_run_day_escaped": morn_escaped,
        }
print(json.dumps(out))
"""

PYTEST_RUNNER = """
import os, sys
from pathlib import Path
scratch = Path(os.environ["SCRATCH_SRC"]).resolve()
sys.path.insert(0, str(scratch))
import bts
assert Path(bts.__file__).resolve().is_relative_to(scratch)
import pytest
raise SystemExit(pytest.main([
    "-q", "-x", "--no-header",
    "tests/test_wr_e_texts.py",
    "tests/test_scheduler_skip_visibility.py",
    "tests/test_scheduler_tail_objective.py",
    "tests/scripts/season_walk/",
]))
"""


def main() -> None:
    # Part 1: Interrupt containment parity
    interrupts = {}
    for label, src in (("base", BASE_SRC), ("head", REPO / "src")):
        env = dict(os.environ, REPO_ROOT=str(REPO), TARGET_SRC=str(src.resolve()))
        r = subprocess.run(
            [sys.executable, "-c", INTERRUPT_WORKER],
            cwd=REPO, env=env, check=True, capture_output=True, text=True, timeout=30,
        )
        interrupts[label] = json.loads(r.stdout)

    assert interrupts["base"]["bts_file"] != interrupts["head"]["bts_file"]
    assert interrupts["base"]["cases"] == interrupts["head"]["cases"] == {
        "RuntimeError": {"delivery_escaped": None, "morning_run_day_escaped": None},
        "KeyboardInterrupt": {"delivery_escaped": None, "morning_run_day_escaped": "KeyboardInterrupt"},
        "SystemExit": {"delivery_escaped": None, "morning_run_day_escaped": "SystemExit"},
    }

    # Part 2: Check bool-streak and negative-streak mutations against all skip test files
    extra_muts = [
        (
            "bool_streak_accepted",
            "if type(streak) is not int or streak < 0:",
            "if not isinstance(streak, int) or streak < 0:",
        ),
        (
            "negative_streak_accepted",
            "if type(streak) is not int or streak < 0:",
            "if type(streak) is not int:",
        ),
    ]
    mut_results = {}
    for name, old, new in extra_muts:
        with tempfile.TemporaryDirectory(dir=HERE, prefix=f"gap-{name}-") as tmp:
            scratch_src = Path(tmp) / "src"
            shutil.copytree(REPO / "src/bts", scratch_src / "bts", ignore=shutil.ignore_patterns("__pycache__"))
            p = scratch_src / "bts" / "scheduler.py"
            txt = p.read_text()
            assert txt.count(old) == 1
            p.write_text(txt.replace(old, new, 1))
            env = dict(os.environ, SCRATCH_SRC=str(scratch_src), PYTHONPATH=f"{scratch_src}{os.pathsep}{REPO}")
            res = subprocess.run(
                [sys.executable, "-c", PYTEST_RUNNER],
                cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
            )
            mut_results[name] = {
                "returncode": res.returncode,
                "survived_all_skip_suites": res.returncode == 0,
            }

    out = {"interrupt_containment": interrupts, "streak_guard_mutations": mut_results}
    (HERE / "probe_final_gaps.json").write_text(json.dumps(out, indent=2))
    print("probe_final_gaps: COMPLETED:", json.dumps(out))


if __name__ == "__main__":
    main()
