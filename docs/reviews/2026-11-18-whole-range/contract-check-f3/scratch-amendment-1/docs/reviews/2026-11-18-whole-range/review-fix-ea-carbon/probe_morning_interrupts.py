"""Verify real morning-route vs delivery-route interrupt containment on 63669c42 and HEAD."""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_SRC = HERE / "base_63669c42" / "src"

WORKER = """
import contextlib, io, json, os, sys, tempfile
from datetime import date, datetime
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
        cfg["orchestrator"]["heartbeat_path"] = str(Path(tmp) / ".hb")
        game = {
            "gamePk": 70001, "gameType": "R", "gameDate": "2027-07-15T23:10:00Z",
            "teams": {"away": {"team": {"id": 1}}, "home": {"team": {"id": 2}}},
        }
        with walk.offline(now), \\
                patch.object(scheduler, "_now_et", lambda: now), \\
                patch.object(scheduler, "fetch_schedule", lambda _d: [game]), \\
                patch.object(contest_state, "load_decision_streak_state", lambda *a, **k: st_dec), \\
                patch.object(alert, "dispatch_dm_for_health_alerts", raise_morn), \\
                patch.object(scheduler, "_watchdog_ping_sleep", lambda _s: None), \\
                patch.object(scheduler, "run_single_check", lambda **k: {
                    "skipped": False, "new_lineups": 0, "should_post": False,
                    "pick_result": None, "pick_name": None, "pick_p": None, "selection": None,
                }), \\
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


def main() -> None:
    results = {}
    for label, src in (("base", BASE_SRC), ("head", REPO / "src")):
        env = dict(os.environ, REPO_ROOT=str(REPO), TARGET_SRC=str(src.resolve()))
        r = subprocess.run(
            [sys.executable, "-c", WORKER],
            cwd=REPO, env=env, check=True, capture_output=True, text=True, timeout=30,
        )
        results[label] = json.loads(r.stdout)

    assert results["base"]["bts_file"] != results["head"]["bts_file"]
    assert results["base"]["cases"] == results["head"]["cases"] == {
        "RuntimeError": {"delivery_escaped": None, "morning_run_day_escaped": None},
        "KeyboardInterrupt": {"delivery_escaped": None, "morning_run_day_escaped": "KeyboardInterrupt"},
        "SystemExit": {"delivery_escaped": None, "morning_run_day_escaped": "SystemExit"},
    }
    (HERE / "probe_morning_interrupts.json").write_text(json.dumps(results, indent=2))
    print("probe_morning_interrupts: PASSED:", json.dumps(results["head"]["cases"]))


if __name__ == "__main__":
    main()
