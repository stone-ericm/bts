"""Reproduce Carbon slice-2 B1 (saver flag after a dated correction), B2 part 2 (post_failure on a
private_locked day) and B3 (EOD late_delivery CRITICAL repeating a daytime postponed_pick page)."""
import datetime as dt
import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.audit.season_walk import walk

ET = ZoneInfo("America/New_York")
REAL = walk.REAL_DATETIME

# B1
with tempfile.TemporaryDirectory() as t:
    picks = Path(t) / "picks"; picks.mkdir()
    d1 = date(2027, 7, 15)
    with walk.offline(REAL.combine(d1, dt.time(9), ET)):
        print("B1 setup", walk.install_state(picks, d1, walk.State("automatic", 14, 15, "active", 0.9)))
        code, out = walk._invoke("set-contest-streak", "--picks-dir", picks, "--date", d1.isoformat(),
                                 "--streak", 14, "--best-streak", 15, "--saver", "used")
        print("B1 dated write exit", code)
        from bts.contest_state import load_decision_streak_state
        s1 = load_decision_streak_state(picks, require_contest_state=True, now=REAL.combine(d1, dt.time(13), ET))
        print("B1 day1 kind", s1.observation_kind if hasattr(s1, "observation_kind") else s1.source, "saver", s1.saver_available)
        print("B1 flag after dated write", json.loads((picks / "account_state/saver_state.json").read_text())["state"])
        d2 = d1 + timedelta(days=1)
        walk._set_clock(REAL.combine(d2, dt.time(2), ET))
        with walk._fetch_stubs(14, 15, d1):
            code, out = walk._invoke("fetch-contest-streak", "--picks-dir", picks, "--expected-username", walk.ACCOUNT)
        print("B1 overnight fetch exit", code)
        walk._set_clock(REAL.combine(d2, dt.time(13), ET))
        s2 = load_decision_streak_state(picks, require_contest_state=True, now=REAL.combine(d2, dt.time(13), ET))
        print("B1 day2 saver_available", s2.saver_available, "streak", s2.streak,
              "flag", json.loads((picks / "account_state/saver_state.json").read_text())["state"])

# B2 part 2
from bts.health import post_failure
from bts.daily_decision import write_decision  # noqa: F401  (writer exists)
with tempfile.TemporaryDirectory() as t:
    picks = Path(t) / "picks"; picks.mkdir()
    day = date(2027, 3, 26)
    (picks / f"{day}.json").write_text(json.dumps({"date": str(day), "pick": {"batter_name": "X", "game_pk": 1},
                                                   "bluesky_posted": False, "bluesky_uri": None}))
    (picks / str(day)).mkdir()
    (picks / str(day) / "decision.json").write_text(json.dumps({"schema_version": "bts_daily_decision_v3",
        "date": str(day), "action": "single", "scoreable": True, "delivery_status": "private_locked"}))
    alerts = post_failure.check(picks, today=day, now=REAL.combine(day, dt.time(23), ET))
    print("B2.2 post_failure alerts", [(a.level, a.source, a.message[:80]) for a in alerts])

# B3
from bts.health import late_delivery
with tempfile.TemporaryDirectory() as t:
    picks = Path(t) / "picks"; (picks / "2027-07-15").mkdir(parents=True)
    (picks / "2027-07-15/scheduler_state.json").write_text(json.dumps({"delivery_refusals": [
        {"at": "2027-07-15T17:00:00-04:00", "label": "single", "reason": "game_not_playable",
         "void_games": {"7301": "Postponed"}}]}))
    alerts = late_delivery.check(picks, today=date(2027, 7, 15))
    print("B3 EOD alerts", [(a.level, a.source) for a in alerts])
