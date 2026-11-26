"""Exercise slate-date delivery, receipt behavior and runner interaction offline."""

import contextlib
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.audit.season_walk.walk import ET, _set_clock, offline
from bts import picks, scheduler
from bts.contest_state import ContestStateError
from bts.health import alert, runner


MODE = sys.argv[1]
SCRATCH = MODE == "scratch"
logging.disable(logging.CRITICAL)


def local(day, hm):
    return datetime.fromisoformat(f"{day}T{hm}").replace(tzinfo=ET)


def dms_at(case, root, day, at, cfg):
    if case == "contest_state":
        kwargs = {"slate_date": day} if SCRATCH else {}
        scheduler._alert_contest_state_failure(cfg, ContestStateError("switch unreadable"), **kwargs)
        return
    batter = picks.Pick(batter_name="Test Batter", batter_id=1, team="CLE",
                        lineup_position=1, pitcher_name="Other Player", pitcher_id=2,
                        p_game_hit=.8, flags=[], game_pk=1, projected_lineup=False,
                        game_time=(at + timedelta(minutes=30)).isoformat())
    daily = picks.DailyPick(day, at.isoformat(), batter, None, None)
    if case == "missed_pick":
        picks.save_pick(daily, root)
    state = scheduler.SchedulerState(day, at.isoformat(), [], [], [], False, None, None, None)
    if case == "missed_pick":
        with patch.object(scheduler, "_watchdog_ping_sleep", lambda sec: _set_clock(at + timedelta(seconds=sec))):
            scheduler._maybe_alert_missed_pick(cfg, day, root, 10, None, state)
    elif case == "late_delivery":
        scheduler._refuse_late_delivery(daily, cfg, root, state, day, "fallback", at - timedelta(minutes=1), at)
    else:
        scheduler._refuse_void_game_delivery(daily, cfg, root, state, day, "fallback", {1: "Cancelled"})


@contextlib.contextmanager
def world():
    with TemporaryDirectory() as name, offline(local("2027-06-14", "20:00")):
        root = Path(name) / "data/picks"
        root.mkdir(parents=True)
        record = root.parent / "health_state/health_dm_delivery_status.json"
        cfg = {"orchestrator": {"picks_dir": str(root)}, "bluesky": {"dm_recipient": "op.invalid"},
               "scheduler": {"contest_state_required": True}}
        sent = []
        with patch.object(alert, "send_dm", lambda _, body: sent.append(body) or "wire"):
            yield root, record, cfg, sent


def sequences(days):
    d, next_day = days
    for case in ("contest_state", "missed_pick", "late_delivery", "postponed_pick"):
        for name, steps in (
            ("E4/E5/E6", [(d, next_day, "00:02"), (next_day, next_day, "08:00")]),
            ("pre/post-midnight same slate", [(d, d, "23:30"), (d, next_day, "00:02")]),
            ("new then old then new", [(next_day, next_day, "00:01"), (d, next_day, "00:02"),
                                       (next_day, next_day, "08:00")]),
        ):
            with world() as (root, record, cfg, sent):
                rows = []
                for slate, wall, hm in steps:
                    at = local(wall, hm)
                    _set_clock(at)
                    prev = len(sent)
                    dms_at(case, root, slate, at, cfg)
                    status = json.loads(record.read_text())
                    rows.append([len(sent) - prev, status["sent_day"]])
                if name == "E4/E5/E6":
                    assert [r[0] for r in rows] == ([1, 1] if SCRATCH else [1, 0])
                print(next_day, case, name, rows)



for case in ("missed_pick", "late_delivery", "postponed_pick"):
    for day in ("bad", "", None):
        with world() as (root, record, cfg, sent):
            try:
                dms_at(case, root, day, local("2027-06-14", "20:00"), cfg)
            except TypeError as exc:
                assert day is None and not sent
                print("S7 guard", case, repr(day), "raises before dispatch", repr(exc))
            else:
                assert day is not None and len(sent) == 1
                assert json.loads(record.read_text())["sent_day"] == "2027-06-14"
                print("S7 guard", case, repr(day), "wall date")
