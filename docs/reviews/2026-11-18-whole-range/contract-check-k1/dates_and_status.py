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


sequences(("2027-06-14", "2027-06-15"))
sequences(("2027-12-31", "2028-01-01"))

with world() as (root, record, cfg, sent):
    at = local("2027-06-14", "23:42")
    _set_clock(at)
    dms_at("missed_pick", root, "2027-06-14", at, cfg)
    status = json.loads(record.read_text())
    assert status["sent_day"] == ("2027-06-14" if SCRATCH else "2027-06-15")
    assert "00:02" in status["updated_at"] or "04:02" in status["updated_at"]
    print("real wait crossing", status["sent_day"], status["updated_at"])

with world() as (root, record, cfg, sent):
    day = "2027-06-14"
    def daily(case, hm):
        at = local(day, hm)
        _set_clock(at)
        dms_at(case, root, day, at, cfg)

    daily("contest_state", "14:00")
    daily("contest_state", "15:00")
    assert len(sent) == 1  # S1
    daily("missed_pick", "16:00")
    assert len(sent) == 2  # S2
    print("S1/S2", len(sent), json.loads(record.read_text())["sent_sources"])

for case in ("missed_pick", "late_delivery", "postponed_pick"):
    for live in (True, False) if case == "missed_pick" else (True,):
        with world() as (root, record, cfg, sent):
            cfg["scheduler"]["contest_state_required"] = live
            dms_at(case, root, "2027-06-14", local("2027-06-14", "20:00"), cfg)
            assert len(sent) == 1
            print("S4", json.dumps(sent[0]))

with world() as (root, record, cfg, sent):
    at = local("2027-06-14", "20:00")
    cfg["bluesky"].clear()
    dms_at("contest_state", root, "2027-06-14", at, cfg)
    assert not sent and json.loads(record.read_text())["status"] == "skipped_no_recipient"
    cfg["bluesky"]["dm_recipient"] = "op.invalid"
    with patch.object(alert, "send_dm", side_effect=RuntimeError("line gone")):
        dms_at("contest_state", root, "2027-06-14", at, cfg)
    assert json.loads(record.read_text())["status"] == "failed"
    dms_at("contest_state", root, "2027-06-14", at, cfg)
    assert len(sent) == 1 and sent[0].startswith("⚠️ Previous health alert DM FAILED")
    print("S3/S5/S6 passed")

for day in ("bad", None, ""):
    with world() as (root, record, cfg, sent):
        dms_at("contest_state", root, day, local("2027-06-14", "20:00"), cfg)
        assert len(sent) == 1 and json.loads(record.read_text())["sent_day"] == "2027-06-14"
        print("S7", repr(day), sent[0].splitlines()[1][:70])

for shape in ("malformed", "FIFO", "directory", "interrupt during send", "after send before write", "no ID"):
    with world() as (root, record, cfg, sent):
        at = local("2027-06-14", "20:00")
        record.parent.mkdir()
        if shape == "malformed":
            record.write_text("{")
        elif shape == "FIFO":
            os.mkfifo(record)
        elif shape == "directory":
            record.mkdir()
        try:
            if shape in ("interrupt during send", "after send before write"):
                target = "send_dm" if shape == "interrupt during send" else "_persist_health_dm_delivery_status"
                with patch.object(alert, target, side_effect=KeyboardInterrupt):
                    dms_at("contest_state", root, "2027-06-14", at, cfg)
                raise AssertionError("interrupt swallowed")
            if shape == "no ID":
                with patch.object(alert, "send_dm", return_value=None):
                    dms_at("contest_state", root, "2027-06-14", at, cfg)
                assert json.loads(record.read_text())["status"] == "sent"
        except KeyboardInterrupt:
            assert not record.exists()
        dms_at("contest_state", root, "2027-06-14", at, cfg)
        assert len(sent) == {"after send before write": 2, "no ID": 0}.get(shape, 1)
        print("standing", shape, len(sent))

with world() as (root, record, cfg, sent):
    day = "2027-06-14"
    at = local("2027-06-15", "00:02")
    _set_clock(at)
    account = root / "account_state"
    account.mkdir()
    (account / "contest_streak.json").write_text(json.dumps({
        "active_streak": 12, "source_date": "2027-06-13",
    }))
    (account / "season_contract.json").write_text("{")
    from bts.contest_state import load_decision_streak_state
    try:
        load_decision_streak_state(root, require_contest_state=True)
    except ContestStateError as error:
        assert "corrupt season switch" in str(error)
        kwargs = {"slate_date": day} if SCRATCH else {}
        scheduler._alert_contest_state_failure(cfg, error, **kwargs)
    # While the old run completes, a distinct bad observation appears. Isolate the
    # real contest check and attention/dispatch from unrelated local health checks.
    (account / "contest_streak.json").write_text("[]")
    checked = runner._safe_run
    with patch.object(runner, "_safe_run", lambda name, fn: checked(name, fn) if name == "contest_state" else []):
        alerts = runner.run_all_checks(root, root.parent / "models", "op.invalid",
                                       today=datetime.fromisoformat(day).date(), contest_state_expected=True)
    assert any(a.source == "contest_state" and "Prior slate" in a.message and "contest streak file is malformed"
               in a.message for a in alerts)
    print("EOD changed reason", len(sent), json.loads(record.read_text())["sent_day"],
          [a.message for a in alerts if a.source == "contest_state"])
    # The base's first attempted long DM need not clear the project's limit in
    # this test; run a separate short same-source control below.

with world() as (root, record, cfg, sent):
    day = "2027-06-14"
    at = local("2027-06-15", "00:02")
    _set_clock(at)
    dms_at("late_delivery", root, day, at, cfg)
    # A second ordinary refusal adds distinct detail to the same EOD source.
    state = scheduler.load_state(day, root)
    state.delivery_refusals.append({**state.delivery_refusals[0], "batter": "Second Batter"})
    scheduler.save_state(state, root)
    checked = runner._safe_run
    with patch.object(runner, "_safe_run", lambda name, fn: checked(name, fn) if name == "late_delivery" else []):
        alerts = runner.run_all_checks(root, root.parent / "models", "op.invalid",
                                       today=datetime.fromisoformat(day).date())
    assert any(a.source == "late_delivery" and "Second Batter" in a.message for a in alerts)
    assert len(sent) == (1 if SCRATCH else 2)
    print("EOD real late-delivery detail", len(sent), [a.message for a in alerts if a.source == "late_delivery"])
