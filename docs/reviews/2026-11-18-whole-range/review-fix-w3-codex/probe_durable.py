"""Durable recovery, both before and inside the added fallback lookups."""

import json
import os
from pathlib import Path
import sys

from bts import daily_decision, picks, scheduler

from review_world import DAY, at, events, health, offline, pending_window


IS_BASE = "base/src" in scheduler.__file__
EXPECT_ALERT = IS_BASE or "scratch/repair/src" in scheduler.__file__


def snapshot(world, *, game_pk=101, status="private_locked"):
    daily = picks.load_pick(DAY, world.picks)
    primary = scheduler._row_from_daily(daily.pick)
    if game_pk != 101:
        primary = dict(primary, game_pk=game_pk, batter_id=999, batter_name="Other Commit")
    assert daily_decision.write_decision(
        DAY, world.picks, action="double" if daily.double_down else "single",
        source="mdp", primary=primary, double_down=scheduler._row_from_daily(daily.double_down),
        delivery_status=status, scoreable=True,
    )
    return daily_decision.decision_path(DAY, world.picks).read_bytes()


def during_poll(case):
    with offline(at(18, 25)) as (world, mp):
        world.dated(0)
        pending_window(world, mp, scheduler)
        starts = []
        poll_calls = []
        original_record = None
        arrival = None
        stage = None
        real_check = scheduler.run_single_check
        check_returned = False
        path = daily_decision.decision_path(DAY, world.picks)

        def check(*args, **kwargs):
            nonlocal check_returned
            result = real_check(*args, **kwargs)
            if not check_returned:
                assert result["pick_result"].daily.pick.game_pk == 101
            check_returned = True
            return result

        def publish():
            nonlocal original_record, arrival
            arrival = world.now
            daily = picks.load_pick(DAY, world.picks)
            if case in {"attempt-deliver", "attempt-defer", "locked-unconfirmed",
                        "attempt-invalid-record", "attempt-invalid-action"}:
                daily.delivery_attempted = True
                picks.save_pick(daily, world.picks)
            if case == "locked-unconfirmed":
                original_record = snapshot(world, status="locked_unconfirmed")
            if case == "attempt-invalid-record":
                path.parent.mkdir(exist_ok=True)
                path.write_text("{")
            if case == "attempt-invalid-action":
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(dict(schema_version=daily_decision.DECISION_SCHEMA,
                                                date=DAY, action=[], scoreable=False)))
            if case == "scoreable":
                original_record = snapshot(world)
            if case == "mismatched":
                original_record = snapshot(world, game_pk=202)
            if case == "delivered":
                daily.delivery_attempted = True
                daily.notification_sent = True
                daily.notification_id = "prior-message"
                daily.delivered_at = world.now.isoformat()
                picks.save_pick(daily, world.picks)

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                world.advance(22 * 60)
                return world.frame((0.99, 0.70), (True, True))
            return world.frame((0.99, 0.70), (True, True))

        def confirmations(game_pks):
            nonlocal stage
            if check_returned and game_pks == [102] and arrival is None:
                stage = "legacy pre-refresh" if IS_BASE else "extra confirmation poll"
                publish()
            return {pk: {"away", "home"} for pk in game_pks
                    if pk == 101 or (arrival is not None and case != "attempt-defer")}

        world.predictions = forecast
        mp.setattr(scheduler, "run_single_check", check)
        mp.setattr(scheduler, "check_confirmed_lineups", confirmations)
        mp.setattr(scheduler, "run_result_polling",
                   lambda pk, *_args, **_kwargs: poll_calls.append(pk) or "captured-poll")
        error = None
        try:
            scheduler.run_day(DAY, world.config)
        except TypeError as exc:
            if case != "attempt-invalid-action":
                raise
            error = str(exc)
        record = None if error else daily_decision.load_decision(DAY, world.picks)
        state = scheduler.load_state(DAY, world.picks)
        current = picks.load_pick(DAY, world.picks)
        facts = {"base": IS_BASE, "case": case, "published_in": stage,
                 "arrived": arrival.isoformat() if arrival else None,
                 "cascades": [start.isoformat() for start in starts],
                 "pick_dm": events(world, "dm"), "health": health(world), "poll": poll_calls,
                 "attempt": bool(current and current.delivery_attempted),
                 "record_status": record and record["delivery_status"],
                 "record_kept": original_record == path.read_bytes() if original_record else None,
                 "locked": state.pick_locked, "error": error}
        print("DURABLE", json.dumps(facts), flush=True)
        assert arrival is not None
        if case == "locked-unconfirmed":
            assert not facts["pick_dm"] and facts["attempt"] and facts["locked"]
            assert facts["record_status"] == "locked_unconfirmed"
            assert any("OUTCOME UNKNOWN" in line for line in facts["health"]) is EXPECT_ALERT
            if not IS_BASE:
                assert facts["record_kept"]
        elif case == "attempt-invalid-action":
            assert not facts["pick_dm"] and facts["attempt"]
            assert bool(error) is not EXPECT_ALERT
            assert any("OUTCOME UNKNOWN" in line for line in facts["health"]) is EXPECT_ALERT
        elif not IS_BASE and case == "attempt-invalid-record":
            assert len(starts) == 1 and len(state.runs_completed) == 2
            assert any("OUTCOME UNKNOWN" in line for line in facts["health"])
        elif not IS_BASE:
            assert not facts["pick_dm"] and facts["locked"] and len(starts) == 1
            if case in {"attempt-deliver", "attempt-defer"}:
                assert facts["record_status"] == "locked_unconfirmed"
                assert any("OUTCOME UNKNOWN" in line for line in facts["health"])
            elif case in {"scoreable", "mismatched"}:
                assert facts["record_kept"] and poll_calls == ([101] if case == "scoreable" else [])
            elif case == "delivered":
                assert facts["record_status"] == "delivered" and current.notification_id == "prior-message"


def readers(case):
    with offline(at(18, 25)) as (world, mp):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.dated(0)
        original = scheduler.run_single_check
        opener = os.open
        path = daily_decision.decision_path(DAY, world.picks)
        called = []
        initial = []

        def forecast():
            called.append(world.now)
            if len(called) == 1:
                world.advance(35 * 60)
                return world.frame((0.99,), (True,))
            initial.append((path.is_fifo(), path.is_symlink(), path.is_dir()))
            return None

        def check(*args, **kwargs):
            result = original(*args, **kwargs)
            path.parent.mkdir(exist_ok=True)
            if case == "fifo":
                os.mkfifo(path)
            elif case == "directory":
                path.mkdir()
            elif case == "dangling":
                path.symlink_to(path.with_name("absent"))
            elif case == "wrong-date":
                record = json.loads(snapshot(world))
                path.write_text(json.dumps(dict(record, date="2027-03-25")))
            elif case == "regular-link":
                snapshot(world)
                link_target = path.with_name("retained.json")
                path.rename(link_target)
                path.symlink_to(link_target)
            elif case == "partial-scoreable":
                path.write_text(json.dumps(dict(schema_version=daily_decision.DECISION_SCHEMA,
                                                action="single", scoreable=True, date=DAY)))
            elif case != "absent":
                path.write_bytes({"malformed": b"{", "empty": b"", "unreadable": b"{}",
                                  "bad-action": b'[]', "bad-encoding": b'\xff'}[case])
            return result

        def open_file(file, flags, *args, **kwargs):
            if case == "unreadable" and Path(file) == path and flags & os.O_NONBLOCK:
                raise PermissionError("fixture EACCES")
            return opener(file, flags, *args, **kwargs)

        world.predictions = forecast
        mp.setattr(scheduler, "run_single_check", check)
        mp.setattr(os, "open", open_file)
        scheduler.run_day(DAY, world.config)
        state = scheduler.load_state(DAY, world.picks)
        facts = {"case": case, "cascades": len(called), "legacy_view": initial,
                 "send": bool(world.sends("dm")), "health": health(world),
                 "scoreable": state.committed_pick_written,
                 "reuse": [r["reused_check_cascade"] for r in state.fallback_refreshes or []]}
        print("DECISION", json.dumps(facts), flush=True)
        if case in {"regular-link", "partial-scoreable"}:
            assert len(called) == 1 and not facts["send"] and facts["scoreable"]
        else:
            assert facts["reuse"] == [case == "absent"]
            assert len(called) == (1 if case == "absent" else 2)
            if case != "directory":  # existing decision writer cannot replace a directory
                assert facts["send"] and facts["scoreable"]


if __name__ == "__main__":
    if sys.argv[1] == "poll":
        for name in sys.argv[2:] or ["locked-unconfirmed", "attempt-deliver", "attempt-defer",
                                    "delivered", "scoreable", "mismatched", "attempt-invalid-record"]:
            during_poll(name)
    else:
        for name in ["absent", "fifo", "directory", "dangling", "regular-link", "wrong-date",
                     "malformed", "empty", "unreadable", "bad-action", "bad-encoding", "partial-scoreable"]:
            readers(name)
