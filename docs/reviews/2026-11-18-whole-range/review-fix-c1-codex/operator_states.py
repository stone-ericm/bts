"""Real HEAD read-only paths under the H5 operator scenarios."""

from datetime import datetime, timedelta
import datetime as datetime_module
from contextlib import redirect_stderr
import io
import logging
import os
from pathlib import Path
import signal
import socket
from tempfile import TemporaryDirectory
from unittest.mock import patch

from historical_copy import HERE, POINTER, allowed, expect, probe, run


def read_only_status(picks, wall):
    from bts.cli import cli
    from click.testing import CliRunner

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return wall.astimezone(tz)

    with patch.object(datetime_module, "datetime", Clock):
        return CliRunner().invoke(cli, ["season-contract", "status", "--picks-dir", str(picks)])


def snapshot(picks):
    return {str(file.relative_to(picks)): file.read_bytes() for file in picks.rglob("*.json")}


def check(root):
    from bts import scheduler
    from bts.health import alert, contest_state, runner
    from bts.picks import DailyPick, Pick, save_pick, submission_cutoff_et

    day = probe.YEAR_END
    following = day + timedelta(days=1)
    wall = probe.at(following, 0, 30)

    picks = root / "pending" / "picks"
    probe.auto(picks, day)
    probe.switch(picks, day, status="revoked")
    probe.saver_flag(picks, following)
    prior_entry = Pick("Prior", 1, "CLE", 1, "Pitcher", 2, 0.6, [], False, 7, probe.at(day, 22).isoformat())
    prior = DailyPick(str(day), probe.at(day, 20).isoformat(), prior_entry, None, None,
                      delivery_attempted=True, result=None)
    save_pick(prior, picks)
    before = snapshot(picks)
    alerts, dms = run(picks, day, wall=wall, isolate=True)
    assert [a.message for a in alerts] == [
        expect(day, "The saver flag is for another season. The saver counted as spent."),
        expect(day, "The season switch is revoked."),
    ]
    status = read_only_status(picks, wall)
    assert status.exit_code == 0 and f"--date {following}" in status.stdout
    assert "live decision now: refused" in status.stdout
    assert snapshot(picks) == before and len(dms) == 1
    print("Year rollover, pending/unconfirmed prior entry, unsafe live recovery template appears in read-only status only.")

    picks = root / "overnight-fetch" / "picks"
    probe.switch(picks, day)
    at_two = probe.at(following, 2)
    assert len(run(picks, day, wall=at_two, isolate=True)[0]) == 2
    probe.auto(picks, day, source_date=str(day), recorded_at=probe.at(following, 1, 10).isoformat(), activation_id="b" * 32)
    probe.saver_flag(picks, day)
    before = snapshot(picks)
    items, _ = run(picks, day, wall=at_two, isolate=True)
    assert [a.message for a in items] == [expect(day, "The automatic observation and season switch have different activation IDs.")]
    assert snapshot(picks) == before
    probe.auto(picks, day, source_date=str(day), recorded_at=probe.at(following, 1, 10).isoformat())
    assert run(picks, day, wall=at_two, isolate=True)[0] == []
    print("01:10 replacement changed the retained diagnosis, first to a mismatch, then silence with a matching activation.")

    for label in ("known no entry", "unconfirmed entry"):
        picks = root / label / "picks"
        probe.switch(picks, day)
        if label == "unconfirmed entry":
            save_pick(prior, picks)
        for hour, minute in ((0, 15), (2, 0)):
            when = probe.at(following, hour, minute)
            items, _ = run(picks, day, wall=when, isolate=True)
            assert [a.message for a in items] == [
                expect(day, "Contest streak observation is missing from account_state."),
                expect(day, "There is no automatic observation for the switch activation."),
            ]
    print("Missing required observation stays a CRITICAL at 00:15 and 02:00 whether or not local bytes suggest prior entry.")

    picks = root / "past-cutoff" / "picks"
    probe.auto(picks, day)
    probe.switch(picks, day, status="revoked")
    probe.saver_flag(picks, day, "not_earned")
    start = probe.at(following, 14)
    pick = Pick("Early opener", 8, "ATL", 1, "Pitcher", 2, 0.7, [], False, 42, start.isoformat())
    next_pick = DailyPick(str(following), probe.at(following, 11).isoformat(), pick, None, None,
                          delivery_attempted=True)
    save_pick(next_pick, picks)
    after_cutoff = probe.at(following, 14, 1)
    assert after_cutoff > submission_cutoff_et(next_pick)
    before = snapshot(picks)
    items, dms = run(picks, day, wall=after_cutoff, isolate=True)
    assert len(items) == 2 and len(dms) == 1
    assert all(item.message.endswith(POINTER) for item in items)
    for item in items:
        allowed(item.message, day)
    assert snapshot(picks) == before
    print("Historical DM after real 13:55 cutoff names no entry or retry action.")

    picks = root / "morning" / "picks"
    probe.switch(picks, day, status="revoked")
    assert run(picks, day, wall=at_two, isolate=True)[0]
    config = {"bluesky": {"dm_recipient": "offline.test"}}
    with redirect_stderr(io.StringIO()), patch.object(alert, "send_dm", return_value="offline") as sent:
        scheduler._morning_state_check(config, picks, str(following), probe.at(following, 10), probe.at(following, 11))
        assert len(sent.call_args_list) == 1 and f"Morning check for {following}" in sent.call_args.args[1]
        sent.reset_mock()
        probe.auto(picks, following, source_date=str(day), recorded_at=probe.at(following, 9).isoformat())
        probe.switch(picks, following)
        probe.saver_flag(picks, following)
        scheduler._morning_state_check(config, picks, str(following), probe.at(following, 10), probe.at(following, 11))
        sent.assert_not_called()
    print("A later morning check refused absent state, then accepted a newer matching snapshot without a DM.")

    # Observe order at the real runner while leaving calibration and contest checks operational.
    previous = probe.at(day, 23, 59)
    clock = [previous]
    events = []
    calibrate = runner.calibration.check
    live = contest_state.check

    class ModeClock(datetime):
        @classmethod
        def now(cls, tz=None):
            events.append("mode")
            now = clock[0]
            clock[0] = wall
            return now.astimezone(tz)

    class LiveClock(datetime):
        @classmethod
        def now(cls, tz=None):
            events.append("live")
            return clock[0].astimezone(tz)

    def observed_calibration(*args, **kwargs):
        events.append("calibration")
        return calibrate(*args, **kwargs)

    picks = root / "crossed" / "picks"
    probe.switch(picks, day, status="revoked")
    with (patch.object(runner, "datetime", ModeClock), patch.object(contest_state, "datetime", LiveClock),
          patch.object(runner.calibration, "check", observed_calibration),
          patch.object(contest_state, "check", wraps=live) as called,
          patch.object(alert, "send_dm", return_value="offline") as sent):
        runner.run_all_checks(picks, picks.parent / "models", "offline.test", today=day, contest_state_expected=True)
    assert events == ["calibration", "mode", "live"]
    called.assert_called_once_with(picks, expected=True)
    assert f"--date {following}" in sent.call_args.args[1]
    print("Order calibration, one mode sample, live check sample. Rollover kept default kwargs and next-day template.")


if __name__ == "__main__":
    signal.alarm(50)
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory(dir=HERE) as scratch:
        root = Path(scratch)
        os.chdir(root)
        with (patch.object(socket, "socket", side_effect=AssertionError("offline")),
              patch.object(socket, "create_connection", side_effect=AssertionError("offline"))):
            check(root)
