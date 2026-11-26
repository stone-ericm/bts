"""Rehearse the DM and checklist steps through the real pinned CLI and scheduler."""

import json
import re
import sys
from contextlib import redirect_stderr
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import MonkeyPatch

from bts import contest_state, dm, orchestrator, picks, scheduler, strategy
from bts.daily_decision import load_decision
from bts.health import alert
from scripts.audit.season_walk.walk import ET, _invoke, _set_clock, offline
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, set_automatic
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def status(root):
    before = snapshot(root)
    code, output = _invoke("season-contract", "status", "--picks-dir", root)
    assert code == 0 and snapshot(root) == before, output
    return output


def dated(root, day, streak=57, best=60):
    return _invoke("set-contest-streak", "--picks-dir", root, "--date", day,
                   "--streak", streak, "--best-streak", best, "--saver", "used")


def late_start(repair, last_only=False):
    with TemporaryDirectory(prefix="wrk1-text-start-") as tmp, offline(NOW), MonkeyPatch.context() as mp:
        root = Path(tmp)
        mp.chdir(root)
        cfg, directory, now, _ = world.__wrapped__(root, mp)
        prepare_run_day(mp)
        cfg["scheduler"]["entry_intent"] = "enter"
        if last_only:
            cfg["scheduler"]["cluster_min"] = 45
        mp.setattr(scheduler, "notify_ready", lambda: None)
        mp.setattr(scheduler, "notify_watchdog", lambda: None)
        mp.setattr(strategy, "_mdp_cache", None)
        journal = StringIO()
        diagnosed = []
        frame, tier = orchestrator.run_cascade.return_value
        orchestrator.run_cascade.return_value = (frame.assign(p_game_hit=[0.97, 0.81]), tier)

        def inspect_message(recipient, body):
            assert "Morning state check skipped:" in journal.getvalue()
            assert re.search(rf"No pick was sent for {DAY} at this check\. Reason: .* switch is absent\.", body)
            assert "scheduler journal for this refusal" in body
            assert "use the season-start checklist (B8a)" in body
            live = status(directory)
            assert "live decision now: refused;" in live
            diagnosed.append((now[0], body))
            return "captured"

        def advance(seconds):
            if repair and diagnosed and len(dm.send_dm.call_args_list) == 0 and not (
                    directory / "account_state/contest_streak.manual.json").exists():
                # A verified MLB observation filed between cycles, while the
                # later game's entry window remains open.
                assert dated(directory, DAY)[0] == 0
            now[0] += timedelta(seconds=seconds)
            _set_clock(now[0])

        mp.setattr(scheduler, "_watchdog_ping_sleep", advance)
        mp.setattr(alert, "send_dm", inspect_message)
        with redirect_stderr(journal):
            scheduler.run_day(DAY, cfg)
        assert len(diagnosed) == 1, journal.getvalue()
        state = scheduler.load_state(DAY, directory)
        recorded = load_decision(DAY, directory)
        if repair and not last_only:
            assert state.pick_locked and recorded["delivery_status"] == "delivered", journal.getvalue()
            assert dm.send_dm.call_count == 1 and "Streak: 57" in dm.send_dm.call_args.args[1]
        else:
            assert recorded is None and dm.send_dm.call_count == 0 and not state.pick_locked, journal.getvalue()
        if last_only:
            assert dated(directory, DAY)[0] == 0
            assert "live decision now: accepted" in status(directory)
            assert load_decision(DAY, directory) is None and not dm.send_dm.called
        print("late start", "after last check" if last_only else ("between checks" if repair else "leave state absent"),
              "checks", len(state.runs_completed), "official sends", dm.send_dm.call_count)


def guarded_delivery():
    for case in ("revoked", "saved changed"):
        with TemporaryDirectory(prefix="wrk1-text-guard-") as tmp, offline(NOW), MonkeyPatch.context() as mp:
            cfg, directory, now, _ = world.__wrapped__(Path(tmp), mp)
            saved = daily_pick()
            if case == "revoked":
                set_automatic(directory, now=now[0] - timedelta(hours=3), streak=12)
                assert "live decision now: accepted" in status(directory)
                code, output = _invoke("season-contract", "revoke", "--season", 2027, "--reason",
                                       "offline rehearsal", "--picks-dir", directory)
                assert code == 0, output
                expected = "switch is revoked"
                current = "refused"
            else:
                assert dated(directory, DAY, 12, 16)[0] == 0
                state = contest_state.load_decision_streak_state(directory, require_contest_state=True)
                saved.chosen_on = picks.chosen_on_record(state, required=True)
                assert dated(directory, DAY, 13, 16)[0] == 0
                expected = "it was chosen on dated"
                current = "accepted"
            path = picks.save_pick(saved, directory)
            original = path.read_bytes()
            journal = StringIO()
            with redirect_stderr(journal):
                assert scheduler._deliver_and_lock_pick(saved, cfg, directory, scheduler_state(), DAY, case) is False
            wire = alert.send_dm.call_args.args[1]
            assert expected in wire and expected in journal.getvalue()
            live = status(directory)
            assert f"live decision now: {current}" in live
            assert path.read_bytes() == original and not dm.send_dm.called
            assert load_decision(DAY, directory) is None
            print(case, "status", current, "pick wire calls", dm.send_dm.call_count)


def overnight(caller_results):
    for slate, later in (("2027-06-14", "2027-06-15"), ("2027-12-31", "2028-01-01")):
        result = caller_results[f"{slate}/final"]
        record, = [row for row in result["trace"] if row[2] == "final"]
        wall = datetime.fromisoformat(record[3]).astimezone(ET)
        body, = [body for body in result["wire"] if "- [contest_state]" in body]
        assert f"No pick was sent for {slate} at this check." in body
        assert "Before changing state for a new slate, verify in MLB that prior entries settled or were absent." in body
        assert "Use a later official pick only while its entry window is open." in body
        assert wall.date().isoformat() == later and record[4] == slate
        assert "FALLBACK BLOCKED" in result["journal"] and result["picks"] == []
        with TemporaryDirectory(prefix="wrk1-text-overnight-") as tmp, offline(wall):
            root = Path(tmp) / "picks"
            live = status(root)
            assert f"set-contest-streak --date {later}" in live
            assert "set-contest-streak --date " + slate not in live
            code, output = dated(root, slate)
            assert code != 0 and f"must be today ({later})" in output
            assert not snapshot(root)
            # With a prior entry pending, the DM's verification instruction
            # blocks use of the current-date template. The CLI itself would
            # accept that template. Exercise that path only in this control.
            code, output = dated(root, later)
            assert code == 0, output
            assert "live decision now: accepted" in status(root)
            print(slate, "final fallback wall", wall.isoformat(), "old date refused; new date writable")


def main():
    checklist = Path("docs/ops/2027-season-start.md").read_text()
    assert "Enter it in the morning, before the scheduler's first lineup check." in checklist
    assert "after their entry windows closed" in checklist
    assert "if a later lineup check or the fallback refresh still runs" in checklist
    late_start(False)
    late_start(True)
    late_start(False, last_only=True)
    guarded_delivery()
    overnight(json.loads(Path(sys.argv[1]).read_text()))


if __name__ == "__main__":
    main()
