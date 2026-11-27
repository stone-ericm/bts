"""Follow B8a with a pinned clock, the real scheduler and the real dated CLI."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from pytest import MonkeyPatch

from bts import dm, orchestrator, scheduler, strategy
from bts.daily_decision import load_decision
from bts.health import alert
from scripts.audit.season_walk.walk import ET, _invoke, _set_clock, offline
from tests.test_pick_chosen_on_cs5a import DAY, NOW
from tests.test_saved_pick_paths_cs5a import prepare_run_day, world


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "docs/reviews/2026-11-18-whole-range/review-fix-k1-codex"
sys.path.insert(0, str(REVIEW))
from probe_operator import snapshot, status
import replay_callers


def file_observation(day, directory=None, streak=57):
    args = ["set-contest-streak", "--date", day, "--streak", streak,
            "--best-streak", 60, "--saver", "used"]
    if directory is not None:
        args += ["--picks-dir", directory]
    return _invoke(*args)


def run_day(case, *, repair):
    with TemporaryDirectory(prefix="wrk1-r2-operator-") as tmp, offline(NOW), MonkeyPatch.context() as mp:
        root = Path(tmp)
        mp.chdir(root)
        cfg, _, now, statuses = world.__wrapped__(root, mp)
        prepare_run_day(mp)
        default = root / "data/picks"
        directory = root / "live-picks" if case == "custom-directory" else default
        cfg["orchestrator"]["picks_dir"] = str(directory)
        cfg["scheduler"]["entry_intent"] = "enter"
        mp.setattr(scheduler, "notify_ready", lambda: None)
        mp.setattr(scheduler, "notify_watchdog", lambda: None)
        mp.setattr(strategy, "_mdp_cache", None)
        journal = StringIO()
        times = []
        filings = []
        changed_authority = False
        frame, tier = orchestrator.run_cascade.return_value
        frame = frame.assign(p_game_hit=[0.99, 0.97])

        def tick(at):
            now[0] = at
            _set_clock(at)

        if case == "no-fallback":
            cfg["scheduler"]["cluster_min"] = 45
        if case == "later-check":
            games = scheduler.fetch_schedule(DAY)
            third = dict(games[-1], gamePk=103, gameDate=f"{DAY}T00:40:00-04:00")
            third["gameDate"] = (NOW + timedelta(hours=2, minutes=5)).isoformat()
            games.append(third)
            statuses[103] = statuses[102]
            scheduler.fetch_schedule.side_effect = lambda day: games if day == DAY else []
            frame = frame.assign(p_game_hit=[0.82, 0.99], game_pk=[101, 103],
                                 game_time=[games[0]["gameDate"], third["gameDate"]])

        def predictions(*args):
            if case == "later-check" and orchestrator.run_cascade.call_count == 1:
                return None, None
            return frame, tier

        orchestrator.run_cascade.side_effect = predictions
        if case == "in-loop-fallback":
            tick(NOW - timedelta(minutes=15))
            assert file_observation(DAY, directory)[0] == 0
            original = scheduler._lock_decision_from_predictions

            def hold_first_check(*args, **kwargs):
                if changed_authority:
                    return original(*args, **kwargs)
                return scheduler.LockDecision(False, None, False)

            mp.setattr(scheduler, "_lock_decision_from_predictions", hold_first_check)
        else:
            tick(NOW)

        def capture(_, body):
            if "No pick was sent for" in body:
                assert f"No pick was sent for {DAY} at this check" in body
                assert "use the season-start checklist (B8a)" in body
                assert "live decision now: refused;" in status(directory)
                times.append(now[0].strftime("%H:%M"))
            return "review-health-dm"

        def advance(seconds):
            nonlocal changed_authority
            if case == "in-loop-fallback" and now[0].strftime("%H:%M") == "18:25":
                code, output = _invoke("season-contract", "revoke", "--season", "2027",
                                       "--picks-dir", directory, "--reason", "review rehearsal")
                assert code == 0, output
                changed_authority = True
            if times and not filings and (case == "custom-directory" or repair):
                target = directory if repair else None
                code, output = file_observation(DAY, target)
                assert code == 0, output
                filings.append(output.splitlines()[0])
                if case == "two-operators":
                    prior = snapshot(directory)
                    code, output = file_observation(DAY, target)
                    assert code == 0 and "Observation unchanged" in output, output
                    assert snapshot(directory) == prior
                    filings.append("second operator verified the same 57/60/used, no state change")
            tick(now[0] + timedelta(seconds=seconds))

        mp.setattr(alert, "send_dm", capture)
        mp.setattr(scheduler, "_watchdog_ping_sleep", advance)
        with redirect_stderr(journal), redirect_stdout(StringIO()):
            scheduler.run_day(DAY, cfg)
        assert times, journal.getvalue()
        if case == "in-loop-fallback":
            assert "FALLBACK BLOCKED" in journal.getvalue(), journal.getvalue()
            assert changed_authority
        if case == "custom-directory" and not repair:
            assert "live decision now: accepted" in status(default)
            assert "live decision now: refused" in status(directory)
        if case == "no-fallback":
            assert "FALLBACK BLOCKED" not in journal.getvalue()
            assert not filings
        if repair:
            assert dm.send_dm.call_count == 1 and "Streak: 57" in dm.send_dm.call_args.args[1], journal.getvalue()
            assert load_decision(DAY, directory)["delivery_status"] == "delivered"
        else:
            assert dm.send_dm.call_count == 0, journal.getvalue()
            assert load_decision(DAY, directory) is None
        print(case, "correct-directory filing control" if repair else "literal B8a",
              "refusal at", times, "official sends", dm.send_dm.call_count, flush=True)
        for filing in filings:
            print(filing, flush=True)


def overnight_pending():
    slate = "2027-12-31"
    today = "2028-01-01"
    midnight = datetime(2028, 1, 1, tzinfo=ET)
    result = replay_callers.replay("final", slate, midnight, True)
    record, = [row for row in result["trace"] if row[2] == "final"]
    at = datetime.fromisoformat(record[3])
    assert at.date().isoformat() == today and record[4] == slate
    with TemporaryDirectory(prefix="wrk1-r2-pending-") as tmp, offline(at):
        directory = Path(tmp) / "picks"
        assert f"set-contest-streak --date {today}" in status(directory)
        assert not snapshot(directory)
        # D differs from today, and MLB still shows D's entry pending. B8a waits.
        code, output = file_observation(slate, directory)
        assert code != 0 and f"must be today ({today})" in output and not snapshot(directory)
        code, output = file_observation(today, directory)
        assert code == 0 and "live decision now: accepted" in status(directory), output
    print("overnight with prior entry pending: follow B8a and leave the row absent; "
          "old-day write rejected; next-day write succeeds only in the control", flush=True)


if __name__ == "__main__":
    probe = sys.argv[1]
    if probe == "overnight-pending":
        overnight_pending()
    else:
        run_day(probe, repair=len(sys.argv) == 3 and sys.argv[2] == "repair")
