import datetime as dt
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner
from pytest import MonkeyPatch
from tests.test_contest_fetch import _mock_fetch
from bts import cli, contest_state, daily_decision, orchestrator, scheduler

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "docs/reviews/2026-11-18-whole-range/closing-test/carbon-slice3"))
import test_slice3_interactions as carbon

carbon.REPO_ROOT = ROOT
DAY = "2027-03-24"
ET = carbon.ET
START = dt.datetime(2027, 3, 24, 10, tzinfo=ET)
SKIP_FIELDS = ("streak", "saver_available", "state_source", "state_status", "contest_source_date")


def run_case(root, name):
    intent = "research" if name == "legacy" else "enter"
    delivery = "private" if intent == "research" else "dm"
    with carbon._harness(root, START, delivery=delivery, intent=intent) as env:
        directory, clock = env["picks_dir"], env["clock"]
        runner = CliRunner()
        read_kind = []
        check_times = []
        reader = contest_state.load_decision_streak_state
        config = root / "orchestrator.toml"
        config.write_text(
            f'[scheduler]\npick_delivery = "{delivery}"\nentry_intent = "{intent}"\n'
        )
        env["confirmed"][101] = {"home", "away"}

        def command(*args):
            with patch("datetime.datetime", carbon._make_Clock(clock)):
                result = runner.invoke(cli.cli, [*args, "--picks-dir", str(directory)])
            assert result.exit_code == 0, (args, result.output, result.exception)
            return result

        def dated(streak, saver="used"):
            command("set-contest-streak", "--date", DAY, "--streak", str(streak),
                    "--best-streak", "30", "--saver", saver)

        def legacy(streak):
            command("set-contest-streak", "--streak", str(streak), "--best-streak", "30")

        def revoke():
            command("season-contract", "revoke", "--season", "2027", "--reason", "restored from snapshot")

        def automatic(streak):
            with MonkeyPatch.context() as mp:
                _mock_fetch(mp, {"activeStreak": streak, "seasonBestStreak": 30,
                                 "predictions": [{"roundId": 1, "result": "hit"}]},
                            {1: dt.date.fromisoformat(DAY)})
                command("fetch-contest-streak", "--expected-username", "stonehengee")
            account = directory / "account_state"
            assert json.loads((account / "contest_streak.json").read_text())["active_streak"] == streak
            assert json.loads((account / "contest_ledger.jsonl").read_text().splitlines()[-1])["active_streak"] == streak

        carbon._write_auto_obs(directory, START - dt.timedelta(days=1), 30,
                               source_date=DAY, saver_flag="used", recorded_at=START.isoformat())
        if name in {"dated", "same_final", "saver_only", "legacy_replacement"}:
            dated(30)
        if name == "legacy":
            legacy(30)
        if name in {"enable_before", "replaced_before"}:
            if name == "replaced_before":
                dated(1, "not_earned")
            revoke()
            command("season-contract", "enable", "--season", "2027")
            automatic(30)
        if name == "revoke_before_dated":
            revoke()
            dated(30)

        def capture_state(*args, **kwargs):
            state = reader(*args, **kwargs)
            read_kind.append(state.observation_kind)
            return state

        def cascade(_tiers, day):
            check_times.append(clock[0].strftime("%H:%M"))
            if len(check_times) == 1:
                return carbon._pred_frame(day, p1=0.72, p2=None), "local"
            if name == "revoke":
                return carbon._pred_frame(day, p1=0.88, p2=None), "local"
            return None, None

        changed = False

        def sleep(secs, *args, **kwargs):
            nonlocal changed
            intended = clock[0] + dt.timedelta(seconds=max(secs, 0))
            change_at = START.replace(hour=16, minute=30)
            if check_times and not changed and clock[0] < change_at <= intended:
                env["advance"]((change_at - clock[0]).total_seconds())
                changed = True
                if name in {"dated", "same_final", "both"}:
                    dated(1, "not_earned")
                if name == "same_final":
                    dated(30)
                if name in {"revoke", "both"}:
                    revoke()
                if name == "saver_only":
                    dated(30, "active")
                if name == "legacy_replacement":
                    legacy(1)
                if name == "automatic_replacement":
                    automatic(1)
                env["advance"]((intended - clock[0]).total_seconds())
            else:
                env["advance"](secs)

        with patch.object(orchestrator, "run_cascade", side_effect=cascade), \
             patch.object(contest_state, "load_decision_streak_state", side_effect=capture_state), \
             patch.object(scheduler, "_watchdog_ping_sleep", side_effect=sleep):
            scheduler.run_day(DAY, env["config"])

        record = daily_decision.load_decision(DAY, directory)
        assert record and (record["action"], record["source"], record["streak"]) == ("skip", "mdp", 30), record
        assert check_times[0] == "16:10" and changed, check_times
        assert not (directory / f"{DAY}.json").exists()
        assert read_kind, name
        before = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
        n = len(env["pick_dms"])
        clock[0] = START.replace(day=25, hour=8, minute=30)
        with patch("datetime.datetime", carbon._make_Clock(clock)):
            result = runner.invoke(cli.cli, [
                "day-outcome", "--date", DAY, "--config", str(config), "--picks-dir", str(directory),
                "--expected-username", "operator", "--dm-recipient", "operator.bsky.social",
            ])
        outcome = json.loads(result.output)
        after = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
        assert before == after
        if "--base" in sys.argv:
            assert (outcome["state"], result.exit_code, env["pick_dms"][n:]) == ("deliberate_skip", 0, [])
        print(json.dumps({
            "case": name, "checks_with_cascade": check_times, "selection_kinds": read_kind,
            "decision": {key: record[key] for key in ("action", "source", *SKIP_FIELDS, "finalized_at")},
            "outcome": outcome, "exit": result.exit_code, "morning_dm": env["pick_dms"][n:],
            "day_dms": env["pick_dms"][:n], "health_dms": env["health_dms"], "files_unchanged": before == after,
        }), flush=True)


with tempfile.TemporaryDirectory() as temp:
    for name in ("dated", "revoke", "same_final", "legacy", "legacy_replacement",
                 "automatic_replacement", "saver_only", "both", "enable_before", "revoke_before_dated", "replaced_before"):
        run_case(Path(temp) / name, name)
