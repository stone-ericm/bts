import datetime as dt
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from click.testing import CliRunner
from pytest import MonkeyPatch
from bts import cli, contest_state, daily_decision, orchestrator, scheduler, util
from tests.test_contest_fetch import _mock_fetch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "docs/reviews/2026-11-18-whole-range/closing-test/carbon-slice3"))
import test_slice3_interactions as carbon

carbon.REPO_ROOT = ROOT
ET = carbon.ET
START = dt.datetime(2027, 3, 24, 10, tzinfo=ET)
DAY = START.date().isoformat()
EXPECTED = {
    "M1": ("dated",), "M2": ("revoke",), "M3": ("dated",), "M4": ("dated", "revoke"),
    "K1": (), "K2a": (), "K2b": (), "K2c": (), "K2d": (),
    "K2c_after_same_day_revoke": ("revoke",),
    "K2a_current_outranked_by_archive": ("dated",),
    "M2_matching_archive": (), "M2_second_revoke_next_morning": (),
    "K2d_next_morning_write": ("revoke",),
    "M1_archived_after_midnight": ("dated",),
    "research_revoke": (), "enter_to_research": (), "research_to_enter": ("revoke",),
    "same_inputs_again": (), "best_only": (), "legacy": (), "automatic": (),
}


def run_case(root, name):
    intent = "research" if name in {"research_revoke", "research_to_enter"} else "enter"
    delivery = "private" if intent == "research" else "dm"
    with carbon._harness(root, START, delivery=delivery, intent=intent) as env:
        clock, picks_dir = env["clock"], env["picks_dir"]
        account = picks_dir / "account_state"
        manual = account / "contest_streak.manual.json"
        runner = CliRunner()
        check_times, read_kinds = [], []
        real_reader = contest_state.load_decision_streak_state
        cfg = root / "config.toml"
        env["confirmed"][101] = {"home", "away"}

        def command(*args, exit_code=0):
            with patch("datetime.datetime", carbon._make_Clock(clock)):
                result = runner.invoke(cli.cli, [*args, "--picks-dir", str(picks_dir)])
            assert result.exit_code == exit_code, (args, result.output, result.exception)
            return result.output

        def dated(streak, saver="used", *, best=30, day=DAY, exit_code=0):
            return command("set-contest-streak", "--date", day, "--streak", str(streak),
                           "--best-streak", str(best), "--saver", saver, exit_code=exit_code)

        def revoke():
            command("season-contract", "revoke", "--season", "2027", "--reason", "restored from snapshot")

        def enable():
            command("season-contract", "enable", "--season", "2027")

        def fetch(streak):
            with MonkeyPatch.context() as mp:
                _mock_fetch(mp, {"activeStreak": streak, "seasonBestStreak": 30,
                                 "predictions": [{"roundId": 1, "result": "hit"}]},
                            {1: START.date()})
                command("fetch-contest-streak", "--expected-username", "stonehengee")

        def interleaved_dated_commands():
            write = util.atomic_write_text
            nested = False

            def delay_before_install(path, text):
                nonlocal nested
                if path == manual and not nested:
                    nested = True
                    clock[0] = START.replace(hour=9, minute=1)
                    dated(1, "not_earned")
                    clock[0] = START.replace(hour=9, minute=2)
                    revoke()
                write(path, text)

            with patch.object(util, "atomic_write_text", side_effect=delay_before_install):
                dated(30)
            live = json.loads(manual.read_text())
            archived, = account.glob("contest_streak.manual.revoked-*.json")
            retired = json.loads(archived.read_text())
            assert live["active_streak"] == 30 and live["recorded_at"] < retired["recorded_at"]
            assert retired["active_streak"] == 1

        carbon._write_auto_obs(picks_dir, START - dt.timedelta(days=1), 30,
                               source_date=DAY, saver_flag="used", recorded_at=START.isoformat())
        clock[0] = START.replace(hour=9)
        if name in {"M3", "K2a", "K2b", "same_inputs_again", "best_only", "legacy"}:
            dated(30)
        if name == "K2c":
            (account / "season_contract.json").unlink()
            enable()
            fetch(30)
        if name in {"K2d", "K2d_next_morning_write"}:
            revoke()
            clock[0] = START.replace(hour=9, minute=5)
            dated(30)
        if name in {"K2c_after_same_day_revoke", "M2_matching_archive"}:
            if name == "M2_matching_archive":
                dated(30)
                clock[0] = START.replace(hour=9, minute=1)
            revoke()
            clock[0] = START.replace(hour=9, minute=2)
            enable()
            fetch(30)
        if name == "K2a_current_outranked_by_archive":
            interleaved_dated_commands()
        clock[0] = START

        def cascade(_tiers, day):
            check_times.append(clock[0].strftime("%H:%M"))
            if len(check_times) == 1:
                return carbon._pred_frame(day, p1=0.72, p2=None), "local"
            if name == "M2":
                return carbon._pred_frame(day, p1=0.88, p2=None), "local"
            return None, None

        def capture_state(*args, **kwargs):
            result = real_reader(*args, **kwargs)
            read_kinds.append(result.observation_kind)
            return result

        changed = False

        def sleep(secs, *_args, **_kwargs):
            nonlocal changed
            target = clock[0] + dt.timedelta(seconds=max(secs, 0))
            at = START.replace(hour=16, minute=30)
            if check_times and not changed and clock[0] < at <= target:
                env["advance"]((at - clock[0]).total_seconds())
                changed = True
                if name in {"M1", "M4", "same_inputs_again", "M1_archived_after_midnight"}:
                    dated(1, "not_earned")
                    env["advance"](60)
                if name in {"M2", "M4", "M2_matching_archive", "M2_second_revoke_next_morning",
                            "research_revoke", "enter_to_research", "research_to_enter"}:
                    revoke()
                if name == "M3":
                    dated(30, "active")
                if name == "K2b":
                    before = manual.read_bytes()
                    with patch.object(util, "atomic_write_text", side_effect=PermissionError("install denied")):
                        message = dated(1, "not_earned", exit_code=1)
                    assert "request was appended" in message and before == manual.read_bytes()
                    log = (account / "contest_manual_requests.jsonl").read_text().splitlines()
                    assert json.loads(log[-1])["active_streak"] == 1
                if name == "same_inputs_again":
                    dated(30)
                if name == "best_only":
                    dated(30, best=31)
                if name == "legacy":
                    command("set-contest-streak", "--streak", "1", "--best-streak", "30")
                if name == "automatic":
                    fetch(1)
            env["advance"]((target - clock[0]).total_seconds())

        with patch.object(orchestrator, "run_cascade", side_effect=cascade), \
             patch.object(contest_state, "load_decision_streak_state", side_effect=capture_state), \
             patch.object(scheduler, "_watchdog_ping_sleep", side_effect=sleep):
            scheduler.run_day(DAY, env["config"])

        decision = daily_decision.load_decision(DAY, picks_dir)
        assert (decision["action"], decision["source"], decision["streak"], decision["saver_available"]) == (
            "skip", "mdp", 30, False), decision
        assert check_times == ["16:10", "17:10", "18:10"] and changed and read_kinds
        assert not (picks_dir / f"{DAY}.json").exists()

        clock[0] = START.replace(day=25, hour=8)
        if name in {"M1_archived_after_midnight", "M2_second_revoke_next_morning"}:
            revoke()
        if name in {"K2d_next_morning_write", "M1_archived_after_midnight"}:
            dated(1, "not_earned", day=clock[0].date().isoformat())
        report_intent = {"enter_to_research": "research", "research_to_enter": "enter"}.get(name, intent)
        report_delivery = "private" if report_intent == "research" else "dm"
        cfg.write_text(f'[scheduler]\npick_delivery = "{report_delivery}"\nentry_intent = "{report_intent}"\n')
        before = {p.relative_to(picks_dir): p.read_bytes() for p in picks_dir.rglob("*") if p.is_file()}
        n = len(env["pick_dms"])
        clock[0] = START.replace(day=25, hour=8, minute=30)
        reasons = () if "--base" in sys.argv else EXPECTED[name]
        expect_code = int(bool(reasons))
        raw = command("day-outcome", "--date", DAY, "--config", str(cfg), "--expected-username", "operator",
                      "--dm-recipient", "operator.bsky.social", exit_code=expect_code)
        result = json.loads(raw)
        morning_dms = env["pick_dms"][n:]
        assert len(morning_dms) == expect_code and result["ok"] == (expect_code == 0)
        assert result["state"] == ("skip_state_changed" if reasons else "deliberate_skip")
        assert ("A dated observation" in result["detail"]) == ("dated" in reasons)
        assert ("records a revocation" in result["detail"]) == ("revoke" in reasons)
        assert before == {p.relative_to(picks_dir): p.read_bytes() for p in picks_dir.rglob("*") if p.is_file()}
        print(json.dumps({"case": name, "selection": read_kinds, "intent": intent, "morning_intent": report_intent,
                          "decision": decision, "outcome": result, "exit": expect_code,
                          "DM": morning_dms, "health_DMs": len(env["health_dms"]), "files_unchanged": True}),
              flush=True)


with tempfile.TemporaryDirectory() as temp:
    for name in EXPECTED:
        run_case(Path(temp) / name, name)
