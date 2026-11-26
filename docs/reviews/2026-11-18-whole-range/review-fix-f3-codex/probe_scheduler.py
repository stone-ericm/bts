"""Run real scheduler skips, state commands, and morning CLI under an offline ET clock."""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import cli, contest_state, daily_decision, orchestrator, scheduler, util
from click.testing import CliRunner
from pytest import MonkeyPatch
from tests.test_contest_fetch import _mock_fetch

from review_support import CURRENT, ET, REPO, REQUESTS, SWITCH, identify_package, put, snapshot


HARNESS = REPO / "docs/reviews/2026-11-18-whole-range/closing-test/carbon-slice3/test_slice3_interactions.py"
SPEC = importlib.util.spec_from_file_location("review_scheduler_harness", HARNESS)
carbon = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(carbon)
carbon.REPO_ROOT = REPO

CASES = {
    "M1": ("installed",), "M2": ("revocation",), "M3": ("installed",),
    "M4": ("archived", "revocation"), "K1": (), "K2a": (), "K2b": (), "K2c": (), "K2d": (),
    "K2c_reenable": (), "K2a_newer_archive": (), "K3": (), "K4": (), "K5": (), "K7": (),
    "carbon_dated": ("installed",), "carbon_revoke": ("revocation",),
    "reenable_retains_mismatch": ("archived",), "later_revoke_hides_M2": (),
    "matching_archive_hides_M2": (), "K2d_next_day_replaces_match": ("revocation",),
    "research_revoke": (), "research_dated": ("installed",),
    "enter_to_research": (), "research_to_enter": ("revocation",),
    "dec31_2350": ("revocation",),
}


def run_case(root, name, base):
    dec31 = name in {"K7", "dec31_2350"}
    start = dt.datetime(2027, 12, 31, 10, tzinfo=ET) if dec31 else dt.datetime(2027, 3, 24, 10, tzinfo=ET)
    day = start.date().isoformat()
    tomorrow = start + dt.timedelta(days=1)
    intent = "research" if name.startswith("research_") else "enter"
    with carbon._harness(root, start, delivery="private" if intent == "research" else "dm", intent=intent) as env:
        clock, picks = env["clock"], env["picks_dir"]
        account = picks / "account_state"
        manual = account / CURRENT
        cfg = root / "morning.toml"
        runner = CliRunner()
        env["confirmed"][101] = {"home", "away"}
        hhmm = "00:10:00" if dec31 else "23:10:00"
        if dec31:
            game = env["games_by_date"][day][0]
            game["gameDate"] = f"{tomorrow.date().isoformat()}T{hhmm}Z"
        checks, observations = [], []

        def call(*args):
            with patch("datetime.datetime", carbon._make_Clock(clock)):
                return runner.invoke(cli.cli, [*args, "--picks-dir", str(picks)])

        def command(*args, code=0):
            result = call(*args)
            assert result.exit_code == code, (name, args, result.exit_code, result.output, result.exception)
            return result

        def dated(streak, saver="used", *, code=0):
            return command("set-contest-streak", "--date", clock[0].date().isoformat(), "--streak", str(streak),
                           "--best-streak", str(streak if streak == 2 else 30), "--saver", saver, code=code)

        def revoke():
            return command("season-contract", "revoke", "--season", str(clock[0].year), "--reason", "restore drill")

        def enable_and_fetch():
            command("season-contract", "enable", "--season", str(clock[0].year))
            with MonkeyPatch.context() as mp:
                _mock_fetch(mp, {"activeStreak": 30, "seasonBestStreak": 30,
                                "predictions": [{"roundId": 1, "result": "hit"}]}, {1: start.date()})
                command("fetch-contest-streak", "--expected-username", "stonehengee")

        carbon._write_auto_obs(picks, start - dt.timedelta(days=1), 30, source_date=day,
                               saver_flag="used", recorded_at=start.isoformat())
        clock[0] = start.replace(hour=9)
        if name in {"K2a", "K2b", "M3", "carbon_dated", "matching_archive_hides_M2"}:
            dated(30)
        if name in {"reenable_retains_mismatch", "K2d", "K2d_next_day_replaces_match"}:
            dated(1, "not_earned")
        if name in {"K2c_reenable", "K2d", "K2d_next_day_replaces_match",
                    "matching_archive_hides_M2", "reenable_retains_mismatch"}:
            env["advance"](60)
            revoke()
            env["advance"](60)
            if name in {"K2d", "K2d_next_day_replaces_match"}:
                dated(30)
            else:
                enable_and_fetch()
        if name == "K2c":
            (account / SWITCH).unlink()
            enable_and_fetch()
        if name == "K2a_newer_archive":
            atomic_write = util.atomic_write_text
            nested = False

            def interleave(path, body):
                nonlocal nested
                if path == manual and not nested:
                    nested = True
                    env["advance"](60)
                    dated(1, "not_earned")
                    env["advance"](60)
                    revoke()
                atomic_write(path, body)

            with patch.object(util, "atomic_write_text", side_effect=interleave):
                dated(30)
            archive, = account.glob("contest_streak.manual.revoked-*.json")
            assert json.loads(archive.read_text())["recorded_at"] > json.loads(manual.read_text())["recorded_at"]
        clock[0] = start
        actual_reader = contest_state.load_decision_streak_state

        def capture_reader(*args, **kwargs):
            reading = actual_reader(*args, **kwargs)
            observations.append((clock[0].strftime("%H:%M"), reading.observation_kind,
                                 reading.streak, reading.saver_available))
            return reading

        def cascade(_tiers, slate):
            checks.append(clock[0].strftime("%H:%M"))
            if len(checks) == 1:
                frame = carbon._pred_frame(slate, p1=0.72, p2=None, hhmm_utc=hhmm)
                if dec31:
                    frame.loc[:, "game_time"] = env["games_by_date"][day][0]["gameDate"]
                return frame, "local"
            if name in {"M2", "carbon_revoke"}:
                return carbon._pred_frame(slate, p1=0.88, p2=None), "local"
            return None, None

        changed = False

        def sleep(seconds, *_args, **_kwargs):
            nonlocal changed
            target = clock[0] + dt.timedelta(seconds=max(0, seconds))
            at = start.replace(hour=16, minute=40 if name.startswith("carbon_") else 30)
            if checks and not changed and clock[0] < at <= target:
                env["advance"]((at - clock[0]).total_seconds())
                changed = True
                if name in {"M1", "M4", "research_dated"}:
                    dated(1, "not_earned")
                    env["advance"](1)
                if name == "carbon_dated":
                    dated(2, "not_earned")
                if name in {"M2", "M4", "carbon_revoke", "later_revoke_hides_M2", "matching_archive_hides_M2",
                            "research_revoke", "enter_to_research", "research_to_enter"}:
                    revoke()
                if name == "M3":
                    dated(30, "active")
                if name == "K2b":
                    saved = manual.read_bytes()
                    with patch.object(util, "atomic_write_text", side_effect=PermissionError("test install failure")):
                        message = dated(1, "not_earned", code=1).output
                    assert "request was appended" in message and manual.read_bytes() == saved
                    assert json.loads((account / REQUESTS).read_text().splitlines()[-1])["active_streak"] == 1
            env["advance"]((target - clock[0]).total_seconds())

        with patch.object(orchestrator, "run_cascade", side_effect=cascade), \
             patch.object(contest_state, "load_decision_streak_state", side_effect=capture_reader), \
             patch.object(scheduler, "_watchdog_ping_sleep", side_effect=sleep):
            scheduler.run_day(day, env["config"])
        decision = daily_decision.load_decision(day, picks)
        assert decision is not None, (name, checks, observations)
        assert (decision["action"], decision["source"], decision["streak"], decision["saver_available"]) == (
            "skip", "mdp", 30, False), (name, decision)
        assert changed and checks == ["16:10", "17:10", "18:10"], (name, checks)
        assert not (picks / f"{day}.json").exists()
        observed_at_check = [r for r in observations if r[0] == "16:10"]
        assert observed_at_check, (name, observations)

        if name == "dec31_2350":
            clock[0] = start.replace(hour=23, minute=50)
            revoke()
        if name in {"K7", "later_revoke_hides_M2"}:
            clock[0] = tomorrow.replace(hour=0, minute=30)
            revoke()
        if name == "K2d_next_day_replaces_match":
            clock[0] = tomorrow.replace(hour=8)
            dated(1, "not_earned")
        if name == "K3":
            put(daily_decision.decision_path(day, picks), decision | {"source": "operator"})
            put(account / MANUAL_BAD_NAME, {"not": "a dated observation"})
        if name == "K4":
            env["games_by_date"][day][0]["status"]["detailedState"] = "Postponed"
        if name == "K5":
            manual.write_bytes(b"\xff")

        report_intent = {"enter_to_research": "research", "research_to_enter": "enter"}.get(name, intent)
        report_delivery = "private" if report_intent == "research" else "dm"
        cfg.write_text(f'[scheduler]\nentry_intent = "{report_intent}"\npick_delivery = "{report_delivery}"\n')
        clock[0] = tomorrow.replace(hour=8, minute=30)
        before = snapshot(root)
        sends_before = len(env["pick_dms"])
        reasons = () if base else CASES[name]
        expected = "skip_state_review" if reasons else {"K3": "skip_not_deliberate", "K4": "no_games"}.get(
            name, "deliberate_skip")
        exit_code = int(bool(reasons) or name == "K3")
        result = command("day-outcome", "--date", day, "--config", str(cfg), "--expected-username", "operator",
                         "--dm-recipient", "operator.invalid", code=exit_code)
        morning_dms = env["pick_dms"][sends_before:]
        data = json.loads(result.stdout)
        assert data["state"] == expected and data["ok"] == (exit_code == 0), (name, data)
        assert len(morning_dms) == exit_code and snapshot(root) == before
        tokens = {"installed": "The installed dated observation", "archived": "An archived dated observation",
                  "revocation": "The season switch records a revocation"}
        assert all((text in data["detail"]) == (reason in reasons) for reason, text in tokens.items()), (name, data)
        if name in {"reenable_retains_mismatch", "matching_archive_hides_M2"}:
            assert ("16:10", "automatic", 30, False) in observed_at_check
            assert json.loads((account / "contest_streak.json").read_text())["active_streak"] == 30
        print(json.dumps({"case": name, "checks": checks, "skip_inputs": observed_at_check, "exit": result.exit_code,
                          "data": data, "DM": morning_dms, "files_unchanged": True}), flush=True)


MANUAL_BAD_NAME = "contest_streak.manual.revoked-invalid.json"


if __name__ == "__main__":
    identify_package()
    with TemporaryDirectory() as temp:
        for case in CASES:
            run_case(Path(temp) / case, case, "--base" in sys.argv)
