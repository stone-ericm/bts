"""Exercise real CLI commands and downstream base reads with offline I/O."""
import datetime as calendar
import json
import os
import socket
import sys
import tempfile
from datetime import datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

from click.testing import CliRunner
from pytest import MonkeyPatch

from bts import cli, contest_fetch, contest_state, dm, orchestrator, picks, scheduler, strategy
from bts.health import contest_state as health
from bts.leaderboard import auth
from bts.saver_state import load_saver_state
from probe_text import offline, run_case
from tests.test_pick_chosen_on_cs5a import DAY, NOW, set_automatic
from tests.test_saved_pick_paths_cs5a import scheduler_state, world
from tests.test_stale_delivery_cs4c import seed_automatic

ET = ZoneInfo("America/New_York")


def invoke(*words, directory=None):
    options = ["--picks-dir", str(directory)] if directory is not None else []
    return CliRunner().invoke(cli.cli, [*words, *options], catch_exceptions=False)


def observation(day, directory, streak=12, best=15, saver="used"):
    return invoke("set-contest-streak", "--date", day, "--streak", str(streak), "--best-streak", str(best),
                  "--saver", saver, directory=directory)


def state_at(directory, now):
    state = contest_state.load_decision_streak_state(directory, require_contest_state=True, now=now)
    return state.observation_kind, state.streak, state.saver_available


def sent_pick(root, mp, repair):
    config, directory, clock, statuses = world.__wrapped__(root, mp)
    seed_automatic(directory, streak=7)
    auto = directory / "account_state/contest_streak.json"
    auto.write_text(json.dumps(dict(json.loads(auto.read_text()), source_date=None)))
    mp.setattr(calendar, "datetime", contest_state.datetime)
    mp.setattr(scheduler, "count_new_confirmations", lambda *a, **k: 0)
    mp.setattr(auth, "load_session_cookies", lambda: {})
    mp.setattr(auth, "extract_uid", lambda _: "1")
    mp.setattr(auth, "fetch_login_session", lambda **_: SimpleNamespace(user_id=1, username="offline", xsid="x"))
    mp.setattr(contest_fetch, "fetch_profile", lambda *a: {"predictions": []})
    mp.setattr(contest_fetch, "fetch_pending_predictions", lambda *a: [])
    mp.setattr(cli, "_fetch_rounds", lambda: {1: calendar.date.fromisoformat(DAY)})
    mp.setattr(cli, "_fetch_bts_to_mlb", lambda: {})
    mp.setattr(scheduler, "fetch_schedule", lambda *a: [{"gamePk": 101}, {"gamePk": 102}])
    cfg = root / "test.toml"
    cfg.write_text('[scheduler]\npick_delivery="dm"\nentry_intent="enter"\ncontest_state_required=true\n')
    result = run_case(config)
    sel = result["selection"]
    daily = sel.pick_result.daily
    scheduler._deliver_and_lock_pick(daily, config, directory, scheduler_state(), DAY, "probe", selection=sel)
    assert picks.pick_was_delivered(picks.load_pick(DAY, directory))
    pick_file = directory / f"{DAY}.json"
    decision = directory / DAY / "decision.json"
    old_bytes = pick_file.read_bytes(), decision.read_bytes()
    if repair:
        filed = observation(DAY, directory)
        assert filed.exit_code == 0, filed.output
        print("filed", filed.output.strip())
    scheduler._morning_state_check(config, directory, DAY, NOW, NOW + timedelta(minutes=30))
    restarted = run_case(config)
    assert old_bytes == (pick_file.read_bytes(), decision.read_bytes())
    watch = invoke("check-pick-entered", "--now-et", f"{DAY}T18:45:00", "--expected-username", "offline",
                   "--dm-recipient", "offline.test", directory=directory)
    assert old_bytes == (pick_file.read_bytes(), decision.read_bytes())
    mp.setattr(scheduler, "poll_game_result", lambda *a: "final")
    mp.setattr(picks, "resolve_daily_slot_results", lambda *a: {"primary": "hit"})
    grade = scheduler.run_result_polling(101, DAY, directory)
    slate = NOW.date()
    clock[0] = datetime.combine(slate + timedelta(days=1), time(0, 5), ET)
    eod = health.check(directory, expected=True, now=datetime.combine(slate, time.max, ET), historical_day=slate)
    clock[0] += timedelta(hours=8, minutes=25)
    outcome = invoke("day-outcome", "--date", DAY, "--config", str(cfg), "--expected-username", "offline",
                     "--dm-recipient", "offline.test", directory=directory)
    refusal = observation(DAY, directory)
    print("repair", repair, "restart locked", restarted["pick_result"].locked, "entry", watch.exit_code, watch.output,
          "grade", grade, picks.load_pick(DAY, directory).result, picks.load_streak(directory, season=2027),
          "EOD", [(a.level, a.message) for a in eod], "08:30", outcome.exit_code, outcome.output,
          "next auto", state_at(directory, clock[0]), "late dated", refusal.exit_code, "DMs", dm.send_dm.call_args_list)


def saver_matrix(root, mp):
    config, _, clock, _ = world.__wrapped__(root, mp)
    base_clock = contest_state.datetime
    mp.setattr(calendar, "datetime", base_clock)
    mp.setattr(cli, "datetime", base_clock)
    for dated in (False, True):
        for prior in ("active", "not_earned", "used", "uninitialized"):
            directory = root / f"{dated}-{prior}"
            clock[0] = NOW
            set_automatic(directory, streak=10)
            if prior != "uninitialized":
                r = invoke("saver-state", "--init", prior, "--season", "2027", directory=directory)
                assert r.exit_code == 0, r.output
            if dated:
                r = observation(DAY, directory, streak=10, best=10, saver="used")
                assert r.exit_code == 0, r.output
            before = state_at(directory, NOW)
            tomorrow = NOW + timedelta(days=1)
            untreated = state_at(directory, tomorrow)
            r = invoke("saver-state", "--use", directory=directory)
            result = state_at(directory, NOW)
            next_day = state_at(directory, tomorrow)
            print(dated, prior, before, r.exit_code, r.output.strip().replace("\n", " | "), result,
                  "next untreated/treated", untreated, next_day)
    wrong = root / "wrong"
    wrong.mkdir()
    mp.chdir(wrong)
    r = observation(DAY, None)
    print("wrong cwd write", r.exit_code, (wrong / "data/picks/account_state/contest_streak.manual.json").exists(),
          "saver", invoke("saver-state", "--use").output.strip())
    clock[0] = datetime(2027, 12, 31, 23, 59, tzinfo=ET)
    year = root / "year"
    print("year init old", invoke("saver-state", "--init", "active", directory=year).output.strip())
    clock[0] += timedelta(minutes=2)
    print("new year implicit", invoke("saver-state", "--use", directory=year).output.strip())
    print("new init", invoke("saver-state", "--init", "active", directory=year).output.strip())
    print("old explicit init", invoke("saver-state", "--init", "used", "--season", "2027", directory=year).output.strip())
    print("year flags", [load_saver_state(year, season=y).state for y in [2027, 2028]],
          "old dated exit", observation("2027-12-31", year).exit_code)
    mdp = strategy._load_mdp()
    for p in (.75, .79, .80, .81, .82, .83, .95):
        for streak in range(10, 16):
            ctx = strategy.DecisionContext(p, p, True, "2027-07-16", True, mdp, streak, "trusted")
            actions = [strategy.resolve_policy_decision(ctx, streak, saver).action for saver in (True, False)]
            if actions[0] != actions[1]:
                print("policy effect", streak, p, actions)


def ops_cli(root, mp):
    config, directory, clock, _ = world.__wrapped__(root, mp)
    mp.setattr(calendar, "datetime", contest_state.datetime)
    clock[0] = datetime(2026, 12, 31, 23, 59, tzinfo=ET)
    args = ("season-contract", "revoke", "--season", "2027", "--reason", "offline restore")
    r = invoke(*args, directory=directory)
    print("revoke before 2027", r.exit_code, r.output.strip())
    clock[0] += timedelta(minutes=3)
    r = invoke(*args, directory=directory)
    print("revoke after midnight", r.exit_code, r.output.splitlines()[:4])
    r = invoke("season-contract", "status", directory=directory)
    print("status revoked", r.exit_code, r.output.splitlines()[:6])
    r = invoke("season-contract", "enable", "--season", "2027", directory=directory)
    print("enable", r.exit_code, r.output.splitlines()[:4])
    mp.setattr(auth, "load_session_cookies", lambda: {})
    mp.setattr(auth, "extract_uid", lambda _: "1")
    for error in (auth.RateLimitedLoginError("429"), auth.TransientAuthError("upstream"), auth.AuthError("login error")):
        mp.setattr(auth, "fetch_login_session", Mock(side_effect=error))
        r = invoke("fetch-contest-streak", directory=directory)
        print(type(error).__name__, r.exit_code, r.output.strip())


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory, MonkeyPatch.context() as mp:
        mp.setattr(socket.socket, "connect", offline)
        mp.setattr(socket, "create_connection", offline)
        root = Path(directory)
        mode = sys.argv[1]
        if mode == "repair" or mode == "control":
            sent_pick(root, mp, mode == "repair")
        elif mode == "saver":
            saver_matrix(root, mp)
        else:
            ops_cli(root, mp)
