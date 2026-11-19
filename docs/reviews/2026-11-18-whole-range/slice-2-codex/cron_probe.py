"""Render both crontabs, parse their actual argv, and drive cron-facing CLI branches offline."""

from contextlib import ExitStack
from datetime import timedelta
import gzip
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import day_probe as probe
import offline_fixtures as fixture


RESEARCH = '[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n'
ENTER = ('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n'
         'contest_state_required = true\n[health_checks]\ncontest_state_expected = true\n')
MARKER = "# BTS-HETZNER"
RECIPIENT = "stonehengee.bsky.social"
USERNAME = "stonehengee"
CLI = "from bts.cli import cli; cli()"


def make_executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o700)


def load_table(root, source, intent):
    home = root / "home"
    app = home / "projects/bts"
    app.mkdir(parents=True)
    (home / "logs").mkdir()
    config = home / ".bts-orchestrator.toml"
    config.write_text(ENTER if intent == "enter" else RESEARCH)
    (app / ".env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://heartbeat.invalid/fixture\n")
    table = root / "crontab.txt"
    table.write_text(f"17 0 * * * true\n39 7 * * * true {MARKER}\n")
    before = table.read_bytes()
    commands = root / "commands.jsonl"
    uv = home / ".local/bin/uv"
    bin_path = root / "bin"
    shim = f'#!/bin/sh\nexec "{sys.executable}" "{Path(__file__).resolve()}" --shim "$0" "$@"\n'
    make_executable(uv, shim)
    for cmd in ("crontab", "curl", "flock", "date"):
        make_executable(bin_path / cmd, shim)
    env = dict(os.environ, HOME=str(home), PATH=f"{bin_path}:/usr/bin:/bin",
               PYTHONPATH=str(source / "src"), HEALTHCHECKS_PING_URL="https://hc.invalid/fixture",
               WR2_CRON_TABLE=str(table), WR2_CRON_COMMANDS=str(commands))
    shown = subprocess.run(["/bin/bash", str(source / "scripts/cron-setup-hetzner.sh"), "show"],
                           capture_output=True, text=True, env=env, cwd=root, timeout=12)
    assert shown.returncode == 0, shown
    assert table.read_bytes() == before
    entries = []
    for line in shown.stdout.split("Would install:\n")[1].splitlines():
        if not line or line.startswith("#"):
            continue
        minute, hour, dom, month, dow, cmd = line.split(maxsplit=5)
        cmd = cmd.replace(r"\%", "%").replace("/usr/bin/flock", shlex.quote(str(bin_path / "flock")))
        ran = subprocess.run(["/bin/sh", "-c", cmd], env=env, cwd=root, capture_output=True,
                             text=True, timeout=10)
        assert ran.returncode == 0, (line, ran)
        calls = [json.loads(log) for log in commands.read_text().splitlines()]
        commands.write_text("")
        entries.append({"spec": [minute, hour, dom, month, dow], "source": line, "calls": calls})
    return {"intent": intent, "version": source.name, "show": shown.stdout, "entries": entries}


def shim():
    name = Path(sys.argv[2]).name
    args = sys.argv[3:]
    if name == "crontab":
        assert args == ["-l"], args
        print(Path(os.environ["WR2_CRON_TABLE"]).read_text(), end="")
        return
    if name == "date":
        values = {("-d", "yesterday", "+%Y-%m-%d"): "2027-07-15",
                  ("-d", "7 days ago", "+%Y-%m-%d"): "2027-07-09",
                  ("+%Y-%m-%d",): "2027-07-16", ("+%Y",): "2027"}
        print(values[tuple(args)])
        return
    if name == "uv" and args[:3] == ["run", "bts", "entry-intent"]:
        os.execv(sys.executable, [sys.executable, "-c", CLI, *args[2:]])
    with Path(os.environ["WR2_CRON_COMMANDS"]).open("a") as stream:
        stream.write(json.dumps([name, *args]) + "\n")
    if name == "flock":
        assert args[:2] == ["-n", "/home/bts/.check-results.lock"], args
        os.execv(args[2], args[2:])
    assert name in {"uv", "curl"}, name


def validate_cli(table):
    from bts.cli import cli

    validations = []
    for entry in table["entries"]:
        for call in entry["calls"]:
            if call[:3] == ["uv", "run", "bts"]:
                command = cli
                tokens = call[3:]
                parents = []
                context = command.make_context("bts", [], resilient_parsing=True)
                parents.append(context)
                names = []
                try:
                    while hasattr(command, "get_command"):
                        name, *tokens = tokens
                        names.append(name)
                        command = command.get_command(context, name)
                        assert command is not None, call
                        if hasattr(command, "get_command"):
                            context = command.make_context(name, [], parent=context, resilient_parsing=True)
                            parents.append(context)
                    with command.make_context(names[-1], tokens.copy(), parent=context) as parsed:
                        validations.append({"spec": entry["spec"], "call": call, "params": parsed.params})
                    help_result = probe.invoke([*names, *tokens, "--help"])
                    assert help_result["exit"] == 0, help_result
                finally:
                    for parent in reversed(parents):
                        parent.close()
            elif call[:4] == ["uv", "run", "python", "scripts/check_heartbeat.py"]:
                options = call[4:]
                assert options == ["--heartbeat-path", "data/.heartbeat", "--ping-url",
                                   "https://heartbeat.invalid/fixture"], options
                help_result = subprocess.run([sys.executable, str(fixture.REPO / call[3]), "--help"],
                                             capture_output=True, text=True, timeout=5)
                assert help_result.returncode == 0 and all(
                    arg in help_result.stdout for arg in options if arg.startswith("--")), help_result
            elif call[0] == "curl":
                assert call == ["curl", "-fsS", "--max-time", "5", "https://hc.invalid/fixture"], call
            else:
                assert call[0] == "flock", call
    return validations


def dm_events(world, start=0):
    return [event for event in world.events[start:] if event[1] == "dm"]


def run_fetches(root):
    from bts.leaderboard import auth

    def auth_failed():
        raise auth.AuthError("synthetic session rejected")

    results = {}
    for intent in ("enter", "research"):
        case = root / intent
        case.mkdir()
        world = probe.World(probe.MID, "normal")
        picks = case / "data/picks"
        fixture.seed(picks, world.day)
        args = ["fetch-contest-streak", "--picks-dir", str(picks), "--expected-username", USERNAME]
        if intent == "enter":
            args += ["--dm-recipient", RECIPIENT]
        with fixture.installed(world, case), patch.object(auth, "load_session_cookies", auth_failed):
            attempts = []
            for hour, minute in ((1, 10), (2, 10), (10, 30), (13, 30)):
                world.now = probe.at(world.day, hour, minute)
                start = len(world.events)
                output = probe.invoke(args)
                attempts.append({"result": output, "dm": dm_events(world, start)})
            results[intent] = {"attempts": attempts, "status": probe.read(
                case / "data/health_state/contest_streak_fetch_status.json")}
    return results


def run_entry(root, private=False, short=False):
    import bts.cli as cli_module
    from bts import contest_fetch, picks as picks_io, scheduler
    from bts.leaderboard import auth

    world = probe.World(probe.MID, "normal")
    if short:
        original_at = probe.at
        # Present a one-minute send window between adjacent 15-minute cron ticks.
        world.game_hours = {101: 15, 102: 18, 103: 21}
        world.now = original_at(world.day, 14, 54)
    picks = root / "data/picks"
    fixture.seed(picks, world.day)
    mode, intent = ("private", "research") if private else ("dm", "enter")
    config = probe.config_for(picks, delivery=mode, intent=intent, required=not private)
    settings = probe.toml(root / "orchestrator.toml", config)
    args = ["check-pick-entered", "--picks-dir", str(picks),
            "--expected-username", USERNAME, "--dm-recipient", RECIPIENT]
    session = SimpleNamespace(user_id=100, username=USERNAME, xsid="fixture")
    pending = []
    outcomes = []

    def attempt_at(hour, minute):
        world.now = probe.at(world.day, hour, minute)
        start = len(world.events)
        outcomes.append({"at": world.now.isoformat(), "cli": probe.invoke(args),
                         "dm": dm_events(world, start)})

    with fixture.installed(world, root), ExitStack() as scope:
        scope.enter_context(patch.object(auth, "load_session_cookies", return_value={}))
        scope.enter_context(patch.object(auth, "extract_uid", return_value=100))
        scope.enter_context(patch.object(auth, "fetch_login_session", return_value=session))
        scope.enter_context(patch.object(cli_module, "_fetch_rounds", return_value={42: world.now.date()}))
        scope.enter_context(patch.object(cli_module, "_fetch_bts_to_mlb", return_value={701: 101}))
        scope.enter_context(patch.object(contest_fetch, "fetch_profile", return_value={"predictions": []}))
        scope.enter_context(patch.object(contest_fetch, "fetch_pending_predictions", return_value=pending))
        if short:
            attempt_at(14, 45)
            world.now = probe.at(world.day, 14, 54)
        scheduler.run_day(world.day, config)
        daily = picks_io.load_pick(world.day, picks)
        assert daily and daily.pick.game_pk == 101, daily
        digest = (picks / world.day / "decision.json").read_text()
        if short:
            attempt_at(15, 0)
        else:
            for hour, minute in ((15, 0), (15, 15), (15, 30), (15, 45)):
                attempt_at(hour, minute)
            pending.append({"roundId": 42, "unitId": 801, "playerId": 701,
                            "number": 1, "result": None})
            snapshots = root / "data/leaderboard/static_snapshots/units"
            snapshots.mkdir(parents=True)
            data = {"units": [{"id": 801, "roundId": 42, "unitType": "ROUND", "feedId": 101,
                               "startDate": world.day + "T16:00:00Z"}]}
            (snapshots / "2027-07-15T190000Z.json.gz").write_bytes(gzip.compress(json.dumps(data).encode()))
            attempt_at(15, 46)
            attempt_at(15, 56)
        world.now = probe.at("2027-07-16", 8, 30)
        day = probe.invoke(["day-outcome", "--date", world.day, "--config", str(settings), "--picks-dir",
                            str(picks), "--expected-username", USERNAME, "--dm-recipient", RECIPIENT])
        assert (picks / world.day / "decision.json").read_text() == digest
        return {"entry": outcomes, "day_outcome": day, "events": dm_events(world),
                "decision": json.loads(digest)}


def scorer_overlap(root):
    from bts import picks as picks_io, scheduler

    class ScoredAfterRestart(Exception):
        pass

    world = probe.World(probe.MID, "normal")
    picks = root / "data/picks"
    fixture.seed(picks, world.day)
    config = probe.config_for(picks, delivery="public")
    sleep = world.sleep

    def suspend_after_send(seconds):
        existing = picks_io.load_pick(world.day, picks)
        if existing and existing.bluesky_posted:
            raise ScoredAfterRestart()
        sleep(seconds)

    world.sleep = suspend_after_send
    log = {}
    with fixture.installed(world, root):
        try:
            scheduler.run_day(world.day, config)
        except ScoredAfterRestart:
            pass
        else:
            raise AssertionError("expected a committed pick, still unscored")
        world.now = probe.at("2027-07-16", 1, 0)
        original = picks_io.resolve_daily_slot_results
        nested = False

        def just_before_scoring(*args, **kwargs):
            nonlocal nested
            if not nested:
                nested = True
                log["cron_01"] = probe.invoke(["check-results", "--date", world.day,
                                                "--wait-deadline-et", "06:00", "--picks-dir", str(picks)])
            return original(*args, **kwargs)

        with patch.object(picks_io, "resolve_daily_slot_results", just_before_scoring):
            log["poll"] = scheduler.run_result_polling(101, world.day, picks)
        log["scored"] = probe.read(picks / "streak.json")
        for hour, minute in ((2, 0), (7, 40)):
            world.now = probe.at("2027-07-16", hour, minute)
            log[f"reconcile_{hour:02}{minute:02}"] = probe.invoke(["reconcile", "--picks-dir", str(picks)])
        log["after_reconcile"] = probe.read(picks / "streak.json")
        log["events"] = [event for event in world.events if event[1] in {"dm", "post", "reply"}]
    return log


def main():
    if sys.argv[1:2] == ["--shim"]:
        shim()
        return
    with TemporaryDirectory(prefix="cron-", dir=Path(__file__).resolve().parent) as temp:
        root = Path(temp)
        tables = {}
        for revision, source in (("head", fixture.REPO), ("base", Path(__file__).parent.resolve() / "base")):
            for intent in ("research", "enter"):
                tables[f"{revision}_{intent}"] = load_table(root / f"{revision}_{intent}", source, intent)
                if revision == "head":
                    tables[f"{revision}_{intent}"]["validated"] = validate_cli(tables[f"{revision}_{intent}"])
        for name in ("fetches", "entry", "private_stale_tab", "entry_short", "scorer"):
            (root / name).mkdir()
        result = {"tables": tables, "fetches": run_fetches(root / "fetches"),
                  "entry": run_entry(root / "entry"),
                  "private_stale_tab": run_entry(root / "private_stale_tab", private=True),
                  "entry_short": run_entry(root / "entry_short", short=True),
                  "scorer_overlap": scorer_overlap(root / "scorer")}
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
