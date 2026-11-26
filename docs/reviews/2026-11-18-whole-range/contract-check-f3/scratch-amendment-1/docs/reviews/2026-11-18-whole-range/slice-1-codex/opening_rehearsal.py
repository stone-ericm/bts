"""Offline full scheduler and Click rehearsal. Run with the repository venv.

The legacy seed runs in its own subprocess with archived 619abda7 on PYTHONPATH.
Pass --case to run one scenario, and --restart to run its restart counterpart.
"""

import argparse
import builtins
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta, timezone
import gzip
from importlib import import_module
from io import StringIO
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ET = ZoneInfo("America/New_York")
DAYS = ("2027-03-24", "2027-03-25", "2027-03-26")
OLD_DAY = "2026-09-27"
DM = "operator.example"
ACCOUNT = "synthetic_account"
UID = "synthetic_uid"
CASES = ("right", "forgot", "before_first", "after_last", "corrected",
         "automatic_match", "automatic_old_file")


def at(day, hhmm):
    return datetime.fromisoformat(f"{day}T{hhmm}").replace(tzinfo=ET)


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def no_op(*args, **kwargs):
    pass


class SimulatedRestart(BaseException):
    pass


class Clock:
    def __init__(self, now):
        self.now = now
        self.events = []

    def when(self, moment, callback):
        self.events.append((moment, callback))
        self.events.sort(key=lambda event: event[0])

    def sleep(self, seconds, **kwargs):
        self.move(self.now + timedelta(seconds=seconds))

    def move(self, target):
        assert target >= self.now, (self.now, target)
        while self.events and self.events[0][0] <= target:
            self.now, callback = self.events.pop(0)
            callback()
        self.now = target


@contextmanager
def sandbox(clock, directory):
    """Scoped wall clock, isolated cwd and a fail-fast socket guard."""
    import datetime as datetime_module

    class PinnedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return clock.now.replace(tzinfo=None)
            return clock.now.astimezone(tz)

    class PinnedDate(date):
        @classmethod
        def today(cls):
            return clock.now.date()

    def fail_network(*args, **kwargs):
        # This represents a harness configuration error. Do not let application
        # fault handlers turn an unexpected attempt into an ordinary failure.
        raise NetworkAttempt("unexpected network syscall")

    class NetworkAttempt(BaseException):
        pass

    import_original = builtins.__import__
    pinned_module = SimpleNamespace(**vars(datetime_module))
    pinned_module.datetime = PinnedDateTime
    pinned_module.date = PinnedDate

    def import_with_cli_clock(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "datetime" and (globals or {}).get("__name__") == "bts.cli":
            return pinned_module
        return import_original(name, globals, locals, fromlist, level)

    # Load only exercised modules. No prediction model import here or at module scope.
    for name in (
        "cli", "contest_state", "daily_decision", "entry_receipt", "heartbeat",
        "picks", "saver_state", "scheduler", "strategy", "slate", "posting",
        "health.alert", "health.contest_state", "health.park_drag_freshness",
        "health.post_failure", "health.runner", "features.park_drag_producer",
    ):
        import_module(f"bts.{name}")
    start = Path.cwd()
    with ExitStack() as stack:
        os.chdir(directory)
        stack.callback(os.chdir, start)
        for name, module in tuple(sys.modules.items()):
            if not name.startswith("bts."):
                continue
            for attr, value in tuple(vars(module).items()):
                if value is datetime:
                    stack.enter_context(patch.object(module, attr, PinnedDateTime))
                elif value is date:
                    stack.enter_context(patch.object(module, attr, PinnedDate))
        stack.enter_context(patch.object(builtins, "__import__", import_with_cli_clock))
        for method in ("__init__", "connect", "connect_ex", "sendto", "sendmsg"):
            if hasattr(socket.socket, method):
                stack.enter_context(patch.object(socket.socket, method, fail_network))
        for method in ("create_connection", "getaddrinfo", "gethostbyname", "gethostbyname_ex"):
            stack.enter_context(patch.object(socket, method, fail_network))
        yield stack


def auth_patches(stack, profile, rounds):
    from bts import contest_fetch
    from bts.leaderboard import auth
    from bts import cli as cli_module

    session = SimpleNamespace(user_id=UID, username=ACCOUNT, xsid="synthetic_xsid")
    stack.enter_context(patch.object(auth, "load_session_cookies", lambda: {}))
    stack.enter_context(patch.object(auth, "extract_uid", lambda cookies: UID))
    stack.enter_context(patch.object(auth, "fetch_login_session", lambda **kw: session))
    stack.enter_context(patch.object(contest_fetch, "fetch_profile", lambda *args: profile()))
    stack.enter_context(patch.object(cli_module, "_fetch_rounds", lambda: rounds))


def invoke(*args, expected=0):
    from bts.cli import cli

    result = CliRunner().invoke(cli, list(map(str, args)), catch_exceptions=False)
    if expected is not None:
        assert result.exit_code == expected, (args, result.exit_code, result.output)
    return result


def seed_legacy(directory):
    """Use prior writers and a synthetic 2026 account record and scored pick."""
    from bts import picks, scheduler
    from bts.health.alert import Alert, dispatch_dm_for_health_alerts

    clock = Clock(at("2026-10-21", "10:00"))
    picks_dir = directory / "data/picks"
    with sandbox(clock, directory) as stack:
        auth_patches(stack, lambda: {"activeStreak": 17, "seasonBestStreak": 26,
                                    "predictions": [{"roundId": 26, "result": "hit"}]},
                     {26: date.fromisoformat(OLD_DAY)})
        stack.enter_context(patch("bts.dm.send_dm", lambda *args: "old-alert"))
        stack.enter_context(patch("bts.health.alert.send_dm", lambda *args: "old-alert"))
        picks.save_streak(19, picks_dir, saver_available=False)
        invoke("fetch-contest-streak", "--picks-dir", picks_dir, "--expected-username", ACCOUNT)
        invoke("saver-state", "--init", "used", "--force", "--season", 2026, "--picks-dir", picks_dir)
        historical_pick = picks.Pick("Prior Batter", 26, "OLD", 1, "Prior Pitcher",
                                    28, .86, [], False, 260, f"{OLD_DAY}T23:05:00Z")
        picks.save_pick(picks.DailyPick(
            date=OLD_DAY, run_time=f"{OLD_DAY}T22:15:00Z", pick=historical_pick,
            double_down=None, runner_up=None, result="hit", notification_sent=True,
            notification_channel="bluesky_dm"), picks_dir)
        scheduler.save_state(scheduler.SchedulerState(
            date=OLD_DAY, schedule_fetched_at=f"{OLD_DAY}T14:00:00Z",
            games=[{"game_pk": 260, "game_time_et": f"{OLD_DAY}T19:05:00-04:00"}],
            confirmed_game_pks=[260], runs_completed=[], pick_locked=True,
            pick_locked_at=f"{OLD_DAY}T22:15:00Z", result_status="final",
            next_wakeup=None), picks_dir)
        dispatch_dm_for_health_alerts(
            [Alert("CRITICAL", "contest_state", "synthetic prior-season health alert")],
            DM, status_path=picks_dir.parent / "health_state/health_dm_delivery_status.json")
    write_json(directory / "data/external/park_drag/park_drag_manifest.json",
               {"max_source_game_date": OLD_DAY})
    (directory / "data/external/park_drag/park_drag_export.csv").write_text(
        f"venue_id,date,park_drag_delta\n3313,{OLD_DAY},0.001\n")


class Season:
    def __init__(self, directory):
        self.root = directory
        self.picks = directory / "data/picks"
        self.account_dir = self.picks / "account_state"
        self.config = directory / ".bts-orchestrator.toml"
        self.clock = Clock(at(DAYS[0], "00:01"))
        self.journal = StringIO()
        self.messages = []
        self.predictions = []
        self.profile_predictions = [{"roundId": 26, "result": "hit"}]
        self.streak = 0
        self.rounds = {26: date.fromisoformat(OLD_DAY)}
        self.game_starts = {DAYS[0]: ("19:05",), DAYS[1]: ("16:10", "16:11", "16:13"),
                            DAYS[2]: ("17:05", "17:07", "17:09")}
        self.entries = {}
        self.restarts = []
        self.cut_after_transport = None
        self.first_lineups = {DAYS[0]: "16:00", DAYS[1]: "15:00", DAYS[2]: "16:00"}
        self.projected_on = {}
        self.cascade_outages = set()
        self.fetch_failure = None

    def old_seed(self):
        base = HERE / "base/src"
        if not (base / "bts/picks.py").is_file():
            raise RuntimeError("Archive 619abda7 src into .codex-review/whole-range/base first")
        env = dict(os.environ, PYTHONPATH=str(base), PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--seed", str(self.root)],
                       check=True, cwd=self.root, env=env)

    def write_config(self, mode, intent, required):
        flags = str(required).lower()
        self.config.write_text(f"""
[orchestrator]
picks_dir = "data/picks"
models_dir = "data/models"
data_dir = "data/processed"
heartbeat_path = "data/.heartbeat"
[scheduler]
pick_delivery = "{mode}"
entry_intent = "{intent}"
contest_state_required = {flags}
live_forward_capture_on_lock = false
[health_checks]
enabled = true
contest_state_expected = {flags}
[bluesky]
dm_recipient = "{DM}"
[[tiers]]
name = "synthetic"
""")

    def restore(self):
        self.write_config("private", "research", False)
        self.command("entry-intent", "--config", self.config, contains="research")
        self.write_config("dm", "enter", False)
        self.command("entry-intent", "--config", self.config, expected=1, contains="contest")
        self.command("schedule", "--config", self.config, expected=2, contains="refusing to start")
        assert not (self.root / "data/.heartbeat").exists()
        self.write_config("dm", "enter", True)
        self.command("entry-intent", "--config", self.config, contains="enter")
        self.command("season-contract", "revoke", "--season", 2027, "--reason",
                     "restored from the offseason snapshot")
        self.status(contains="live decision now: refused")
        # Health checks need existence only. The cascade never reads or trains these.
        models = self.root / "data/models"
        models.mkdir(exist_ok=True)
        for day in (*DAYS, "2027-03-27"):
            (models / f"blend_{day}.pkl").touch()
        assert not (self.account_dir / "saver_state.json").exists()
        assert read_json(self.account_dir / "contest_streak.json")["active_streak"] == 17

    def command(self, *args, expected=0, contains=None):
        file_arg = ("--picks-dir", self.picks)
        if args[0] in {"entry-intent", "schedule", "data", "park-drag-refresh"}:
            file_arg = ()
        result = invoke(*args, *file_arg, expected=expected)
        tail = ""
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            tail = f" [{result.exception!r}]"
        print(self.clock.now.isoformat(), "bts", *args, f"(exit {result.exit_code}){tail}",
              "\n" + result.output.rstrip(), file=self.journal)
        if contains:
            assert contains in result.output, result.output
        return result.output

    def dated(self, day, streak, saver="not_earned", best=None):
        self.command("set-contest-streak", "--date", day, "--streak", streak,
                     "--best-streak", streak if best is None else best, "--saver", saver)

    def status(self, contains=None):
        return self.command("season-contract", "status", contains=contains)

    def fetch(self, expected=0, contains=None):
        return self.command("fetch-contest-streak", "--expected-username", ACCOUNT,
                            "--dm-recipient", DM, expected=expected, contains=contains)

    def profile(self):
        if self.fetch_failure is not None:
            raise self.fetch_failure
        return {"activeStreak": self.streak, "seasonBestStreak": self.streak,
                "predictions": list(self.profile_predictions)}

    def schedule(self, day):
        games = []
        for i, time in enumerate(self.game_starts.get(day, ())):
            start = at(day, time)
            finished = self.clock.now >= start + timedelta(hours=2)
            games.append({
                "gamePk": 27000 + 10 * int(day[-2:]) + i,
                "gameDate": start.astimezone(timezone.utc).isoformat(),
                "gameType": "R",
                "status": {"abstractGameCode": "F" if finished else "P",
                           "detailedState": "Final" if finished else "Scheduled"},
                "teams": {"home": {"team": {"name": f"Synthetic Home {i}"}},
                          "away": {"team": {"name": f"Synthetic Away {i}"}}},
            })
        return games

    def lineups(self, pks):
        day = self.clock.now.date().isoformat()
        sides = {"home", "away"} if self.clock.now >= at(day, self.first_lineups[day]) else set()
        return {pk: sides for pk in pks}

    def detailed(self, day):
        return {game["gamePk"]: {"abstract": game["status"]["abstractGameCode"],
                                "detailed": game["status"]["detailedState"]}
                for game in self.schedule(day)}

    def mark_predictions(self, day):
        import pandas as pd

        projected_until = self.projected_on.get(day)
        projected = projected_until is not None and self.clock.now < at(day, projected_until)
        missing_lineups = self.clock.now < at(day, self.first_lineups[day])
        outage = (day, self.clock.now.strftime("%H:%M")) in self.cascade_outages
        self.predictions.append({"at": self.clock.now.isoformat(), "date": day,
                                 "available": not outage and (projected or not missing_lineups),
                                 "projected": projected})
        if outage or (missing_lineups and not projected):
            return pd.DataFrame()
        rows = []
        for i, game in enumerate(self.schedule(day)):
            rows.append({"batter_name": f"Batter {game['gamePk']}", "batter_id": game["gamePk"],
                         "team": f"S{i}", "lineup": 1, "pitcher_name": f"Pitcher {i}",
                         "pitcher_id": i + 1, "p_game_hit": .92 - i * .02,
                         "flags": "PROJECTED" if projected else "",
                         "game_pk": game["gamePk"], "game_time": game["gameDate"],
                         "status": "Scheduled"})
        return pd.DataFrame(rows)

    def record_transport(self, channel, recipient, body):
        ident = f"synthetic-{len(self.messages) + 1:03}"
        self.messages.append({"at": self.clock.now.isoformat(), "channel": channel,
                              "recipient": recipient, "body": body, "id": ident})
        if self.cut_after_transport is not None and body.startswith("Today's pick"):
            callback, self.cut_after_transport = self.cut_after_transport, None
            callback()
        return ident

    def dm(self, recipient, body):
        return self.record_transport("dm", recipient, body)

    def post(self, body, *args):
        return self.record_transport("post", "public", body)

    def poll_game(self, pk):
        day = self.clock.now.date().isoformat()
        for game in self.schedule(day):
            if game["gamePk"] == pk:
                return "final" if game["status"]["abstractGameCode"] == "F" else "live"
        raise AssertionError(pk)

    def pending(self, *args):
        return self.entries.get(self.clock.now.date().isoformat(), [])

    def enter(self, day, *, expected=0):
        from bts.picks import load_pick, pick_was_delivered

        daily = load_pick(day, self.picks)
        if daily is not None and pick_was_delivered(daily):
            round_id = int(day[-2:]) + 100
            self.rounds[round_id] = date.fromisoformat(day)
            self.entries[day] = [
                {"roundId": round_id, "unitId": slot.game_pk, "number": i,
                 "playerId": slot.batter_id, "result": "pending"}
                for i, slot in enumerate((daily.pick, daily.double_down), start=1) if slot is not None
            ]
            units = [{"id": game["gamePk"], "unitType": "ROUND", "feedId": game["gamePk"],
                      "roundId": round_id, "startDate": f"{day}T04:00:00Z",
                      "endDate": f"{day}T23:59:59Z"} for game in self.schedule(day)]
            snapshot = self.root / "data/leaderboard/static_snapshots/units" / f"{day.replace('-', '')}T120000Z.json.gz"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(gzip.compress(json.dumps({"units": units}).encode()))
        self.command("check-pick-entered", "--expected-username", ACCOUNT, "--dm-recipient",
                     DM, "--now-et", self.clock.now.strftime("%Y-%m-%dT%H:%M:%S"), expected=expected)

    def settle(self, day):
        if self.entries.get(day):
            rows = [dict(row, result="hit") for row in self.entries[day]]
            self.profile_predictions.extend(rows)
            self.streak += len(rows)

    def start(self, day):
        # Real Click command, config loader and run_day. A simulated process death
        # unwinds the stack; a new invocation can recover only what landed on disk.
        with redirect_stderr(self.journal), redirect_stdout(self.journal):
            try:
                self.command("schedule", "--config", self.config)
            except SimulatedRestart:
                self.restarts.append(self.clock.now.isoformat())
                self.clock.sleep(60)
                return self.start(day)

    def outcome(self, day, expected):
        return self.command("day-outcome", "--date", day, "--config", self.config,
                            "--expected-username", ACCOUNT, "--dm-recipient", DM,
                            expected=expected)

    def snapshot(self, day):
        account = {path.name: read_json(path) for path in sorted(self.account_dir.glob("*.json"))}
        health_dir = self.picks.parent / "health_state"
        return {
            "day": day, "ended_at": self.clock.now.isoformat(),
            "pick": read_json(self.picks / f"{day}.json"),
            "decision": read_json(self.picks / day / "decision.json"),
            "scheduler": read_json(self.picks / day / "scheduler_state.json"),
            "model": read_json(self.picks / "streak.json"),
            "account": account,
            "health_status": {path.name: read_json(path) for path in sorted(health_dir.glob("*.json"))},
        }

    @contextmanager
    def run_offline(self):
        from bts import cli, orchestrator, picks, scheduler
        from bts.features import park_drag
        from bts.health import runner

        with sandbox(self.clock, self.root) as stack:
            auth_patches(stack, self.profile, self.rounds)
            stack.enter_context(patch("bts.contest_fetch.fetch_pending_predictions", self.pending))
            mocks = (
                (scheduler, "fetch_schedule", self.schedule),
                (scheduler, "check_confirmed_lineups", self.lineups),
                (scheduler, "poll_game_result", self.poll_game),
                (scheduler, "_check_hits_midgame", lambda *args: []),
                (scheduler, "_watchdog_ping_sleep", self.clock.sleep),
                (scheduler, "_poll_interval_sleep", lambda path, seconds: self.clock.sleep(seconds)),
                (scheduler, "_idle_until_next_wakeup", no_op),
                (scheduler, "notify_ready", no_op),
                (scheduler, "notify_watchdog", no_op),
                (scheduler, "heartbeat_watchdog", lambda *a, **k: ExitStack()),
                (runner, "read_systemd_nrestarts", lambda: None),
                (orchestrator, "run_cascade", lambda tiers, day: (self.mark_predictions(day), "synthetic")),
                (picks, "get_game_statuses_detailed", self.detailed),
                (picks, "get_game_statuses", lambda day: {pk: g["abstract"] for pk, g in self.detailed(day).items()}),
                (picks, "check_hit", lambda *a, **k: True),
                (cli, "_fetch_bts_to_mlb", lambda: {g["gamePk"]: g["gamePk"]
                                                   for d in DAYS for g in self.schedule(d)}),
            )
            for module, name, replacement in mocks:
                stack.enter_context(patch.object(module, name, replacement))
            stack.enter_context(patch("bts.dm.send_dm", self.dm))
            stack.enter_context(patch("bts.health.alert.send_dm", self.dm))
            stack.enter_context(patch("bts.posting.post_to_bluesky", self.post))
            stack.enter_context(patch("bts.posting.reply_to_bluesky", self.post))
            stack.enter_context(patch("bts.strategy._mdp_cache", None))
            stack.enter_context(patch.object(park_drag, "_CACHE", {}))
            stack.enter_context(patch.object(park_drag, "_WARNED", set()))
            yield


def stop_scheduler():
    raise SimulatedRestart()


def run_case(season, case, restart):
    from bts.daily_decision import decision_path
    from bts.picks import load_pick, pick_was_delivered

    snapshots = []
    outcomes = []
    season.restore()
    for index, day in enumerate(DAYS):
        if index:
            prior = DAYS[index - 1]
            prior_model = read_json(season.picks / "streak.json")
            season.clock.move(at(day, "01:00"))
            season.command("check-results", "--date", prior, "--wait-deadline-et", "06:00")
            assert read_json(season.picks / "streak.json") == prior_model
            for time in ("02:00", "07:40"):
                season.clock.move(at(day, time))
                result = season.command("reconcile", contains="No scoring changes detected")
                assert f"Streak: {season.streak}" in result, result
                model = read_json(season.picks / "streak.json")
                assert model["season"] == 2027 and model["streak"] == season.streak
            season.clock.move(at(day, "08:30"))
            prior_pick = load_pick(prior, season.picks)
            outcome = season.outcome(prior, expected=0 if prior_pick and pick_was_delivered(prior_pick) else 1)
            outcomes.append(json.loads(outcome)["state"])
        season.clock.move(at(day, "09:30"))
        if index < 2 or case not in {"automatic_match", "automatic_old_file"}:
            season.fetch(contains="reading recorded in the ledger")
            assert read_json(season.account_dir / "contest_streak.json")["active_streak"] == 17
        if index == 0 and case in {"forgot", "before_first", "after_last", "corrected"}:
            if case == "before_first":
                season.clock.when(at(day, "11:45"), lambda: season.dated(day, 0))
            elif case == "after_last":
                season.clock.when(at(day, "18:23"), lambda: season.dated(day, 0))
            elif case == "corrected":
                season.dated(day, 11, saver="used")
                season.first_lineups[day] = "18:15"
                season.projected_on[day] = "18:15"
                season.clock.when(at(day, "17:10"), lambda: season.dated(day, 0))
        elif index == 2 and case in {"automatic_match", "automatic_old_file"}:
            season.status()
            season.command("season-contract", "enable", "--season", 2027)
            if case == "automatic_match":
                season.fetch(contains="wrote")
                season.status(contains="live decision now: accepted")
                # Fetch already cold-initializes below 10 on a settled 2027 round.
                season.command("saver-state", "--init", "not_earned", "--season", 2027,
                               expected=1, contains="already initialized")
            else:
                season.status(contains="live decision now: refused")
        else:
            season.dated(day, season.streak)
        season.clock.move(at(day, "10:00"))
        first_game = at(day, season.game_starts[day][0])
        if index == 0 and restart:
            restart_at = "17:20" if case == "corrected" else "12:00"
            season.clock.when(at(day, restart_at), stop_scheduler)
        if index == 2 and restart and case in {"automatic_match", "automatic_old_file"}:
            season.clock.when(at(day, "12:00"), stop_scheduler)
        season.clock.when(first_game - timedelta(minutes=15), lambda: season.enter(day))
        season.clock.when(first_game + timedelta(hours=2, minutes=5), lambda: season.settle(day))
        season.start(day)
        # No-pick days can take the idle path at the last lineup check, leaving
        # cron/operator events to proceed independently of the scheduler.
        season.clock.move(max(season.clock.now, first_game + timedelta(hours=2, minutes=5)))
        snap = season.snapshot(day)
        snap["decision"] = read_json(decision_path(day, season.picks))
        snapshots.append(snap)
    season.clock.move(at("2027-03-27", "08:30"))
    last_pick = load_pick(DAYS[2], season.picks)
    outcome = season.outcome(DAYS[2], expected=0 if last_pick and pick_was_delivered(last_pick) else 1)
    outcomes.append(json.loads(outcome)["state"])

    assert not any(message["channel"] == "post" for message in season.messages)
    assert all(message["recipient"] == DM for message in season.messages)
    assert bool(season.restarts) == restart
    missing = {0} if case in {"forgot", "after_last"} else {2} if case == "automatic_old_file" else set()
    for i, snap in enumerate(snapshots):
        if i in missing:
            assert snap["decision"] is None and snap["pick"] is None, (case, snap)
            assert outcomes[i] == "no_decision", outcomes
        else:
            assert snap["decision"]["delivery_status"] == "delivered", (case, snap)
            pick = snap["pick"]
            assert pick["notification_sent"] and pick["result"] == "hit", (case, pick)
            assert pick["chosen_on"]["kind"] == ("automatic" if i == 2 and case == "automatic_match" else "dated")
            assert pick["chosen_on"]["streak"] == sum(
                1 + bool(earlier["pick"]["double_down"])
                for earlier in snapshots[:i] if earlier["pick"] is not None)
            assert outcomes[i] in {"entered", "entered_game_unverified"}, outcomes
            assert snap["model"]["season"] == 2027
            assert snap["model"]["streak"] < 10
    pick_ids = {snap["pick"]["notification_id"] for snap in snapshots if snap["pick"] is not None}
    pick_messages = [message for message in season.messages if message["body"].startswith("Today's pick")]
    assert len(pick_messages) == 3 - len(missing)
    assert {message["id"] for message in pick_messages} == pick_ids
    assert not snapshots[0]["pick"] or snapshots[0]["pick"]["double_down"] is None
    for i in missing:
        assert any("Morning check for" in message["body"] and DAYS[i] in message["body"]
                   for message in season.messages)
    return {"case": case, "restart": restart, "restarts": season.restarts,
            "predictions": season.predictions, "dms": season.messages, "days": snapshots,
            "outcomes": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--logs", type=Path, default=HERE)
    args = parser.parse_args()
    if args.seed:
        seed_legacy(args.seed)
        return
    runs = [(args.case, args.restart)] if args.case else [(case, restart) for case in CASES for restart in (False, True)]
    for case, restart in runs:
        name = f"{case}{'_restart' if restart else ''}"
        with TemporaryDirectory(prefix="wr-opening-") as tmp:
            season = Season(Path(tmp))
            season.old_seed()
            try:
                with season.run_offline():
                    evidence = run_case(season, case, restart)
                write_json(args.logs / f"{name}.json", evidence)
                print(name, evidence["outcomes"], flush=True)
            finally:
                (args.logs / f"{name}.journal.log").write_text(season.journal.getvalue())


if __name__ == "__main__":
    main()
