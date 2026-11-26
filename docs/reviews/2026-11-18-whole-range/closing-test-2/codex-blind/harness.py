"""Offline boundary adapters shared by the whole-range probes."""
import datetime as dt
import importlib
import json
import os
import socket
import sys
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
from click.testing import CliRunner

from bts import cli, contest_fetch, contest_state, dm, orchestrator, picks, posting, scheduler
from bts.health import alert, realized_calibration, runner
from bts.leaderboard import auth

REPO = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent / "base" / "src"
IS_BASE = Path(scheduler.__file__).is_relative_to(BASE)
ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
REAL = dt.datetime
ACCOUNT = "offline-account"
RECIPIENT = "offline-recipient.example"


class Clock(REAL):
    at = None

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls.at.astimezone(ET).replace(tzinfo=None)
        return cls.at.astimezone(tz)

    @classmethod
    def utcnow(cls):
        return cls.at.astimezone(UTC).replace(tzinfo=None)


def wall(day, hm):
    return REAL.fromisoformat(f"{day}T{hm}").replace(tzinfo=ET)


def blocked(*args, **kwargs):
    raise AssertionError("network or child process requested outside the fixture")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


class PowerLoss(BaseException):
    pass


class World:
    def __init__(self, root, day, hm="10:00", *, health=True, live=True, required=True,
                 intent=None):
        self.root = root
        self.picks = root / "data/picks"
        self.picks.mkdir(parents=True, exist_ok=True)
        (root / "data/models").symlink_to(REPO / "data/models", target_is_directory=True)
        self.config_path = root / "orchestrator.toml"
        intent = intent or ("enter" if live and required else "research")
        self.config_path.write_text(
            f'[orchestrator]\npicks_dir = "{self.picks}"\n'
            f'heartbeat_path = "{root / "data/.heartbeat"}"\n'
            f'models_dir = "{root / "data/models"}"\n'
            f'data_dir = "{root / "data/processed"}"\n'
            f'[[tiers]]\nname = "offline"\ntype = "local"\n'
            f'[bluesky]\ndm_recipient = "{RECIPIENT}"\n'
            f'[scheduler]\npick_delivery = "{"dm" if live else "private"}"\n'
            f'entry_intent = "{intent}"\n'
            f'lineup_check_offset_min = 60\n'
            f'contest_state_required = {str(required).lower()}\n'
            f'live_forward_capture_on_lock = false\n'
            f'[health_checks]\nenabled = {str(health).lower()}\n'
            f'contest_state_expected = {str(required).lower()}\n'
        )
        self.config = orchestrator.load_config(self.config_path)
        self.day = day
        self.hm = hm
        self.events = []
        self.actions = []
        self.games = {}
        self.rows = {}
        self.finish = {}
        self.slots = {}
        self.profile = (0, 0, None)
        self.pending = []
        self.crosswalk = {}
        self.send_failure = None
        self.coverage = {}
        self.prediction_rule = lambda day: True

    def event(self, typ, **fields):
        self.events.append(dict(at=Clock.at.isoformat(), type=typ, **fields))

    def set_day(self, day, hm):
        self.day = day
        Clock.at = wall(day, hm)

    def at(self, hm, func):
        self.actions.append((wall(self.day, hm), func))
        self.actions.sort(key=lambda action: action[0])

    def advance(self, seconds):
        target = Clock.at + dt.timedelta(seconds=seconds)
        while self.actions and self.actions[0][0] <= target:
            when, callback = self.actions.pop(0)
            Clock.at = max(when, Clock.at)
            callback()
        Clock.at = max(Clock.at, target)

    def slate(self, day, times=("19:05",), probabilities=(.97, .96, .90),
              *, projected_until=None, doubleheader=False):
        self.games[day] = [
            dict(gamePk=700 + i, gameType="R", gameDate=wall(day, hm).astimezone(UTC).isoformat(),
                 doubleHeader="Y" if doubleheader else "N", gameNumber=i + 1,
                 teams={"away": {"team": {"id": 1, "name": "Away"}},
                        "home": {"team": {"id": 2, "name": "Home"}}})
            for i, hm in enumerate(times)
        ]
        self.rows[day] = [
            dict(batter_name=f"Fixture Batter {i}", batter_id=100 + i, team=f"T{i}", lineup=1 + i,
                 pitcher_name=f"Fixture Pitcher {i}", pitcher_id=900 + i, p_game_hit=p,
                 game_pk=self.games[day][i % len(times)]["gamePk"],
                 game_time=self.games[day][i % len(times)]["gameDate"], pitcher_team=f"U{i}")
            for i, p in enumerate(probabilities)
        ]
        self.coverage[day] = wall(day, projected_until) if projected_until else None

    def status_for(self, game, day=None):
        key = (day or self.day, game["gamePk"])
        if self.slots.get(key) == "void":
            return ("F", "Postponed")
        start = scheduler.game_time_et(game)
        finish = self.finish.get(key, start + dt.timedelta(minutes=40))
        if Clock.at >= finish:
            return ("F", "Final")
        if Clock.at >= start:
            return ("L", "In Progress")
        return ("P", "Scheduled")

    def schedule(self, day):
        self.event("schedule", day=day)
        output = []
        for game in self.games.get(day, []):
            abstract, detail = self.status_for(game, day)
            output.append(dict(game, status={"abstractGameState": abstract, "detailedState": detail}))
        return output

    def statuses(self, day):
        return {game["gamePk"]: dict(zip(("abstract", "detailed"), self.status_for(game, day)))
                for game in self.games.get(day, [])}

    def predictions(self, tiers, day):
        rows = self.rows.get(day)
        available = bool(rows and self.prediction_rule(day))
        self.event("predictions", available=available)
        if not available:
            return None, None
        cutoff = self.coverage[day]
        flags = "PROJECTED" if cutoff and Clock.at < cutoff else ""
        return pd.DataFrame([dict(row, flags=flags) for row in rows]), "offline"

    def send(self, recipient, message):
        self.event("dm", recipient=recipient, message=message)
        if self.send_failure and message.startswith(("Today's pick:", "Today's picks ")):
            fail, self.send_failure = self.send_failure, None
            raise fail
        return f"offline-dm-{len(self.events)}"

    def post(self, message):
        self.event("post", message=message)
        return f"at://offline/{len(self.events)}"

    def poll(self, pk):
        game = next(game for game in self.games[self.day] if game["gamePk"] == pk)
        state, _ = self.status_for(game)
        return {"F": "final", "P": "preview", "L": "live"}[state]

    def profile_data(self, *args):
        streak, best, settled_day = self.profile
        predictions = [{"roundId": 1, "result": "hit"}] if settled_day else []
        return {"activeStreak": streak, "seasonBestStreak": best, "predictions": predictions}

    def rounds(self):
        return {1: dt.date.fromisoformat(self.profile[2] or self.day)}

    @contextmanager
    def switched_on(self):
        old_cwd, old_clock = Path.cwd(), Clock.at
        os.chdir(self.root)
        Clock.at = wall(self.day, self.hm)
        with ExitStack() as stack:
            stack.callback(os.chdir, old_cwd)
            stack.callback(setattr, Clock, "at", old_clock)
            stack.enter_context(patch.dict(os.environ, {"HOME": str(self.root)}))
            # Import CLI-local modules before replacing datetime so subsequent worlds
            # use the same reversible bindings.
            for name in ("bts.day_outcome", "bts.season_contract"):
                if not IS_BASE:
                    importlib.import_module(name)
            stack.enter_context(patch.object(dt, "datetime", Clock))
            for name, module in list(sys.modules.items()):
                if name.startswith("bts.") and getattr(module, "datetime", None) is REAL:
                    stack.enter_context(patch.object(module, "datetime", Clock))
            for method in ("connect", "connect_ex", "sendto", "bind"):
                stack.enter_context(patch.object(socket.socket, method, blocked))
            stack.enter_context(patch.object(socket, "getaddrinfo", blocked))
            stack.enter_context(patch("subprocess.Popen", blocked))
            for module, name, stub in (
                (scheduler, "fetch_schedule", self.schedule),
                (scheduler, "check_confirmed_lineups", lambda pks: {pk: {"home", "away"} for pk in pks}),
                (scheduler, "notify_watchdog", lambda: None),
                (scheduler, "notify_ready", lambda: self.event("ready")),
                (scheduler, "heartbeat_watchdog", lambda *a, **kw: nullcontext()),
                (scheduler, "_watchdog_ping_sleep", self.advance),
                (scheduler, "_poll_interval_sleep", lambda path, seconds: self.advance(seconds)),
                (scheduler, "_idle_until_next_wakeup", lambda when, path: self.event("idle", until=when)),
                (scheduler, "poll_game_result", self.poll),
                (scheduler, "_check_hits_midgame", lambda *a: [False, False]),
                (picks, "get_game_statuses_detailed", self.statuses),
                (picks, "get_game_statuses", lambda day: {k: v["abstract"] for k, v in self.statuses(day).items()}),
                (picks, "check_hit", lambda pk, *a, **kw: self.slots.get((self.day, pk), "hit")),
                (picks, "_git_head_sha", lambda *a: "619abda7" if IS_BASE else "d0bdcdba"),
                (orchestrator, "run_cascade", self.predictions),
                (dm, "send_dm", self.send),
                (alert, "send_dm", self.send),
                (posting, "post_to_bluesky", self.post),
                (realized_calibration, "_current_deploy_iso", lambda *a: None),
                (runner, "read_systemd_nrestarts", lambda: 0),
                (auth, "load_session_cookies", lambda: {}),
                (auth, "extract_uid", lambda cookies: 4242),
                (auth, "fetch_login_session", lambda **kw: SimpleNamespace(user_id=4242, username=ACCOUNT, xsid="offline")),
                (contest_fetch, "fetch_profile", self.profile_data),
                (contest_fetch, "fetch_pending_predictions", lambda *a: self.pending),
                (cli, "_fetch_rounds", self.rounds),
                (cli, "_fetch_bts_to_mlb", lambda: self.crosswalk),
            ):
                stack.enter_context(patch.object(module, name, stub))
            yield self

    def invoke(self, *args):
        result = CliRunner().invoke(cli.cli, [str(arg) for arg in args])
        error = result.exception
        self.event("cli", args=[str(arg) for arg in args], code=result.exit_code,
                   text=result.output, error=None if isinstance(error, SystemExit) else repr(error))
        return result

    def dated(self, streak=0, best=0, saver="not_earned"):
        result = self.invoke("set-contest-streak", "--picks-dir", self.picks, "--date",
                             self.day, "--streak", streak, "--best-streak", best, "--saver", saver)
        assert result.exit_code == 0, result.output

    def fetch(self, streak, best, settled_day=None):
        self.profile = streak, best, settled_day
        return self.invoke("fetch-contest-streak", "--picks-dir", self.picks,
                           "--expected-username", ACCOUNT, "--dm-recipient", RECIPIENT)

    def run(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr
        logs = io.StringIO()
        try:
            with redirect_stdout(logs), redirect_stderr(logs):
                scheduler.run_day(self.day, self.config)
        finally:
            self.event("run_day", journal=logs.getvalue())

    def snapshot(self):
        paths = (*self.picks.glob("*.json"), *self.picks.glob("account_state/*.json"),
                 *self.picks.glob("*/decision.json"), *self.picks.glob("*/scheduler_state.json"),
                 *(self.root / "data/health_state").rglob("*.json"))
        out = {}
        for path in sorted(set(paths)):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                data = {"fixture_read_error": str(error)}
            out[str(path.relative_to(self.root))] = data
        return out
