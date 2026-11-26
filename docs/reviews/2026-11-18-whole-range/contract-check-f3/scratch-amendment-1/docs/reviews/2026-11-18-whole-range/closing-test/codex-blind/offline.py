"""Shared synthetic clock, MLB responses and transport capture."""

import datetime as calendar
import io
import json
import os
import socket
import sys
from contextlib import ExitStack, chdir, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pandas as pd

from bts import cli, contest_fetch, contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
from bts.health import alert, runner
from bts.health import contest_state as contest_health
from bts.leaderboard import auth
from bts import entry_receipt, reconcile_receipt, saver_state

ET = ZoneInfo("America/New_York")
REAL_DT = calendar.datetime
UTC = calendar.timezone.utc
REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]


def at(day, hm="10:00"):
    return REAL_DT.fromisoformat(f"{day}T{hm}").replace(tzinfo=ET)


class Interrupted(BaseException):
    pass


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected socket operation")


class World:
    def __init__(self, day, root=None):
        self.day = day
        self.now = at(day)
        self.root = root
        self.events = []
        self.games = {}
        self.finish = at(day, "22:00")
        self.results = {}
        self.profile = {"activeStreak": 0, "seasonBestStreak": 0, "predictions": []}
        self.rounds = {1: at(day).date()}
        self.pending = []
        self.actions = []
        self.prediction = None
        self.on_send = None
        self.game("19:10")

    def game(self, when, pk=101, *, day=None, detailed="Scheduled", pair="A"):
        day = day or self.day
        start = at(day, when)
        game = {"gamePk": pk, "gameType": "R", "gameDate": start.astimezone(UTC).isoformat(),
                "status": {"detailedState": detailed},
                "teams": {side: {"team": {"name": f"{pair}-{side}"}} for side in ("home", "away")}}
        self.games.setdefault(day, []).append(game)
        return game

    def __enter__(self):
        self.stack = ExitStack()
        try:
            self._prepare()
        except BaseException:
            self.stack.close()
            raise
        return self

    def _prepare(self):
        stack = self.stack
        if self.root is None:
            self.root = Path(stack.enter_context(TemporaryDirectory(dir=REVIEW)))
        stack.enter_context(chdir(self.root))
        stack.enter_context(patch.dict(os.environ, {"TZ": "America/New_York"}))
        for method in ("__init__", "connect", "connect_ex", "sendto", "bind"):
            stack.enter_context(patch.object(socket.socket, method, forbidden))
        stack.enter_context(patch.object(socket, "getaddrinfo", forbidden))
        stack.enter_context(patch.object(socket, "socketpair", forbidden))
        self.directory = self.root / "data/picks"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.models = self.root / "data/models"
        self.models.mkdir(parents=True, exist_ok=True)
        for name in ("mdp_policy.npz", "mdp_tail_policy.npz"):
            target = self.models / name
            if not target.exists():
                target.symlink_to(ROOT / "data/models" / name)
        self.config = {
            "orchestrator": {"picks_dir": str(self.directory), "heartbeat_path": str(self.root / "data/.heartbeat"),
                             "models_dir": str(self.models), "data_dir": str(self.root / "data/processed")},
            "bluesky": {"dm_recipient": "offline.test"},
            "scheduler": {"contest_state_required": True, "pick_delivery": "dm", "entry_intent": "enter",
                          "live_forward_capture_on_lock": False},
            "health_checks": {"enabled": True, "contest_state_expected": True},
            "tiers": [],
        }
        world = self

        class Clock(REAL_DT):
            @classmethod
            def now(cls, tz=None):
                return world.now.astimezone(tz) if tz else world.now.replace(tzinfo=None)

        from bts.model import predict
        self.clock_modules = set()
        for name, module in tuple(sys.modules.items()):
            if name.startswith("bts.") and getattr(module, "datetime", None) is REAL_DT:
                self.patch(module, "datetime", Clock)
                self.clock_modules.add(name)
        self.clock = Clock
        self.patch(strategy, "_mdp_cache", None)
        self.patch(scheduler, "_now_et", lambda: self.now)
        self.patch(cli, "_now_et", lambda: self.now)
        self.patch(scheduler, "_watchdog_ping_sleep", self.wait)
        self.patch(scheduler, "_poll_interval_sleep", lambda _, duration: self.wait(duration))
        self.patch(scheduler, "_idle_until_next_wakeup", lambda *args: self.record("idle", args[0]))
        self.patch(scheduler, "heartbeat_watchdog", lambda *args, **kwargs: nullcontext())
        self.patch(scheduler, "retry_urlopen", self.schedule_response)
        self.patch(scheduler, "check_confirmed_lineups", self.confirmed)
        self.patch(scheduler, "poll_game_result", lambda _: "final" if self.now >= self.finish else "live")
        self.patch(scheduler, "_check_hits_midgame", lambda *_: [False, False])
        self.patch(picks, "get_game_statuses_detailed", self.statuses)
        self.patch(picks, "get_game_statuses", lambda day: {pk: row["abstract"] for pk, row in self.statuses(day).items()})
        self.patch(picks, "resolve_daily_slot_results", self.grade)
        if hasattr(strategy, "get_game_statuses_detailed"):
            self.patch(strategy, "get_game_statuses_detailed", self.statuses)
        if hasattr(strategy, "get_game_statuses"):
            self.patch(strategy, "get_game_statuses",
                       lambda day: {pk: row["abstract"] for pk, row in self.statuses(day).items()})
        self.patch(orchestrator, "run_cascade", self.cascade)
        self.patch(predict, "run_pipeline", lambda *args, **kwargs: self.cascade(*args, **kwargs)[0])
        self.patch(dm, "send_dm", self.send)
        self.patch(alert, "send_dm", self.send)
        self.patch(posting, "post_to_bluesky", lambda body: self.send("public", body))
        self.patch(posting, "reply_to_bluesky", lambda body, uri: self.send(uri, body))
        self.patch(runner, "read_systemd_nrestarts", lambda: 0)
        self.patch(auth, "load_session_cookies", lambda: {})
        self.patch(auth, "extract_uid", lambda _: "offline")
        self.patch(auth, "fetch_login_session", lambda **_: SimpleNamespace(user_id=1, username="offline", xsid="x"))
        self.patch(contest_fetch, "fetch_profile", lambda *_: self.profile)
        self.patch(contest_fetch, "fetch_pending_predictions", lambda *_: self.pending)
        self.patch(cli, "_fetch_rounds", lambda: self.rounds)
        self.patch(cli, "_fetch_bts_to_mlb", lambda: {1: 100001, 2: 100002})

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def patch(self, module, name, value):
        self.stack.enter_context(patch.object(module, name, value))

    def record(self, kind, *values):
        self.events.append((self.now.isoformat(), kind, *values))

    def wait(self, duration):
        end = self.now + calendar.timedelta(seconds=duration)
        while self.actions and self.actions[0][0] <= end:
            self.now, action = self.actions.pop(0)
            action()
        self.now = end

    def schedule_response(self, url, **kwargs):
        if "/api/v1/schedule?" not in url:
            forbidden(url)
        from urllib.parse import parse_qs, urlsplit
        day = parse_qs(urlsplit(url).query)["date"][0]
        self.record("schedule", day)
        return io.BytesIO(json.dumps({"dates": [{"games": self.games.get(day, [])}]}).encode())

    def statuses(self, day):
        table = {}
        for game in self.games.get(day, []):
            detail = game["status"]["detailedState"]
            if detail == "Scheduled" and self.now >= self.finish:
                detail = "Final"
            table[game["gamePk"]] = {"abstract": {"Final": "F", "Postponed": "F"}.get(detail, "P"),
                                    "detailed": detail, "code": "S" if detail == "Scheduled" else "F"}
        return table

    def confirmed(self, game_pks):
        return {pk: {"away", "home"} for pk in game_pks}

    def grade(self, daily, day, receipt=None):
        return {key: self.results.get(pick.game_pk, "hit")
                for key, pick in (("pick", daily.pick), ("double_down", daily.double_down)) if pick}

    def frame(self, probs=(0.99, 0.85), projected=False, day=None):
        games = self.games[day or self.day]
        return pd.DataFrame([
            dict(batter_id=100001 + i, batter_name=f"Batter {i + 1}", team="AWY", lineup=i+1,
                 pitcher_id=200001, pitcher_name="Fixture Pitcher", opp="HME", pitcher_team="HME",
                 p_hit_pa=0.32, p_game_hit=p, flags="PROJECTED" if projected else "", projected_lineup=projected,
                 game_pk=games[i % len(games)]["gamePk"], game_time=games[i % len(games)]["gameDate"])
            for i, p in enumerate(probs)
        ])

    def cascade(self, *args, **kwargs):
        self.record("predict")
        if self.prediction is None:
            return self.frame(), "offline"
        return self.prediction()

    def send(self, recipient, body):
        self.record("send", recipient, body)
        if self.on_send is not None:
            self.on_send(recipient, body)
        return f"offline-{len(self.events)}"

    def toml(self):
        path = self.root / "config.toml"
        lines = ["tiers = []"]
        for table, row in self.config.items():
            if isinstance(row, dict):
                lines.extend([f"[{table}]", *(f"{key} = {json.dumps(value)}" for key, value in row.items())])
        path.write_text("\n".join(lines) + "\n")
        return path

    def invoke(self, *args, expect=None):
        with patch.object(calendar, "datetime", self.clock):
            result = CliRunner().invoke(cli.cli, list(args))
        for name, module in tuple(sys.modules.items()):
            if name.startswith("bts.") and getattr(module, "datetime", None) is self.clock:
                # Newly imported modules must also revert their clock on exit.
                if name not in self.clock_modules:
                    self.stack.callback(setattr, module, "datetime", REAL_DT)
                    self.clock_modules.add(name)
        self.record("cli", list(args), result.exit_code, result.output,
                    repr(result.exception) if result.exception else None)
        if expect is not None and result.exit_code != expect:
            raise AssertionError((args, result.exit_code, result.output, result.exception))
        return result

    def dated(self, streak=0, best=None, saver="not_earned", day=None, expect=0):
        return self.invoke("set-contest-streak", "--date", day or self.now.date().isoformat(),
                           "--streak", str(streak), "--best-streak", str(streak if best is None else best),
                           "--saver", saver, expect=expect)

    def switch(self, action, *args, season=None, expect=0):
        opts = ["--season", str(season or self.now.year)] if action != "status" or season else []
        return self.invoke("season-contract", action, *opts, *args, expect=expect)

    def fetch(self, expect=0):
        return self.invoke("fetch-contest-streak", "--expected-username", "offline",
                           "--dm-recipient", "offline.test", expect=expect)

    def grade_command(self, day=None, expect=None):
        return self.invoke("check-results", "--date", day or self.day, expect=expect)

    def run(self, day=None, via_cli=False, dry=False):
        if via_cli:
            return self.invoke("schedule", "--date", day or self.day, "--config", str(self.toml()),
                               *(["--dry-run"] if dry else []))
        return scheduler.run_day(day or self.day, self.config, dry_run=dry)

    def outcome(self, day=None):
        return self.invoke("day-outcome", "--date", day or self.day, "--config", str(self.toml()),
                           "--expected-username", "offline", "--dm-recipient", "offline.test")

    def watchdog(self):
        return self.invoke("check-pick-entered", "--expected-username", "offline", "--dm-recipient", "offline.test")

    def snapshot(self, day=None):
        day = day or self.day
        files = {}
        for path in sorted(self.directory.parent.rglob("*.json")):
            if path.name[:4].isdigit() and not path.name.startswith(day):
                continue
            if path.is_file():
                try:
                    files[str(path.relative_to(self.root))] = json.loads(path.read_bytes())
                except (ValueError, OSError):
                    continue
        self.record("snapshot", day, files)
        return files

    def dump(self, name):
        (REVIEW / f"{name}.json").write_text(json.dumps(self.events, indent=2, default=str) + "\n")
