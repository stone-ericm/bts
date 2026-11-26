"""Offline scheduler and scorer replay. Select the revision through PYTHONPATH."""

import argparse
from contextlib import ExitStack, contextmanager, nullcontext, redirect_stderr
import datetime as datetime_module
from datetime import datetime, timedelta
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")
MID = "2027-07-15"
LATE = "2027-09-20"
ACTIVATION = "b" * 32
AUTO = "bts_contest_streak_auto_v2"


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def at(day, hour=10, minute=0):
    return datetime.fromisoformat(day).replace(hour=hour, minute=minute, tzinfo=ET)


def seed(picks, day, *, streak=12, best=None, season=2027, age=8, switch=True,
         model=None, old=False):
    now = at(day)
    account = picks / "account_state"
    observed_day = (now - timedelta(days=1)).date().isoformat() if not old else "2026-09-13"
    write(picks / "streak.json", {
        "streak": streak if model is None else model, "saver_available": True,
        **({"season": season} if not old else {}),
    })
    write(account / "contest_streak.json", {
        "active_streak": streak, "best_streak": streak if best is None else best,
        "source": "probe_mlb", "source_date": observed_day, "recorded_at": (
            now - timedelta(hours=age)).isoformat() if not old else "2026-09-14T06:00:00-04:00",
        "username": "stonehengee", "user_id": 100,
        **({"schema_version": AUTO, "season": season, "activation_id": ACTIVATION} if not old else
           {"schema_version": "bts_contest_streak_auto_v1"}),
    })
    write(account / "saver_state.json", {"season": season, "state": "active", "source": "probe"})
    if switch:
        write(account / "season_contract.json", {
            "schema_version": "bts_season_contract_v1", "season": season, "status": "automatic",
            "activation_id": ACTIVATION, "switched_at": now.isoformat(),
            "revoked_at": None, "note": "synthetic",
        })


def config_for(picks, delivery="dm", intent="enter", required=True):
    return {
        "orchestrator": {"picks_dir": str(picks), "heartbeat_path": str(picks.parent / ".heartbeat")},
        "scheduler": {"pick_delivery": delivery, "entry_intent": intent,
                      "contest_state_required": required, "live_forward_capture_on_lock": False,
                      "lineup_check_offset_min": 60},
        "health_checks": {"enabled": False, "contest_state_expected": required},
        "bluesky": {"dm_recipient": "probe.invalid"},
        "tiers": {"local": {}},
    }


def toml(path, config):
    path.write_text("\n".join(
        f"[{key}]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in table.items()
                                if not isinstance(v, dict))
        for key, table in config.items()
    ))
    return path


def deny_socket(*args, **kwargs):
    raise AssertionError("unexpected network transport")


class World:
    def __init__(self, day, kind):
        self.day = day
        self.kind = kind
        self.now = at(day)
        self.origin = self.now
        self.events = []
        self.details = 0
        self.cascades = 0
        self.game_hours = {101: 16, 102: 18, 103: 21}
        if kind == "doubleheader":
            self.game_hours = {101: 16, 102: 18, 103: 17}

    def event(self, kind, *args):
        self.events.append([self.now.isoformat(), kind, *args])

    def sleep(self, seconds):
        assert 0 <= seconds <= 24 * 3600, seconds
        self.now += timedelta(seconds=seconds)

    def status(self, pk):
        if pk == 101 and (
            self.kind == "postpone" and self.details >= 3 or
            self.kind == "postpone_double" and self.details >= 3
        ):
            return "P", "Postponed"
        if self.now >= at(self.day, self.game_hours[pk], 12):
            return "F", "Final"
        if self.now >= at(self.day, self.game_hours[pk]):
            return "L", "In Progress"
        return "P", "Scheduled"

    def schedule(self, day):
        if day != self.day or self.kind == "no_games":
            return {"dates": []}
        games = []
        for pk, hour in self.game_hours.items():
            code, detailed = self.status(pk)
            pair = 101 if pk == 102 and self.kind == "doubleheader" else pk
            games.append({
                "gamePk": pk, "gameType": "R", "gameDate": at(day, hour).isoformat(),
                "status": {"abstractGameCode": code, "detailedState": detailed},
                "teams": {side: {"team": {"name": f"{side}-{pair}"}} for side in ("away", "home")},
            })
        return {"dates": [{"games": games}]}

    def feed(self, pk):
        code, detailed = self.status(pk)
        no_ab = pk == 101 and self.kind in {"void", "void_double"}
        confirmed = pk != 102 or self.kind != "doubleheader" or self.now >= at(self.day, 16, 30)
        players = {}
        for batter in (pk, 1000 + pk):
            players[f"ID{batter}"] = {
                "person": {"fullName": f"Batter {batter}"},
                "battingOrder": "100" if confirmed else "",
                "stats": {"batting": {
                    "hits": int(code == "F" and not no_ab),
                    "atBats": int(code == "F" and not no_ab),
                    "baseOnBalls": int(code == "F" and no_ab),
                }},
            }
        return {
            "gameData": {"status": {"abstractGameCode": code, "detailedState": detailed}},
            "liveData": {"boxscore": {"teams": {
                "away": {"players": players}, "home": {"players": {}},
            }}},
        }

    def request(self, url, **kwargs):
        parsed = urlsplit(url)
        if parsed.path == "/api/v1/schedule":
            qs = parse_qs(parsed.query)
            day = qs["date"][0]
            if day == self.day and "hydrate" not in qs:
                self.details += 1
            result = self.schedule(day)
            self.event("schedule", day, "hydrate" in qs,
                       [g["status"]["detailedState"] for d in result["dates"] for g in d["games"]])
        elif parsed.path.startswith("/api/v1.1/game/") and parsed.path.endswith("/feed/live"):
            pk = int(parsed.path.split("/")[4])
            result = self.feed(pk)
            self.event("feed", pk, result["gameData"]["status"]["detailedState"])
        else:
            raise AssertionError(f"unexpected fixture URL: {url}")
        return BytesIO(json.dumps(result).encode())

    def predictions(self, tiers, day):
        import pandas as pd

        self.cascades += 1
        self.event("predictions", day)
        if self.kind in {"postpone", "postpone_double", "cascade_flake"} and self.cascades > 1:
            return None, None
        rows = []
        for pk, hour in self.game_hours.items():
            rank = pk - 101
            if self.kind == "doubleheader":
                rank = {102: 0, 101: 1, 103: 2}[pk]
            p = (0.78 if self.kind == "skip" else 0.96) - rank * 0.02
            flags = "PROJECTED" if self.kind == "doubleheader" and pk == 102 and self.now < at(day, 16, 30) else ""
            if self.kind == "cascade_flake":
                flags = "PROJECTED"
            rows.append({
                "batter_id": pk, "batter_name": f"Batter {pk}", "team": f"T{pk}",
                "game_pk": pk, "lineup": 1, "pitcher_name": f"Pitcher {pk}",
                "game_time": at(day, hour).isoformat(), "p_game_hit": p,
                "flags": flags, "status": "Scheduled",
            })
        return pd.DataFrame(sorted(rows, key=lambda row: -row["p_game_hit"])), "probe"

    def send(self, recipient, message):
        self.event("dm", recipient, message)
        return f"dm-{sum(event[1] == 'dm' for event in self.events)}"

    def post(self, message, *args, **kwargs):
        self.event("post", message)
        return "at://probe/post"

    def reply(self, message, uri):
        self.event("reply", message, uri)
        return "at://probe/reply"

    def idle(self, wakeup, heartbeat):
        self.event("idle", wakeup)


def artifact(path):
    import numpy as np

    # Use the production loader and lookup with a synthetic table. Its final
    # day picks a strong candidate, as do all available-day slices.
    table = np.full((58, 181, 2, 3), 1, dtype=np.int8)
    table[:, 0, :, :] = 0
    table[:, :, :, 0] = 0
    table[:, 1:, 1, 2] = 2
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, policy_table=table, boundaries=[0.80, 0.99], season_length=180)


@contextmanager
def installed(world, root, *, bridge=False):
    import bts.cli as cli
    from bts import contest_state, daily_decision, dm, heartbeat, orchestrator, picks
    from bts import posting, saver_state, scheduler, slate, strategy
    from bts.health import alert
    from bts.simulate import mdp

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return world.now.replace(tzinfo=None) if tz is None else world.now.astimezone(tz)

        @classmethod
        def utcnow(cls):
            return cls.now(datetime_module.timezone.utc).replace(tzinfo=None)

    previous = os.getcwd()
    try:
        os.chdir(root)
        artifact(Path(mdp.DEFAULT_POLICY_PATH))
        with ExitStack() as scope:
            for module in (cli, contest_state, daily_decision, heartbeat, picks, saver_state,
                           scheduler, slate, strategy, alert, datetime_module):
                if hasattr(module, "datetime"):
                    scope.enter_context(patch.object(module, "datetime", FrozenDateTime))
            if bridge:
                assert not hasattr(strategy, "SEASON_END_DATES")
                scope.enter_context(patch.object(strategy, "SEASON_END_DATE", "2027-09-26"))
            scope.enter_context(patch.object(socket.socket, "connect", deny_socket))
            scope.enter_context(patch.object(socket.socket, "connect_ex", deny_socket))
            scope.enter_context(patch.object(socket.socket, "sendto", deny_socket))
            scope.enter_context(patch.object(scheduler, "retry_urlopen", world.request))
            scope.enter_context(patch.object(picks, "retry_urlopen", world.request))
            scope.enter_context(patch.object(orchestrator, "run_cascade", world.predictions))
            scope.enter_context(patch.object(scheduler, "_watchdog_ping_sleep", world.sleep))
            scope.enter_context(patch.object(scheduler, "_poll_interval_sleep",
                                             lambda path, seconds: world.sleep(seconds)))
            scope.enter_context(patch.object(scheduler, "_idle_until_next_wakeup", world.idle))
            scope.enter_context(patch.object(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext()))
            scope.enter_context(patch.object(scheduler, "notify_ready", lambda: world.event("READY")))
            scope.enter_context(patch.object(scheduler, "notify_watchdog", lambda: None))
            scope.enter_context(patch.object(dm, "send_dm", world.send))
            scope.enter_context(patch.object(posting, "post_to_bluesky", world.post))
            scope.enter_context(patch.object(posting, "reply_to_bluesky", world.reply))
            scope.enter_context(patch.object(strategy, "_mdp_cache", None))
            yield
    finally:
        os.chdir(previous)


def snapshot(root):
    items = {}
    for path in sorted((root / "data").rglob("*")):
        if not path.is_file() or path.suffix == ".npz":
            continue
        rel = str(path.relative_to(root))
        raw = path.read_text()
        try:
            items[rel] = json.loads(raw)
        except ValueError:
            items[rel] = raw
    return items


def invoke(args):
    from bts.cli import cli
    from click.testing import CliRunner

    result = CliRunner().invoke(cli, args)
    exception = result.exception
    return {"exit": result.exit_code, "stdout": result.stdout, "stderr": result.stderr,
            "exception": repr(exception) if exception is not None and
            not isinstance(exception, SystemExit) else None}


CASES = [
    ("mid_double", MID, "normal", 12, 12, 0.995, "dm", 8),
    ("mid_public", MID, "normal", 12, 12, 0.96, "public", 8),
    ("mid_skip", MID, "skip", 12, 12, 0.78, "dm", 8),
    ("mid_postpone", MID, "postpone", 12, 12, 0.96, "dm", 8),
    ("mid_postpone_double", MID, "postpone_double", 12, 12, 0.995, "dm", 8),
    ("mid_void_double", MID, "void_double", 12, 12, 0.995, "dm", 8),
    ("late_doubleheader", LATE, "doubleheader", 46, 46, 0.96, "dm", 16),
    ("late_void", LATE, "void", 46, 46, 0.96, "dm", 8),
    ("late_tail", LATE, "normal", 13, 18, 0.96, "dm", 8),
    ("late_tail_skip", LATE, "normal", 13, 30, 0.96, "dm", 8),
    ("last_date", "2027-09-26", "normal", 56, 56, 0.96, "dm", 8),
    ("boundary", "2027-09-20", "normal", 43, 56, 0.96, "dm", 8),
]


def run_case(root, case, bridge):
    import bts
    from bts import scheduler, strategy

    name, day, kind, streak, best, top, delivery, age = case
    world = World(day, kind)
    if top != 0.96 and kind != "skip":
        original = world.predictions

        def ranked(tiers, date):
            frame, tier = original(tiers, date)
            if frame is not None:
                frame.loc[frame.index[0], "p_game_hit"] = top
            return frame, tier

        world.predictions = ranked
    picks = root / "data/picks"
    seed(picks, day, streak=streak, best=best, age=age)
    config = config_for(picks, delivery=delivery)
    journal = StringIO()
    with installed(world, root, bridge=bridge), redirect_stderr(journal):
        scheduler.run_day(day, config)
        files = snapshot(root)
        days_remaining = strategy._days_remaining(day)
        world.now = at((at(day) + timedelta(days=1)).date().isoformat(), 7, 40)
        scorer = invoke(["check-results", "--date", day, "--picks-dir", str(picks)])
        reconciler = invoke(["reconcile", "--picks-dir", str(picks)])
        day_check = invoke(["day-outcome", "--date", day, "--picks-dir", str(picks),
                            "--config", str(toml(root / "config.toml", config)),
                            "--dm-recipient", "probe.invalid"])
        final = snapshot(root)
    return {"case": name, "day": day, "module": bts.__file__, "table_days": days_remaining,
            "events": world.events, "journal": journal.getvalue(), "after_scheduler": files,
            "scorer": scorer, "reconcile": reconciler, "day_outcome": day_check,
            "after_cron": final}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", action="store_true", help="BASE only, isolate changes after its date bug")
    parser.add_argument("--case", choices=[case[0] for case in CASES])
    args = parser.parse_args()
    for case in CASES:
        if args.case is None or args.case == case[0]:
            with TemporaryDirectory(prefix=f"{case[0]}-", dir=Path(__file__).resolve().parent) as temp:
                print(json.dumps(run_case(Path(temp), case, args.bridge)), flush=True)


if __name__ == "__main__":
    main()
