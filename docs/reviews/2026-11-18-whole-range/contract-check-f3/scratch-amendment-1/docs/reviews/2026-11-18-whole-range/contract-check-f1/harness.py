"""Synthetic file-backed contest state and captured CLI transports."""
from contextlib import ExitStack
from datetime import date, datetime, timezone
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from bts import entry_receipt, picks
from bts.cli import cli
from bts.daily_decision import write_decision
from bts.health import pick_entry

DAY = "2027-03-24"
XW = {1: 1010, 2: 1020, 3: 1030}
RIGHT = {"id": 77, "roundId": 1, "feedId": 101}
WRONG = {**RIGHT, "feedId": 102}


def row(unit=77, player=1, number=1, rnd=1):
    return {"roundId": rnd, "unitId": unit, "playerId": player, "number": number, "result": None}


class World:
    def __init__(self, root, day=DAY):
        self.root, self.day = root, day
        self.picks = root / "picks"
        self.picks.mkdir(parents=True, exist_ok=True)
        self.marker = root / "health_state" / "pick_entry_check.json"
        self.units = entry_receipt.units_snapshot_dir(self.picks)
        self.units.mkdir(parents=True, exist_ok=True)
        self.config = root / "config.toml"
        self.config.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
        self.runner = CliRunner()

    def commit(self, *, game=101, dd=None, pitch=None, delivery="delivered"):
        pitch = pitch if pitch is not None else f"{self.day}T19:10:00-04:00"

        def slot(batter, game_pk, game_time):
            return picks.Pick(batter_name=f"Batter {batter}", batter_id=batter, team="NYY",
                              lineup_position=1, pitcher_name="Pitcher", pitcher_id=500,
                              p_game_hit=0.8, flags=[], projected_lineup=False, game_pk=game_pk,
                              game_time=game_time)

        primary = slot(1010, game, pitch)
        second = slot(*dd) if dd else None
        daily = picks.DailyPick(self.day, f"{self.day}T16:00:00-04:00", primary, second, None,
                                notification_sent=True, notification_channel="bluesky_dm", notification_id="fixture")
        picks.save_pick(daily, self.picks)
        if delivery is not None:
            decision = write_decision(self.day, self.picks, action="double" if dd else "single", source="mdp",
                                      delivery_status=delivery, scoreable=True,
                                      primary={"batter_name": primary.batter_name},
                                      double_down={"batter_name": second.batter_name} if second else None,
                                      now=datetime.fromisoformat(f"{self.day}T20:00:00+00:00"))
            assert decision is not None
        return daily

    def capture(self, units, stamp=None):
        stamp = stamp or f"{self.day}T210000Z"
        path = self.units / f"{stamp}.json.gz"
        path.write_bytes(gzip.compress(json.dumps({"units": units}).encode(), mtime=0))
        return path

    def read_marker(self):
        try:
            return json.loads(self.marker.read_text())
        except (OSError, ValueError):
            return None

    def set_marker(self, marker, raw=None):
        self.marker.unlink(missing_ok=True)
        if marker is not None or raw is not None:
            self.marker.parent.mkdir(exist_ok=True)
            self.marker.write_text(raw if raw is not None else json.dumps(marker))

    def check(self, at, pending=(), *, profile=None, crosswalk=None, rounds=None, dm="success", elapsed=0,
              more_patches=(), publish=True, fetch_failure=None):
        messages, attempted, fetches = [], [], []

        def send(recipient, message):
            attempted.append(message)
            if dm == "failure":
                raise RuntimeError("synthetic send failure")
            messages.append(message)

        def fetch(name, data):
            def call(*args, **kwargs):
                fetches.append(name)
                if fetch_failure == name:
                    raise ValueError("synthetic fetch failure")
                return data
            return call

        def stable_mono():
            if not anchor:
                anchor.append(True)
                return 1000
            return 1000 + elapsed

        session = SimpleNamespace(user_id="fixture", username="operator", xsid="fixture")
        leaves = {
            "bts.leaderboard.auth.load_session_cookies": lambda: {},
            "bts.leaderboard.auth.extract_uid": lambda c: "fixture",
            "bts.leaderboard.auth.fetch_login_session": lambda *a, **k: session,
            "bts.contest_fetch.fetch_profile": fetch("profile", profile or {"predictions": []}),
            "bts.contest_fetch.fetch_pending_predictions": fetch("pending", list(pending)),
            "bts.cli._fetch_rounds": fetch("rounds", rounds if rounds is not None else {1: date.fromisoformat(self.day)}),
            "bts.cli._fetch_bts_to_mlb": fetch("crosswalk", crosswalk if crosswalk is not None else XW),
            "bts.dm.send_dm": send,
            "bts.picks._git_head_sha": lambda *a, **k: "fixture-fixed-head",
            "time.monotonic": stable_mono,
        }
        anchor = []
        args = ["check-pick-entered", "--picks-dir", str(self.picks), "--now-et", f"{self.day}T{at}:00",
                "--expected-username", "operator"]
        if dm != "none":
            args += ["--dm-recipient", "captured.invalid"]
        receipts = []
        with ExitStack() as stack:
            for target, leaf in (*leaves.items(), *more_patches):
                stack.enter_context(patch(target, leaf))
            if not publish:
                stack.enter_context(patch.object(entry_receipt.EntryReceipt, "publish",
                                                 lambda rec: receipts.append(rec.record())))
            result = self.runner.invoke(cli, args)
        if publish:
            receipts = entry_receipt.discover(self.picks, self.day)
        return {"exit": result.exit_code, "exception": type(result.exception).__name__ if result.exception else None,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip(), "marker": self.read_marker(),
                "dm": messages, "attempted_dm": attempted, "fetches": fetches, "receipts": receipts}

    def morning(self):
        sent = []
        with patch("bts.scheduler.fetch_schedule", lambda _: [{"gamePk": 101}, {"gamePk": 102}]), \
             patch("bts.dm.send_dm", lambda _, msg: sent.append(msg)):
            result = self.runner.invoke(cli, ["day-outcome", "--date", self.day, "--config", str(self.config),
                                             "--picks-dir", str(self.picks), "--expected-username", "operator",
                                             "--dm-recipient", "captured.invalid"])
        assert result.exit_code in (0, 1), result.output
        return {"exit": result.exit_code, "outcome": json.loads(result.stdout), "dm": sent}

    def health(self):
        return [{"level": a.level, "message": a.message}
                for a in pick_entry.check(self.picks, date.fromisoformat(self.day),
                                          datetime(2030, 1, 1, tzinfo=timezone.utc))]
