"""Shared offline sandbox harness for F1 round-2 adversarial review probes."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import date, datetime, timezone
import gzip
import json
from pathlib import Path
import re
import shutil
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

import bts
from bts import entry_receipt, picks
from bts.cli import cli
from bts.daily_decision import write_decision
from bts.health import pick_entry

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

DAY = "2027-03-24"
PITCH = f"{DAY}T19:40:00-04:00"
DEFAULT_XW = {1: 1010, 2: 1020, 3: 1030, 4: 1040}
UNIT_RIGHT_1 = {"id": 77, "roundId": 1, "feedId": 101}
UNIT_WRONG_1 = {"id": 77, "roundId": 1, "feedId": 102}
UNIT_RIGHT_2 = {"id": 78, "roundId": 1, "feedId": 201}
UNIT_WRONG_2 = {"id": 78, "roundId": 1, "feedId": 202}


def make_row(unit: object = 77, player: object = 1, number: object = 1, rnd: object = 1, result: object = None) -> dict:
    return {"roundId": rnd, "unitId": unit, "playerId": player, "number": number, "result": result}


class SandboxWorld:
    """Isolated per-case filesystem world invoking the real Click commands offline."""

    def __init__(self, root: Path, day: str = DAY):
        self.root = Path(root)
        self.day = day
        self.picks = self.root / "picks"
        self.picks.mkdir(parents=True, exist_ok=True)
        self.marker = self.root / "health_state" / "pick_entry_check.json"
        self.units_dir = entry_receipt.units_snapshot_dir(self.picks)
        self.units_dir.mkdir(parents=True, exist_ok=True)
        self.config = self.root / "config.toml"
        self.config.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
        self.runner = CliRunner()

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def commit(
        self,
        *,
        batter: object = 1010,
        game: object = 101,
        pitch: str = PITCH,
        name: object = "Batter 1010",
        dd: tuple | None = None,
        delivery: str | None = "delivered",
    ) -> picks.DailyPick:
        def build_slot(b_id, g_pk, g_time, b_name):
            p = picks.Pick(
                batter_name=f"Batter {b_id}",
                batter_id=b_id,
                team="NYY",
                lineup_position=1,
                pitcher_name="Pitcher",
                pitcher_id=500,
                p_game_hit=0.80,
                flags=[],
                projected_lineup=False,
                game_pk=g_pk,
                game_time=g_time,
            )
            p.batter_name = b_name
            return p

        primary = build_slot(batter, game, pitch, name)
        second = None
        if dd is not None:
            dd_batter, dd_game, dd_pitch = dd[:3]
            dd_name = dd[3] if len(dd) > 3 else f"Batter {dd_batter}"
            second = build_slot(dd_batter, dd_game, dd_pitch, dd_name)

        notif_sent = delivery is not None
        daily = picks.DailyPick(
            self.day,
            f"{self.day}T16:00:00-04:00",
            primary,
            second,
            None,
            notification_sent=notif_sent,
            notification_channel="bluesky_dm" if notif_sent else None,
            notification_id="fixture" if notif_sent else None,
        )
        picks.save_pick(daily, self.picks)
        if delivery == "not_applicable":
            write_decision(
                self.day,
                self.picks,
                action="skip",
                source="mdp",
                delivery_status="not_applicable",
                scoreable=False,
                now=datetime.fromisoformat(f"{self.day}T20:00:00+00:00"),
            )
        elif delivery not in (None, "legacy_no_decision"):
            dec = write_decision(
                self.day,
                self.picks,
                action="double" if second else "single",
                source="mdp",
                delivery_status=delivery,
                scoreable=True,
                primary={"batter_name": primary.batter_name if isinstance(primary.batter_name, str) else "Batter"},
                double_down=(
                    {"batter_name": second.batter_name if isinstance(second.batter_name, str) else "Batter DD"}
                    if second
                    else None
                ),
                now=datetime.fromisoformat(f"{self.day}T20:00:00+00:00"),
            )
            assert dec is not None
        return daily

    def capture(self, units_list: list, stamp: str | None = None) -> Path:
        stamp = stamp or f"{self.day}T210000Z"
        path = self.units_dir / f"{stamp}.json.gz"
        path.write_bytes(gzip.compress(json.dumps({"units": units_list}).encode(), mtime=0))
        return path

    def set_marker(self, data: dict) -> None:
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        self.marker.write_text(json.dumps(data))

    def read_marker(self):
        if not self.marker.is_file():
            return None
        try:
            return json.loads(self.marker.read_text())
        except (OSError, ValueError):
            return "<UNREADABLE>"

    def read_marker_bytes_norm(self) -> str | None:
        if not self.marker.is_file():
            return None
        raw = self.marker.read_bytes()
        norm = re.sub(rb'("(?:receipt|checked_at)"\s*:\s*)"[^"]*"', rb'\1"<NORM>"', raw)
        return norm.decode("utf-8", errors="replace")

    def check(
        self,
        at_hhmm: str,
        pending: list | tuple = (),
        *,
        profile: dict | None = None,
        crosswalk: dict | None = None,
        rounds: dict | None = None,
        dm: str = "success",
        elapsed: float = 0.0,
        fetch_failure: str | None = None,
        username: str = "operator",
        expected_username: str | None = "operator",
        window_min: int | None = None,
        extra_patches: tuple | list = (),
    ) -> dict:
        sent_dms: list[str] = []
        attempted_dms: list[str] = []
        fetches: list[str] = []
        mono_calls: list[int] = []

        def fake_send(recipient, msg):
            attempted_dms.append(msg)
            if dm == "failure":
                raise RuntimeError("simulated DM failure")
            sent_dms.append(msg)

        def wrap_fetch(tag, payload):
            def _fn(*args, **kwargs):
                fetches.append(tag)
                if fetch_failure == tag:
                    raise ValueError(f"simulated {tag} failure")
                return payload
            return _fn

        def fake_login(*args, **kwargs):
            fetches.append("auth")
            if fetch_failure == "auth":
                from bts.leaderboard.auth import AuthError
                raise AuthError("simulated auth failure")
            return SimpleNamespace(user_id="uid-1", username=username, xsid="xsid-1")

        def fake_mono():
            mono_calls.append(1)
            return 1000.0 if len(mono_calls) == 1 else 1000.0 + elapsed

        patches = {
            "bts.leaderboard.auth.load_session_cookies": lambda: {},
            "bts.leaderboard.auth.extract_uid": lambda c: "uid-1",
            "bts.leaderboard.auth.fetch_login_session": fake_login,
            "bts.contest_fetch.fetch_profile": wrap_fetch("profile", profile if profile is not None else {"predictions": []}),
            "bts.contest_fetch.fetch_pending_predictions": wrap_fetch("pending", list(pending)),
            "bts.cli._fetch_rounds": wrap_fetch(
                "rounds", rounds if rounds is not None else {1: date.fromisoformat(self.day)}
            ),
            "bts.cli._fetch_bts_to_mlb": wrap_fetch(
                "crosswalk", crosswalk if crosswalk is not None else dict(DEFAULT_XW)
            ),
            "bts.dm.send_dm": fake_send,
            "bts.picks._git_head_sha": lambda *a, **k: "pinned-head-sha",
            "time.monotonic": fake_mono,
        }

        args = ["check-pick-entered", "--picks-dir", str(self.picks), "--now-et", f"{self.day}T{at_hhmm}:00"]
        if expected_username is not None:
            args += ["--expected-username", expected_username]
        if window_min is not None:
            args += ["--window-min", str(window_min)]
        if dm != "none":
            args += ["--dm-recipient", "captured.invalid"]

        with ExitStack() as stack:
            for target, fn in (*patches.items(), *extra_patches):
                stack.enter_context(patch(target, fn))
            res = self.runner.invoke(cli, args)

        receipts = entry_receipt.discover(self.picks, self.day)
        return {
            "exit": res.exit_code,
            "exception": type(res.exception).__name__ if res.exception else None,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "dm": sent_dms,
            "attempted_dm": attempted_dms,
            "fetches": fetches,
            "marker": self.read_marker(),
            "marker_bytes": self.read_marker_bytes_norm(),
            "receipts": receipts,
        }

    def morning(self, *, games: list | None = None, dm: str = "success") -> dict:
        sent: list[str] = []
        sched = [{"gamePk": 101}, {"gamePk": 102}] if games is None else games

        def fake_send(_, msg):
            if dm == "failure":
                raise RuntimeError("morning DM failed")
            sent.append(msg)

        with patch("bts.scheduler.fetch_schedule", lambda _: sched), patch("bts.dm.send_dm", fake_send):
            res = self.runner.invoke(
                cli,
                [
                    "day-outcome",
                    "--date",
                    self.day,
                    "--config",
                    str(self.config),
                    "--picks-dir",
                    str(self.picks),
                    "--expected-username",
                    "operator",
                    "--dm-recipient",
                    "captured.invalid",
                ],
            )
        return {
            "exit": res.exit_code,
            "stdout": json.loads(res.stdout) if res.stdout.strip() else None,
            "stderr": res.stderr,
            "dm": sent,
        }

    def health(self, now_utc: datetime | None = None) -> list[dict]:
        when = now_utc or datetime(2030, 1, 1, tzinfo=timezone.utc)
        alerts = pick_entry.check(self.picks, date.fromisoformat(self.day), when)
        return [{"level": a.level, "source": a.source, "message": a.message} for a in alerts]


def normalize_obs(value, root: Path):
    """Strip run-unique IDs and timestamps while preserving every semantic & layout field."""
    skip_keys = {
        "receipt",
        "attempt_id",
        "confirmed_by",
        "checked_at",
        "started_at",
        "published_at",
        "response_completed_at",
        "failed_at",
    }
    if isinstance(value, dict):
        return {k: normalize_obs(v, root) for k, v in value.items() if k not in skip_keys}
    if isinstance(value, list):
        return [normalize_obs(x, root) for x in value]
    if isinstance(value, str):
        return value.replace(str(root), "<ROOT>")
    return value


def bts_source_path() -> str:
    return str(Path(bts.__file__).resolve())
