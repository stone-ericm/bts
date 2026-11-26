"""Shared synthetic harness for Carbon round-1 review probes of change set F1.

Replaces all auth, contest, schedule, and DM transports and blocks any real network call.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import date, datetime, timezone
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

import bts
from bts import entry_receipt, picks
from bts.cli import cli
from bts.daily_decision import write_decision
from bts.health import pick_entry

DEFAULT_DAY = "2027-03-24"
DEFAULT_PITCH = f"{DEFAULT_DAY}T19:40:00-04:00"  # cutoff at 19:35 ET
DEFAULT_XW = {1: 1010, 2: 1020, 3: 1030}
UNIT_RIGHT_1 = {"id": 77, "roundId": 1, "feedId": 101}
UNIT_WRONG_1 = {"id": 77, "roundId": 1, "feedId": 102}
UNIT_RIGHT_2 = {"id": 78, "roundId": 1, "feedId": 201}
UNIT_WRONG_2 = {"id": 78, "roundId": 1, "feedId": 202}


def _forbid_network(*args, **kwargs):
    raise AssertionError(f"Network access is forbidden in review probes: {args!r}")


def make_row(unit=77, player=1, number=1, rnd=1, result=None, **extra):
    out = {"roundId": rnd, "unitId": unit, "playerId": player, "number": number, "result": result}
    out.update(extra)
    return out


class ReviewWorld:
    def __init__(self, root: Path, day: str = DEFAULT_DAY):
        self.root = Path(root)
        self.day = day
        self.picks_dir = self.root / "picks"
        self.picks_dir.mkdir(parents=True, exist_ok=True)
        self.marker_path = self.root / "health_state" / "pick_entry_check.json"
        self.units_dir = entry_receipt.units_snapshot_dir(self.picks_dir)
        self.units_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.root / "config.toml"
        self.config_path.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
        self.runner = CliRunner()

    def write_pick_and_decision(
        self,
        *,
        batter_id=1010,
        batter_name="Batter 1010",
        game_pk=101,
        game_time=DEFAULT_PITCH,
        dd=None,
        delivery="delivered",
        scoreable=True,
        notification_sent=True,
        raw_pick_patch=None,
        decision_date=None,
    ):
        p1 = picks.Pick(
            batter_name="Batter 1010" if batter_name is None else str(batter_name),
            batter_id=1010 if batter_id is None else (batter_id if isinstance(batter_id, int) else 1010),
            team="NYY",
            lineup_position=1,
            pitcher_name="Pitcher A",
            pitcher_id=500,
            p_game_hit=0.80,
            flags=[],
            projected_lineup=False,
            game_pk=101 if game_pk is None else (game_pk if isinstance(game_pk, int) else 101),
            game_time=game_time if isinstance(game_time, str) else DEFAULT_PITCH,
        )
        p2 = None
        if dd is not None:
            dd_bid, dd_gpk, dd_time, dd_name = (
                dd[0],
                dd[1],
                dd[2] if len(dd) > 2 else DEFAULT_PITCH,
                dd[3] if len(dd) > 3 else f"Batter {dd[0]}",
            )
            p2 = picks.Pick(
                batter_name="Batter DD" if dd_name is None else str(dd_name),
                batter_id=1020 if dd_bid is None else (dd_bid if isinstance(dd_bid, int) else 1020),
                team="BOS",
                lineup_position=2,
                pitcher_name="Pitcher B",
                pitcher_id=501,
                p_game_hit=0.78,
                flags=[],
                projected_lineup=False,
                game_pk=201 if dd_gpk is None else (dd_gpk if isinstance(dd_gpk, int) else 201),
                game_time=dd_time if isinstance(dd_time, str) else DEFAULT_PITCH,
            )
        daily = picks.DailyPick(
            date=self.day,
            run_time=f"{self.day}T16:00:00-04:00",
            pick=p1,
            double_down=p2,
            runner_up=None,
            notification_sent=notification_sent,
            notification_channel="bluesky_dm" if notification_sent else None,
            notification_id="dm-fixture" if notification_sent else None,
        )
        pick_file = picks.save_pick(daily, self.picks_dir)
        # Apply exact JSON-level overrides (null/non-int game_pk, null batter_name, custom game_time, etc.)
        body = json.loads(pick_file.read_text())
        body["pick"]["batter_id"] = batter_id
        body["pick"]["batter_name"] = batter_name
        body["pick"]["game_pk"] = game_pk
        body["pick"]["game_time"] = game_time
        if dd is not None:
            body["double_down"]["batter_id"] = dd[0]
            body["double_down"]["game_pk"] = dd[1]
            body["double_down"]["game_time"] = dd[2] if len(dd) > 2 else DEFAULT_PITCH
            body["double_down"]["batter_name"] = dd[3] if len(dd) > 3 else f"Batter {dd[0]}"
        if raw_pick_patch:
            raw_pick_patch(body)
        pick_file.write_text(json.dumps(body, indent=2))

        dec_path = self.picks_dir / (decision_date or self.day) / "decision.json"
        if delivery is None:
            if dec_path.exists():
                dec_path.unlink()
        else:
            write_decision(
                decision_date or self.day,
                self.picks_dir,
                action="double" if dd is not None else "single",
                source="mdp",
                delivery_status=delivery,
                scoreable=scoreable,
                primary={"batter_name": batter_name if isinstance(batter_name, str) else ""},
                double_down={"batter_name": (dd[3] if len(dd) > 3 else f"Batter {dd[0]}") or ""} if dd else None,
                now=datetime.fromisoformat(f"{self.day}T20:00:00+00:00"),
            )
        return pick_file

    def clear_captures(self):
        if self.units_dir.exists():
            for child in self.units_dir.iterdir():
                if child.is_dir():
                    child.rmdir()
                else:
                    child.unlink()

    def write_capture(self, units_list, stamp: str | None = None) -> Path:
        stamp = stamp or f"{self.day.replace('-', '')}T210000Z"
        path = self.units_dir / f"{stamp}.json.gz"
        raw = gzip.compress(json.dumps({"units": units_list}).encode("utf-8"), mtime=0)
        path.write_bytes(raw)
        return path

    def set_marker(self, marker_obj=None, raw_bytes: bytes | None = None):
        if marker_obj is None and raw_bytes is None:
            if self.marker_path.exists():
                if self.marker_path.is_dir():
                    self.marker_path.rmdir()
                else:
                    self.marker_path.unlink()
            return
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_bytes is not None:
            self.marker_path.write_bytes(raw_bytes)
        else:
            self.marker_path.write_text(json.dumps(marker_obj))

    def read_marker(self):
        if not self.marker_path.exists() or self.marker_path.is_dir():
            return None
        try:
            return json.loads(self.marker_path.read_text())
        except (OSError, ValueError):
            return {"_unreadable": True}

    def run_watchdog(
        self,
        at_hhmm: str = "18:30",
        pending=(),
        *,
        now_et_full: str | None = None,
        profile=None,
        rounds=None,
        crosswalk=None,
        dm_mode: str = "success",
        elapsed_sec: float = 0.0,
        session_username: str = "operator",
        expected_username: str | None = "operator",
        fetch_error: tuple[str, Exception] | None = None,
        extra_patches=(),
        window_min: int | None = None,
    ) -> dict:
        sent_dms: list[tuple[str, str]] = []
        attempted_dms: list[tuple[str, str]] = []
        fetches: list[str] = []
        mono_calls: list[int] = []

        def fake_send_dm(recipient, msg):
            attempted_dms.append((recipient, msg))
            if dm_mode == "failure":
                raise RuntimeError("synthetic DM failure")
            sent_dms.append((recipient, msg))

        def make_leaf(name, payload):
            def _leaf(*args, **kwargs):
                fetches.append(name)
                if fetch_error and fetch_error[0] == name:
                    raise fetch_error[1]
                return payload
            return _leaf

        def fake_monotonic():
            mono_calls.append(1)
            return 1000.0 if len(mono_calls) == 1 else 1000.0 + elapsed_sec

        session = SimpleNamespace(user_id="42", username=session_username, xsid="xsid-fixed")
        rounds_val = {1: date.fromisoformat(self.day)} if rounds is None else rounds
        xw_val = DEFAULT_XW if crosswalk is None else crosswalk
        prof_val = {"predictions": []} if profile is None else profile

        patches = [
            ("httpx.get", _forbid_network),
            ("httpx.post", _forbid_network),
            ("urllib.request.urlopen", _forbid_network),
            ("bts.leaderboard.auth.load_session_cookies", lambda: {}),
            ("bts.leaderboard.auth.extract_uid", lambda c: "42"),
            ("bts.leaderboard.auth.fetch_login_session", make_leaf("login", session)),
            ("bts.contest_fetch.fetch_profile", make_leaf("profile", prof_val)),
            ("bts.contest_fetch.fetch_pending_predictions", make_leaf("pending", list(pending))),
            ("bts.cli._fetch_rounds", make_leaf("rounds", rounds_val)),
            ("bts.cli._fetch_bts_to_mlb", make_leaf("crosswalk", xw_val)),
            ("bts.dm.send_dm", fake_send_dm),
            ("bts.picks._git_head_sha", lambda *a, **k: "fixed-head-sha"),
            ("time.monotonic", fake_monotonic),
            *extra_patches,
        ]
        now_arg = now_et_full or f"{self.day}T{at_hhmm}:00"
        args = ["check-pick-entered", "--picks-dir", str(self.picks_dir), "--now-et", now_arg]
        if expected_username is not None:
            args += ["--expected-username", expected_username]
        if dm_mode != "none":
            args += ["--dm-recipient", "operator.invalid"]
        if window_min is not None:
            args += ["--window-min", str(window_min)]

        with ExitStack() as stack:
            for target, replacement in patches:
                stack.enter_context(patch(target, replacement))
            res = self.runner.invoke(cli, args)

        rec_day = now_arg[:10]
        sealed = entry_receipt.discover(self.picks_dir, rec_day)
        return {
            "bts_file": bts.__file__,
            "exit_code": res.exit_code,
            "exception": type(res.exception).__name__ if (res.exception and not isinstance(res.exception, SystemExit)) else None,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "dm_sent": [m for _, m in sent_dms],
            "dm_attempted": [m for _, m in attempted_dms],
            "fetches": fetches,
            "marker": self.read_marker(),
            "receipts": sealed,
        }

    def run_day_outcome(self, games=None) -> dict:
        dms = []
        sched = [{"gamePk": 101}, {"gamePk": 102}] if games is None else games
        with patch("httpx.get", _forbid_network), \
             patch("bts.scheduler.fetch_schedule", lambda _: sched), \
             patch("bts.dm.send_dm", lambda _, msg: dms.append(msg)):
            res = self.runner.invoke(
                cli,
                [
                    "day-outcome",
                    "--date", self.day,
                    "--config", str(self.config_path),
                    "--picks-dir", str(self.picks_dir),
                    "--expected-username", "operator",
                    "--dm-recipient", "operator.invalid",
                ],
            )
        return {
            "exit_code": res.exit_code,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "outcome": json.loads(res.stdout) if res.stdout.strip().startswith("{") else None,
            "dm_sent": dms,
        }

    def run_health_pick_entry(self, now_utc: datetime | None = None) -> list[dict]:
        when = now_utc or datetime(2027, 3, 25, 4, 0, tzinfo=timezone.utc)
        alerts = pick_entry.check(self.picks_dir, date.fromisoformat(self.day), when)
        return [{"level": a.level, "source": a.source, "message": a.message} for a in alerts]


VOLATILE_KEYS = frozenset({
    "receipt",
    "attempt_id",
    "confirmed_by",
    "checked_at",
    "started_at",
    "published_at",
    "response_completed_at",
    "failed_at",
    "bts_file",
})


def strip_volatile(obj, picks_dir: Path | str | None = None):
    if isinstance(obj, dict):
        return {k: strip_volatile(v, picks_dir) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [strip_volatile(x, picks_dir) for x in obj]
    if isinstance(obj, str) and picks_dir is not None:
        return obj.replace(str(picks_dir), "<PICKS_DIR>")
    return obj
