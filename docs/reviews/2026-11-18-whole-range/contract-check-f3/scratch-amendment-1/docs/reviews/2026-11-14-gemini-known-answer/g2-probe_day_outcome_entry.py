"""Adversarial probes for bts day-outcome, entry_receipt, and check-pick-entered."""
from __future__ import annotations

import gzip
import json
import tempfile
from datetime import date as date_cls
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from bts.cli import cli
from bts.daily_decision import write_decision
from bts.day_outcome import gather
from bts.entry_receipt import discover, receipts_root
from bts.picks import DailyPick, Pick, save_pick


def _setup_committed_day(root: Path, date: str, batter_id: int = 592450,
                         game_time_utc: str = "2027-05-15T23:10:00Z") -> Path:
    picks = root / "picks"
    picks.mkdir(parents=True, exist_ok=True)
    daily = DailyPick(
        date=date,
        run_time="17:00 ET",
        pick=Pick(
            batter_name="Aaron Judge", batter_id=batter_id, team="NYY",
            lineup_position=2, pitcher_name="Brayan Bello", pitcher_id=678394,
            p_game_hit=0.81, flags=[], projected_lineup=False,
            game_pk=9001, game_time=game_time_utc,
        ),
        double_down=None,
        runner_up=None,
        delivered_at=f"{date}T17:05:00-04:00",
        notification_sent=True, notification_channel="bluesky_dm",
    )
    save_pick(daily, picks)
    write_decision(
        date, picks, action="single", source="mdp",
        primary={"batter_id": batter_id, "batter_name": "Aaron Judge", "p_game_hit": 0.81},
        delivery_status="delivered", scoreable=True,
    )
    return picks


def _write_units_snapshot(root: Path, date: str, feed_id: int | None, round_id: int = 42,
                          unit_id: int = 700, snap_name: str = "2027-05-15T180000Z") -> None:
    units_dir = root / "leaderboard" / "static_snapshots" / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "captured_at": snap_name,
        "units": [{
            "id": unit_id,
            "roundId": round_id,
            "unitType": "ROUND",
            "feedId": feed_id,
            "startDate": f"{date}T04:00:00Z",
            "endDate": f"{date}T23:59:59Z",
            "title": date,
        }],
    }
    gz_path = units_dir / f"{snap_name}.json.gz"
    gz_path.write_bytes(gzip.compress(json.dumps(payload).encode()))


def _mock_contest_env(date: str, bts_player_id: int = 99, mlb_id: int = 592450,
                      round_id: int = 42, unit_id: int = 700):
    d_obj = date_cls.fromisoformat(date)
    session = SimpleNamespace(user_id="uid-1", username="stonehengee", xsid="xsid-1")
    profile = {"current_streak": 5, "predictions": []}
    pending = [{
        "roundId": round_id,
        "unitId": unit_id,
        "playerId": bts_player_id,
        "number": 1,
        "result": "pending",
    }]
    rounds = {round_id: d_obj}
    crosswalk = {bts_player_id: mlb_id}
    return session, profile, pending, rounds, crosswalk


def probe_premature_confirmed_marker_poisons_day_outcome():
    """Run check-pick-entered BEFORE units.json has feedId, then static capture gets feedId,
    then re-run check-pick-entered, then run day-outcome."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        date = "2027-05-15"
        picks = _setup_committed_day(root, date)
        # Early static capture before feedId bound (or before first capture):
        _write_units_snapshot(root, date, feed_id=None, round_id=42, unit_id=700, snap_name="2027-05-15T180000Z")
        session, profile, pending, rounds, crosswalk = _mock_contest_env(date)

        runner = CliRunner()
        with patch("bts.leaderboard.auth.load_session_cookies", return_value={}), \
             patch("bts.leaderboard.auth.extract_uid", return_value="uid-1"), \
             patch("bts.leaderboard.auth.fetch_login_session", return_value=session), \
             patch("bts.contest_fetch.fetch_profile", return_value=profile), \
             patch("bts.contest_fetch.fetch_pending_predictions", return_value=pending), \
             patch("bts.cli._fetch_rounds", return_value=rounds), \
             patch("bts.cli._fetch_bts_to_mlb", return_value=crosswalk):
            # Run 1 at 18:05 ET (65 min before 19:10 first pitch)
            r1 = runner.invoke(cli, [
                "check-pick-entered", "--picks-dir", str(picks),
                "--expected-username", "stonehengee", "--now-et", f"{date}T18:05:00",
            ])
            # Static capture refreshes at 18:10 ET with feedId=9001!
            _write_units_snapshot(root, date, feed_id=9001, round_id=42, unit_id=700, snap_name="2027-05-15T221000Z")
            # Run 2 at 18:20 ET (50 min before 19:10) and Run 3 at 18:35 ET
            r2 = runner.invoke(cli, [
                "check-pick-entered", "--picks-dir", str(picks),
                "--expected-username", "stonehengee", "--now-et", f"{date}T18:20:00",
            ])
            r3 = runner.invoke(cli, [
                "check-pick-entered", "--picks-dir", str(picks),
                "--expected-username", "stonehengee", "--now-et", f"{date}T18:35:00",
            ])

        marker = json.loads((picks.parent / "health_state" / "pick_entry_check.json").read_text())
        config = {"scheduler": {"entry_intent": "enter", "pick_delivery": "dm"}}
        with patch("bts.scheduler.fetch_schedule", return_value=[{"gamePk": 9001, "gameDate": f"{date}T23:10:00Z"}]):
            outcome = gather(date, picks, config, expected_username="stonehengee")

        receipts = discover(picks, date)
        outcomes = [(r.get("outcome"),
                     (r.get("qualification") or {}).get("all_confirmed"),
                     (r.get("qualification") or {}).get("slots"))
                    for r in receipts]

        print("[PROBE 3A premature confirmed marker + delayed units.json -> day-outcome]")
        print(f"  r1={r1.output.strip()} (exit={r1.exit_code})")
        print(f"  r2={r2.output.strip()} (exit={r2.exit_code})")
        print(f"  r3={r3.output.strip()} (exit={r3.exit_code})")
        print(f"  marker={marker}")
        print(f"  receipts={outcomes}")
        print(f"  day_outcome: state={outcome.state}, ok={outcome.ok}, detail={outcome.detail}")


def probe_ordinary_day_56_receipts_and_already_confirmed():
    """Simulate an ordinary day with 56 check-pick-entered runs (*/15 from 10:00 to 23:45)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        date = "2027-05-15"
        picks = _setup_committed_day(root, date, game_time_utc=f"{date}T23:10:00Z")  # 19:10 ET
        _write_units_snapshot(root, date, feed_id=9001, round_id=42, unit_id=700)
        session, profile, pending, rounds, crosswalk = _mock_contest_env(date)
        runner = CliRunner()

        with patch("bts.leaderboard.auth.load_session_cookies", return_value={}), \
             patch("bts.leaderboard.auth.extract_uid", return_value="uid-1"), \
             patch("bts.leaderboard.auth.fetch_login_session", return_value=session), \
             patch("bts.contest_fetch.fetch_profile", return_value=profile), \
             patch("bts.contest_fetch.fetch_pending_predictions", return_value=pending), \
             patch("bts.cli._fetch_rounds", return_value=rounds), \
             patch("bts.cli._fetch_bts_to_mlb", return_value=crosswalk):
            for hour in range(10, 24):
                for minute in (0, 15, 30, 45):
                    ts = f"{date}T{hour:02d}:{minute:02d}:00"
                    res = runner.invoke(cli, [
                        "check-pick-entered", "--picks-dir", str(picks),
                        "--expected-username", "stonehengee", "--now-et", ts,
                    ])
                    assert res.exit_code == 0, res.output

        receipts = discover(picks, date)
        config = {"scheduler": {"entry_intent": "enter", "pick_delivery": "dm"}}
        with patch("bts.scheduler.fetch_schedule", return_value=[{"gamePk": 9001, "gameDate": f"{date}T23:10:00Z"}]):
            outcome = gather(date, picks, config, expected_username="stonehengee")
        print("[PROBE 3B ordinary day with 56 receipts]")
        print(f"  receipt_count={len(receipts)}, day_outcome: state={outcome.state}, ok={outcome.ok}, detail={outcome.detail}")


def probe_doubleheader_units_multiple_rounds():
    """On a doubleheader (same game in two snapshots or two units), how does qualify behave?"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        date = "2027-05-15"
        picks = _setup_committed_day(root, date, game_time_utc=f"{date}T17:10:00Z")  # 13:10 ET
        units_dir = root / "leaderboard" / "static_snapshots" / "units"
        units_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "captured_at": f"{date}T160000Z",
            "units": [
                {"id": 700, "roundId": 42, "unitType": "ROUND", "feedId": 9001},
                {"id": 701, "roundId": 43, "unitType": "ROUND", "feedId": 9002},
            ],
        }
        (units_dir / f"{date}T160000Z.json.gz").write_bytes(gzip.compress(json.dumps(payload).encode()))
        session, profile, pending, rounds, crosswalk = _mock_contest_env(date)
        runner = CliRunner()
        with patch("bts.leaderboard.auth.load_session_cookies", return_value={}), \
             patch("bts.leaderboard.auth.extract_uid", return_value="uid-1"), \
             patch("bts.leaderboard.auth.fetch_login_session", return_value=session), \
             patch("bts.contest_fetch.fetch_profile", return_value=profile), \
             patch("bts.contest_fetch.fetch_pending_predictions", return_value=pending), \
             patch("bts.cli._fetch_rounds", return_value=rounds), \
             patch("bts.cli._fetch_bts_to_mlb", return_value=crosswalk):
            runner.invoke(cli, [
                "check-pick-entered", "--picks-dir", str(picks),
                "--expected-username", "stonehengee", "--now-et", f"{date}T12:15:00",
            ])
        config = {"scheduler": {"entry_intent": "enter", "pick_delivery": "dm"}}
        with patch("bts.scheduler.fetch_schedule", return_value=[{"gamePk": 9001, "gameDate": f"{date}T17:10:00Z"}]):
            outcome = gather(date, picks, config, expected_username="stonehengee")
        receipts = discover(picks, date)
        rec_data = receipts[0]
        print("[PROBE 3C doubleheader / two units on same date]")
        print(f"  receipt outcome={rec_data.get('outcome')}, qual={rec_data.get('qualification')}")
        print(f"  day_outcome: state={outcome.state}, ok={outcome.ok}, detail={outcome.detail}")


if __name__ == "__main__":
    probe_premature_confirmed_marker_poisons_day_outcome()
    probe_ordinary_day_56_receipts_and_already_confirmed()
    probe_doubleheader_units_multiple_rounds()
