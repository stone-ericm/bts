"""Probe 1: Round-1 closure (B1, B2, refresh None) and Item 2 (build.py vs e3edbf58)."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import stat
import subprocess
import tempfile
from contextlib import redirect_stderr
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
from click.testing import CliRunner

from bts.cli import cli
from bts.data.build import build_season as head_build_season
from bts.model.predict import _refresh_season_data
import bts.scheduler as sched

REPO = Path(__file__).resolve().parents[3]
ET = ZoneInfo("America/New_York")


def load_parent_build():
    src = subprocess.run(
        ["git", "show", "e3edbf58:src/bts/data/build.py"],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp(prefix="parent_build_")) / "parent_build.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location("parent_build_e3edbf58", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_season


def make_regular_feed(pk: int = 999999, season: str = "2027", date: str = "2027-03-24", gtype: str = "R") -> dict:
    return {
        "gameData": {
            "datetime": {"officialDate": date},
            "teams": {
                "away": {"id": 134, "name": "Pittsburgh Pirates"},
                "home": {"id": 121, "name": "New York Mets"},
            },
            "venue": {"id": 3289, "name": "Citi Field", "fieldInfo": {"roofType": "Open"}},
            "weather": {"condition": "Sunny", "temp": "78", "wind": "9 mph, Out To CF"},
            "game": {"pk": pk, "season": season, "type": gtype},
        },
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {
                        "batters": [100001],
                        "players": {
                            "ID100001": {"person": {"id": 100001, "fullName": "Test Batter"}, "battingOrder": "300"},
                        },
                    },
                    "home": {
                        "batters": [200001],
                        "players": {
                            "ID200001": {"person": {"id": 200001, "fullName": "Home Batter"}, "battingOrder": "100"},
                        },
                    },
                },
                "officials": [{"official": {"id": 427215, "fullName": "Test Umpire"}, "officialType": "Home Plate"}],
            },
            "plays": {
                "allPlays": [
                    {
                        "result": {"eventType": "single", "description": "Test Batter singles."},
                        "about": {"halfInning": "top", "inning": 1, "hasReview": False},
                        "matchup": {
                            "batter": {"id": 100001},
                            "batSide": {"code": "R", "description": "Right"},
                            "pitcher": {"id": 300001},
                            "pitchHand": {"code": "L", "description": "Left"},
                        },
                        "count": {"balls": 1, "strikes": 2},
                        "playEvents": [
                            {
                                "isPitch": True,
                                "pitchNumber": 1,
                                "details": {
                                    "call": {"code": "X", "description": "In play, no out"},
                                    "type": {"code": "FF", "description": "Four-Seam Fastball"},
                                    "hasReview": False,
                                },
                                "pitchData": {
                                    "coordinates": {"pX": -0.2, "pZ": 2.5},
                                    "strikeZoneTop": 3.4,
                                    "strikeZoneBottom": 1.7,
                                    "startSpeed": 94.0,
                                    "endSpeed": 86.0,
                                    "extension": 6.4,
                                    "breaks": {"spinRate": 2350, "breakVertical": -14.0, "breakHorizontal": 9.0},
                                },
                                "hitData": {
                                    "launchSpeed": 98.3,
                                    "launchAngle": 12.0,
                                    "trajectory": "line_drive",
                                    "hardness": "hard",
                                    "totalDistance": 310.0,
                                },
                                "count": {"balls": 1, "strikes": 2},
                            }
                        ],
                    }
                ]
            },
        },
    }


def probe_b1_outage_simulation():
    print("=== PROBE 1A: Round-1 B1 (thin-slate outage between 12:05 and 12:10) ===")
    diff = subprocess.run(
        ["git", "diff", "e3edbf58", "HEAD", "--", "src/bts/scheduler.py"],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout
    print(f"git diff e3edbf58 HEAD -- src/bts/scheduler.py bytes: {len(diff)}")

    date = "2027-04-15"
    games = [
        {
            "gamePk": 101,
            "gameType": "R",
            "gameDate": "2027-04-15T17:10:00Z",  # 13:10 EDT
            "officialDate": date,
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled", "statusCode": "S"},
            "teams": {"away": {"team": {"name": "NYY"}}, "home": {"team": {"name": "BOS"}}},
        },
        {
            "gamePk": 202,
            "gameType": "R",
            "gameDate": "2027-04-15T18:05:00Z",  # 14:05 EDT
            "officialDate": date,
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled", "statusCode": "S"},
            "teams": {"away": {"team": {"name": "NYM"}}, "home": {"team": {"name": "ATL"}}},
        },
    ]
    clock = [datetime(2027, 4, 15, 9, 0, tzinfo=ET)]
    runs = sched.plan_run_times(games, offset_min=60, cluster_min=10, now=clock[0])
    print("Planned run times:", [(r["time_et"].strftime("%H:%M"), r["game_pks"]) for r in runs])

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        picks_dir = tmp / "data" / "picks"
        models_dir = tmp / "data" / "models"
        models_dir.mkdir(parents=True)
        config = {
            "orchestrator": {
                "picks_dir": str(picks_dir),
                "models_dir": str(models_dir),
                "heartbeat_path": str(tmp / "data" / ".heartbeat"),
            },
            "bluesky": {"dm_recipient": "did:plc:test"},
            "scheduler": {
                "pick_delivery": "dm",
                "entry_intent": "enter",
                "lineup_check_offset_min": 60,
                "cluster_min": 10,
                "early_lock_gap": 0.03,
                "fallback_deadline_min": 35,
                "live_forward_capture_on_lock": False,
                "shadow_model": False,
            },
            "health_checks": {"enabled": False},
            "tiers": [],
        }

        def fake_sleep(secs):
            if secs > 0:
                from datetime import timedelta
                clock[0] = clock[0] + timedelta(seconds=secs)

        call_times = []

        def fake_predict(tiers_, d):
            now_hm = clock[0].strftime("%H:%M")
            call_times.append(now_hm)
            # Outage through 12:08 ET -> 10:10, 11:05, 12:05 all fail; 12:10 succeeds!
            if clock[0] < datetime(2027, 4, 15, 12, 8, tzinfo=ET):
                return None, "outage"
            df = pd.DataFrame([
                {
                    "batter_name": "Aaron Judge", "batter_id": 592450, "team": "NYY",
                    "lineup": 2, "pitcher_name": "Brayan Bello", "pitcher_id": 678394,
                    "pitcher_team": "BOS", "p_game_hit": 0.81, "flags": "",
                    "game_pk": 101, "game_time": "2027-04-15T17:10:00Z",
                },
                {
                    "batter_name": "Juan Soto", "batter_id": 665742, "team": "NYM",
                    "lineup": 2, "pitcher_name": "Max Fried", "pitcher_id": 608331,
                    "pitcher_team": "ATL", "p_game_hit": 0.76, "flags": "",
                    "game_pk": 202, "game_time": "2027-04-15T18:05:00Z",
                },
            ])
            return df, "local"

        dms = []
        with patch.object(sched, "_now_et", side_effect=lambda: clock[0]), \
             patch.object(sched, "_watchdog_ping_sleep", side_effect=fake_sleep), \
             patch.object(sched, "_idle_until_next_wakeup", side_effect=lambda *a, **k: None), \
             patch.object(sched, "fetch_schedule", side_effect=lambda d: games if d == date else []), \
             patch.object(sched, "check_confirmed_lineups", return_value={101: {"away", "home"}, 202: {"away", "home"}}), \
             patch("bts.orchestrator.run_cascade", side_effect=fake_predict), \
             patch.object(sched, "_delivery_game_statuses", return_value={
                 101: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
                 202: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
             }), \
             patch("bts.picks.get_game_statuses_detailed", return_value={
                 101: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
                 202: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
             }), \
             patch.object(sched, "run_result_polling", return_value="final"), \
             patch("bts.dm.send_dm", side_effect=lambda r, m: dms.append((clock[0].strftime("%H:%M"), r, m)) or "dm-1"):
            sched.run_day(date, config)

        dec = json.loads((picks_dir / date / "decision.json").read_text())
        saved = json.loads((picks_dir / f"{date}.json").read_text())
        print("Cascade call times:", call_times)
        print("DMs sent:", [(t, r, m.splitlines()[0]) for t, r, m in dms])
        print("Decision record:", {k: dec[k] for k in ("action", "delivery_status", "scoreable", "primary")})
        print("Delivered_at:", saved["delivered_at"])


def run_build_case(fn, raw_dir: Path, out_path: Path):
    buf = io.StringIO()
    try:
        with redirect_stderr(buf):
            res = fn(raw_dir, out_path, 2027)
        exists = out_path.exists()
        rows = len(res) if res is not None else None
        return {"status": "ok", "rows": rows, "wrote": exists, "stderr": buf.getvalue().strip()}
    except Exception as exc:
        return {
            "status": f"{type(exc).__name__}: {exc}",
            "rows": None,
            "wrote": out_path.exists(),
            "stderr": buf.getvalue().strip(),
        }


def probe_item2_matrix():
    print("\n=== PROBE 1B: Item 2 Matrix (HEAD build_season vs e3edbf58 build_season) ===")
    parent_build = load_parent_build()

    cases = [
        ("missing_dir", lambda raw: None),
        ("empty_dir", lambda raw: (raw / "2027").mkdir(parents=True)),
        ("weather_only", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1_weather.json").write_text("{}"))),
        ("S_only", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text(json.dumps(make_regular_feed(gtype="S"))))),
        ("season_path_is_regular_file", lambda raw: (raw.mkdir(parents=True, exist_ok=True), (raw / "2027").write_text("not a dir"))),
        ("season_dir_chmod_000", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text(json.dumps(make_regular_feed())), os.chmod(raw / "2027", 0))),
        ("typeless_dict_alone", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text("{}"))),
        ("typeless_dict_with_R", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text("{}"), (raw / "2027" / "2.json").write_text(json.dumps(make_regular_feed())))),
        ("empty_str_type_with_R", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text('{"gameData":{"game":{"type":""}}}'), (raw / "2027" / "2.json").write_text(json.dumps(make_regular_feed())))),
        ("ws_str_type_with_R", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text('{"gameData":{"game":{"type":"   "}}}'), (raw / "2027" / "2.json").write_text(json.dumps(make_regular_feed())))),
        ("null_root_with_R", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text("null"), (raw / "2027" / "2.json").write_text(json.dumps(make_regular_feed())))),
        ("list_root_with_R", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text("[]"), (raw / "2027" / "2.json").write_text(json.dumps(make_regular_feed())))),
        ("int_root_with_R", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text("500"), (raw / "2027" / "2.json").write_text(json.dumps(make_regular_feed())))),
        ("str_root_with_R", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text('"Gateway Timeout"'), (raw / "2027" / "2.json").write_text(json.dumps(make_regular_feed())))),
        ("gameData_null_with_R", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text('{"gameData": null}'), (raw / "2027" / "2.json").write_text(json.dumps(make_regular_feed())))),
        ("game_null_with_R", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text('{"gameData": {"game": null}}'), (raw / "2027" / "2.json").write_text(json.dumps(make_regular_feed())))),
        ("type_list_with_R", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text('{"gameData": {"game": {"type": []}}}'), (raw / "2027" / "2.json").write_text(json.dumps(make_regular_feed())))),
        ("type_dict_with_R", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text('{"gameData": {"game": {"type": {}}}}'), (raw / "2027" / "2.json").write_text(json.dumps(make_regular_feed())))),
        ("type_int_with_R", lambda raw: ((raw / "2027").mkdir(parents=True), (raw / "2027" / "1.json").write_text('{"gameData": {"game": {"type": 7}}}'), (raw / "2027" / "2.json").write_text(json.dumps(make_regular_feed())))),
    ]

    for name, setup in cases:
        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            raw1, out1 = Path(td1) / "raw", Path(td1) / "proc" / "pa_2027.parquet"
            raw2, out2 = Path(td2) / "raw", Path(td2) / "proc" / "pa_2027.parquet"
            try:
                setup(raw1)
                setup(raw2)
                r_parent = run_build_case(parent_build, raw1, out1)
                r_head = run_build_case(head_build_season, raw2, out2)
            finally:
                for p in (raw1 / "2027", raw2 / "2027"):
                    if p.exists() and not os.access(p, os.R_OK):
                        os.chmod(p, stat.S_IRWXU)
            diff_marker = "DIFF!" if (r_parent["status"] != r_head["status"] or r_parent["rows"] != r_head["rows"] or r_parent["wrote"] != r_head["wrote"]) else "same"
            print(f"[{diff_marker:5s}] {name:28s} | parent=({r_parent['status'][:40]}, rows={r_parent['rows']}, wrote={r_parent['wrote']}) | head=({r_head['status'][:40]}, rows={r_head['rows']}, wrote={r_head['wrote']})")


def probe_corrupt_feed_nightly_and_refresh():
    print("\n=== PROBE 1C: Corrupt cached feed beside R game -> Nightly CLI && chain + _refresh_season_data ===")
    parent_build = load_parent_build()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        raw = tmp / "raw"
        proc = tmp / "proc"
        (raw / "2027").mkdir(parents=True)
        # Game 900001 (Opening Night, valid R feed, 1 PA)
        (raw / "2027" / "900001.json").write_text(json.dumps(make_regular_feed(pk=900001, date="2027-03-24")))
        # Game 900002 (Day 2 Final game whose API response cached `{"gameData": null}` or `null` via pull_feeds)
        (raw / "2027" / "900002.json").write_text('{"gameData": null}')

        # Pre-existing parquet with sentinel bytes
        proc.mkdir(parents=True)
        (proc / "pa_2027.parquet").write_bytes(b"PRIOR_PARQUET")

        # 1. Parent behavior on same directory
        try:
            parent_build(raw, tmp / "parent_out.parquet", 2027)
            parent_outcome = "succeeded"
        except Exception as e:
            parent_outcome = f"RAISED {type(e).__name__}: {e}"
        print("Parent on {900001=R, 900002={'gameData':null}}:", parent_outcome)

        # 2. HEAD CLI behavior (`bts data build --seasons 2027`)
        res = CliRunner().invoke(cli, ["data", "build", "--seasons", "2027", "--raw-dir", str(raw), "--out-dir", str(proc)])
        df_written = pd.read_parquet(proc / "pa_2027.parquet")
        print(f"HEAD CLI exit_code={res.exit_code}, overwrote_parquet={len(df_written)} rows (dropped 900002!), stderr={res.stderr.strip()!r}")

        # 3. HEAD _refresh_season_data memoization
        proc2 = tmp / "proc2"
        with patch("bts.data.pull.pull_feeds", return_value=[]):
            _refresh_season_data("2027-03-26", raw_dir=str(raw), processed_dir=str(proc2))
        markers = [p.name for p in proc2.glob(".refreshed_*")]
        print(f"HEAD _refresh_season_data markers written: {markers}, pa_2027 rows={len(pd.read_parquet(proc2 / 'pa_2027.parquet'))}")


def probe_parent_build_fixture_bug():
    print("\n=== PROBE 1D: tests/test_data_build_empty.py:102 `git show main:src/bts/data/build.py` ===")
    r_main = subprocess.run(
        ["git", "show", "main:src/bts/data/build.py"],
        cwd=REPO, capture_output=True, text=True,
    )
    print(f"`git show main:src/bts/data/build.py` -> returncode={r_main.returncode}, stderr={r_main.stderr.strip()!r}")
    r_sha = subprocess.run(
        ["git", "show", "e3edbf58:src/bts/data/build.py"],
        cwd=REPO, capture_output=True, text=True,
    )
    print(f"`git show e3edbf58:src/bts/data/build.py` -> returncode={r_sha.returncode}, bytes={len(r_sha.stdout)}")


if __name__ == "__main__":
    probe_b1_outage_simulation()
    probe_item2_matrix()
    probe_corrupt_feed_nightly_and_refresh()
    probe_parent_build_fixture_bug()
