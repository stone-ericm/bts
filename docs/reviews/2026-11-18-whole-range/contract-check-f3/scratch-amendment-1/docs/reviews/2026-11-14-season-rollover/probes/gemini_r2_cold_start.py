"""Probe R1 closure + Requirement 3 real cold-start pipeline trace.

Builds real multi-date 2025 + 2026 regular-season parquets via build_season,
uses real compute_all_features, train_model, train_blend, _build_feature_lookups,
predict, select_pick (with real mdp_policy.npz + mdp_tail_policy.npz),
run_single_check, deliver_pick, write_decision, and bts preview.
Mocks ONLY MLB HTTP endpoints, DM/Bluesky transports, and systemd/watchdog.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import shutil
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO = Path("/Users/eric/projects/bts-agy-g3")


def make_historical_feed(
    pk: int,
    official_date: str,
    season: int,
    away_id: int,
    home_id: int,
    away_batters: list[int],
    home_batters: list[int],
    away_pitcher: int,
    home_pitcher: int,
    venue_id: int,
    ump_id: int,
    rng: random.Random,
    *,
    game_type: str = "R",
) -> dict:
    def mk_players(batters, catcher_id):
        d = {}
        for idx, bid in enumerate(batters, start=1):
            d[f"ID{bid}"] = {
                "person": {"id": bid, "fullName": f"Batter {bid}"},
                "battingOrder": str(idx * 100),
                "allPositions": [{"code": "2"}] if bid == catcher_id else [],
            }
        return d

    plays = []
    # Give every batter 4 PAs with mix of hits and outs + Statcast data
    for _pa in range(4):
        for half, batters, pitcher in (
            ("top", away_batters, home_pitcher),
            ("bottom", home_batters, away_pitcher),
        ):
            for bid in batters:
                is_hit = rng.random() < 0.25
                ev = "single" if is_hit else "field_out"
                la = 12.0 if is_hit else rng.choice([5.0, 25.0])
                ls = 96.0 if is_hit else 85.0
                calls = [rng.choice(["C", "S", "B"]), "X"]
                plays.append(
                    {
                        "about": {
                            "inning": 1 + _pa,
                            "halfInning": half,
                            "isTopInning": half == "top",
                            "startTime": f"{official_date}T23:10:00Z",
                        },
                        "matchup": {
                            "batter": {"id": bid},
                            "pitcher": {"id": pitcher, "fullName": f"Pitcher {pitcher}"},
                            "batSide": {"code": "R"},
                            "pitchHand": {"code": "R" if pitcher % 2 == 0 else "L"},
                        },
                        "playEvents": [
                            {
                                "isPitch": True,
                                "details": {"call": {"code": c}, "type": {"code": "FF"}},
                                "pitchData": {
                                    "startSpeed": 94.0,
                                    "endSpeed": 85.0,
                                    "extension": 6.4,
                                    "strikeZoneTop": 3.4,
                                    "strikeZoneBottom": 1.6,
                                    "coordinates": {"pX": 0.1, "pZ": 2.4},
                                    "breaks": {"spinRate": 2300, "breakVertical": -14.0, "breakHorizontal": 6.0},
                                },
                                **({"hitData": {"launchSpeed": ls, "launchAngle": la, "trajectory": "line_drive", "hardness": "medium", "totalDistance": 220}} if i == len(calls) - 1 else {}),
                            }
                            for i, c in enumerate(calls)
                        ],
                        "result": {"eventType": ev},
                        "count": {"balls": 1, "strikes": 1},
                    }
                )

    return {
        "gameData": {
            "game": {"pk": pk, "type": game_type, "season": str(season), "doubleHeader": "N"},
            "datetime": {"officialDate": official_date},
            "venue": {"id": venue_id, "fieldInfo": {"roofType": "Open"}},
            "weather": {"temp": "72", "wind": "7 mph, Out To CF"},
            "teams": {
                "away": {"id": away_id, "abbreviation": f"T{away_id}"},
                "home": {"id": home_id, "abbreviation": f"T{home_id}"},
            },
            "officials": [{"officialType": "Home Plate", "official": {"id": ump_id, "fullName": f"Ump {ump_id}"}}],
        },
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"players": mk_players(away_batters, away_batters[-1])},
                    "home": {"players": mk_players(home_batters, home_batters[-1])},
                },
                "officials": [{"officialType": "Home Plate", "official": {"id": ump_id}}],
            },
            "plays": {"allPlays": plays},
        },
    }


def make_live_schedule_and_feed(
    date_str: str,
    games_spec: list[dict],
) -> tuple[dict, dict[int, dict]]:
    games = []
    feeds = {}
    for spec in games_spec:
        pk = spec["pk"]
        away_id = spec["away_id"]
        home_id = spec["home_id"]
        away_p = spec["away_pitcher"]
        home_p = spec["home_pitcher"]
        games.append(
            {
                "gamePk": pk,
                "gameType": spec.get("gameType", "R"),
                "officialDate": date_str,
                "gameDate": spec.get("gameDate", f"{date_str}T23:10:00Z"),
                "status": {
                    "abstractGameCode": spec.get("abstract", "P"),
                    "detailedState": spec.get("detailed", "Scheduled"),
                    "statusCode": "S",
                },
                "teams": {
                    "away": {"team": {"id": away_id}, "probablePitcher": {"id": away_p, "fullName": f"Pitcher {away_p}"}},
                    "home": {"team": {"id": home_id}, "probablePitcher": {"id": home_p, "fullName": f"Pitcher {home_p}"}},
                },
            }
        )
        def _players(batters, posted: bool):
            if not posted:
                return {}
            return {
                f"ID{bid}": {"person": {"id": bid, "fullName": f"Batter {bid}"}, "battingOrder": str(i * 100)}
                for i, bid in enumerate(batters, start=1)
            }

        feeds[pk] = {
            "gameData": {
                "game": {"pk": pk, "type": spec.get("gameType", "R"), "season": date_str[:4], "doubleHeader": "N"},
                "datetime": {"officialDate": date_str},
                "venue": {"id": spec.get("venue_id", 10), "fieldInfo": {"roofType": "Open"}},
                "weather": {"temp": "68", "wind": "6 mph, Out To CF"},
                "teams": {
                    "away": {"id": away_id, "abbreviation": spec.get("away_abbr", f"T{away_id}")},
                    "home": {"id": home_id, "abbreviation": spec.get("home_abbr", f"T{home_id}")},
                },
                "officials": [{"officialType": "Home Plate", "official": {"id": spec.get("ump_id", 901)}}],
            },
            "liveData": {
                "boxscore": {
                    "teams": {
                        "away": {"players": _players(spec["away_batters"], spec.get("away_posted", False))},
                        "home": {"players": _players(spec["home_batters"], spec.get("home_posted", False))},
                    },
                    "officials": [{"officialType": "Home Plate", "official": {"id": spec.get("ump_id", 901)}}],
                },
                "plays": {"allPlays": []},
            },
        }
    return {"dates": [{"date": date_str, "games": games}]}, feeds


class FakeResp:
    def __init__(self, obj: dict | bytes):
        self.raw = obj if isinstance(obj, bytes) else json.dumps(obj).encode("utf-8")

    def read(self):
        return self.raw


def main():
    from click.testing import CliRunner
    from zoneinfo import ZoneInfo
    from bts.cli import cli
    from bts.data.build import build_season
    from bts.model.predict import run_pipeline, load_blend
    from bts.orchestrator import run_and_pick
    from bts.scheduler import run_single_check, _deliver_and_lock_pick, SchedulerState, save_state
    from bts.picks import load_pick, load_streak
    from bts.daily_decision import load_decision
    from bts.features import park_drag as _park_drag

    _park_drag._reset_cache()

    rng = random.Random(42)
    away_batters = list(range(101, 110))
    home_batters = list(range(201, 210))
    t3_batters = list(range(301, 310))
    t4_batters = list(range(401, 410))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        old_cwd = Path.cwd()
        os.chdir(tmp)
        try:
            raw_dir = tmp / "data" / "raw"
            proc_dir = tmp / "data" / "processed"
            models_dir = tmp / "data" / "models"
            picks_dir = tmp / "data" / "picks"
            ext_drag = tmp / "data" / "external" / "park_drag"
            (tmp / "data" / "health_state").mkdir(parents=True)
            models_dir.mkdir(parents=True)
            picks_dir.mkdir(parents=True)
            ext_drag.mkdir(parents=True)

            # Copy real MDP artifacts (read-only copy) into temp cwd
            shutil.copy(REPO / "data/models/mdp_policy.npz", models_dir / "mdp_policy.npz")
            shutil.copy(REPO / "data/models/mdp_tail_policy.npz", models_dir / "mdp_tail_policy.npz")
            # Copy framing CSV if present so lookups match prod codepath
            if (REPO / "data/external/savant_catcher_framing.csv").exists():
                shutil.copy(
                    REPO / "data/external/savant_catcher_framing.csv",
                    tmp / "data/external/savant_catcher_framing.csv",
                )
            # Stale 2026 park_drag table + manifest (simulates offseason stale external table on 2027-03-24)
            pd.DataFrame(
                [{"venue_id": 10, "date": "2026-09-27", "park_drag_delta": 0.012}]
            ).to_csv(ext_drag / "park_drag_export.csv", index=False)
            (ext_drag / "park_drag_manifest.json").write_text(
                json.dumps({"max_source_game_date": "2026-09-27", "generated_at": "2026-09-28T07:45:00"})
            )

            # Build real historical raw + parquet for 2025 and 2026
            pk = 700001
            for season, start_d, end_d in (
                (2025, date(2025, 8, 1), date(2025, 9, 28)),
                (2026, date(2026, 8, 1), date(2026, 9, 27)),
            ):
                sdir = raw_dir / str(season)
                sdir.mkdir(parents=True)
                d = start_d
                i = 0
                while d <= end_d:
                    f1 = make_historical_feed(
                        pk, d.isoformat(), season, 108, 119,
                        away_batters, home_batters, 5001 + (i % 2), 6001 + (i % 2), 10, 901, rng
                    )
                    (sdir / f"{pk}.json").write_text(json.dumps(f1))
                    pk += 1
                    f2 = make_historical_feed(
                        pk, d.isoformat(), season, 110, 111,
                        t3_batters, t4_batters, 7001 + (i % 2), 8001 + (i % 2), 11, 902, rng
                    )
                    (sdir / f"{pk}.json").write_text(json.dumps(f2))
                    pk += 1
                    d += timedelta(days=2)
                    i += 1
                df_built = build_season(raw_dir, proc_dir / f"pa_{season}.parquet", season)
                print(f"Built pa_{season}.parquet: {len(df_built)} PAs across {df_built['date'].nunique()} dates")

            # Raw 2027 has Spring Training ("S") feeds ONLY initially
            (raw_dir / "2027").mkdir(parents=True)
            sp = make_historical_feed(
                900001, "2027-03-21", 2027, 108, 119,
                away_batters, home_batters, 5001, 6001, 10, 901, rng, game_type="S"
            )
            (raw_dir / "2027" / "900001.json").write_text(json.dumps(sp))

            # State store for fake MLB endpoints
            state_http = {
                "schedules_by_date": {},
                "feeds_by_pk": {},
                "team_schedules": {},
            }

            # Spring Final game on 2027-03-21 (does NOT count as regular season!)
            state_http["schedules_by_date"]["2027-03-21"] = {
                "dates": [{
                    "date": "2027-03-21",
                    "games": [{
                        "gamePk": 900001,
                        "gameType": "S",
                        "officialDate": "2027-03-21",
                        "status": {"abstractGameCode": "F", "detailedState": "Final", "statusCode": "F"},
                    }],
                }]
            }
            state_http["feeds_by_pk"][900001] = sp

            # Opening Night 2027-03-24: 1 regular-season game (pk=910001) BEFORE lineups post
            # Includes vet batters 101..108 + rookie debut batter 99109; vet pitcher 6001 + debut pitcher 99601
            on_away_batters = away_batters[:-1] + [99109]
            on_home_batters = home_batters
            sched_0324_pre, feeds_0324_pre = make_live_schedule_and_feed(
                "2027-03-24",
                [{
                    "pk": 910001, "away_id": 108, "home_id": 119,
                    "away_pitcher": 99601, "home_pitcher": 6001,
                    "away_batters": on_away_batters, "home_batters": on_home_batters,
                    "away_posted": False, "home_posted": False, "venue_id": 10, "ump_id": 901,
                }],
            )
            state_http["schedules_by_date"]["2027-03-24"] = sched_0324_pre
            state_http["feeds_by_pk"].update(feeds_0324_pre)

            def fake_urlopen(req, timeout=15):
                url = req.full_url if hasattr(req, "full_url") else str(req)
                if "/api/v1/schedule?" in url and "teamId=" in url:
                    # extract teamId
                    for part in url.split("?")[1].split("&"):
                        if part.startswith("teamId="):
                            tid = int(part.split("=")[1])
                            return FakeResp(state_http["team_schedules"].get(tid, {"dates": []}))
                if "/api/v1/schedule?" in url and "date=" in url:
                    for part in url.split("?")[1].split("&"):
                        if part.startswith("date="):
                            dstr = part.split("=")[1]
                            return FakeResp(state_http["schedules_by_date"].get(dstr, {"dates": []}))
                if "/api/v1.1/game/" in url and "/feed/live" in url:
                    pk_str = url.split("/api/v1.1/game/")[1].split("/")[0]
                    return FakeResp(state_http["feeds_by_pk"][int(pk_str)])
                raise AssertionError(f"Unexpected URL: {url}")

            dms_sent = []

            def fake_send_dm(recipient, text):
                dms_sent.append((recipient, text))
                return "dm-uri-1"

            with (
                patch("bts.model.predict.urlopen", fake_urlopen),
                patch("bts.util.urlopen", fake_urlopen),
                patch("bts.data.pull.urlopen", fake_urlopen),
                patch("bts.scheduler.notify_watchdog", lambda *a, **k: None),
                patch("bts.dm.send_dm", fake_send_dm),
            ):
                runner = CliRunner()

                # --- SCENARIO A: Opening Night 2027-03-24 BEFORE lineups post ---
                print("\n=== SCENARIO A: Opening Night 2027-03-24 pre-lineup ===")
                res_prev = runner.invoke(
                    cli,
                    ["preview", "--date", "2027-03-24", "--data-dir", str(proc_dir),
                     "--picks-dir", str(picks_dir), "--models-dir", str(models_dir)],
                )
                print(f"bts preview exit={res_prev.exit_code} output={res_prev.output!r}")
                print(f"blend_2027-03-24.pkl exists={ (models_dir / 'blend_2027-03-24.pkl').exists() }")
                print(f"pa_2027.parquet exists={ (proc_dir / 'pa_2027.parquet').exists() }")
                print(f"refresh markers={ [p.name for p in proc_dir.glob('.refreshed_*')] }")

                config = {
                    "orchestrator": {
                        "picks_dir": str(picks_dir),
                        "data_dir": str(proc_dir),
                        "models_dir": str(models_dir),
                        "heartbeat_path": str(tmp / "data/.heartbeat"),
                    },
                    "bluesky": {"dm_recipient": "test.bsky.social"},
                    "tiers": [{"name": "local", "type": "local", "data_dir": str(proc_dir), "models_dir": str(models_dir)}],
                    "scheduler": {
                        "pick_delivery": "dm",
                        "early_lock_gap": 0.03,
                        "contest_state_required": False,
                        "live_forward_capture_on_lock": False,
                    },
                    "health_checks": {"enabled": False},
                }

                # Try run_single_check before lineups post
                confirmed = set()
                chk_a = run_single_check("2027-03-24", [910001], confirmed, config, 0.03)
                print("run_single_check (pre-lineup):", {k: v for k, v in chk_a.items() if k != "pick_result"}, "pick_result=", chk_a.get("pick_result"))
                print("2027-03-24.json exists=", (picks_dir / "2027-03-24.json").exists(), "decision.json exists=", (picks_dir / "2027-03-24/decision.json").exists())

                # Try with contest_state_required=True and stale/missing contest_streak.json
                config_req = {**config, "scheduler": {**config["scheduler"], "contest_state_required": True}}
                chk_a_req = run_single_check("2027-03-24", [910001], confirmed, config_req, 0.03)
                print("run_single_check (pre-lineup, contest_state_required=True):", chk_a_req["pick_result"], "selection=", chk_a_req["selection"])

                # --- SCENARIO B: Opening Night 2027-03-24 AFTER lineups post ---
                print("\n=== SCENARIO B: Opening Night 2027-03-24 lineups confirmed ===")
                sched_0324_post, feeds_0324_post = make_live_schedule_and_feed(
                    "2027-03-24",
                    [{
                        "pk": 910001, "away_id": 108, "home_id": 119,
                        "away_pitcher": 99601, "home_pitcher": 6001,
                        "away_batters": on_away_batters, "home_batters": on_home_batters,
                        "away_posted": True, "home_posted": True, "venue_id": 10, "ump_id": 901,
                    }],
                )
                state_http["schedules_by_date"]["2027-03-24"] = sched_0324_post
                state_http["feeds_by_pk"].update(feeds_0324_post)

                preds_b, sel_b, tier_b = run_and_pick(config, "2027-03-24")
                print(f"preds_b shape={preds_b.shape} cols_na_pgame={int(preds_b['p_game_hit'].isna().sum())}")
                print("top 3 rows:", preds_b[["batter_id", "team", "p_game_hit", "days_rest", "park_drag_delta", "flags"]].head(3).to_dict("records"))
                print("debut batter row:", preds_b[preds_b["batter_id"] == 99109][["batter_id", "pitcher_id", "p_game_hit", "days_rest", "flags"]].to_dict("records"))
                print(f"sel_b: action={sel_b.action} source={sel_b.source} reason={sel_b.no_pick_reason} pick_result={bool(sel_b.pick_result)} decision={sel_b.decision}")

                # Check what _deliver_and_lock_pick + decision.json look like if sel_b made a pick
                sched_state = SchedulerState(
                    date="2027-03-24",
                    schedule_fetched_at="2027-03-24T10:00:00-04:00",
                    games=[{"game_pk": 910001, "game_time_et": "2027-03-24T19:10:00-04:00",
                            "lineup_confirmed": True, "is_doubleheader_game2": False}],
                    confirmed_game_pks=[910001],
                    runs_completed=[],
                    pick_locked=False,
                    pick_locked_at=None,
                    result_status=None,
                    next_wakeup=None,
                )
                confirmed2 = set()
                with patch("bts.scheduler._now_et", lambda: datetime(2027, 3, 24, 18, 0, tzinfo=ZoneInfo("America/New_York"))):
                    chk_b = run_single_check("2027-03-24", [910001], confirmed2, config, 0.03)
                    print(f"run_single_check (post-lineup): should_post={chk_b['should_post']} pick_name={chk_b['pick_name']} pick_p={chk_b['pick_p']} action={chk_b['selection'].action if chk_b.get('selection') else None}")
                    if chk_b["pick_result"] and not chk_b["pick_result"].locked and chk_b["should_post"]:
                        ok = _deliver_and_lock_pick(
                            chk_b["pick_result"].daily, config, picks_dir, sched_state,
                            "2027-03-24", "lineup-lock", selection=chk_b["selection"],
                        )
                        print(f"_deliver_and_lock_pick={ok} dm_sent={len(dms_sent)} decision_on_disk={load_decision('2027-03-24', picks_dir)}")

                # Check contest_state_required=True with NO contest_streak.json vs 2026 contest_streak.json
                try:
                    run_and_pick(config_req, "2027-03-24")
                    print("contest_state_required=True missing: DID NOT RAISE")
                except Exception as exc:
                    print(f"contest_state_required=True missing raised: {type(exc).__name__}: {exc}")

                # --- SCENARIO C: Day 2 (2027-03-25), 1 Final game in 2027 (910001) ---
                print("\n=== SCENARIO C: Day 2 (2027-03-25), 1 prior 2027 game ===")
                # Make game 910001 Final on 2027-03-24
                fin_0324 = make_historical_feed(
                    910001, "2027-03-24", 2027, 108, 119,
                    on_away_batters[:-1] + [109], on_home_batters, 5001, 6001, 10, 901, rng, game_type="R"
                )
                state_http["schedules_by_date"]["2027-03-24"] = {
                    "dates": [{
                        "date": "2027-03-24",
                        "games": [{
                            "gamePk": 910001,
                            "gameType": "R",
                            "officialDate": "2027-03-24",
                            "status": {"abstractGameCode": "F", "detailedState": "Final", "statusCode": "F"},
                        }],
                    }]
                }
                state_http["feeds_by_pk"][910001] = fin_0324
                # Team schedules for prior-lineup fallback
                for tid in (108, 119):
                    state_http["team_schedules"][tid] = state_http["schedules_by_date"]["2027-03-24"]
                for tid in (110, 111):
                    state_http["team_schedules"][tid] = {"dates": []}

                # Day 2 schedule: Game 910002 (108@119, unposted) + Game 910003 (110@111, unposted - teams with 0 prior 2027 games!)
                sched_0325_pre, feeds_0325_pre = make_live_schedule_and_feed(
                    "2027-03-25",
                    [
                        {
                            "pk": 910002, "away_id": 108, "home_id": 119,
                            "away_pitcher": 5002, "home_pitcher": 6002,
                            "away_batters": away_batters, "home_batters": home_batters,
                            "away_posted": False, "home_posted": False, "venue_id": 10, "ump_id": 901,
                        },
                        {
                            "pk": 910003, "away_id": 110, "home_id": 111,
                            "away_pitcher": 7001, "home_pitcher": 8001,
                            "away_batters": t3_batters, "home_batters": t4_batters,
                            "away_posted": False, "home_posted": False, "venue_id": 11, "ump_id": 902,
                        },
                    ],
                )
                state_http["schedules_by_date"]["2027-03-25"] = sched_0325_pre
                state_http["feeds_by_pk"].update(feeds_0325_pre)

                preds_c1, sel_c1, _ = run_and_pick(config, "2027-03-25")
                print(f"Day 2 pre-lineup: pa_2027 exists={(proc_dir / 'pa_2027.parquet').exists()} markers={[p.name for p in proc_dir.glob('.refreshed_*')]}")
                print(f"Day 2 pre-lineup: preds shape={preds_c1.shape} teams={sorted(preds_c1['team'].unique().tolist())} pks={sorted(preds_c1['game_pk'].unique().tolist())}")
                print(f"Day 2 pre-lineup sel: action={sel_c1.action} pick_result={bool(sel_c1.pick_result)} dd={bool(sel_c1.pick_result.daily.double_down) if sel_c1.pick_result else None}")
                print("Day 2 sample flags/rest:", preds_c1[["batter_id", "team", "days_rest", "flags", "p_game_hit"]].head(3).to_dict("records"))

                # Test should_lock when 108@119 IS posted on Day 2, while 110@111 is NOT posted (so 110@111 was omitted, not projected!)
                sched_0325_partial, feeds_0325_partial = make_live_schedule_and_feed(
                    "2027-03-25",
                    [
                        {
                            "pk": 910002, "away_id": 108, "home_id": 119,
                            "away_pitcher": 5002, "home_pitcher": 6002,
                            "away_batters": away_batters, "home_batters": home_batters,
                            "away_posted": True, "home_posted": True, "venue_id": 10, "ump_id": 901,
                        },
                        {
                            "pk": 910003, "away_id": 110, "home_id": 111,
                            "away_pitcher": 7001, "home_pitcher": 8001,
                            "away_batters": t3_batters, "home_batters": t4_batters,
                            "away_posted": False, "home_posted": False, "venue_id": 11, "ump_id": 902,
                        },
                    ],
                )
                state_http["schedules_by_date"]["2027-03-25"] = sched_0325_partial
                state_http["feeds_by_pk"].update(feeds_0325_partial)
                chk_c2 = run_single_check("2027-03-25", [910002, 910003], set(), config, 0.03)
                print(f"Day 2 partial (108@119 posted, 110@111 unposted & 0 prior 2027 games!): should_post={chk_c2['should_post']} pick_name={chk_c2['pick_name']}")

        finally:
            os.chdir(old_cwd)


if __name__ == "__main__":
    main()
