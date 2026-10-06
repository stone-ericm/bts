"""C2 step 2a §5 gate: the deterministic synthetic world shared by the golden generator (run at f882411) and the
candidate comparison (design `docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md` §5.1).

Everything here is generated from fixed seeds and constants, with no dependency on the code under test, so both sides
see byte-identical inputs: two seasons of schema-complete synthetic plate appearances, the MLB API responses for the
serving date (schedule, live feeds, a prior game for projected lineups), a history of resolved picks for calibration,
and fixed stand-ins for the trained models (training is the leaf the design stubs, so LightGBM thread nondeterminism
cannot make the comparison flaky).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date as Date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATE = "2026-06-30"
TEAMS = [(101 + i, abbr) for i, abbr in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"))]
# Today's games: (game_pk, away index, home index, first pitch UTC). ET = UTC-4 in June.
TODAY_GAMES = [
    (900001, 0, 1, "2026-06-30T17:10:00Z"),   # 13:10 ET
    (900002, 2, 3, "2026-06-30T23:05:00Z"),   # 19:05 ET
    (900003, 4, 5, "2026-06-30T23:40:00Z"),   # 19:40 ET
    (900004, 6, 7, "2026-07-01T02:10:00Z"),   # 22:10 ET
]
PRIOR_GAME_PK = 899999
POSTED_TEAMS = {101, 102, 103, 104, 105, 106}     # 107 and 108 have no posted lineup: projected from the prior game
EVENTS_HIT = ("single", "double", "triple", "home_run")
EVENTS_OUT = ("field_out", "strikeout", "force_out", "grounded_into_double_play")
EVENTS_OTHER = ("walk", "hit_by_pitch", "sac_fly")


def batter_ids(team_id: int) -> list[int]:
    return [team_id * 100 + k for k in range(1, 10)]


def starters(team_id: int) -> list[int]:
    return [team_id * 100 + 50 + k for k in range(1, 6)]


def relievers(team_id: int) -> list[int]:
    return [team_id * 100 + 60 + k for k in range(1, 4)]


def catcher(team_id: int) -> int:
    return team_id * 100 + 2


def _skill(bid: int) -> float:
    return 0.17 + 0.15 * ((bid * 7919) % 97) / 96.0


def _pitch_factor(pid: int) -> float:
    return 0.85 + 0.3 * ((pid * 104729) % 89) / 88.0


def _game_days(season: int) -> list[Date]:
    if season == 2025:
        start, end, step = Date(2025, 4, 1), Date(2025, 9, 27), 2
    else:
        start, end, step = Date(2026, 3, 30), Date(2026, 6, 29), 1
    out, d = [], start
    while d <= end:
        out.append(d)
        d += timedelta(days=step)
    return out


def _pa_rows(season: int) -> list[dict]:
    rng = np.random.default_rng(1000 + season)
    rows = []
    for di, day in enumerate(_game_days(season)):
        perm = np.random.default_rng(season * 1000 + di).permutation(len(TEAMS))
        for g in range(4):
            away, home = TEAMS[perm[2 * g]], TEAMS[perm[2 * g + 1]]
            game_pk = season * 100000 + di * 10 + g
            ump = 500 + (di + g) % 6
            temp = int(60 + (di * 3 + g) % 30)
            wind = int((di + 2 * g) % 15)
            wind_dir = ("Out To CF", "In From LF", "L To R", "Out To RF")[(di + g) % 4]
            roof = "Dome" if home[0] == 108 else "Open"
            for side, team, opp in (("away", away, home), ("home", home, away)):
                starter = starters(opp[0])[di % 5]
                for pa in range(36):
                    k = pa % 9
                    bid = batter_ids(team[0])[k]
                    pid = starter if pa < 24 else relievers(opp[0])[pa % 3]
                    p_hit = min(0.6, _skill(bid) * _pitch_factor(pid))
                    is_hit = bool(rng.random() < p_hit)
                    n = int(rng.integers(1, 7))
                    calls = [str(c) for c in rng.choice(["B", "C", "S", "F"], size=n - 1)]
                    if is_hit:
                        event, last = EVENTS_HIT[int(rng.integers(0, 4))], "X"
                    else:
                        r = rng.random()
                        event = EVENTS_OTHER[int(rng.integers(0, 3))] if r < 0.12 else EVENTS_OUT[int(rng.integers(0, 4))]
                        last = "S" if event == "strikeout" else ("B" if event == "walk" else "X")
                    calls.append(last)
                    in_play = last == "X"
                    rows.append({
                        "game_pk": game_pk, "date": day.isoformat(), "season": season,
                        "batter_id": bid, "pitcher_id": pid,
                        "bat_side": "L" if bid % 2 else "R", "pitch_hand": "L" if pid % 2 else "R",
                        "lineup_position": k + 1, "is_home": side == "home", "hp_umpire_id": ump,
                        "venue_id": home[0] - 100, "pitch_count": n,
                        "pitch_types": [str(t) for t in rng.choice(["FF", "SL", "CH", "CU"], size=n)],
                        "pitch_calls": calls,
                        "pitch_px": [round(float(x), 3) for x in rng.uniform(-1.6, 1.6, size=n)],
                        "pitch_pz": [round(float(x), 3) for x in rng.uniform(1.0, 4.0, size=n)],
                        "sz_top": round(3.3 + 0.2 * rng.random(), 3), "sz_bottom": round(1.5 + 0.2 * rng.random(), 3),
                        "final_count_balls": int(min(3, calls.count("B"))),
                        "final_count_strikes": int(min(2, calls.count("C") + calls.count("S"))),
                        "launch_speed": round(float(rng.uniform(60, 110)), 1) if in_play else None,
                        "launch_angle": round(float(rng.uniform(-20, 50)), 1) if in_play else None,
                        "trajectory": ("ground_ball", "line_drive", "fly_ball")[int(rng.integers(0, 3))] if in_play else None,
                        "hardness": ("soft", "medium", "hard")[int(rng.integers(0, 3))] if in_play else None,
                        "total_distance": round(float(rng.uniform(5, 420)), 1) if in_play else None,
                        "pitch_speeds": [round(float(x), 1) for x in rng.uniform(80, 99, size=n)],
                        "pitch_end_speeds": [round(float(x), 1) for x in rng.uniform(72, 90, size=n)],
                        "pitch_spin_rates": [int(x) for x in rng.integers(1800, 2700, size=n)],
                        "pitch_extensions": [round(float(x), 2) for x in rng.uniform(5.5, 7.0, size=n)],
                        "pitch_break_vertical": [round(float(x), 1) for x in rng.uniform(-40, 20, size=n)],
                        "pitch_break_horizontal": [round(float(x), 1) for x in rng.uniform(-15, 15, size=n)],
                        "fielding_catcher_id": catcher(opp[0]),
                        "challenge_player_id": None, "challenge_role": None, "challenge_overturned": None,
                        "challenge_team_batting": None,
                        "event_type": event, "is_hit": int(is_hit),
                        "weather_temp": temp, "weather_wind_speed": wind, "weather_wind_dir": wind_dir,
                        "roof_type": roof, "atm_pressure": None, "humidity": None, "is_resumed_portion": False,
                    })
    return rows


def pa_frame(season: int) -> pd.DataFrame:
    return pd.DataFrame(_pa_rows(season))


def _pick_record(day: str, slot_pick: dict, *, result="hit", double_down=None, slot_results=None) -> dict:
    rec = {"date": day, "run_time": f"{day}T15:00:00+00:00", "pick": slot_pick, "double_down": double_down,
           "runner_up": None, "result": result}
    if slot_results is not None:
        rec["slot_results"] = slot_results
    return rec


def _slot_pick(bid: int, p: float, game_pk: int, day: str) -> dict:
    team = next(t for t in TEAMS if t[0] == bid // 100)
    return {"batter_name": f"Batter {bid}", "batter_id": bid, "team": team[1], "lineup_position": bid % 100,
            "pitcher_name": "Synthetic Pitcher", "pitcher_id": None, "p_game_hit": p, "flags": [],
            "projected_lineup": False, "game_pk": game_pk, "game_time": f"{day}T23:05:00Z", "pitcher_team": None}


def pick_history(pa_2026: pd.DataFrame, n_days: int = 40) -> dict[str, dict | str]:
    """Resolved picks for calibration: {filename: record or raw text}. Covers primary + double-down, a void slot,
    a malformed file, an unresolved day and an out-of-window day (design §5.3)."""
    out: dict[str, dict | str] = {}
    by_day = pa_2026.groupby("date")
    days = sorted(by_day.groups)[-n_days:]
    rng = np.random.default_rng(77)
    for i, day in enumerate(days):
        g = by_day.get_group(day)
        bids = sorted(g["batter_id"].unique())
        b1, b2 = int(bids[(i * 5) % len(bids)]), int(bids[(i * 5 + 11) % len(bids)])
        pk1 = int(g.loc[g["batter_id"] == b1, "game_pk"].iloc[0])
        pk2 = int(g.loc[g["batter_id"] == b2, "game_pk"].iloc[0])
        p1 = round(0.70 + 0.18 * float(rng.random()), 4)
        p2 = round(0.68 + 0.18 * float(rng.random()), 4)
        dd = _slot_pick(b2, p2, pk2, day) if i % 3 == 0 else None
        slot_results = {"double_down": "void"} if i == 6 else None
        out[f"{day}.json"] = _pick_record(day, _slot_pick(b1, p1, pk1, day), double_down=dd,
                                          slot_results=slot_results)
    out["2026-04-02.json"] = _pick_record("2026-04-02", _slot_pick(10101, 0.9, 2026000001, "2026-04-02"))
    out["2026-06-28x-unresolved.json"] = _pick_record("2026-06-28", _slot_pick(10102, 0.77, 1, "2026-06-28"),
                                                      result=None)
    out["2026-06-27x-malformed.json"] = '{"date": "2026-06-27", "result": "hit", "pick": '
    return out


def _schedule_game(game_pk, away, home, game_date, *, status="Pre-Game", code="P"):
    return {"gamePk": game_pk, "gameType": "R", "officialDate": DATE, "gameDate": game_date,
            "status": {"abstractGameCode": code, "detailedState": status, "codedGameState": "P"},
            "teams": {"away": {"team": {"id": away[0], "name": away[1]},
                               "probablePitcher": {"id": starters(away[0])[0], "fullName": f"Starter {away[1]}"}},
                      "home": {"team": {"id": home[0], "name": home[1]},
                               "probablePitcher": {"id": starters(home[0])[0], "fullName": f"Starter {home[1]}"}}},
            "venue": {"id": home[0] - 100}}


def schedule() -> dict:
    return {"dates": [{"date": DATE, "games": [_schedule_game(pk, TEAMS[a], TEAMS[h], t)
                                               for pk, a, h, t in TODAY_GAMES]}]}


def _players(team_id: int, posted: bool) -> dict:
    out = {}
    for k, bid in enumerate(batter_ids(team_id), start=1):
        p = {"person": {"id": bid, "fullName": f"Batter {bid}"}}
        if posted:
            p["battingOrder"] = str(k * 100)
        out[f"ID{bid}"] = p
    return out


def feed(game_pk: int) -> dict:
    if game_pk == PRIOR_GAME_PK:
        away, home, posted = TEAMS[6], TEAMS[7], (True, True)
    else:
        _, a, h, _ = next(g for g in TODAY_GAMES if g[0] == game_pk)
        away, home = TEAMS[a], TEAMS[h]
        posted = (away[0] in POSTED_TEAMS, home[0] in POSTED_TEAMS)
    return {"gameData": {"venue": {"id": home[0] - 100, "fieldInfo": {"roofType": "Dome" if home[0] == 108 else "Open"}},
                         "weather": {"temp": "74", "wind": "9 mph, Out To CF"},
                         "officials": [{"officialType": "Home Plate", "official": {"id": 503}}],
                         "teams": {"away": {"abbreviation": away[1], "id": away[0]},
                                   "home": {"abbreviation": home[1], "id": home[0]}}},
            "liveData": {"boxscore": {"teams": {"away": {"players": _players(away[0], posted[0])},
                                                "home": {"players": _players(home[0], posted[1])}}},
                         "plays": {"allPlays": []}}}


def team_schedule(team_id: int) -> dict:
    away, home = TEAMS[6], TEAMS[7]
    return {"dates": [{"date": "2026-06-29", "games": [
        _schedule_game(PRIOR_GAME_PK, away, home, "2026-06-29T23:05:00Z", status="Final", code="F")]}]}


class _Resp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Router:
    """The fake MLB API: every pick-path URL the code under test fetches. Unknown URLs raise and are recorded."""

    def __init__(self, *, statuses: dict | None = None):
        self.calls: list[str] = []
        self.unknown: list[str] = []
        self.statuses = statuses or {}

    def __call__(self, req, timeout=None, *a, **k):
        url = req if isinstance(req, str) else getattr(req, "full_url", str(req))
        self.calls.append(url)
        if "/feed/live" in url:
            pk = int(url.split("/game/")[1].split("/")[0])
            return _Resp(json.dumps(feed(pk)).encode())
        if "/schedule" in url and "teamId=" in url:
            tid = int(url.split("teamId=")[1].split("&")[0])
            return _Resp(json.dumps(team_schedule(tid)).encode())
        if "/schedule" in url and f"date={DATE}" in url:
            sched = schedule()
            for g in sched["dates"][0]["games"]:
                if g["gamePk"] in self.statuses:
                    g["status"].update(self.statuses[g["gamePk"]])
            return _Resp(json.dumps(sched).encode())
        if "/schedule" in url and "date=" in url:
            return _Resp(json.dumps({"dates": []}).encode())        # any other day: no games
        self.unknown.append(url)
        raise OSError(f"golden world: no route for {url}")


def write_world(root: Path, *, repo: Path, calibration_history: bool = True, streak: int = 0,
                saver_available: bool = True) -> dict:
    """Create `root/data/{processed,models,picks}`; returns {relative path: sha256} of every input written."""
    data = root / "data"
    for d in ("processed", "models", "picks"):
        (data / d).mkdir(parents=True, exist_ok=True)
    frames = {s: pa_frame(s) for s in (2025, 2026)}
    for s, df in frames.items():
        df.to_parquet(data / "processed" / f"pa_{s}.parquet", index=False)
    for name in ("mdp_policy.npz", "mdp_tail_policy.npz"):
        src = repo / "data" / "models" / name
        if src.exists():
            shutil.copyfile(src, data / "models" / name)
    picks = data / "picks"
    if calibration_history:
        for name, rec in pick_history(frames[2026]).items():
            (picks / name).write_text(rec if isinstance(rec, str) else json.dumps(rec))
    (picks / "streak.json").write_text(json.dumps({"streak": streak, "saver_available": saver_available,
                                                   "updated": "2026-06-30T08:00:00+00:00"}))
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(data.rglob("*")) if p.is_file()}
