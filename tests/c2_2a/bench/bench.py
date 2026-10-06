"""C2 step 2a §6.1 cost benchmark: paired baseline (f882411) and candidate runs of `run_and_pick`, one fresh process
per run, identical synthetic inputs (design `docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md` §6.1).

    python -m tests.c2_2a.bench.bench prepare --out DIR             # once: the inputs (declared by `declare`)
    python -m tests.c2_2a.bench.bench measure --case CASE --inputs DIR   # one run; prints one JSON line
    python -m tests.c2_2a.bench.bench declare --inputs DIR          # the footprints, before measurement

Run `measure` from the repo root of the side being measured (the f882411 worktree or the candidate), with this file
copied unchanged, `TZ=America/New_York OMP_NUM_THREADS=1`. The driver (`drive.py`) pairs and interleaves the runs.

Inputs (representative of the box on 2026-10-06, read for sizes only: blend cache 9,298,511 bytes; 156 pick files,
437,574 bytes; PA parquets 17-28 MB each):
- **big**: six synthetic PA parquets of about 26 MB (the design's realistic size), a cached blend pickled to about
  9.3 MB, 156 resolved pick files of about 2.8 KB;
- **small**: the golden world's two small seasons, for the cold-train control with REAL LightGBM training.

Cases: `cold_small` (real training and save; the small real-training control), `warm_off` (cache, calibration off),
`warm_on` (calibration fitted and applied), `warm_unavailable` (calibration on, insufficient support: every pick read,
nothing applied). Training is never run in the warm cases (the cache is loaded).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import resource
import shutil
import sys
import tempfile
import time
from datetime import date as Date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tests.c2_2a.golden import fakes, world as W

N_TEAMS_BIG = 30
SEASONS_BIG = (2021, 2022, 2023, 2024, 2025, 2026)
CACHE_TARGET_BYTES = 9_298_511
N_PICKS = 156


# ---------------------------------------------------------------- inputs

def _big_rows(season: int, days: int) -> list[dict]:
    rng = np.random.default_rng(5000 + season)
    # Team, batter, pitcher and venue ids follow world.py's scheme, so today's slate (world.Router: teams 101-108)
    # has real feature history.
    teams = [(101 + i, f"T{i:02d}") for i in range(N_TEAMS_BIG)]
    start = Date(season, 3, 28) if season != 2026 else Date(2026, 3, 30)
    rows = []
    for di in range(days):
        day = (start + timedelta(days=di)).isoformat()
        if season == 2026 and day >= W.DATE:
            break
        perm = np.random.default_rng(season * 7919 + di).permutation(N_TEAMS_BIG)
        for g in range(N_TEAMS_BIG // 2):
            away, home = teams[perm[2 * g]], teams[perm[2 * g + 1]]
            pk = season * 1_000_000 + di * 100 + g
            for side, team, opp in (("away", away, home), ("home", home, away)):
                for pa in range(37):
                    k = pa % 9
                    bid, pid = team[0] * 100 + k + 1, opp[0] * 100 + 50 + (di % 5) + (10 if pa >= 25 else 0)
                    n = int(rng.integers(1, 7))
                    hit = bool(rng.random() < 0.22 + 0.1 * ((bid * 31) % 7) / 6)
                    calls = [str(c) for c in rng.choice(["B", "C", "S", "F"], size=n - 1)] + ["X" if hit else "S"]
                    rows.append({
                        "game_pk": pk, "date": day, "season": season, "batter_id": bid, "pitcher_id": pid,
                        "bat_side": "L" if bid % 2 else "R", "pitch_hand": "L" if pid % 2 else "R",
                        "lineup_position": k + 1, "is_home": side == "home", "hp_umpire_id": 600 + (di + g) % 70,
                        "venue_id": home[0] - 100, "pitch_count": n,
                        "pitch_types": [str(t) for t in rng.choice(["FF", "SL", "CH", "CU", "SI"], size=n)],
                        "pitch_calls": calls,
                        "pitch_px": [float(x) for x in rng.uniform(-1.6, 1.6, size=n)],
                        "pitch_pz": [float(x) for x in rng.uniform(1.0, 4.0, size=n)],
                        "sz_top": round(3.3 + 0.2 * rng.random(), 3), "sz_bottom": round(1.5 + 0.2 * rng.random(), 3),
                        "final_count_balls": int(min(3, calls.count("B"))),
                        "final_count_strikes": int(min(2, calls.count("C") + calls.count("S"))),
                        "launch_speed": round(float(rng.uniform(60, 110)), 1) if hit else None,
                        "launch_angle": round(float(rng.uniform(-20, 50)), 1) if hit else None,
                        "trajectory": "line_drive" if hit else None, "hardness": "hard" if hit else None,
                        "total_distance": round(float(rng.uniform(5, 420)), 1) if hit else None,
                        "pitch_speeds": [float(x) for x in rng.uniform(80, 99, size=n)],
                        "pitch_end_speeds": [round(float(x), 2) for x in rng.uniform(72, 90, size=n)],
                        "pitch_spin_rates": [int(x) for x in rng.integers(1800, 2700, size=n)],
                        "pitch_extensions": [round(float(x), 2) for x in rng.uniform(5.5, 7.0, size=n)],
                        "pitch_break_vertical": [round(float(x), 2) for x in rng.uniform(-40, 20, size=n)],
                        "pitch_break_horizontal": [round(float(x), 2) for x in rng.uniform(-15, 15, size=n)],
                        "fielding_catcher_id": opp[0] * 100 + 2,
                        "challenge_player_id": None, "challenge_role": None, "challenge_overturned": None,
                        "challenge_team_batting": None, "event_type": "single" if hit else "strikeout",
                        "is_hit": int(hit), "weather_temp": 70 + di % 20, "weather_wind_speed": di % 15,
                        "weather_wind_dir": "Out To CF", "roof_type": "Open", "atm_pressure": None, "humidity": None,
                        "is_resumed_portion": False,
                    })
    return rows


# The pickles below are this script's own generated cache files, read back under the same trust model as production's
# own blend cache (bts.model.predict.load_blend); nothing external is unpickled.
def _cache_blend(target: int) -> dict:
    blend = {**fakes.train_blend(None), "_model": fakes.train_model(None)}
    size = len(pickle.dumps(blend))
    pad = max(0, (target - size) // 8)
    blend["_model"].padding = np.arange(len(blend["_model"].padding) + pad, dtype=np.float64)
    return blend


def prepare(out: Path, days: int) -> None:
    """Write the big inputs (and the small golden world) under `out`."""
    big = out / "big" / "data"
    for d in ("processed", "models", "picks"):
        (big / d).mkdir(parents=True, exist_ok=True)
    for s in SEASONS_BIG:
        df = pd.DataFrame(_big_rows(s, days))
        df.to_parquet(big / "processed" / f"pa_{s}.parquet", index=False)
        print(f"  pa_{s}: {len(df)} rows, {(big / 'processed' / f'pa_{s}.parquet').stat().st_size} bytes",
              file=sys.stderr)
    blend = _cache_blend(CACHE_TARGET_BYTES)
    (big / "models" / f"blend_{W.DATE}.pkl").write_bytes(pickle.dumps(blend))
    # The box's 156 picks span more than the synthetic 2026's 92 game days: the rest come from late 2025 (outside the
    # calibration window, but read and hashed like every pick file).
    pa = pd.concat([pd.read_parquet(big / "processed" / f"pa_{s}.parquet") for s in (2025, 2026)])
    by_day = pa.groupby("date")
    days26 = sorted(by_day.groups)[-N_PICKS:]
    rng = np.random.default_rng(91)
    for i, day in enumerate(days26):
        g = by_day.get_group(day)
        bid = int(sorted(g["batter_id"].unique())[i % 40])
        pk = int(g.loc[g["batter_id"] == bid, "game_pk"].iloc[0])
        rec = {"date": day, "run_time": f"{day}T15:00:00+00:00", "result": "hit",
               "pick": {"batter_name": f"Batter {bid}", "batter_id": bid, "team": "T", "lineup_position": 1,
                        "pitcher_name": "P", "pitcher_id": None, "p_game_hit": round(0.7 + 0.2 * rng.random(), 4),
                        "flags": [], "projected_lineup": False, "game_pk": pk, "game_time": f"{day}T23:05:00Z",
                        "pitcher_team": None},
               "double_down": None, "runner_up": None, "notes": "x" * 2400}
        (big / "picks" / f"{day}.json").write_text(json.dumps(rec))
    (big / "picks" / "streak.json").write_text(json.dumps({"streak": 0, "saver_available": True,
                                                         "updated": "2026-06-30T08:00:00+00:00"}))
    small = out / "small"
    small.mkdir(parents=True, exist_ok=True)
    W.write_world(small, repo=Path.cwd())
    for name in ("mdp_policy.npz", "mdp_tail_policy.npz"):
        src = Path.cwd() / "data" / "models" / name
        if src.exists():
            shutil.copyfile(src, big / "models" / name)


def declare(inputs: Path) -> dict:
    """The footprints, reported before measurement (§6.1)."""
    out = {}
    for world in ("big", "small"):
        d = inputs / world / "data"
        parquets = sorted((d / "processed").glob("pa_*.parquet"))
        decoded = {}
        for p in parquets:
            df = pd.read_parquet(p)
            decoded[p.name] = {"rows": len(df), "file_bytes": p.stat().st_size,
                               "decoded_bytes": int(df.memory_usage(deep=True).sum())}
        picks = sorted((d / "picks").glob("20*.json"))
        cache = d / "models" / f"blend_{W.DATE}.pkl"
        out[world] = {"parquets": decoded, "pick_files": len(picks),
                      "pick_bytes": sum(p.stat().st_size for p in picks),
                      "cache": None if not cache.exists() else {
                          "bytes": cache.stat().st_size,
                          "models": len(pickle.loads(cache.read_bytes())) - 1}}
    return out


# ---------------------------------------------------------------- one measured run

class _Phases:
    def __init__(self):
        self.t: dict[str, list[float]] = {}

    def wrap(self, mod, name, key):
        real = getattr(mod, name, None)
        if real is None:
            return

        def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return real(*a, **k)
            finally:
                self.t.setdefault(key, []).append(time.perf_counter() - t0)
        setattr(mod, name, timed)


def measure(case: str, inputs: Path) -> dict:
    world = "small" if case == "cold_small" else "big"
    work = Path(tempfile.mkdtemp(prefix=f"c2-2a-bench-{case}-"))
    shutil.copytree(inputs / world / "data", work / "data")
    if case == "cold_small":
        (work / "data" / "models" / f"blend_{W.DATE}.pkl").unlink(missing_ok=True)
    if case == "warm_unavailable":
        for f in sorted((work / "data" / "picks").glob("20*.json"))[:-20]:
            f.unlink()                                     # 20 samples < 30: insufficient support
    os.chdir(work)
    import bts.orchestrator as O
    import bts.progress
    import bts.slate
    import bts.strategy
    import bts.util
    import bts.model.calibrate as C
    from bts.model import predict as P
    router = W.Router()
    P.urlopen = router
    bts.util.urlopen = router
    P._refresh_season_data = lambda *a, **k: None
    if case != "cold_small":
        P.train_model = lambda *a, **k: (_ for _ in ()).throw(AssertionError("warm cases never train"))
        P.train_blend = P.train_model
    if case in ("warm_on", "warm_unavailable"):
        os.environ["BTS_USE_CALIBRATION"] = "1"
    else:
        os.environ.pop("BTS_USE_CALIBRATION", None)
    ph = _Phases()
    marks = []
    real_mark = bts.progress.mark
    bts.progress.mark = lambda stage: (marks.append((stage, time.perf_counter())), real_mark(stage))[1]
    ph.wrap(O, "predict_local", "predict_local")
    ph.wrap(P, "run_pipeline", "run_pipeline")
    ph.wrap(P, "save_blend", "save")
    ph.wrap(C, "fit_calibrator_from_picks", "calibration_fit")
    ph.wrap(O, "_attach_serving_witness", "witness")
    ph.wrap(bts.slate, "save_slate", "slate")
    ph.wrap(bts.strategy, "select_pick", "selection")
    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    O.run_and_pick(json.loads(json.dumps({"orchestrator": {"picks_dir": "data/picks", "models_dir": "data/models"},
                                          "tiers": [{"name": "local", "type": "local"}],
                                          "scheduler": {"pick_delivery": "private"}})),
                   W.DATE, require_detailed_statuses=False)
    total = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stage_t = {}
    for (s, t), nxt in zip(marks, marks[1:] + [(None, None)]):
        if nxt[1] is not None:
            stage_t[s] = stage_t.get(s, 0.0) + (nxt[1] - t)
    slate = json.loads((work / "data" / "picks" / "slates" / f"{W.DATE}.json").read_text())
    shutil.rmtree(work, ignore_errors=True)
    return {"case": case, "total_s": total, "rss_before": rss_before, "rss_peak": rss_after,
            "rss_unit": "bytes" if sys.platform == "darwin" else "KiB", "phases": {k: sum(v) for k, v in ph.t.items()},
            "stages": stage_t, "slate_schema": slate.get("schema_version"),
            "calibration_status": ((slate.get("serving") or {}).get("calibration") or {}).get("status"),
            "router_unknown": router.unknown}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--days", type=int, default=180)
    m = sub.add_parser("measure")
    m.add_argument("--case", required=True, choices=["cold_small", "warm_off", "warm_on", "warm_unavailable"])
    m.add_argument("--inputs", type=Path, required=True)
    d = sub.add_parser("declare")
    d.add_argument("--inputs", type=Path, required=True)
    a = ap.parse_args(argv)
    if os.environ.get("TZ") != "America/New_York" or os.environ.get("OMP_NUM_THREADS") != "1":
        print("refusing: set TZ=America/New_York and OMP_NUM_THREADS=1", file=sys.stderr)
        return 2
    if a.cmd == "prepare":
        prepare(a.out, a.days)
    elif a.cmd == "declare":
        print(json.dumps(declare(a.inputs), indent=1))
    else:
        print(json.dumps(measure(a.case, a.inputs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
