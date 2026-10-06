"""C2 step 2a: the phase-level memory method (evidence README § "Phase-level memory method"; register row
C2-2a-cost-method). One fresh process per run; imports and input copies happen before the measured section; the
process's peak RSS is read at the phase's end. Every phase runs the side's real code; only the named stop or stand-in
differs, and it is the same on both sides.

    python -m tests.c2_2a.bench.phases measure --phase PHASE --inputs DIR   # prints one JSON line

Run from the side's repo root (the f882411 worktree or the candidate) with this file copied unchanged,
TZ=America/New_York OMP_NUM_THREADS=1.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tests.c2_2a.bench import bench as B
from tests.c2_2a.golden import world as W

SLATE_ROWS = 351          # the largest production slate of 2026 (box, read 2026-10-06; median 270)
PHASES = ("load", "cache", "save", "tail_off", "tail_on", "slate")


class _Stop(BaseException):
    pass


def _rss() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _is_candidate() -> bool:
    try:
        import bts.serving_witness as sw
    except ImportError:
        return False
    return hasattr(sw, "build")


def slate_frame() -> pd.DataFrame:
    """A deterministic full-size predictions frame with every slate row column and the values selection reads."""
    rng = np.random.default_rng(351)
    n = SLATE_ROWS
    p = np.sort(rng.uniform(0.55, 0.82, size=n))[::-1]
    return pd.DataFrame({
        "batter_id": 10000 + np.arange(n), "batter_name": [f"Batter {i}" for i in range(n)],
        "team": [f"T{i % 30:02d}" for i in range(n)], "game_pk": 900000 + (np.arange(n) // 18),
        "lineup": 1 + np.arange(n) % 9, "pitcher_id": 50000 + np.arange(n) // 9,
        "pitcher_name": [f"Pitcher {i // 9}" for i in range(n)], "p_game_hit": p, "p_game_blend": p,
        "p_hit_vs_starter": p / 3.2, "p_hit_vs_reliever": p / 3.4, "est_pas": 4.0 + (np.arange(n) % 9 < 4) * 0.4,
        "starter_pas": 2.8, "reliever_pas": 1.4, "flags": ["" if i % 7 else "PROJECTED lineup" for i in range(n)],
        "projected": [i % 7 == 0 for i in range(n)], "game_time": "2026-06-30T23:05:00Z", "status": "Pre-Game",
    })


def _pipeline_attrs() -> dict:
    """What the candidate's real run_pipeline attaches (model, six inputs, errors)."""
    return {"serving_errors": [], "serving_model": {"source": "cache", "sha256": "a" * 64},
            "serving_inputs": [{"file": f"pa_{s}.parquet", "bytes": 25_000_000, "sha256": "b" * 64}
                               for s in B.SEASONS_BIG]}


def _realistic_witness() -> dict:
    from bts import serving_witness as sw
    cal = {"enabled": True, "applied": True, "status": "applied",
           "pa_input": {"file": "pa_2026.parquet", "bytes": 13_000_000, "sha256": "c" * 64},
           "pick_inputs": [{"file": f"2026-{4 + i // 30:02d}-{1 + i % 28:02d}.json", "bytes": 2800, "sha256": "d" * 64}
                           for i in range(156)],
           "n_fit": 60, "samples": [{"file": "2026-06-01.json", "file_sha256": "e" * 64, "date": "2026-06-01",
                                     "slot": "pick", "batter_id": 10101, "p": 0.7712345, "y": 1} for _ in range(60)],
           "samples_sha256": "f" * 64,
           "map": {"X_thresholds": list(np.linspace(0.6, 0.9, 24)), "y_thresholds": list(np.linspace(0.5, 0.9, 24)),
                   "increasing": True, "out_of_bounds": "clip", "y_min": 0.0, "y_max": 1.0},
           "map_sha256": "0" * 64, "errors": []}
    return sw.build(model=_pipeline_attrs()["serving_model"], inputs=_pipeline_attrs()["serving_inputs"],
                    calibration=cal, errors=[])


def measure(phase: str, inputs: Path) -> dict:
    work = Path(tempfile.mkdtemp(prefix=f"c2-2a-phase-{phase}-"))
    shutil.copytree(inputs / "big" / "data", work / "data")
    if phase in ("tail_off", "tail_on"):
        (work / "data" / "models" / f"blend_{W.DATE}.pkl").unlink()        # the tail only: no cache load
    os.chdir(work)
    import bts.orchestrator as O
    import bts.slate as SL
    import bts.model.calibrate  # noqa: F401
    import sklearn.isotonic  # noqa: F401
    from bts.model import predict as P
    P._refresh_season_data = lambda *a, **k: None
    if phase == "tail_on":
        os.environ["BTS_USE_CALIBRATION"] = "1"
    else:
        os.environ.pop("BTS_USE_CALIBRATION", None)
    candidate = _is_candidate()
    prepared = {}
    if phase == "save":
        prepared["blend"] = B._cache_blend(B.CACHE_TARGET_BYTES)
    if phase in ("tail_off", "tail_on", "slate"):
        prepared["frame"] = slate_frame()
    if phase == "slate" and candidate:
        prepared["frame"].attrs["serving"] = _realistic_witness()
    gc.collect()
    rss_start = _rss()
    peak = {}
    t0 = time.perf_counter()
    try:
        if phase == "load":
            def stop(df):
                peak["rss"] = _rss()
                raise _Stop
            P.compute_all_features = stop
            P.run_pipeline(W.DATE, "data/processed", refresh_data=False)
        elif phase == "cache":
            def stop_entry(*a, **k):
                peak["rss"] = _rss()
                raise _Stop
            P.run_pipeline = stop_entry
            O.predict_local(W.DATE)
        elif phase == "save":
            P.save_blend(prepared["blend"], work / "saved.pkl")
        elif phase in ("tail_off", "tail_on"):
            def stand_in(*a, **k):
                df = prepared["frame"].copy()
                if candidate:
                    df.attrs.update(_pipeline_attrs())
                return df
            P.run_pipeline = stand_in
            out = O.predict_local(W.DATE)
            assert out is not None and len(out) == SLATE_ROWS
        elif phase == "slate":
            assert SL.save_slate(prepared["frame"], W.DATE, Path("data/picks"), "local") is not None
    except _Stop:
        pass
    elapsed = time.perf_counter() - t0
    end = _rss()
    shutil.rmtree(work, ignore_errors=True)
    return {"phase": phase, "candidate": candidate, "rss_start": rss_start, "rss_peak": peak.get("rss", end),
            "rss_end": end, "elapsed_s": elapsed, "rss_unit": "bytes" if sys.platform == "darwin" else "KiB"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["measure"])
    ap.add_argument("--phase", required=True, choices=PHASES)
    ap.add_argument("--inputs", type=Path, required=True)
    a = ap.parse_args(argv)
    if os.environ.get("TZ") != "America/New_York" or os.environ.get("OMP_NUM_THREADS") != "1":
        print("refusing: set TZ=America/New_York and OMP_NUM_THREADS=1", file=sys.stderr)
        return 2
    print(json.dumps(measure(a.phase, a.inputs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
