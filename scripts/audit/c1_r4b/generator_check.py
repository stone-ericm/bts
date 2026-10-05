#!/usr/bin/env python3
"""C1 4b one-block generator reproduction check: Eric's ruling (a) of 2026-10-05, register row C1-4b-generator-commit.

The 4b profiles' generator commit was never recorded. The inferred generator is 8522412: HEAD when the 6/10 profile
run launched; the only later commit that day, 03c06cf, changed no `src/` file. This check re-runs that commit's
walk-forward on one retained block (seed 42, test days 2023-09-25..10-01) **with nothing added**, and compares the
output with the retained W0.1 rows exactly.
- **If the block is equal:** the register records 8522412 as the generator, reproduction-witnessed on that block.
- **If it is not:** the register records the generator as unretained, and the limit is disclosed.

**No result is read.** The retained rows are compared by program only. The manifest records the verdict, the
differing-cell counts per column, and a max absolute difference for the prediction columns only (never for the
outcome label). It holds no row values.

Standalone on purpose: run it as a script, not with -m, with the generator worktree as the working directory and that
worktree's own venv, so the `bts` it imports is the generator's code (checked: `bts.__file__` must be under
`./src`). The worktree must have no `data/raw` and no probable-pitcher lookup cache: the profile boxes had neither, so
`opp_bullpen_hr_30g` was missing for historical rows.

**What the 6/10 boxes ran, and how it is mirrored here:**
- **Source:** `src/` was the controller's tree (audit_driver rsync).
- **PA data:** `data/processed/` was pulled from bts-hetzner (`--data-relay`).
  - pa_2017..pa_2025 on the box are unchanged since 2026-04-11, so they are used as they are.
  - pa_2026 is the daily-refreshed file. It is truncated to dates before 2026-06-10, the launch day, and cut to
    pa_2023's columns (`is_resumed_portion` came later). The 2023 block cannot see 2026 rows (date-guarded features,
    training on earlier dates only).
- **Environment:** every BTS_* variable is cleared. BTS_LGBM_DETERMINISTIC=1 is set, and BTS_LGBM_RANDOM_STATE=42.
  OMP_NUM_THREADS=16 matches the cpx62 boxes' 16 vCPUs, because LGB_PARAMS sets no thread count.
- **Walk-forward:** `run_backtest` computes features once. `blend_walk_forward` (estimated_pa, top 10, no cache)
  retrains at test index 175 (175 % 7 == 0) and predicts 175..181.

    cd ~/projects/bts-gen8522412 && .venv/bin/python <c1 tree>/scripts/audit/c1_r4b/generator_check.py \\
        --pa-dir ~/projects/bts/data/processed --profiles ~/projects/bts/data/hetzner_results/mdp_estpa_run \\
        --out ~/projects/bts/data/hetzner_results/c1/r4b/generator_check/<stamp>
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

SEED = 42
SEASON = 2023
CHECK_FROM, CHECK_TO = "2023-09-25", "2023-10-01"
IDX_FROM, IDX_TO, N_TEST_DATES = 175, 181, 182
RETRAIN_EVERY = 7
PA_2026_CUTOFF = "2026-06-10"
THREADS = "16"
OUTCOME_COLUMNS = ("actual_hit",)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def prepare_env(environ) -> list[str]:
    """Clear every BTS_* variable, then pin determinism and the original thread count. Returns the cleared names."""
    removed = sorted(k for k in environ if k.startswith("BTS_"))
    for k in removed:
        del environ[k]
    environ["BTS_LGBM_DETERMINISTIC"] = "1"
    environ["OMP_NUM_THREADS"] = THREADS
    return removed


def cut_2026(df, ref_cols: list[str]):
    """pa_2026 as the 6/10 run saw it: dates before the launch day, pa_2023's columns in pa_2023's order."""
    import pandas as pd
    missing = [c for c in ref_cols if c not in df.columns]
    if missing:
        raise ValueError(f"pa_2026 lacks pa_2023 columns {missing}")
    return df.loc[pd.to_datetime(df["date"]) < pd.Timestamp(PA_2026_CUTOFF), list(ref_cols)]


def compare(mine, ref) -> dict:
    """Exact equality (DataFrame.equals: same columns, dtypes and values; NaN equals NaN in the same place), with
    diagnostics that carry no row values."""
    import numpy as np
    import pandas as pd
    out = {"rows": int(len(ref)), "rows_mine": int(len(mine)), "columns_equal": list(mine.columns) == list(ref.columns),
           "dtype_mismatch": [], "differing_cells": {}, "max_abs_diff": {}}
    mine, ref = mine.reset_index(drop=True), ref.reset_index(drop=True)
    if out["columns_equal"] and len(mine) == len(ref):
        for c in ref.columns:
            if mine[c].dtype != ref[c].dtype:
                out["dtype_mismatch"].append(c)
            a, b = mine[c].to_numpy(), ref[c].to_numpy()
            try:
                same = (a == b) | (pd.isna(a) & pd.isna(b))
            except (TypeError, ValueError):
                same = np.array([x == y for x, y in zip(a, b)])
            n = int((~np.asarray(same, bool)).sum())
            if n:
                out["differing_cells"][c] = n
                if c not in OUTCOME_COLUMNS and np.issubdtype(np.asarray(a).dtype, np.floating):
                    out["max_abs_diff"][c] = float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))
    out["reproduced"] = bool(mine.equals(ref)) and out["columns_equal"] and not out["dtype_mismatch"] \
        and not out["differing_cells"]
    return out


def main(argv=None) -> int:
    removed = prepare_env(os.environ)           # before numpy/lightgbm/bts load (OpenMP reads its thread count once)
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]   # never shadow modules from the c1 tree
    ap = argparse.ArgumentParser()
    ap.add_argument("--pa-dir", type=Path, required=True, help="production data/processed (read only)")
    ap.add_argument("--profiles", type=Path, required=True, help="the retained mdp_estpa_run root")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cwd = Path.cwd()
    if (cwd / "data" / "raw").exists() or (cwd / "data" / "models" / "probable_pitcher_lookup.json").exists():
        print("refusing: the generator worktree must have no data/raw and no probable_pitcher_lookup.json", file=sys.stderr)
        return 2
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        print(f"refusing: the generator worktree is dirty: {dirty[:200]}", file=sys.stderr)
        return 2
    os.environ["BTS_LGBM_RANDOM_STATE"] = str(SEED)

    import lightgbm
    import numpy as np
    import pandas as pd
    import pyarrow
    import bts
    if not Path(bts.__file__).resolve().is_relative_to(cwd / "src"):
        print(f"refusing: bts imports from {bts.__file__}, not this worktree's src", file=sys.stderr)
        return 2
    from bts.features.compute import TRAIN_START_YEAR, compute_all_features
    from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
    from bts.simulate import backtest_blend as bb
    if not (LGB_PARAMS.get("deterministic") and LGB_PARAMS.get("force_row_wise")):
        print("refusing: deterministic LightGBM params are not active", file=sys.stderr)
        return 2

    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"ruling": "register row C1-4b-generator-commit: Eric 2026-10-05, (a) check first",
                "note": "nothing added; the retained rows are compared by program only and no row value is recorded",
                "generator_commit": commit, "seed": SEED, "block": [CHECK_FROM, CHECK_TO],
                "env": {"cleared_bts_vars": removed, "BTS_LGBM_DETERMINISTIC": "1", "BTS_LGBM_RANDOM_STATE": str(SEED),
                        "OMP_NUM_THREADS": THREADS, "nproc": os.cpu_count()},
                "versions": {"lightgbm": lightgbm.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                             "pyarrow": pyarrow.__version__, "python": sys.version.split()[0]},
                "inputs": {}}

    # PA data exactly as the relay delivered it (see the module docstring)
    dfs, ref_cols = [], None
    for p in sorted(args.pa_dir.glob("pa_*.parquet")):
        b = p.read_bytes()
        manifest["inputs"][p.name] = sha(b)
        df = pd.read_parquet(io.BytesIO(b))
        if p.name == f"pa_{SEASON}.parquet":
            ref_cols = list(df.columns)
        if p.name == "pa_2026.parquet":
            if ref_cols is None:
                raise SystemExit("refusing: pa_2023 was not read before pa_2026")
            df = cut_2026(df, ref_cols)
            manifest["inputs"]["pa_2026_as_used"] = {"rows": int(len(df)),
                                                     "max_date": str(pd.to_datetime(df["date"]).max().date())}
        dfs.append(df)
    full = compute_all_features(pd.concat(dfs, ignore_index=True))   # run_backtest: features computed once

    # blend_walk_forward's body for test indices 175..181 (one retrain at 175)
    full = full.copy()
    full["date"] = pd.to_datetime(full["date"])
    season_data = full[full["season"] == SEASON]
    test_start = season_data["date"].min()
    train_pool = full[(full["date"] < test_start) & (full["season"] >= TRAIN_START_YEAR)].copy()
    test_data = season_data.copy()
    test_dates = sorted(test_data["date"].unique())
    got_dates = (len(test_dates), str(pd.Timestamp(test_dates[IDX_FROM]).date()), str(pd.Timestamp(test_dates[IDX_TO]).date()))
    if got_dates != (N_TEST_DATES, CHECK_FROM, CHECK_TO) or IDX_FROM % RETRAIN_EVERY:
        raise SystemExit(f"refusing: unexpected {SEASON} test dates {got_dates}")
    rows = []
    for i in range(IDX_FROM, IDX_TO + 1):
        day = test_dates[i]
        day_data = test_data[test_data["date"] == day].copy()
        if i == IDX_FROM:
            available = pd.concat([train_pool, test_data[test_data["date"] < day]])
            window = available.copy()
            blend, side = bb._train_blend_for_day(available, BLEND_CONFIGS, LGB_PARAMS, cached_models=None)
        scores = {}
        for name, (model, cols, predict_fn) in blend.items():
            try:
                scores[name] = predict_fn(model, day_data, cols)
            except Exception as e:  # noqa: BLE001 - mirrors blend_walk_forward exactly
                print(f"  ! {name} predict failed on {day}: {e}", file=sys.stderr)
                scores[name] = pd.Series(np.nan, index=day_data.index)
        day_data = bb._attach_pa_blend_scores(day_data, scores, side)
        gp = bb._estimated_pa_game_predictions(day_data, blend=blend, blend_pa_scores=scores,
                                               side_channel_names=side, model_train_window=window, top_n=10,
                                               capture_per_model=False)
        gp["date"] = pd.Timestamp(day).date()
        keep = list(bb.PROFILE_COLUMNS) + [c for c in bb.ESTIMATED_PA_EXTRA_COLUMNS if c in gp.columns]
        rows.append(gp[keep])
    buf = io.BytesIO()
    pd.concat(rows, ignore_index=True).to_parquet(buf, index=False)     # save_profiles' write, read back
    mine = pd.read_parquet(io.BytesIO(buf.getvalue()))

    retained = sorted(args.profiles.glob(f"*/simulation_seed{SEED}/backtest_{SEASON}.parquet"))
    if len(retained) != 1:
        raise SystemExit(f"refusing: expected one retained seed-{SEED} {SEASON} profile, found {len(retained)}")
    rb = retained[0].read_bytes()
    rel = f"{args.profiles.name}/{retained[0].relative_to(args.profiles)}"
    w0 = {ln.split()[1]: ln.split()[0] for ln in (args.profiles.parent / f"{args.profiles.name}.sha256").read_text().splitlines()
          if ln.strip()}
    if w0.get(rel) != sha(rb):
        raise SystemExit(f"refusing: {rel} does not match the W0.1 manifest")
    ref = pd.read_parquet(io.BytesIO(rb))
    in_block = lambda d: (pd.to_datetime(d["date"]) >= CHECK_FROM) & (pd.to_datetime(d["date"]) <= CHECK_TO)  # noqa: E731
    manifest["retained"] = {"path": rel, "sha256": sha(rb), "w0_manifest_match": True}
    manifest["result"] = compare(mine[in_block(mine)], ref[in_block(ref)])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"reproduced={manifest['result']['reproduced']}", file=sys.stderr, flush=True)
    return 0 if manifest["result"]["reproduced"] else 1


if __name__ == "__main__":
    sys.exit(main())
