#!/usr/bin/env python3
"""C1 4b one-block generator reproduction check: Eric's ruling (a) of 2026-10-05, register row C1-4b-generator-commit.

The 4b profiles' generator commit was never recorded. The inferred generator is 8522412: HEAD when the 6/10 profile
run launched; the only later commit that day, 03c06cf, changed no `src/` file. This check re-runs that commit's
walk-forward on one retained block (seed 42, test days 2023-09-25..10-01) **with nothing added**, and compares the
output with the retained W0.1 rows.

**Statuses** (exit codes 0 / 1 / 2):
- `reproduced`: both blocks hold the seven dates with ranks 1–10 each, and they are equal (exact DataFrame
  value/dtype/order equality after a parquet round-trip; colocated NaNs and signed zeros compare equal; this is not
  parquet byte equality).
- `unequal`: a valid, fully covered block that differs. Under the ruling the generator is then recorded as unknown;
  this does not disprove 8522412.
- `refused`: an environment or input problem (identity, pins, versions, coverage). Says nothing about the generator.

**No result is read.** The comparison is made by program only. The manifest holds no row value. Its diagnostics
(differing-cell counts and float magnitudes) cover the non-outcome columns only. The outcome label (`actual_hit`)
affects only the overall equality verdict: no count, position, value or magnitude of it is recorded or printed
(code review r2 N1).

**Refusals before the reference is read** (r2 N3):
- the working tree must be the clean generator commit 8522412, and `bts` must import from its `./src`;
- the W0.1 manifest's own digest must equal the registered pin;
- the retained file must match its W0.1 line;
- the installed lightgbm, numpy, pandas and pyarrow must equal the generator's `uv.lock`.

The worktree must have no `data/raw` and no probable-pitcher lookup cache: the profile boxes had neither, so
`opp_bullpen_hr_30g` was missing for historical rows.

**What the 6/10 boxes ran, and how it is mirrored here:**
- **Source:** `src/` was the controller's tree (audit_driver rsync).
- **PA data:** `data/processed/` was pulled from bts-hetzner (`--data-relay`).
  - pa_2017..pa_2025 on the box are unchanged since 2026-04-11, so they are used as they are; their 6/10 bytes were
    not retained, and the input hashes recorded here are today's.
  - pa_2026 is the daily-refreshed file. It is truncated to dates before 2026-06-10, the launch day, and cut to
    pa_2023's columns (`is_resumed_portion` came later). The 2023 block cannot see 2026 rows (date-guarded features,
    training on earlier dates only).
- **Environment:** every BTS_* variable is cleared, then BTS_LGBM_DETERMINISTIC=1 and BTS_LGBM_RANDOM_STATE=42 are
  set. **The environment reconstruction is incomplete** (r2 N2). LGB_PARAMS sets no thread count, so LightGBM's
  sklearn API passes `num_threads` = the physical-core count; `OMP_NUM_THREADS` does not change it. The 6/10 boxes'
  effective count and their native builds were not retained. The manifest records the count each trained model
  actually used.
- **Walk-forward:** `run_backtest` computes features once. `blend_walk_forward` (estimated_pa, top 10, no cache)
  retrains at test index 175 (175 % 7 == 0) and predicts 175..181.

Standalone on purpose: run it as a script, not with -m, with the generator worktree as the working directory and that
worktree's own venv.

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
from datetime import date, timedelta
from pathlib import Path

GENERATOR_COMMIT = "85224124f97a0d4ee8da3059ff54b1bad228d44e"
W0_MANIFEST_SHA = "d63fefffff0a8868de4d21c1aa680e352f9ee4badff158bd2a8a01c5e359e7c6"
SEED = 42
SEASON = 2023
CHECK_FROM, CHECK_TO = "2023-09-25", "2023-10-01"
BLOCK_DATES = [str(date.fromisoformat(CHECK_FROM) + timedelta(days=i)) for i in range(7)]
RANKS = 10
IDX_FROM, IDX_TO, N_TEST_DATES = 175, 181, 182
RETRAIN_EVERY = 7
PA_2026_CUTOFF = "2026-06-10"
OUTCOME_COLUMNS = ("actual_hit",)
LOCKED = ("lightgbm", "numpy", "pandas", "pyarrow")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def prepare_env(environ) -> list[str]:
    """Clear every BTS_* variable, then pin determinism. Returns the cleared names (values are never recorded)."""
    removed = sorted(k for k in environ if k.startswith("BTS_"))
    for k in removed:
        del environ[k]
    environ["BTS_LGBM_DETERMINISTIC"] = "1"
    return removed


def cut_2026(df, ref_cols: list[str]):
    """pa_2026 as the 6/10 run saw it: dates before the launch day, pa_2023's columns in pa_2023's order."""
    import pandas as pd
    missing = [c for c in ref_cols if c not in df.columns]
    if missing:
        raise ValueError(f"pa_2026 lacks pa_2023 columns {missing}")
    return df.loc[pd.to_datetime(df["date"]) < pd.Timestamp(PA_2026_CUTOFF), list(ref_cols)]


def lock_versions(lock_text: str, names) -> dict:
    import tomllib
    pkgs = {p["name"]: p.get("version") for p in tomllib.loads(lock_text).get("package", [])}
    return {n: pkgs.get(n) for n in names}


def version_mismatches(installed: dict, locked: dict) -> list[str]:
    return [f"{n}: installed {installed[n]}, lock {locked.get(n)}" for n in installed if installed[n] != locked.get(n)]


def block_coverage(df) -> list[str]:
    """The block must hold exactly the seven registered dates, each with ranks 1..10 once (date and rank only)."""
    import pandas as pd
    if len(df) == 0:
        return ["empty block"]
    problems = []
    days = sorted(str(x) for x in pd.to_datetime(df["date"]).dt.date.unique())
    if days != BLOCK_DATES:
        problems.append(f"dates {days[:2]}..{days[-2:]} ({len(days)}) are not the registered seven")
    for d, g in df.groupby(pd.to_datetime(df["date"]).dt.date):
        if sorted(int(r) for r in g["rank"]) != list(range(1, RANKS + 1)):
            problems.append(f"{d}: ranks are not 1..{RANKS}")
    return problems


def compare(mine, ref) -> dict:
    """Exact DataFrame value/dtype/order equality (DataFrame.equals after an index reset: colocated NaNs and signed
    zeros compare equal). The diagnostics carry no row value and cover the non-outcome columns only."""
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
            if n and c not in OUTCOME_COLUMNS:
                out["differing_cells"][c] = n
                if np.issubdtype(np.asarray(a).dtype, np.floating):
                    out["max_abs_diff"][c] = float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))
    out["reproduced"] = bool(mine.equals(ref)) and out["columns_equal"] and not out["dtype_mismatch"] \
        and not out["differing_cells"]
    return out


def status(result: dict, mine, ref) -> tuple[str, list[str]]:
    """Coverage on both sides comes first: an empty or partial block is a refusal, never a reproduction."""
    problems = [f"generated: {p}" for p in block_coverage(mine)] + [f"retained: {p}" for p in block_coverage(ref)]
    if problems:
        return "refused", problems
    return ("reproduced" if result["reproduced"] else "unequal"), []


def main(argv=None) -> int:
    removed = prepare_env(os.environ)           # before numpy/lightgbm/bts load
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]   # never shadow modules from the c1 tree
    ap = argparse.ArgumentParser()
    ap.add_argument("--pa-dir", type=Path, required=True, help="production data/processed (read only)")
    ap.add_argument("--profiles", type=Path, required=True, help="the retained mdp_estpa_run root")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"ruling": "register row C1-4b-generator-commit: Eric 2026-10-05, (a) check first",
                "note": "nothing added; the retained rows are compared by program only and no row value is recorded",
                "check_script_sha256": sha(Path(__file__).read_bytes()), "seed": SEED, "block": [CHECK_FROM, CHECK_TO],
                "environment_reconstruction": ("incomplete: the 6/10 boxes' effective LightGBM thread count and native "
                                               "builds were not retained; see threads_used"),
                "inputs": {}}

    def finish(state: str, reasons: list[str] | None = None) -> int:
        manifest["status"] = state
        if reasons:
            manifest["reasons"] = reasons
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"status={state} {reasons or ''}", file=sys.stderr, flush=True)
        return {"reproduced": 0, "unequal": 1}.get(state, 2)

    cwd = Path.cwd()
    if (cwd / "data" / "raw").exists() or (cwd / "data" / "models" / "probable_pitcher_lookup.json").exists():
        return finish("refused", ["the generator worktree must have no data/raw and no probable_pitcher_lookup.json"])
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    commit = git("rev-parse", "HEAD")
    manifest["generator_commit"] = commit
    if commit != GENERATOR_COMMIT:
        return finish("refused", [f"HEAD {commit} is not the generator commit {GENERATOR_COMMIT}"])
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", "src", "pyproject.toml", "uv.lock")
    if dirty:
        return finish("refused", [f"the generator worktree is dirty: {dirty[:200]}"])
    os.environ["BTS_LGBM_RANDOM_STATE"] = str(SEED)
    manifest["env"] = {"cleared_bts_vars": removed, "BTS_LGBM_DETERMINISTIC": "1", "BTS_LGBM_RANDOM_STATE": str(SEED),
                       "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "nproc": os.cpu_count()}

    import lightgbm
    import numpy as np
    import pandas as pd
    import pyarrow
    import bts
    lock_bytes = (cwd / "uv.lock").read_bytes()
    installed = {"lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                 "pyarrow": pyarrow.__version__}
    manifest["versions"] = {**installed, "python": sys.version.split()[0], "uv_lock_sha256": sha(lock_bytes)}
    bad = version_mismatches(installed, lock_versions(lock_bytes.decode(), LOCKED))
    if bad:
        return finish("refused", bad)
    if not Path(bts.__file__).resolve().is_relative_to(cwd / "src"):
        return finish("refused", [f"bts imports from {bts.__file__}, not this worktree's src"])
    from bts.features.compute import TRAIN_START_YEAR, compute_all_features
    from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS
    from bts.simulate import backtest_blend as bb
    if not (LGB_PARAMS.get("deterministic") and LGB_PARAMS.get("force_row_wise")):
        return finish("refused", ["deterministic LightGBM params are not active"])

    # the reference identity, before any reference row is read
    w0_path = args.profiles.parent / f"{args.profiles.name}.sha256"
    w0_bytes = w0_path.read_bytes()
    if sha(w0_bytes) != W0_MANIFEST_SHA:
        return finish("refused", ["the W0.1 manifest's digest is not the registered pin"])
    retained = sorted(args.profiles.glob(f"*/simulation_seed{SEED}/backtest_{SEASON}.parquet"))
    if len(retained) != 1:
        return finish("refused", [f"expected one retained seed-{SEED} {SEASON} profile, found {len(retained)}"])
    rb = retained[0].read_bytes()
    rel = f"{args.profiles.name}/{retained[0].relative_to(args.profiles)}"
    w0 = dict(reversed(ln.split(None, 1)) for ln in w0_bytes.decode().splitlines() if ln.strip())
    if w0.get(rel) != sha(rb):
        return finish("refused", [f"{rel} does not match the W0.1 manifest"])
    manifest["retained"] = {"path": rel, "sha256": sha(rb), "w0_manifest_sha256": W0_MANIFEST_SHA}

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
                return finish("refused", ["pa_2023 was not read before pa_2026"])
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
        return finish("refused", [f"unexpected {SEASON} test dates {got_dates}"])
    rows = []
    for i in range(IDX_FROM, IDX_TO + 1):
        day = test_dates[i]
        day_data = test_data[test_data["date"] == day].copy()
        if i == IDX_FROM:
            available = pd.concat([train_pool, test_data[test_data["date"] < day]])
            window = available.copy()
            blend, side = bb._train_blend_for_day(available, BLEND_CONFIGS, LGB_PARAMS, cached_models=None)
            manifest["threads_used"] = {name: (getattr(getattr(m, "booster_", None), "params", {}) or {}).get("num_threads")
                                        for name, (m, _, _) in blend.items()}
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
    ref = pd.read_parquet(io.BytesIO(rb))
    in_block = lambda d: (pd.to_datetime(d["date"]) >= CHECK_FROM) & (pd.to_datetime(d["date"]) <= CHECK_TO)  # noqa: E731
    mine_b, ref_b = mine[in_block(mine)], ref[in_block(ref)]
    manifest["result"] = compare(mine_b, ref_b)
    state, problems = status(manifest["result"], mine_b, ref_b)
    return finish(state, problems)


if __name__ == "__main__":
    sys.exit(main())
