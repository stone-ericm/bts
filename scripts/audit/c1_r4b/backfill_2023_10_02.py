#!/usr/bin/env python3
"""C1 4b backfill of 2023-10-02 (gamePk 716404): Eric's ruling of 2026-10-05, register row C1-4b-2023-10-02.

Standalone on purpose: run it as a script, not with -m, from a worktree of the profile generator's source (commit
8522412, most likely what the 6/10 profile run executed), using that worktree's own venv and with that worktree as
the working directory. The bts it imports is then the generator's code. That worktree must contain no `data/raw`
and no probable-pitcher lookup cache: the profile boxes had neither, so `opp_bullpen_hr_30g` was missing for
historical rows, and the backfill must reproduce that.

What it does:
1. **Environment:** clear every BTS_* variable (the original run set none besides the two below), then set
   BTS_LGBM_DETERMINISTIC=1 before importing bts.
2. **Augmented PA data** in a scratch directory:
   - copy pa_2017..pa_2025 unchanged;
   - copy pa_2026 truncated to dates before 2026-06-10, with columns cut to pa_2023's schema;
   - parse the game's feed with the generator's own `parse_game_feed`, applying `build_season`'s filters
     (type R; the 7-inning doubleheader skip);
   - append the game's rows to pa_2023, cast to its schema.

   Every input file is hashed.
3. **Features:** computed once, exactly as `run_backtest` does.
4. **For each of the 24 seeds** (BTS_LGBM_RANDOM_STATE = seed), mirror `blend_walk_forward`'s loop body for test
   indices 175..182 only:
   - retrain at 175 (2023-09-25) and predict 9/25..10/01;
   - retrain at 182 (2023-10-02) and predict 10/02.
5. **Reproduction gate:** each seed's 9/25..10/01 rows must equal the retained W0.1 profile's rows exactly
   (`check_exact`). Only then are its 10/02 rows written as that seed's hashed supplement.

A failed seed writes no supplement.

    cd ~/projects/bts-gen8522412 && .venv/bin/python /path/to/backfill_2023_10_02.py --pa-dir ... --feed ... \\
        --profiles ... --seeds ... --out ...
"""
from __future__ import annotations

import os
import sys

for _k in [k for k in os.environ if k.startswith("BTS_")]:
    del os.environ[_k]
os.environ["BTS_LGBM_DETERMINISTIC"] = "1"

import argparse  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

GAME_PK = 716404
DAY = "2023-10-02"
CHECK_FROM, CHECK_TO = "2023-09-25", "2023-10-01"
IDX_CHECK, IDX_DAY = 175, 182
PA_2026_CUTOFF = "2026-06-10"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pa-dir", type=Path, required=True, help="production data/processed (read only)")
    ap.add_argument("--feed", type=Path, required=True, help="the receipted 716404.json.gz")
    ap.add_argument("--profiles", type=Path, required=True, help="the retained mdp_estpa_run root")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cwd = Path.cwd()
    if (cwd / "data" / "raw").exists() or (cwd / "data" / "models" / "probable_pitcher_lookup.json").exists():
        print("refusing: the generator worktree must have no data/raw and no probable_pitcher_lookup.json", file=sys.stderr)
        return 2
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        print(f"refusing: the generator worktree is dirty: {dirty[:200]}", file=sys.stderr)
        return 2

    import numpy as np
    import pandas as pd
    from bts.data import build
    from bts.features.compute import TRAIN_START_YEAR, compute_all_features
    from bts.simulate import backtest_blend as bb

    out = args.out
    work = out / "pa_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    manifest = {"ruling": "register row C1-4b-2023-10-02 (Eric 2026-10-05): backfill the real game",
                "note": ("reads the retained 2023-09-25..10-01 profile rows only for the exact-equality reproduction "
                         "check; computes no policy, replay or 4b result"),
                "generator_commit": commit, "env": {"BTS_LGBM_DETERMINISTIC": "1", "BTS_LGBM_RANDOM_STATE": "per seed"},
                "inputs": {}, "seeds": {}}

    # 2. augmented PA data
    ref_cols = pq_cols = None
    for p in sorted(args.pa_dir.glob("pa_20*.parquet")):
        b = p.read_bytes()
        manifest["inputs"][p.name] = sha(b)
        season = int(p.stem.split("_")[1])
        if season < 2017:
            continue
        df = pd.read_parquet(p)
        if season == 2023:
            ref_cols, ref_dtypes = list(df.columns), df.dtypes
        if season == 2026:
            df = df[pd.to_datetime(df["date"]) < pd.Timestamp(PA_2026_CUTOFF)]
            df = df[[c for c in df.columns if c != "is_resumed_portion"]]
        if season == 2023:
            fb = args.feed.read_bytes()
            manifest["inputs"]["feed_716404.json.gz"] = sha(fb)
            feed = json.loads(gzip.decompress(fb))
            info = feed.get("gameData", {}).get("game", {})
            if info.get("type") != "R" or int(feed.get("gamePk")) != GAME_PK:
                print("refusing: the feed is not the regular-season game 716404", file=sys.stderr)
                return 2
            if info.get("doubleHeader", "N") in ("Y", "S"):
                plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
                if max((x["about"]["inning"] for x in plays), default=9) <= 7:
                    print("refusing: build_season would skip this game (7-inning doubleheader)", file=sys.stderr)
                    return 2
            new = pd.DataFrame(build.parse_game_feed(feed))
            if (df["game_pk"] == GAME_PK).any():
                print("refusing: pa_2023 already contains 716404", file=sys.stderr)
                return 2
            new = new[ref_cols]
            for c in ref_cols:
                try:
                    new[c] = new[c].astype(ref_dtypes[c])
                except (TypeError, ValueError):
                    pass
            manifest["backfill_rows"] = int(len(new))
            manifest["backfill_dates"] = sorted(map(str, pd.to_datetime(new["date"]).dt.date.unique()))
            df = pd.concat([df, new], ignore_index=True)
        df.to_parquet(work / p.name, index=False)
        manifest["inputs"][f"work/{p.name}"] = sha((work / p.name).read_bytes())

    # 3. features, exactly as run_backtest
    dfs = [pd.read_parquet(p) for p in sorted(work.glob("pa_*.parquet"))]
    full = compute_all_features(pd.concat(dfs, ignore_index=True))

    # 4. mirror blend_walk_forward for test indices 175..182
    full = full.copy()
    full["date"] = pd.to_datetime(full["date"])
    season_data = full[full["season"] == 2023]
    test_start = season_data["date"].min()
    train_pool = full[(full["date"] < test_start) & (full["season"] >= TRAIN_START_YEAR)].copy()
    test_data = season_data.copy()
    test_dates = sorted(test_data["date"].unique())
    if len(test_dates) != 183 or str(pd.Timestamp(test_dates[IDX_CHECK]).date()) != CHECK_FROM \
            or str(pd.Timestamp(test_dates[IDX_DAY]).date()) != DAY:
        print(f"refusing: unexpected 2023 test dates ({len(test_dates)})", file=sys.stderr)
        return 2
    from bts.model.predict import BLEND_CONFIGS as blend_configs, LGB_PARAMS   # blend_walk_forward's defaults
    if not (LGB_PARAMS.get("deterministic") and LGB_PARAMS.get("force_row_wise")):
        print("refusing: deterministic LightGBM params are not active", file=sys.stderr)
        return 2
    cols_to_keep = list(bb.PROFILE_COLUMNS) + list(bb.ESTIMATED_PA_EXTRA_COLUMNS)
    all_ok = True
    for seed in args.seeds:
        os.environ["BTS_LGBM_RANDOM_STATE"] = str(seed)
        rows, blend, side, window = [], None, None, None
        for i in range(IDX_CHECK, IDX_DAY + 1):
            day = test_dates[i]
            day_data = test_data[test_data["date"] == day].copy()
            if i in (IDX_CHECK, IDX_DAY):
                available = pd.concat([train_pool, test_data[test_data["date"] < day]])
                window = available.copy()
                blend, side = bb._train_blend_for_day(available, blend_configs, LGB_PARAMS, cached_models=None)
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
            rows.append(gp[[c for c in cols_to_keep if c in gp.columns]])
        got = pd.concat(rows, ignore_index=True)
        retained = next(args.profiles.glob(f"*/simulation_seed{seed}/backtest_2023.parquet"))
        ref = pd.read_parquet(retained)
        ref = ref[(pd.to_datetime(ref["date"]) >= CHECK_FROM) & (pd.to_datetime(ref["date"]) <= CHECK_TO)]
        mine = got[(pd.to_datetime(got["date"]) >= CHECK_FROM) & (pd.to_datetime(got["date"]) <= CHECK_TO)]
        mine = mine[list(ref.columns)].reset_index(drop=True)
        ref = ref.reset_index(drop=True)
        rec = {"retained_profile_sha256": sha(retained.read_bytes()), "check_rows": int(len(ref))}
        try:
            pd.testing.assert_frame_equal(mine.astype(ref.dtypes.to_dict()), ref, check_exact=True)
            rec["reproduced"] = True
        except AssertionError as e:
            rec["reproduced"] = False
            rec["mismatch"] = str(e)[:800]
            num = [c for c in ref.columns if pd.api.types.is_float_dtype(ref[c])]
            if len(mine) == len(ref) and num:
                rec["max_abs_diff"] = float(np.nanmax(np.abs(mine[num].to_numpy(float) - ref[num].to_numpy(float))))
        if rec["reproduced"]:
            sup = got[pd.to_datetime(got["date"]) == pd.Timestamp(DAY)][list(ref.columns)].reset_index(drop=True)
            path = out / f"supplement_2023-10-02_seed{seed}.parquet"
            sup.to_parquet(path, index=False)
            rec.update({"supplement": path.name, "supplement_rows": int(len(sup)), "supplement_sha256": sha(path.read_bytes())})
        else:
            all_ok = False
        manifest["seeds"][str(seed)] = rec
        print(f"seed {seed}: reproduced={rec['reproduced']}", file=sys.stderr, flush=True)
        (out / "backfill_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    shutil.rmtree(work)
    manifest["all_reproduced"] = all_ok
    (out / "backfill_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
