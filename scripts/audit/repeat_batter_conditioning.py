#!/usr/bin/env python3
"""Stage 1: repeat-batter conditioning of rank-1 calibration.

Mechanism test for the 2026-07-13 run-structure finding
(docs/audit/2026-07-13-dd-p-policy-value-sensitivity.md finding 5, open
thread): if long runs concentrate on recency-hot batters whose true rate
sits below the form-chasing estimate (a serial winner's curse), then rank-1
picks that REPEAT — the batter was already rank-1 on a recent prior slate
day — should realize BELOW stated, while fresh rank-1 picks stay calibrated.
This formulation tests the mechanism on every slate day (~908 distinct
dates), not just the thin run tail (63 runs >=15).

Stage-1 questions (prevalence-first, F9 discipline):
1. Prevalence — how often is rank-1 a repeat of the previous slate day
   (repeat_1) / any of the last 3 (repeat_3) / last 7 (repeat_7)?
   "Previous slate day" is position-based within a profile (the prior pick
   day), not calendar-based.
2. Calibration — mean(realized − stated) for repeat vs fresh, and the
   difference, with a (season, date)-cluster bootstrap CI: the 24 seeds
   re-use the same dates and mostly pick the same batter, so rows are ~24x
   duplicated; clusters carry all seeds of a date together.
3. Stated-p confound — repeats are selected for high stated p, so a
   p-level-dependent calibration error could masquerade as a repeat effect;
   the contrast is re-run within pooled stated-p quintile strata.
4. Per-season signs (5 seasons = the honest robustness unit here).

Pre-registered decision rule: proceed to stage 2 (run-conditional
decomposition) only if the repeat-vs-fresh gap difference is NEGATIVE
(repeats overconfident) with a 95% date-cluster bootstrap CI excluding 0,
and the direction holds in >=4 of 5 seasons. Otherwise the repeat-batter
mechanism is unsupported and the finding-5 candidates fall back to
schedule/regime structure and run-conditional miscalibration generally.

Uses the same estimated_pa profiles as the 7/13 sensitivity analysis
(CLAUDE.md PROFILE BASIS warning applies).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROFILE_ROOT = "data/hetzner_results/mdp_estpa_run"
BOOT_SEED = 20260714
REPEAT_KS = (1, 3, 7)


def load_rank1_frame(root: str = PROFILE_ROOT) -> tuple[pd.DataFrame, list[str]]:
    """One row per (file, slate day): the rank-1 pick with stated p and outcome."""
    files = sorted(glob.glob(f"{root}/**/backtest_*.parquet", recursive=True))
    if not files:
        raise SystemExit(f"no profile parquets under {root}")
    frames = []
    for idx, f in enumerate(files):
        season = int(re.search(r"backtest_(\d{4})\.parquet", f).group(1))
        seed = int(re.search(r"simulation_seed(\d+)", f).group(1))
        df = pd.read_parquet(f, columns=["date", "rank", "batter_id", "p_game_hit", "actual_hit"])
        r1 = df[df["rank"] == 1].sort_values("date")
        frames.append(
            pd.DataFrame(
                {
                    "file_idx": idx,
                    "season": season,
                    "seed": seed,
                    "date": r1["date"].to_numpy(),
                    "batter_id": r1["batter_id"].to_numpy(),
                    "stated": r1["p_game_hit"].to_numpy(dtype=float),
                    "hit": r1["actual_hit"].to_numpy(dtype=float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True), files


def add_repeat_flags(rank1: pd.DataFrame, ks: tuple[int, ...] = REPEAT_KS) -> pd.DataFrame:
    """Flag rank-1 picks whose batter was rank-1 on a recent prior slate day.

    Position-based within each file's date-sorted sequence: repeat_k is True
    when today's batter_id equals the rank-1 batter_id of ANY of the previous
    k slate days (rows), False on the first day(s). Same-file only.
    """
    out = rank1.sort_values(["file_idx", "date"]).reset_index(drop=True).copy()
    for k in ks:
        flags = np.zeros(len(out), dtype=bool)
        for _fi, g in out.groupby("file_idx", sort=False):
            ids = g["batter_id"].to_numpy()
            f = np.zeros(len(ids), dtype=bool)
            for j in range(1, len(ids)):
                lo = max(0, j - k)
                f[j] = ids[j] in ids[lo:j]
            flags[g.index.to_numpy()] = f
        out[f"repeat_{k}"] = flags
    return out


def gap_contrast(
    rank1: pd.DataFrame,
    flag: str,
    n_boot: int = 2000,
    seed: int = BOOT_SEED,
) -> dict:
    """(realized − stated) for repeat vs fresh + difference, date-cluster bootstrap.

    Clusters are (season, date): a resampled cluster carries every seed's row
    for that date, so the ~24x seed duplication cannot manufacture precision.
    """
    df = rank1.copy()
    df["gap"] = df["hit"] - df["stated"]
    rep = df[df[flag]]
    fresh = df[~df[flag]]
    point = {
        "n_repeat_rows": int(len(rep)),
        "n_fresh_rows": int(len(fresh)),
        "repeat_share": float(df[flag].mean()),
        "repeat_gap": float(rep["gap"].mean()),
        "fresh_gap": float(fresh["gap"].mean()),
        "repeat_stated_mean": float(rep["stated"].mean()),
        "fresh_stated_mean": float(fresh["stated"].mean()),
        "diff": float(rep["gap"].mean() - fresh["gap"].mean()),
    }

    clusters = df.groupby(["season", "date"], sort=False)
    keys = list(clusters.groups.keys())
    # per-cluster sufficient statistics: (sum gap, n) for repeat and fresh
    stats = np.zeros((len(keys), 4))
    for i, (key, g) in enumerate(clusters):
        m = g[flag].to_numpy()
        gap = g["gap"].to_numpy()
        stats[i] = [gap[m].sum(), m.sum(), gap[~m].sum(), (~m).sum()]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        take = rng.integers(0, len(keys), len(keys))
        s = stats[take].sum(axis=0)
        if s[1] == 0 or s[3] == 0:
            boots[b] = np.nan
            continue
        boots[b] = s[0] / s[1] - s[2] / s[3]
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    point.update(
        {
            "n_clusters": len(keys),
            "diff_ci95_cluster_boot": [float(lo), float(hi)],
            "boot_reps_used": int(len(boots)),
        }
    )

    per_season = {}
    for s, g in df.groupby("season"):
        r, f = g[g[flag]], g[~g[flag]]
        per_season[int(s)] = {
            "repeat_gap": float(r["gap"].mean()) if len(r) else float("nan"),
            "fresh_gap": float(f["gap"].mean()) if len(f) else float("nan"),
            "diff": float(r["gap"].mean() - f["gap"].mean()) if len(r) and len(f) else float("nan"),
            "repeat_share": float(g[flag].mean()),
        }
    point["per_season"] = per_season
    return point


def stratified_contrast(rank1: pd.DataFrame, flag: str, n_strata: int = 5) -> list[dict]:
    """Repeat-vs-fresh gap difference within pooled stated-p quantile strata."""
    df = rank1.copy()
    df["gap"] = df["hit"] - df["stated"]
    edges = np.quantile(df["stated"], [i / n_strata for i in range(1, n_strata)])
    df["stratum"] = np.digitize(df["stated"], edges)
    out = []
    for s, g in df.groupby("stratum"):
        r, f = g[g[flag]], g[~g[flag]]
        out.append(
            {
                "stratum": int(s),
                "stated_range": [float(g["stated"].min()), float(g["stated"].max())],
                "n_repeat": int(len(r)),
                "n_fresh": int(len(f)),
                "repeat_gap": float(r["gap"].mean()) if len(r) else float("nan"),
                "fresh_gap": float(f["gap"].mean()) if len(f) else float("nan"),
                "diff": (
                    float(r["gap"].mean() - f["gap"].mean()) if len(r) and len(f) else float("nan")
                ),
            }
        )
    return out


def live_primary_contrast(csv_path: str | Path, ks: tuple[int, ...] = REPEAT_KS) -> dict:
    """Directional repeat contrast on scored live primaries (2026 side-check).

    Input: the slot dataset CSV from scripts/audit/build_slot_dataset.py
    (run on the box). Primaries with terminal hit/miss outcomes, date order,
    same positional repeat semantics as the profiles. No bootstrap — n is
    tens; this is archived for the audit trail, labeled directional.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    prim = df[(df["slot"] == "pick") & df["outcome"].isin(["hit", "miss"])].sort_values("date")
    frame = pd.DataFrame(
        {
            "file_idx": 0,
            "season": 2026,
            "seed": 0,
            "date": prim["date"].to_numpy(),
            "batter_id": prim["batter_id"].to_numpy(),
            "stated": prim["p"].to_numpy(dtype=float),
            "hit": (prim["outcome"] == "hit").to_numpy(dtype=float),
        }
    )
    frame = add_repeat_flags(frame, ks=ks)
    frame["gap"] = frame["hit"] - frame["stated"]
    out = {
        "n_days": int(len(frame)),
        "date_range": [str(frame["date"].min()), str(frame["date"].max())],
        # full audit trail (review r2#4): the CSV's identity and every scored
        # row with its repeat assignment, so attributions are reproducible
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "rows": [
            {
                "date": str(r.date),
                "batter_id": int(r.batter_id),
                "stated": float(r.stated),
                "hit": int(r.hit),
                **{f"repeat_{k}": bool(getattr(r, f"repeat_{k}")) for k in ks},
            }
            for r in frame.itertuples(index=False)
        ],
    }
    for k in ks:
        flag = f"repeat_{k}"
        r, f = frame[frame[flag]], frame[~frame[flag]]
        out[flag] = {
            "n_repeat": int(len(r)),
            "n_fresh": int(len(f)),
            "repeat_gap": float(r["gap"].mean()) if len(r) else float("nan"),
            "fresh_gap": float(f["gap"].mean()) if len(f) else float("nan"),
            "diff": float(r["gap"].mean() - f["gap"].mean()) if len(r) and len(f) else float("nan"),
        }
    return out


def _fingerprint(files: list[str]) -> dict:
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode())
        h.update(hashlib.sha256(Path(f).read_bytes()).hexdigest().encode())
    return {"n_files": len(files), "combined_sha256": h.hexdigest()}


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=PROFILE_ROOT)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="docs/audit/2026-07-13-repeat-batter-stage1.json")
    ap.add_argument(
        "--live-csv",
        default=None,
        help="slot dataset CSV (build_slot_dataset.py on the box) for the directional 2026 side-check",
    )
    args = ap.parse_args(argv)

    rank1, files = load_rank1_frame(args.root)
    rank1 = add_repeat_flags(rank1)
    print(
        f"rank-1 frame: {len(rank1)} rows, {rank1.groupby(['season','date']).ngroups} "
        f"distinct (season, date) clusters, pooled gap (realized-stated) "
        f"{(rank1['hit'] - rank1['stated']).mean():+.4f}"
    )

    results = {"flags": {}}
    for k in REPEAT_KS:
        flag = f"repeat_{k}"
        c = gap_contrast(rank1, flag, n_boot=args.n_boot)
        results["flags"][flag] = {"contrast": c, "by_stated_stratum": stratified_contrast(rank1, flag)}
        lo, hi = c["diff_ci95_cluster_boot"]
        print(
            f"\n{flag}: share {c['repeat_share']:.1%} | gap repeat {c['repeat_gap']:+.4f} "
            f"vs fresh {c['fresh_gap']:+.4f} | diff {c['diff']:+.4f} "
            f"[95% cluster CI {lo:+.4f}, {hi:+.4f}] | stated means "
            f"{c['repeat_stated_mean']:.4f} vs {c['fresh_stated_mean']:.4f}"
        )
        for s, v in c["per_season"].items():
            print(
                f"    {s}: diff {v['diff']:+.4f} (repeat {v['repeat_gap']:+.4f} / "
                f"fresh {v['fresh_gap']:+.4f}, share {v['repeat_share']:.1%})"
            )
        print("    by stated-p stratum (diff, n_repeat):")
        for st in results["flags"][flag]["by_stated_stratum"]:
            print(
                f"      p in [{st['stated_range'][0]:.3f}, {st['stated_range'][1]:.3f}]: "
                f"diff {st['diff']:+.4f} (n_rep {st['n_repeat']})"
            )

    live = None
    if args.live_csv:
        live = live_primary_contrast(args.live_csv)
        print(f"\nlive 2026 primaries (directional, n={live['n_days']}):")
        for k in REPEAT_KS:
            v = live[f"repeat_{k}"]
            print(
                f"  repeat_{k}: n_rep {v['n_repeat']} | gap repeat {v['repeat_gap']:+.4f} "
                f"vs fresh {v['fresh_gap']:+.4f} | diff {v['diff']:+.4f}"
            )
    else:
        print(
            "\nNOTE: --live-csv omitted — the written artifact will lack the "
            "live_2026_primaries_directional block (the committed artifact has it)."
        )

    payload = {
        "generated_by": "scripts/audit/repeat_batter_conditioning.py",
        "git_head": _git_head(),
        "params": {"n_boot": args.n_boot, "boot_seed": BOOT_SEED, "repeat_ks": list(REPEAT_KS)},
        "inputs": _fingerprint(files),
        "results": results,
    }
    if live is not None:
        payload["live_2026_primaries_directional"] = live
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
