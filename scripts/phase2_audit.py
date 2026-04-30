#!/usr/bin/env python3
"""Phase 2 audit — single experiment forward+backward at multi-seed pool.

Env-var driven so the same script can run different (experiment, seed-set)
pairs across boxes in parallel.

Required env:
  AUDIT_EXPERIMENT=<name>          experiment registry name
  AUDIT_SEEDS=<csv int>            seeds to pool across (e.g. canonical-n10)
  AUDIT_OUT_DIR=<path>             output dir (relative to bts repo root)

Optional env:
  AUDIT_SEASONS=2024,2025          test seasons (default 2024,2025)

Output: AUDIT_OUT_DIR/result.json + per-step forward/backward logs.
"""
import json
import os
import sys
from pathlib import Path

os.environ["BTS_LGBM_DETERMINISTIC"] = "1"

import pandas as pd
from bts.experiment.registry import load_all_experiments, get_experiment
from bts.experiment.runner import run_selection
from bts.features.compute import compute_all_features

EXPERIMENT = os.environ["AUDIT_EXPERIMENT"]
SEEDS = [int(s) for s in os.environ["AUDIT_SEEDS"].split(",")]
OUT_DIR = Path(os.environ["AUDIT_OUT_DIR"])
TEST_SEASONS = [int(s) for s in os.environ.get("AUDIT_SEASONS", "2024,2025").split(",")]

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[audit] {EXPERIMENT} | seeds={len(SEEDS)} | seasons={TEST_SEASONS}", file=sys.stderr)
print(f"[audit] Loading PA data...", file=sys.stderr)
dfs = [pd.read_parquet(p) for p in sorted(Path("data/processed").glob("pa_*.parquet"))]
df = pd.concat(dfs, ignore_index=True)
df = compute_all_features(df)
print(f"[audit] PA frame: {len(df):,} rows", file=sys.stderr)

load_all_experiments()
exp = get_experiment(EXPERIMENT)
winners = [{"name": EXPERIMENT, "diff": {}, "passed": True}]
experiments_by_name = {EXPERIMENT: exp}

print(f"[audit] Phase 2 forward+backward, multi-seed n={len(SEEDS)}...", file=sys.stderr)
result = run_selection(
    winners, experiments_by_name, df, TEST_SEASONS,
    OUT_DIR, retrain_every=7, seeds=SEEDS,
)

out = OUT_DIR / "result.json"
out.write_text(json.dumps(result, indent=2, default=str))
print(f"\n[audit] Wrote {out}", file=sys.stderr)
print(f"[audit] included={result.get('included')}", file=sys.stderr)
for step in result.get("forward_log", []):
    t = step.get("t_stat", "n/a")
    print(f"  forward {step['name']}: pooled Δ={step.get('delta', 0):+.5f} t={t} kept={step.get('kept')}", file=sys.stderr)
for step in result.get("backward_log", []):
    t = step.get("t_stat", "n/a")
    print(f"  backward {step['name']}: pooled Δ={step.get('delta', 0):+.5f} t={t} kept={step.get('kept')}", file=sys.stderr)
