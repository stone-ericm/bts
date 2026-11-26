"""v2.6 Route A — run all 6 v2.5 attribution cells with n_block_bootstrap=500."""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.mdp import solve_mdp
from bts.validate.ope import corrected_audit_pipeline


CELL_FLAGS = {
    "000": {"params_mode": "pooled",     "rho_pair_mode": "scalar",  "policy_mode": "global"},
    "010": {"params_mode": "pooled",     "rho_pair_mode": "per-bin", "policy_mode": "global"},
    "001": {"params_mode": "pooled",     "rho_pair_mode": "scalar",  "policy_mode": "per-fold"},
    "011": {"params_mode": "pooled",     "rho_pair_mode": "per-bin", "policy_mode": "per-fold"},
    "101": {"params_mode": "fold-local", "rho_pair_mode": "scalar",  "policy_mode": "per-fold"},
    "111": {"params_mode": "fold-local", "rho_pair_mode": "per-bin", "policy_mode": "per-fold"},
}


def _to_jsonable(obj):
    """Recursively convert numpy types + NaN to JSON-safe Python types."""
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def main():
    out_dir = Path("data/validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading parquets...")
    t0 = time.time()
    profiles = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob("data/simulation/profiles_seed*_season*.parquet")))
    pa_df = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob("data/simulation/pa_predictions_seed*_season*.parquet")))
    seasons = sorted(profiles["season"].unique().tolist())
    print(f"  profiles: {len(profiles):,} rows, pa_df: {len(pa_df):,} rows ({time.time()-t0:.1f}s)")
    print(f"  seasons: {seasons}")

    def _solve(corrected_bins):
        return solve_mdp(corrected_bins, season_length=153, late_phase_days=30)

    summary_rows = []
    for cell, flags in CELL_FLAGS.items():
        print(f"\n=== Cell {cell}: {flags} ===")
        t_cell = time.time()
        result = corrected_audit_pipeline(
            profiles, pa_df,
            fold_seasons=seasons,
            mdp_solve_fn=_solve,
            n_bootstrap=300,
            rho_pair_n_permutations=300,
            pa_n_bootstrap=300,
            seed=42,
            n_block_bootstrap=500,
            expected_block_length=7,
            **flags,
        )
        runtime = time.time() - t_cell
        print(f"  point: {result.point_estimate:.6f}")
        print(f"  ci: [{result.ci_lower}, {result.ci_upper}]")
        print(f"  runtime: {runtime/60:.1f} min")

        cell_out = {
            "cell": cell,
            "params_mode": flags["params_mode"],
            "rho_pair_mode": flags["rho_pair_mode"],
            "policy_mode": flags["policy_mode"],
            "n_block_bootstrap": 500,
            "expected_block_length": 7,
            "n_bootstrap": 300,
            "rho_pair_n_permutations": 300,
            "pa_n_bootstrap": 300,
            "seed": 42,
            "point": float(result.point_estimate),
            "ci_lower": _to_jsonable(result.ci_lower),
            "ci_upper": _to_jsonable(result.ci_upper),
            "n_folds": len(result.fold_metadata),
            "runtime_seconds": runtime,
            "fold_metadata": _to_jsonable(result.fold_metadata),
        }
        cell_path = out_dir / f"falsification_harness_v2.6_n500_cell{cell}.json"
        cell_path.write_text(json.dumps(cell_out, indent=2))
        print(f"  wrote {cell_path}")
        summary_rows.append({
            "cell": cell,
            **{k: flags[k] for k in flags},
            "point": float(result.point_estimate),
            "ci_lower": _to_jsonable(result.ci_lower),
            "ci_upper": _to_jsonable(result.ci_upper),
            "runtime_seconds": runtime,
        })

    summary_path = out_dir / "v2_6_n500_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    print(f"\n=== Wrote summary: {summary_path} ===")
    print("DONE all 6 cells.")


if __name__ == "__main__":
    main()
